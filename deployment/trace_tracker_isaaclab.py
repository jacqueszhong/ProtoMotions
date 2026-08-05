# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""IsaacLab ground-truth trace harness for tracker deployment debugging.

Runs a tracker checkpoint in IsaacLab on a single pinned motion, one env,
deterministically, and records **what training actually produces** so the
standalone deployment drivers can be measured against it rather than against
the MotionLib reference.

Why this is a separate script and not a flag on ``inference_agent.py``
------------------------------------------------------------------------
``MimicEvaluator.simple_test_policy``
(``protomotions/agents/evaluators/base_evaluator.py``) is an unhooked
``while True`` on the training hot path -- there is nowhere to hang a recorder
without editing the loop every algorithm shares.  This harness also has to own
its ``import_simulator_before_torch("isaaclab")`` + ``AppLauncher`` boilerplate
at module scope (IsaacLab must be imported before torch), which is exactly the
shape ``inference_agent.py`` already has and cannot host twice.

What it emits
-------------
``trace_isaaclab.json``
    Per-control-step rows in the canonical
    :data:`deployment.state_utils.TRACE_COLUMNS` schema -- byte-comparable with
    ``test_tracker_isaacsim.py --trace-out`` and ``test_tracker_mujoco.py
    --trace-out``.

``context_isaaclab.npz``
    Per-control-step arrays: every ONNX-contract context tensor (the 13 keys in
    the exported YAML's ``_runtime.obs_context_keys``), each actor observation,
    the assembled observation vector, the post-normalizer vector, ``mean_action``,
    ``processed_action``, and the full robot state trajectory.  This is the input
    to ``deployment/check_onnx_parity.py`` (Stage 2) and the action tape for
    ``test_tracker_isaacsim.py --action-tape`` (Stage 3).

``init_state.json``
    The post-reset state read back **from the simulator**, not reconstructed.
    Reading it back captures the ``ref_respawn_offset`` z bump, the sampled XY
    location and any FK settling in one shot, so the open-loop replay can start
    from the identical initial condition instead of reasoning about how it was
    built.

Alignment is load-bearing
-------------------------
Every row is recorded **before** ``env.step()``, from the state that produced
that step's action, with ``frame = round(motion_time / env.dt)``.  That is
where ``TrackerPolicy._record_trace`` sits inside ``_compute_action`` in the
Isaac Sim driver, so rows with equal ``frame`` describe the same instant.

A note on the +0.05 m spawn offset
-----------------------------------
``MimicControl.populate_context`` offsets the reference body positions into
world space via
``get_spawn_to_ref_pose_offset_with_terrain_height_correction``, which resolves
z from the *terrain*, so IsaacLab's ``ref_h`` equals the raw MotionLib height
the deployment drivers use (measured: identical to 0.0 at frame 0).  The robot,
however, is spawned at ``ref + env.config.ref_respawn_offset``
(``base_env/env.py``), which is 0.05 m on the G1.  So an IsaacLab episode starts
5 cm *above* its own reference and settles onto it over the first ~5 control
steps, while the deployment drivers start flush.  ``init_state.json`` records
the post-reset root position and ``respawn_root_offset`` so Stage 3 can
reproduce or ablate that initial condition rather than infer it.

Usage
-----
::

    python deployment/trace_tracker_isaaclab.py \\
        --checkpoint results/g1_walk_box/score_based.ckpt \\
        --motion-file results/g1_walk_box/g1_walk_box.pt \\
        --out-dir /tmp/stage1 --headless
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path as _Path

# Ensure the repo root is importable so `deployment.*` resolves regardless of
# where this is invoked from -- same guard as test_tracker_isaacsim.py.
_REPO_ROOT = _Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Record an IsaacLab ground-truth trace for a tracker checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", required=True, help="Path to the .ckpt to run")
    p.add_argument(
        "--motion-file",
        default=None,
        help="Motion .pt to override the checkpoint's own motion file",
    )
    p.add_argument(
        "--motion-index",
        type=int,
        default=0,
        help="Clip index to pin (via motion_manager.subset_method=[index])",
    )
    p.add_argument("--out-dir", required=True, help="Directory for the three artifacts")
    p.add_argument(
        "--max-steps",
        type=int,
        default=2000,
        help="Stop after this many control steps even if the episode has not ended",
    )
    p.add_argument(
        "--headless", action="store_true", default=False, help="Run without a viewport"
    )
    p.add_argument("--seed", type=int, default=0, help="Seed for torch/numpy")
    p.add_argument(
        "--action-tape",
        default=None,
        help=(
            "Replay the mean_action sequence from a previous run's "
            "context_isaaclab.npz instead of querying the policy. This is the "
            "control experiment for the standalone driver's open-loop replay: it "
            "measures how much of the divergence is the open-loop test itself "
            "rather than a physics difference."
        ),
    )
    return p.parse_args()


args = _parse_args()

# IsaacLab must be imported before torch -- same contract as inference_agent.py.
from protomotions.utils.simulator_imports import (  # noqa: E402
    import_simulator_before_torch,
)

AppLauncher = import_simulator_before_torch("isaaclab")

import json  # noqa: E402
import logging  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from lightning.fabric import Fabric  # noqa: E402

from deployment.state_utils import make_trace_row, summarize_trace  # noqa: E402
from protomotions.utils.fabric_config import FabricConfig  # noqa: E402
from protomotions.utils.hydra_replacement import get_class  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s", force=True)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Context resolution
# ---------------------------------------------------------------------------

#: Context paths the exported ONNX contract consumes, from the deployment
#: YAML's ``_runtime.obs_context_keys``.  Eight of these become ONNX inputs;
#: the rest are folded into constants during tracing, but are recorded anyway
#: because a mismatch in a "passthrough" value is exactly the kind of thing
#: that makes the baked constants wrong.
CONTEXT_KEYS = (
    "current.anchor_pos",
    "current.anchor_rot",
    "current.dof_pos",
    "current.dof_vel",
    "current.root_local_ang_vel",
    "historical.processed_actions",
    "mimic.future_anchor_ang_vel",
    "mimic.future_anchor_pos",
    "mimic.future_anchor_rot",
    "mimic.future_anchor_vel",
    "mimic.future_dof_pos",
    "mimic.future_dof_vel",
    "mimic.ref_anchor_pos",
)


def resolve_context_path(path: str, context):
    """Resolve a dotted context path (e.g. ``"mimic.future_dof_pos"``) to a tensor."""
    obj = context
    for attr in path.split("."):
        obj = getattr(obj, attr)
    return obj


def to_np(tensor) -> np.ndarray:
    """Detach a tensor to a float32 NumPy array on the host."""
    return np.asarray(tensor.detach().cpu().numpy(), dtype=np.float32)


# ---------------------------------------------------------------------------
# Determinism gate
# ---------------------------------------------------------------------------


def assert_deterministic(env_config, robot_config, simulator_config) -> None:
    """Verify the inference config is already deterministic; never silently fix it.

    ``resolved_configs_inference.pt`` is expected to carry
    ``init_start_prob=1.0`` (episodes begin at motion frame 0),
    ``domain_randomization=None`` and ``reset_noise=None``.  Setting them here
    instead of checking would paper over a **stale checkpoint** whose frozen
    configs still carry training-time randomization -- which would make every
    number this harness produces unreproducible for reasons invisible in the
    output.  So: assert, and log what was asserted.
    """
    problems = []

    init_start_prob = env_config.motion_manager.init_start_prob
    if init_start_prob != 1.0:
        problems.append(
            f"motion_manager.init_start_prob={init_start_prob} (expected 1.0; "
            "episodes would not start at motion frame 0)"
        )
    if simulator_config.domain_randomization is not None:
        problems.append(
            f"simulator.domain_randomization={simulator_config.domain_randomization} "
            "(expected None)"
        )
    if robot_config.reset_noise is not None:
        problems.append(f"robot.reset_noise={robot_config.reset_noise} (expected None)")

    if problems:
        raise SystemExit(
            "Inference configs are not deterministic:\n  - "
            + "\n  - ".join(problems)
            + "\nThis is a stale-checkpoint symptom -- re-export the inference "
            "configs (--create-config-only) rather than overriding here."
        )

    log.info(
        "Determinism gate OK: init_start_prob=1.0, domain_randomization=None, "
        "reset_noise=None"
    )


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------


class TraceRecorder:
    """Accumulates per-control-step trace rows and context/state arrays."""

    def __init__(self, actor_in_keys: list, anchor_idx: int, dt: float) -> None:
        self.actor_in_keys = list(actor_in_keys)
        self.anchor_idx = anchor_idx
        self.dt = dt
        self.trace: list = []
        self.arrays: dict = {}

    def _append(self, name: str, value: np.ndarray) -> None:
        self.arrays.setdefault(name, []).append(value)

    def record_pre_step(self, env, context, obs_td, mean_action) -> None:
        """Record everything derivable from the state that produced ``mean_action``.

        Called after the model forward but **before** ``env.step`` -- the
        tensordict already carries the normalizer output by then (see
        ``MLPWithConcat.forward``), and the context still describes the state the
        action was computed from.
        """
        motion_time = float(env.motion_manager.motion_times[0].item())
        frame = int(round(motion_time / self.dt))

        cur, mimic = context.current, context.mimic
        root_pos = to_np(cur.root_pos[0])
        anchor_rot = to_np(cur.anchor_rot[0])
        dof_pos = to_np(cur.dof_pos[0])
        dof_vel = to_np(cur.dof_vel[0])

        ref_pos = to_np(mimic.ref_state.rigid_body_pos[0])
        ref_rot = to_np(mimic.ref_state.rigid_body_rot[0])
        ref_dof_pos = to_np(mimic.ref_state.dof_pos[0])

        self.trace.append(
            make_trace_row(
                loop=0,
                frame=frame,
                root_h=float(root_pos[2]),
                ref_h=float(ref_pos[0, 2]),
                anchor_rot_xyzw=anchor_rot,
                ref_anchor_rot_xyzw=ref_rot[self.anchor_idx],
                dof_pos=dof_pos,
                ref_dof_pos=ref_dof_pos,
                dof_vel=dof_vel,
            )
        )

        self._append("frame", np.asarray(frame, dtype=np.int64))
        self._append("motion_time", np.asarray(motion_time, dtype=np.float32))

        # ONNX-contract context tensors, env 0.
        for key in CONTEXT_KEYS:
            self._append(
                f"ctx__{key.replace('.', '_')}",
                to_np(resolve_context_path(key, context)[0]),
            )

        # Actor observations, the assembled vector, and the normalizer output.
        obs_parts = []
        for key in self.actor_in_keys:
            value = to_np(obs_td[key][0])
            self._append(f"obs__{key}", value)
            obs_parts.append(value.reshape(-1))
        self._append("obs_concat", np.concatenate(obs_parts))

        norm_key = f"norm_{self.actor_in_keys[0]}"
        if norm_key in obs_td.keys():
            self._append("obs_norm", to_np(obs_td[norm_key][0]))

        self._append("mean_action", to_np(mean_action[0]))

        # Full state trajectory -- the ground truth everything downstream is
        # measured against.
        self._append("state__root_pos", root_pos)
        self._append("state__root_rot", to_np(cur.root_rot[0]))
        self._append("state__root_vel", to_np(cur.root_vel[0]))
        self._append("state__root_ang_vel", to_np(cur.root_ang_vel[0]))
        self._append("state__dof_pos", dof_pos)
        self._append("state__dof_vel", dof_vel)
        self._append("state__anchor_rot", anchor_rot)
        self._append("state__rigid_body_pos", to_np(cur.rigid_body_pos[0]))
        self._append("state__rigid_body_rot", to_np(cur.rigid_body_rot[0]))

        # Reference, with and without the respawn/terrain offset, so the two
        # harnesses can be diffed without the 0.05 m bump confusing the result.
        self._append("ref__dof_pos", ref_dof_pos)
        self._append("ref__root_pos", ref_pos[0])
        self._append("ref__anchor_rot", ref_rot[self.anchor_idx])

    def record_post_step(self, env) -> None:
        """Record the action the environment actually applied for this step."""
        self._append("processed_action", to_np(env._current_processed_action[0]))

    def write(self, out_dir: Path, metadata: dict) -> None:
        trace_path = out_dir / "trace_isaaclab.json"
        with open(trace_path, "w") as f:
            json.dump(self.trace, f)
        log.info(
            f"\n=== Tracking trace ({len(self.trace)} control steps) -> {trace_path} ===\n"
            + summarize_trace(self.trace)
        )

        stacked = {k: np.stack(v) for k, v in self.arrays.items()}
        # Ragged only if a step failed mid-record; catch it loudly rather than
        # shipping an npz whose rows do not line up.
        lengths = {k: len(v) for k, v in stacked.items()}
        if len(set(lengths.values())) != 1:
            raise RuntimeError(f"Recorded arrays have mismatched lengths: {lengths}")

        npz_path = out_dir / "context_isaaclab.npz"
        np.savez_compressed(npz_path, **stacked, **metadata)
        log.info(f"Context/state arrays ({len(self.trace)} steps) -> {npz_path}")


def dump_init_state(env, out_dir: Path, joint_names: list) -> None:
    """Read the post-reset state back from the simulator and write it to JSON.

    Read back rather than reconstructed: this is what PhysX actually holds after
    ``env.reset()`` has applied the reference pose, the sampled XY spawn, the
    ``ref_respawn_offset`` z bump and any FK settling.  Stage 3 writes it into
    the standalone driver verbatim so the open-loop replay starts from the same
    initial condition instead of from motion frame 0.
    """
    state = env.simulator.get_robot_state()
    root_pos = to_np(state.rigid_body_pos[0, 0])
    root_rot = to_np(state.rigid_body_rot[0, 0])
    init = {
        "joint_names": list(joint_names),
        "root_pos": root_pos.tolist(),
        "root_rot_xyzw": root_rot.tolist(),
        "root_lin_vel": to_np(state.rigid_body_vel[0, 0]).tolist(),
        "root_ang_vel": to_np(state.rigid_body_ang_vel[0, 0]).tolist(),
        "dof_pos": to_np(state.dof_pos[0]).tolist(),
        "dof_vel": to_np(state.dof_vel[0]).tolist(),
        "respawn_root_offset": to_np(env.respawn_root_offset[0]).tolist(),
        "ref_respawn_offset": float(env.config.ref_respawn_offset),
        "motion_id": int(env.motion_manager.motion_ids[0].item()),
        "motion_time": float(env.motion_manager.motion_times[0].item()),
    }
    path = out_dir / "init_state.json"
    with open(path, "w") as f:
        json.dump(init, f, indent=2)
    log.info(
        f"Post-reset state -> {path}  root_pos={root_pos.round(4).tolist()} "
        f"respawn_offset={init['respawn_root_offset']}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    checkpoint = Path(args.checkpoint)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    resolved_path = checkpoint.parent / "resolved_configs_inference.pt"
    if not resolved_path.exists():
        raise SystemExit(f"Could not find resolved configs at {resolved_path}")
    log.info(f"Loading resolved configs from {resolved_path}")
    resolved = torch.load(resolved_path, map_location="cpu", weights_only=False)

    robot_config = resolved["robot"]
    simulator_config = resolved["simulator"]
    terrain_config = resolved.get("terrain")
    scene_lib_config = resolved["scene_lib"]
    motion_lib_config = resolved["motion_lib"]
    env_config = resolved["env"]
    agent_config = resolved["agent"]

    current_simulator = simulator_config._target_.split(".")[-3]
    if current_simulator != "isaaclab":
        from protomotions.simulator.factory import update_simulator_config_for_test

        log.info(f"Switching simulator '{current_simulator}' -> 'isaaclab'")
        simulator_config = update_simulator_config_for_test(
            current_simulator_config=simulator_config,
            new_simulator="isaaclab",
            robot_config=robot_config,
        )

    assert_deterministic(env_config, robot_config, simulator_config)

    # One env, one clip, pinned. `subset_method=[i]` requires len == num_envs.
    simulator_config.num_envs = 1
    simulator_config.headless = args.headless
    env_config.motion_manager.subset_method = [args.motion_index]
    if args.motion_file is not None:
        motion_lib_config.motion_file = args.motion_file
    log.info(
        f"Pinned motion index {args.motion_index} of "
        f"{motion_lib_config.motion_file} on 1 env"
    )

    fabric = Fabric(
        **FabricConfig(
            accelerator="gpu", devices=1, num_nodes=1, loggers=[], callbacks=[]
        ).as_kwargs()
    )
    fabric.launch()

    app_launcher = AppLauncher(
        {"headless": args.headless, "device": str(fabric.device)}
    )

    from protomotions.simulator.base_simulator.utils import (
        convert_friction_for_simulator,
    )

    terrain_config, simulator_config = convert_friction_for_simulator(
        terrain_config, simulator_config
    )

    from protomotions.utils.component_builder import build_all_components

    components = build_all_components(
        terrain_config=terrain_config,
        scene_lib_config=scene_lib_config,
        motion_lib_config=motion_lib_config,
        simulator_config=simulator_config,
        robot_config=robot_config,
        device=fabric.device,
        save_dir=getattr(env_config, "save_dir", None),
        simulation_app=app_launcher.app,
    )

    from protomotions.envs.base_env.env import BaseEnv

    EnvClass = get_class(env_config._target_)
    env: BaseEnv = EnvClass(
        config=env_config,
        robot_config=robot_config,
        device=fabric.device,
        terrain=components["terrain"],
        scene_lib=components["scene_lib"],
        motion_lib=components["motion_lib"],
        simulator=components["simulator"],
    )

    from protomotions.agents.base_agent.agent import BaseAgent

    AgentClass = get_class(agent_config._target_)
    agent: BaseAgent = AgentClass(
        config=agent_config, env=env, fabric=fabric, root_dir=checkpoint.parent
    )
    agent.setup()
    agent.load(str(checkpoint), load_env=False, load_training_state=False)
    agent.eval()

    joint_names = list(robot_config.kinematic_info.dof_names)
    anchor_idx = robot_config.anchor_body_index
    actor_in_keys = list(agent_config.model.actor.in_keys)
    recorder = TraceRecorder(actor_in_keys, anchor_idx, env.dt)

    tape = None
    if args.action_tape is not None:
        tape_data = np.load(args.action_tape, allow_pickle=True)
        tape = torch.as_tensor(
            tape_data["mean_action"], dtype=torch.float32, device=fabric.device
        )
        log.info(
            f"Replaying {tape.shape[0]} recorded actions from {args.action_tape}; "
            "the policy is out of the loop."
        )

    try:
        obs, _ = env.reset(None)
        dump_init_state(env, out_dir, joint_names)

        step = 0
        while step < args.max_steps:
            obs = agent.add_agent_info_to_obs(obs)
            obs_td = agent.obs_dict_to_tensordict(obs)

            # `env.context` is the state that produces this step's action; it is
            # invalidated at the top of env.step(), so grab it first.
            context = env.context

            with torch.no_grad():
                model_outs = agent.model(obs_td)
            action = model_outs["mean_action"]
            if tape is not None:
                if step >= tape.shape[0]:
                    log.info(f"Action tape exhausted after {step} steps.")
                    break
                action = tape[step : step + 1]

            recorder.record_pre_step(env, context, obs_td, action)

            obs, _, dones, terminated, _ = env.step(action)
            recorder.record_post_step(env)
            step += 1

            if bool(dones[0].item()):
                log.info(
                    f"Episode ended at control step {step} "
                    f"(terminated={bool(terminated[0].item())})"
                )
                break
        else:
            log.warning(f"Hit --max-steps={args.max_steps} without the episode ending.")

        recorder.write(
            out_dir,
            metadata={
                "meta__joint_names": np.array(joint_names),
                "meta__anchor_body_index": np.asarray(anchor_idx, dtype=np.int64),
                "meta__actor_in_keys": np.array(actor_in_keys),
                "meta__dt": np.asarray(env.dt, dtype=np.float32),
                "meta__checkpoint": np.array(str(checkpoint)),
                "meta__motion_file": np.array(str(motion_lib_config.motion_file)),
                "meta__motion_index": np.asarray(args.motion_index, dtype=np.int64),
            },
        )
    finally:
        if hasattr(env.simulator, "shutdown"):
            env.simulator.shutdown()


if __name__ == "__main__":
    main()
