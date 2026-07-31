# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Analyze per-joint torques of a trained policy during inference.

Runs a checkpoint for a bounded number of steps, records the applied torque of
every DOF each step, then prints per-joint statistics (mean / std / RMS /
peak |tau| / peak %% of effort limit) and saves a bar chart.

Example
-------
    python scripts/analyze_joint_torques.py \\
        --checkpoint data/pretrained_models/motion_tracker/g1-bones-deploy/last.ckpt \\
        --motion-file data/motion_for_trackers/g1_bones_seed_mini.pt \\
        --simulator mujoco --num-envs 1 --num-steps 300 --headless

Note on backend semantics: ``dof_forces`` is the *applied actuator torque* for
MuJoCo / Newton / IsaacLab, but a *measured joint-force sensor* reading for
IsaacGym. The active backend is printed alongside the results.
"""


def create_parser():
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="Analyze per-joint torques of a trained agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to checkpoint file"
    )
    parser.add_argument(
        "--simulator",
        type=str,
        required=True,
        help="Simulator to use (e.g., 'isaaclab', 'mujoco', 'newton', 'isaacgym')",
    )
    parser.add_argument(
        "--motion-file",
        type=str,
        default=None,
        help="Motion file for inference. Defaults to the checkpoint's motion file.",
    )
    parser.add_argument(
        "--num-envs", type=int, default=1, help="Number of parallel environments"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run simulation in headless mode",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=300,
        help="Number of simulation steps to record",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/joint_torque_analysis.png",
        help="Path to save the per-joint torque bar chart",
    )
    return parser


# Parse arguments first (argparse is safe, doesn't import torch)
import argparse  # noqa: E402

parser = create_parser()
args = parser.parse_args()

# Import simulator before torch - isaacgym/isaaclab must be imported before torch.
from protomotions.utils.simulator_imports import (  # noqa: E402
    import_simulator_before_torch,
)

AppLauncher = import_simulator_before_torch(args.simulator)

# Now safe to import torch and everything else.
import logging  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from lightning.fabric import Fabric  # noqa: E402

from protomotions.utils.fabric_config import FabricConfig  # noqa: E402
from protomotions.utils.hydra_replacement import get_class  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")
log = logging.getLogger(__name__)


def build_agent_and_env(args, AppLauncher):
    """Build env + agent from a checkpoint, mirroring inference_agent.main()."""
    checkpoint = Path(args.checkpoint)
    resolved_configs_path = checkpoint.parent / "resolved_configs_inference.pt"
    assert (
        resolved_configs_path.exists()
    ), f"Could not find resolved configs at {resolved_configs_path}"

    log.info(f"Loading resolved configs from {resolved_configs_path}")
    resolved_configs = torch.load(
        resolved_configs_path, map_location="cpu", weights_only=False
    )

    robot_config = resolved_configs["robot"]
    simulator_config = resolved_configs["simulator"]
    terrain_config = resolved_configs.get("terrain")
    scene_lib_config = resolved_configs["scene_lib"]
    motion_lib_config = resolved_configs["motion_lib"]
    env_config = resolved_configs["env"]
    agent_config = resolved_configs["agent"]

    # Switch simulator if the requested one differs from training.
    current_simulator = simulator_config._target_.split(".")[-3]
    if args.simulator != current_simulator:
        log.info(
            f"Switching simulator from '{current_simulator}' (training) to "
            f"'{args.simulator}' (inference)"
        )
        from protomotions.simulator.factory import update_simulator_config_for_test

        simulator_config = update_simulator_config_for_test(
            current_simulator_config=simulator_config,
            new_simulator=args.simulator,
            robot_config=robot_config,
        )

    # CLI runtime overrides.
    simulator_config.num_envs = args.num_envs
    simulator_config.headless = args.headless
    if args.motion_file is not None:
        motion_lib_config.motion_file = args.motion_file

    # MuJoCo is CPU-only, so force CPU accelerator.
    accelerator = "cpu" if args.simulator == "mujoco" else "gpu"
    fabric_config = FabricConfig(
        accelerator=accelerator,
        devices=1,
        num_nodes=1,
        loggers=[],
        callbacks=[],
    )
    fabric = Fabric(**fabric_config.as_kwargs())
    fabric.launch()

    # Setup IsaacLab simulation_app if needed.
    simulator_extra_params = {}
    if args.simulator == "isaaclab":
        app_launcher = AppLauncher(
            {"headless": args.headless, "device": str(fabric.device)}
        )
        simulator_extra_params["simulation_app"] = app_launcher.app

    from protomotions.simulator.base_simulator.utils import (
        convert_friction_for_simulator,
    )

    terrain_config, simulator_config = convert_friction_for_simulator(
        terrain_config, simulator_config
    )

    from protomotions.utils.component_builder import build_all_components

    save_dir_for_weights = getattr(env_config, "save_dir", None)
    components = build_all_components(
        terrain_config=terrain_config,
        scene_lib_config=scene_lib_config,
        motion_lib_config=motion_lib_config,
        simulator_config=simulator_config,
        robot_config=robot_config,
        device=fabric.device,
        save_dir=save_dir_for_weights,
        **simulator_extra_params,
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
    agent.load(args.checkpoint, load_env=False, load_training_state=False)

    return agent, env


def collect_torques(agent, env, num_steps):
    """Run the policy for ``num_steps`` and collect env-0 DOF torques."""
    agent.eval()
    done_indices = None
    torques = []

    print(f"Collecting torques over {num_steps} steps... (Ctrl+C to stop early)")
    step = 0
    while step < num_steps:
        obs, _ = env.reset(done_indices)
        agent.pre_collect_step(step)
        obs = agent.add_agent_info_to_obs(obs)
        obs_td = agent.obs_dict_to_tensordict(obs)

        model_outs = agent.model(obs_td)
        action = (
            model_outs["mean_action"]
            if "mean_action" in model_outs
            else model_outs["action"]
        )

        _, _, dones, _, _ = env.step(action)

        dof_forces = env.simulator.get_dof_forces().dof_forces  # [num_envs, num_dof]
        torques.append(dof_forces[0].detach().float().cpu())

        done_indices = dones.nonzero(as_tuple=False).squeeze(-1)
        step += 1

    return torch.stack(torques, dim=0)  # [T, num_dof]


def report(torques, dof_names, effort_limits, simulator_name, output_path):
    """Print a per-joint torque table and save a bar chart."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tau = torques.numpy()  # [T, num_dof]
    abs_tau = np.abs(tau)
    mean = tau.mean(axis=0)
    std = tau.std(axis=0)
    rms = np.sqrt((tau**2).mean(axis=0))
    peak = abs_tau.max(axis=0)

    if effort_limits is not None:
        limits = np.asarray(effort_limits, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            peak_pct = np.where(limits > 0, 100.0 * peak / limits, np.nan)
    else:
        limits = None
        peak_pct = np.full_like(peak, np.nan)

    print("\n" + "=" * 88)
    print(
        f"Per-joint torque over {tau.shape[0]} steps  "
        f"(simulator: {simulator_name}, {tau.shape[1]} DOFs)"
    )
    if simulator_name == "isaacgym":
        print("NOTE: IsaacGym reports a measured joint-force sensor, not applied torque.")
    print("=" * 88)
    header = "{:<28} {:>9} {:>9} {:>9} {:>9} {:>8}".format(
        "joint", "mean", "std", "rms", "peak|t|", "%limit"
    )
    print(header)
    print("-" * 88)
    order = np.argsort(-peak)  # highest peak torque first
    for i in order:
        name = dof_names[i] if i < len(dof_names) else f"dof_{i}"
        pct = "-" if np.isnan(peak_pct[i]) else "{:.1f}".format(peak_pct[i])
        print(
            "{:<28} {:>9.2f} {:>9.2f} {:>9.2f} {:>9.2f} {:>8}".format(
                name, mean[i], std[i], rms[i], peak[i], pct
            )
        )
    print("=" * 88 + "\n")

    # Bar chart: RMS (bars) with peak |tau| overlaid, joints ordered by peak.
    num_dof = tau.shape[1]
    labels = [
        dof_names[i] if i < len(dof_names) else f"dof_{i}" for i in order
    ]
    y = np.arange(num_dof)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.32 * num_dof)))
    ax.barh(y, rms[order], color="#4C72B0", label="RMS torque")
    ax.plot(peak[order], y, "o", color="#C44E52", label="peak |torque|")
    if limits is not None:
        ax.plot(limits[order], y, "|", color="#55A868", markersize=12, label="effort limit")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("torque (N·m)")
    ax.set_title(f"Per-joint torque ({simulator_name}, {tau.shape[0]} steps)")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved per-joint torque chart to: {output_path}")


def main():
    agent, env = build_agent_and_env(args, AppLauncher)

    dof_names = env.robot_config.kinematic_info.dof_names
    effort_limits = None
    limits_t = getattr(env.simulator, "_torque_limits_common", None)
    if limits_t is not None:
        effort_limits = limits_t.detach().float().cpu().numpy()

    try:
        torques = collect_torques(agent, env, args.num_steps)
    except KeyboardInterrupt:
        print("\nInterrupted; nothing recorded.")
        return
    finally:
        if hasattr(env.simulator, "shutdown"):
            env.simulator.shutdown()

    report(torques, dof_names, effort_limits, args.simulator, args.output)


if __name__ == "__main__":
    main()
