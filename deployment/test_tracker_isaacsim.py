# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Standalone Isaac Sim inference for tracker ONNX policies.

Isaac Sim analogue of ``deployment/test_tracker_mujoco.py`` -- see that
file's module docstring for the full deployment-contract background (ONNX
I/O names, quaternion conventions, PD target post-processing, motion
realignment notes). This script swaps MuJoCo physics for Isaac Sim
(``isaacsim.core.api.World`` + a USD robot asset), following the same
standalone-script shape as NVIDIA's own policy examples, e.g.
``isaacsim.robot.policy.examples.robots.h1.H1FlatTerrainPolicy`` /
``isaacsim.robot.policy.examples.controllers.PolicyController``: create
``SimulationApp`` first, build a ``World``, spawn the robot, drive it from a
physics callback, then ``world.step()`` in a loop.

Unlike the H1 example (which loads a TorchScript policy + IsaacLab-style
``env.yaml``), this script reuses the *same* ONNX + YAML deployment contract
as the MuJoCo script, so a single export works for both simulators.

Everything lives in this one file on purpose: the CLI, the ``World`` setup and
the :class:`TrackerPolicy` driver. Kit forces the layout -- ``isaacsim.core.*``
is only importable once ``SimulationApp`` exists, so every import below the
boot line has to stay below it.

Isaac Sim specifics (read before pointing this at a new robot)
----------------------------------------------------------------
- **DOF ordering**: the referenced USD's articulation ``dof_names`` order is
  not guaranteed to match the MJCF-derived ``joint_names`` order baked into
  the ONNX model (the USD was produced by ``usd_convert``'s MJCF importer,
  which may reorder joints).  Reordering happens at the read/write boundary
  -- see ``TrackerPolicy.initialize`` (builds ``self._isaac_to_policy``).

- **Body poses must come from the physics view, never from USD.**  The
  anchor body (G1's ``torso_link``) is read via
  ``ArticulationView._physics_view.get_link_transforms()``.  The obvious
  alternative -- wrapping the link prim in a ``SingleXFormPrim`` and calling
  ``get_world_pose()`` -- is **wrong**: that path resolves to
  ``XFormPrim.get_world_poses(usd=True)``, a pure USD stage read with no
  physics-view branch at all.  PhysX does not author USD xforms for
  articulation links during simulation (the live data lives in Fabric), and
  these robot assets are built ``make_instanceable: true``, so the link
  prims are instance proxies whose USD transforms can never reflect
  per-instance simulation.  Reading the anchor that way returns the spawn
  pose forever, the policy never sees the robot tipping, and it falls over.

  Note the quaternion convention trap: ``get_link_transforms()`` returns
  ``[x, y, z, qx, qy, qz, qw]`` -- already ProtoMotions' **xyzw** order,
  unlike every other Isaac Sim pose API here (which is wxyz).  Do not run it
  through :func:`wxyz_to_xyzw`.

- **Joint properties**: training applies per-joint armature, effort limits
  and solver iteration counts from the ProtoMotions robot config (see
  ``protomotions/simulator/isaaclab/utils/scene.py``).  The exported
  deployment YAML does not carry all of these, so when a robot name is
  available they are resolved from ``protomotions.robot_configs`` at runtime
  by :func:`load_robot_joint_properties`.  This matters for the G1, whose PD
  gains are *derived from* armature (``kp = armature * w_n**2``), so the
  closed-loop joint dynamics are wrong without it.

- **Root articulation prim**: found by searching for
  ``UsdPhysics.ArticulationRootAPI`` (:meth:`TrackerPolicy._find_articulation_root`),
  *not* by assuming a path shape.  For these assets the bodies live under
  ``{prim_path}/pelvis/`` while the articulation root is one level deeper, on
  ``{prim_path}/pelvis/pelvis``.  Wrapping the wrong one still produces a
  ``SingleArticulation`` with correct ``dof_names``/``body_names`` whose reads
  track the live simulation, so nothing looks broken.  Override with
  ``--root-prim-name`` if a new asset needs it.

- **USD angular drive gains are per-*degree*; the policy's are per-radian.**
  ``usd_convert`` writes ``drive:angular:physics:stiffness`` and ``damping`` as
  **0** -- training supplies them (IsaacLab does it through its actuator configs
  at spawn time).  :meth:`TrackerPolicy._author_drive_gains` fills them in so the
  drives are sane during the physics steps between ``world.reset()`` and the
  first ``set_gains()``, but it must convert: ``UsdPhysics`` angular drives are
  authored in Nm/deg, and Isaac Sim's own ``set_gains(save_to_usd=True)`` writes
  ``kp * pi/180`` (``isaacsim/core/prims/impl/articulation.py``).  Writing the raw
  Nm/rad number instead makes PhysX solve at **57.2958x** the intended stiffness.
  Armature and effort limits do *not* need this -- ``usd_convert`` already
  authors ``physxJoint:armature`` and ``drive:angular:physics:maxForce``
  correctly, and those are not angle-unit scaled.

  ``ArticulationView.set_gains()`` *does* reach PhysX (read back after
  ``initialize()`` it reports the configured values, and the robot responds to
  actions), so the USD authoring is belt-and-braces, not the mechanism that makes
  control work.  An earlier version of this docstring claimed the opposite; that
  was an artifact of wrapping the wrong prim before ``_find_articulation_root``
  existed -- see the articulation-root note above.

- **The default experience has no physics *authoring* UI.**  ``SimulationApp`` with
  no ``experience=`` falls back to ``isaacsim.exp.base.python.kit``, a trimmed
  "app for python samples" that enables the physics *runtime*
  (``omni.physics.physx``, ``omni.physx.tensors``) but nothing to author it
  with -- no ``Create > Physics`` menu, no ``+ Add > Physics`` section in the
  Property window.  Those come from ``omni.physx.bundle`` (``omni.physx.ui``,
  ``omni.kit.property.physx``, ``omni.physx.commands``, ...), which only
  ``isaacsim.exp.full.kit`` lists.  Windowed runs enable that bundle at runtime
  right after boot (see ``--no-physics-ui``); booting ``full.kit`` instead also
  works but drags in ``isaacsim.app.setup``, the examples browser and the
  replicator UI, which take over the window layout for no benefit here.

- **``post_reset()`` reverts gains.**  ``ArticulationView._on_post_reset()``
  (``isaacsim/core/prims/impl/articulation.py``) calls
  ``set_gains(kps=self._default_kps, kds=self._default_kds)``, discarding
  whatever ``initialize()`` configured, so anything gain-related is re-applied
  after it.

Requirements
------------
- A pre-exported ``unified_pipeline.onnx`` + ``.yaml`` sidecar. Use
  ``deployment/export_bm_tracker_onnx_isaacsim.py`` -- it resolves the timing
  block from the training config's Isaac-family rates. The MuJoCo-targeted
  ``deployment/export_bm_tracker_onnx.py`` also works, but its YAML carries
  MuJoCo's 1 kHz / decimation-20 timing, which has to be overridden here with
  ``--physics-dt``.
- A pre-built robot USD asset (see ``protomotions/data/assets/usd/<robot>/``,
  produced offline by ``usd_convert/``). Pass its path via ``--usd``.

Usage
-----
::

    python deployment/test_tracker_isaacsim.py \
        --onnx data/pretrained_models/motion_tracker/g1-bones-deploy/compiled_models/unified_pipeline.onnx \
        --motion data/motion_for_trackers/g1_random_subset_tiny.pt

    python deployment/test_tracker_isaacsim.py \
        --onnx results/g1_walk_box/compiled_models/unified_pipeline.onnx \
        --motion results/g1_walk_box/g1_walk_box.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

_DEFAULT_USD = "protomotions/data/assets/usd/g1_holo_compat/g1_holo_compat.usda"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run a ProtoMotions tracker ONNX policy in Isaac Sim",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--onnx", required=True, help="Path to unified_pipeline.onnx")
    p.add_argument(
        "--motion",
        required=True,
        help="Path to motion .pt file (raw ProtoMotions or pre-cached)",
    )
    p.add_argument(
        "--usd",
        default=_DEFAULT_USD,
        help="Path to the robot USD asset (absolute, or relative to the repo root)",
    )
    p.add_argument(
        "--robot",
        default="g1",
        help=(
            "Robot name for protomotions.robot_configs. Supplies the per-joint "
            "armature, effort limits and solver iteration counts that the "
            "deployment YAML does not carry. Pass '' to skip, at the cost of "
            "joint dynamics that no longer match training."
        ),
    )
    p.add_argument(
        "--prim-path",
        default="/World/Robot",
        help="Stage prim path to spawn the robot at",
    )
    p.add_argument(
        "--root-prim-name",
        default=None,
        help=(
            "Name of the articulation-root sub-prim under --prim-path. "
            "Defaults to the robot's root body name from the YAML metadata (e.g. 'pelvis')."
        ),
    )
    p.add_argument(
        "--motion-index",
        type=int,
        default=0,
        help="Clip index in multi-motion .pt library",
    )
    p.add_argument(
        "--cache-motion",
        action="store_true",
        default=False,
        help="After loading a raw motion file, write a 50fps cache next to it",
    )
    p.add_argument(
        "--loops",
        type=int,
        default=None,
        help="Number of times to loop the motion (default: infinite unless --headless, then 1)",
    )
    p.add_argument(
        "--headless", action="store_true", default=False, help="Run without a viewport"
    )
    p.add_argument(
        "--no-physics-ui",
        action="store_true",
        default=False,
        help=(
            "Skip loading the physics authoring UI extensions (omni.physx.bundle) "
            "in windowed runs -- slightly faster startup, but no 'Create > Physics' "
            "menu and no '+ Add > Physics' in the Property window. --headless never "
            "loads them."
        ),
    )
    p.add_argument(
        "--trace-out",
        type=str,
        default=None,
        help=(
            "Write a per-control-step tracking trace (root height, torso tilt and "
            "joint error against the reference, plus joint velocities) as JSON, and "
            "print the summary table. Comparable to the IsaacLab/MuJoCo numbers in "
            "docs/isaacsim_g1_tracker_instability_findings.md."
        ),
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help=(
            "Physics device, e.g. 'cuda:0'. Default (unset) runs PhysX on the CPU "
            "with the numpy backend -- a different solver path from the GPU "
            "pipeline used in training. Pass cuda:0 to match training."
        ),
    )
    p.add_argument(
        "--no-realtime",
        action="store_true",
        default=False,
        help=(
            "Disable real-time pacing of the windowed run (go as fast as physics "
            "allows). --headless never paces."
        ),
    )
    p.add_argument(
        "--action-ema-alpha",
        type=float,
        default=None,
        help="EMA filter on PD targets. Overrides the YAML metadata's control.action_ema_alpha.",
    )
    p.add_argument(
        "--physics-dt",
        type=float,
        default=None,
        help=(
            "Physics timestep. Defaults to the YAML metadata's timing.physics_dt, "
            "which is correct for YAMLs from export_bm_tracker_onnx_isaacsim.py but "
            "carries MuJoCo's 1 kHz rate for YAMLs from export_bm_tracker_onnx.py. "
            "control_dt is preserved; decimation is re-derived."
        ),
    )
    p.add_argument(
        "--ground-friction",
        type=float,
        default=1.0,
        help="Ground plane static/dynamic friction (training default: 1.0)",
    )
    return p.parse_args()


# Parse CLI args before creating SimulationApp (headless flag is needed for its config,
# and this keeps failure on bad args fast, without paying Kit's boot cost).
args = _parse_args()

# Ensure the repo root is on sys.path so `deployment.*` imports work
# regardless of where the script is invoked from.
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# SimulationApp must be created before importing any other omni/isaacsim module.
from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": args.headless})

# The default experience (isaacsim.exp.base.python.kit) enables the physics *runtime*
# but none of the physics *authoring* UI, so the viewport has no "Create > Physics"
# menu and the Property panel no "+ Add > Physics" section. Both come from
# omni.physx.bundle, which only isaacsim.exp.full.kit lists -- pull it in here rather
# than booting the whole editor experience. See the module docstring.
if not args.headless and not args.no_physics_ui:
    from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

    if enable_extension("omni.physx.bundle"):
        simulation_app.update()  # let the extensions register their menus

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
import yaml  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.prims import SingleArticulation  # noqa: E402
from isaacsim.core.utils.prims import create_prim  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402
from isaacsim.core.utils.types import ArticulationAction  # noqa: E402
from isaacsim.core.utils.stage import get_current_stage  # noqa: E402
from isaacsim.core.utils.viewports import set_camera_view  # noqa: E402
from pxr import PhysxSchema, Usd, UsdPhysics  # noqa: E402

from deployment.motion_utils import MotionPlayer  # noqa: E402
from deployment.state_utils import (  # noqa: E402
    apply_heading_offset_np,
    compute_root_local_ang_vel_np,
    compute_yaw_offset_np,
)

# `force=True` is load-bearing: SimulationApp has already installed Kit's
# `_CarbLogHandler` on the root logger by this point, and a plain basicConfig()
# is a documented no-op once the root logger has any handler. Without the
# override every line below silently goes to Kit's log file instead of stderr.
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s", force=True)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def wxyz_to_xyzw(wxyz) -> np.ndarray:
    """Convert an Isaac Sim quaternion (wxyz) to ProtoMotions xyzw convention."""
    return np.asarray(wxyz)[..., [1, 2, 3, 0]]


def _to_numpy(array) -> np.ndarray:
    """Convert a physics-view tensor (numpy / torch / warp) to a NumPy array."""
    if isinstance(array, np.ndarray):
        return array
    if hasattr(array, "detach"):  # torch.Tensor
        return array.detach().cpu().numpy()
    if hasattr(array, "numpy"):  # warp.array
        return array.numpy()
    return np.asarray(array)


def _tilt_deg_xyzw(quat_xyzw) -> float:
    """Angle in degrees between the body's local +z axis and world +z."""
    x, y, _z, _w = np.asarray(quat_xyzw, dtype=np.float64)
    # R[2, 2] -- the z component of the body's z axis in world coordinates.
    cos_tilt = 1.0 - 2.0 * (x * x + y * y)
    return float(np.degrees(np.arccos(np.clip(cos_tilt, -1.0, 1.0))))


def _set_prim_attr(prim, name: str, value) -> bool:
    """Set an already-declared USD attribute, creating it only if the schema knows it.

    ``Apply()``-ing the PhysX API adds the attribute declarations but leaves them
    unauthored, so ``GetAttribute()`` returns a valid-but-empty handle that
    ``Set()`` authors correctly. If a given Isaac Sim version does not declare the
    attribute at all, skip it rather than authoring a stray property PhysX will
    ignore.
    """
    attr = prim.GetAttribute(name)
    if not attr:
        log.debug(f"{prim.GetPath()}: no attribute '{name}' to set")
        return False
    attr.Set(value)
    return True


def resolve_usd_path(usd_path: str) -> str:
    """Resolve a (possibly relative) USD path to an absolute filesystem path."""
    p = Path(usd_path)
    if p.is_absolute() and p.exists():
        return str(p)
    candidates = [
        _REPO_ROOT / usd_path,
        _REPO_ROOT / "protomotions" / "data" / "assets" / usd_path,
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    raise FileNotFoundError(
        f"Cannot find USD '{usd_path}'. Tried: {[str(c) for c in candidates]}"
    )


def build_robot_config(robot_name: str):
    """Build a ProtoMotions ``RobotConfig``, independent of the caller's CWD.

    ``RobotConfig.__post_init__`` parses the robot's MJCF through the
    *relative* default ``asset_root="protomotions/data/assets"``, so
    constructing one from anywhere but the repo root raises
    ``FileNotFoundError``.  Pin the CWD for the duration of the call.
    """
    from protomotions.robot_configs.factory import robot_config

    cwd = os.getcwd()
    try:
        os.chdir(_REPO_ROOT)
        return robot_config(robot_name)
    finally:
        os.chdir(cwd)


def load_robot_joint_properties(robot_name: str, joint_names: list) -> dict:
    """Resolve per-joint physics properties from the ProtoMotions robot config.

    The exported deployment YAML carries stiffness/damping but not armature or
    velocity limits, and its ``effort_limits`` is often ``null``.  Training
    applies all of them (``protomotions/simulator/isaaclab/utils/scene.py``),
    so pull them from the same source the exporter reads its gains from:
    ``RobotConfig.control.control_info``, which ``RobotConfig.__post_init__``
    has already expanded from regex patterns to one entry per DOF.

    Args:
        robot_name: Robot key accepted by ``protomotions.robot_configs.factory``.
        joint_names: Policy joint order (from the YAML metadata).

    Returns:
        Dict with ``armature`` / ``effort_limit`` / ``velocity_limit`` arrays in
        *policy* joint order (or ``None`` per entry if the config does not
        define that property for every joint), plus the solver iteration counts,
        the IsaacLab physics rate/decimation, and the rigid-body and collision
        properties training spawns the robot with.
    """
    config = build_robot_config(robot_name)
    control_info = config.control.control_info

    def _column(attr: str):
        values = []
        for joint in joint_names:
            info = control_info.get(joint)
            value = getattr(info, attr, None) if info is not None else None
            if value is None:
                return None
            values.append(float(value))
        return np.asarray(values, dtype=np.float32)

    sim = getattr(config.simulation_params, "isaaclab", None)
    physx = getattr(sim, "physx", None)
    asset = config.asset
    return {
        "armature": _column("armature"),
        "effort_limit": _column("effort_limit"),
        "velocity_limit": _column("velocity_limit"),
        "solver_position_iterations": getattr(physx, "num_position_iterations", None),
        "solver_velocity_iterations": getattr(physx, "num_velocity_iterations", None),
        # Timing: training's IsaacLab rate, not the YAML's MuJoCo-flavoured one.
        "sim_fps": getattr(sim, "fps", None),
        "sim_decimation": getattr(sim, "decimation", None),
        # RigidBodyPropertiesCfg / CollisionPropertiesCfg equivalents -- the USD
        # ships different values for several of these (see _author_body_properties).
        "linear_damping": getattr(asset, "linear_damping", None),
        "angular_damping": getattr(asset, "angular_damping", None),
        "max_linear_velocity": getattr(asset, "max_linear_velocity", None),
        "max_angular_velocity": getattr(asset, "max_angular_velocity", None),
        "max_depenetration_velocity": getattr(
            physx, "max_depenetration_velocity", None
        ),
        "contact_offset": getattr(physx, "contact_offset", None),
        "rest_offset": getattr(physx, "rest_offset", None),
        "bounce_threshold_velocity": getattr(physx, "bounce_threshold_velocity", None),
    }


def build_onnx_inputs(
    dof_pos: np.ndarray,
    dof_vel: np.ndarray,
    anchor_rot: np.ndarray,
    root_local_ang_vel: np.ndarray,
    future_refs: dict,
    anchor_body_index: int,
    onnx_name_to_key: dict,
    num_dofs: int,
    prev_actions: np.ndarray | None = None,
) -> dict:
    """Assemble ONNX input dict from live robot state + motion futures.

    Unlike ``test_tracker_mujoco.build_onnx_inputs``, ``anchor_rot`` and
    ``root_local_ang_vel`` are passed in pre-computed rather than derived
    from a full per-body rotation array -- Isaac Sim's articulation API has
    no cheap "all body world poses" query, so callers fetch only the two
    bodies actually needed (root + anchor).
    """
    if prev_actions is None:
        prev_actions = np.zeros(num_dofs, dtype=np.float32)

    future_anchor_rot = future_refs["body_rot"][:, anchor_body_index, :]  # [nsteps, 4]

    key_to_array = {
        "current.dof_pos": dof_pos[None],
        "current.dof_vel": dof_vel[None],
        "current.anchor_rot": anchor_rot[None],
        "current.root_local_ang_vel": root_local_ang_vel[None],
        "historical.processed_actions": prev_actions[None, None],
        "mimic.future_anchor_rot": future_anchor_rot[None],
        "mimic.future_rot": future_refs["body_rot"][None],
        "mimic.future_dof_pos": future_refs["dof_pos"][None],
        "mimic.future_dof_vel": future_refs["dof_vel"][None],
    }

    onnx_inputs: dict[str, np.ndarray] = {}
    for onnx_name, sem_key in onnx_name_to_key.items():
        if sem_key in key_to_array:
            onnx_inputs[onnx_name] = key_to_array[sem_key].astype(np.float32)
        else:
            log.warning(f"No value for ONNX input '{onnx_name}' (key='{sem_key}')")
    return onnx_inputs


# ---------------------------------------------------------------------------
# Tracker policy controller
# ---------------------------------------------------------------------------


class TrackerPolicy:
    """Drives a ProtoMotions tracker ONNX policy on an Isaac Sim articulation.

    Mirrors ``PolicyController``/``H1FlatTerrainPolicy``'s shape (load
    policy + metadata, ``initialize()`` after ``world.reset()``,
    ``post_reset()``, per-physics-step ``forward()``), but speaks the
    ONNX + YAML deployment contract instead of TorchScript + IsaacLab's
    ``env.yaml``.
    """

    def __init__(
        self,
        meta: dict,
        prim_path: str,
        usd_path: str,
        onnx_path: str,
        motion_file: str,
        root_prim_name: str | None = None,
        cache_motion: bool = False,
        action_ema_alpha: float | None = None,
        motion_index: int = 0,
        robot_name: str | None = None,
        physics_dt: float | None = None,
        log_every: int = 100,
    ) -> None:
        robot_meta = meta["robot"]
        timing = meta["timing"]
        motion_meta = meta["motion"]
        control = meta["control"]
        runtime = meta["_runtime"]

        self.anchor_body_index = robot_meta["anchor_body_index"]
        self.anchor_body_name = robot_meta.get("anchor_body_name", "torso_link")
        self.root_body_name = robot_meta.get("root_body_name", "pelvis")
        self.joint_names = list(robot_meta["joint_names"])
        self.num_dofs = robot_meta["num_dofs"]
        self.control_dt = timing["control_dt"]
        self.future_step_indices = list(motion_meta["future_step_indices"])
        self.stiffness = control["stiffness"]
        self.damping = control["damping"]
        self.effort_limits = control.get("effort_limits")
        self.pd_target_max_accel = control.get("pd_target_max_accel")
        self.action_ema_alpha = (
            action_ema_alpha
            if action_ema_alpha is not None
            else control.get("action_ema_alpha", 1.0)
        )
        self.onnx_name_to_key = runtime["onnx_name_to_in_key"]
        self.log_every = log_every

        # Timing. A YAML from the MuJoCo exporter resolves physics_dt through the
        # *MuJoCo* sim params (1 kHz / decimation 20), not the backend the policy
        # was trained on. Callers can override; control_dt is what the policy
        # actually cares about, so decimation is re-derived to preserve it.
        self.physics_dt = (
            float(physics_dt) if physics_dt else float(timing["physics_dt"])
        )
        self.decimation = max(1, int(round(self.control_dt / self.physics_dt)))
        if self.decimation * self.physics_dt != self.control_dt:
            log.warning(
                f"control_dt={self.control_dt}s is not an integer multiple of "
                f"physics_dt={self.physics_dt}s; using decimation={self.decimation} "
                f"(effective control_dt={self.decimation * self.physics_dt}s)."
            )

        # Per-joint physics properties the YAML does not carry (armature,
        # velocity limits, solver iterations). Only available with a robot name.
        self.joint_properties: dict | None = None
        if robot_name:
            try:
                self.joint_properties = load_robot_joint_properties(
                    robot_name, self.joint_names
                )
            except Exception as e:
                log.warning(
                    f"Could not load joint properties for robot '{robot_name}': {e}. "
                    "Armature and effort limits will fall back to the USD defaults."
                )
        else:
            log.warning(
                "No robot name given -- armature, velocity limits and solver "
                "iteration counts cannot be matched to training. Pass a robot "
                "name for a physics setup that matches the training config."
            )

        log.info(
            f"Robot: {self.num_dofs} DOFs, anchor={self.anchor_body_name}"
            f"[{self.anchor_body_index}], root={self.root_body_name}"
        )
        log.info(
            f"control_dt={self.control_dt}s  physics_dt={self.physics_dt}s  decimation={self.decimation}"
        )
        log.info(f"Future steps: {self.future_step_indices}")

        self.session = ort.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )
        self.onnx_out_names = [o.name for o in self.session.get_outputs()]
        log.info(f"ONNX inputs:  {[i.name for i in self.session.get_inputs()]}")
        log.info(f"ONNX outputs: {self.onnx_out_names}")

        self.motion_player = MotionPlayer(
            motion_file, motion_index=motion_index, control_dt=self.control_dt
        )
        if cache_motion:
            motion_p = Path(motion_file)
            cache_p = motion_p.parent / (motion_p.stem + ".50fps.pt")
            if not cache_p.exists():
                self.motion_player.cache_to_file(str(cache_p))
        log.info(
            f"Motion: {self.motion_player.total_frames} frames @ "
            f"{1.0 / self.control_dt:.0f} Hz"
        )

        add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
        self._prim_path = prim_path
        if root_prim_name:
            articulation_path = f"{prim_path}/{root_prim_name}"
        else:
            articulation_path = self._find_articulation_root(prim_path)
        log.info(f"Articulation root prim: {articulation_path}")
        self._author_drive_gains(prim_path)
        self._author_body_properties(prim_path)
        self.robot = SingleArticulation(
            prim_path=articulation_path, name="tracker_robot"
        )

        self._decimation_counter = 0
        self._frame_idx = 0
        self._loop_idx = 0
        self._total_steps = 0
        self.num_loops = 1
        self.done = False
        # Per-control-step tracking trace; None disables recording (see --trace-out).
        self.trace: list | None = None
        # Set from inside the physics callback, consumed by the main loop between
        # world.step() calls -- see reset_episode()'s note on why the reset itself
        # must not happen mid-step.
        self._pending_reset = False

        # Run-wide statistics for the end-of-run summary (never reset per episode).
        self.total_ort_ms = 0.0
        self.max_ref_err_run = 0.0

        # Episode-local filter/history state -- reset every episode in reset_episode().
        self._isaac_to_policy: np.ndarray | None = None
        self._anchor_link_index: int | None = None
        self._prev_actions: np.ndarray | None = None
        self._prev_pd: np.ndarray | None = None
        self._prev_prev_pd: np.ndarray | None = None
        self._ema_prev_targets: np.ndarray | None = None
        self._heading_offset: np.ndarray | None = None
        self._pd_targets_isaac: np.ndarray | None = None
        self._max_ref_err = 0.0

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _author_drive_gains(self, prim_path: str) -> None:
        """Write the PD gains onto the USD joint drives, before the sim starts.

        ``usd_convert`` authors these assets with ``drive:angular:physics:stiffness``
        and ``damping`` at **0**; the gains are expected to come from the training
        framework (IsaacLab applies them through its actuator configs at spawn time).
        ``initialize()``/``post_reset()`` do apply them through
        ``ArticulationView.set_gains()``, but that only takes effect once the
        physics view exists -- every physics step in between runs on whatever the
        stage says. Author them here so that window is not spent at zero gain.

        Units: ``UsdPhysics`` **angular** drives are per-degree, while the policy's
        gains (and ``set_gains()``) are per-radian. Isaac Sim's own
        ``set_gains(save_to_usd=True)`` writes ``kp * pi/180``
        (``isaacsim/core/prims/impl/articulation.py``), so do the same -- writing
        the raw Nm/rad value makes PhysX solve at 57.2958x the intended stiffness.
        """
        stage = get_current_stage()
        deg = math.pi / 180.0  # Nm/rad -> Nm/deg, matching set_gains(save_to_usd=True)
        by_name = {
            name: (float(kp) * deg, float(kd) * deg)
            for name, kp, kd in zip(self.joint_names, self.stiffness, self.damping)
        }
        applied = 0
        for prim in Usd.PrimRange(stage.GetPrimAtPath(prim_path)):
            gains = by_name.get(prim.GetName())
            if gains is None or not prim.HasAPI(UsdPhysics.DriveAPI):
                continue
            drive = UsdPhysics.DriveAPI.Get(prim, "angular")
            if not drive:
                continue
            drive.CreateStiffnessAttr().Set(gains[0])
            drive.CreateDampingAttr().Set(gains[1])
            applied += 1
        log.info(
            f"Authored PD gains on {applied}/{len(by_name)} USD joint drives "
            "(converted Nm/rad -> Nm/deg)."
        )

    def _author_body_properties(self, prim_path: str) -> None:
        """Apply training's RigidBodyPropertiesCfg / CollisionPropertiesCfg to the stage.

        ``protomotions/simulator/isaaclab/utils/scene.py`` spawns the robot with
        explicit rigid-body and collision properties taken from the robot config;
        this script previously spawned the USD as-authored, so several of them
        differed from training. The measured gaps on the G1 asset were a per-link
        ``angularDamping`` of 0.05 (a global drag that exists in neither training
        nor MuJoCo), a ``maxDepenetrationVelocity`` of 3.0 vs training's 1.0, and
        unpinned contact/rest offsets.

        Units: PhysX authors ``maxAngularVelocity`` in **deg/s**, and IsaacLab
        writes the config value into it verbatim (``sim/schemas/schemas.py`` does
        no conversion -- assets that want rad/s scale it themselves, cf.
        ``isaaclab_assets/robots/allegro.py``). So the raw config number is passed
        through here too, which is what training actually runs with.
        """
        props = self.joint_properties or {}
        rigid_body_attrs = {
            "physxRigidBody:linearDamping": props.get("linear_damping"),
            "physxRigidBody:angularDamping": props.get("angular_damping"),
            "physxRigidBody:maxLinearVelocity": props.get("max_linear_velocity"),
            "physxRigidBody:maxAngularVelocity": props.get("max_angular_velocity"),
            "physxRigidBody:maxDepenetrationVelocity": props.get(
                "max_depenetration_velocity"
            ),
            "physxRigidBody:retainAccelerations": False,
        }
        collision_attrs = {
            "physxCollision:contactOffset": props.get("contact_offset"),
            "physxCollision:restOffset": props.get("rest_offset"),
        }
        rigid_body_attrs = {k: v for k, v in rigid_body_attrs.items() if v is not None}
        collision_attrs = {k: v for k, v in collision_attrs.items() if v is not None}

        stage = get_current_stage()
        root_prim = stage.GetPrimAtPath(prim_path)

        # The colliders are USD *instances*: each body's `collisions` scope is an
        # instanceable prim, so its children are instance proxies. A default
        # Usd.PrimRange does not descend into them (159 prims vs 320 for this
        # asset) and authoring on one raises "authoring to an instance proxy is
        # not allowed" -- which is why the collision offsets silently reached
        # zero colliders before. De-instancing exposes them and makes the writes
        # legal; for a single robot the lost instancing costs nothing.
        if collision_attrs:
            for prim in Usd.PrimRange.Stage(
                stage, Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)
            ):
                if (
                    prim.GetPath().HasPrefix(root_prim.GetPath())
                    and prim.IsInstanceable()
                ):
                    prim.SetInstanceable(False)

        n_bodies = n_colliders = 0
        for prim in Usd.PrimRange(root_prim):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
                for name, value in rigid_body_attrs.items():
                    _set_prim_attr(prim, name, value)
                n_bodies += 1
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                PhysxSchema.PhysxCollisionAPI.Apply(prim)
                for name, value in collision_attrs.items():
                    _set_prim_attr(prim, name, value)
                n_colliders += 1
        log.info(
            f"Authored rigid-body properties on {n_bodies} bodies and collision "
            f"offsets on {n_colliders} colliders."
        )

    def _find_articulation_root(self, prim_path: str) -> str:
        """Locate the prim carrying ``UsdPhysics.ArticulationRootAPI``.

        Do not guess this from the body layout. ``usd_convert`` authors these
        assets with the articulation root one level *below* the Xform that holds
        the body subtree -- for the G1 the bodies live at
        ``/World/Robot/pelvis/<body>`` while the root API sits on
        ``/World/Robot/pelvis/pelvis`` (note the doubled name). Wrapping the
        subtree Xform instead still yields a ``SingleArticulation`` that reports
        the right ``dof_names``/``body_names`` and whose *reads* track the live
        simulation, so the mistake is invisible -- but every drive write
        (position targets, gains) is silently dropped and the robot falls limp,
        identically no matter what the policy commands.
        """
        stage = get_current_stage()
        root_prim = stage.GetPrimAtPath(prim_path)
        if root_prim.IsValid():
            for prim in Usd.PrimRange(root_prim):
                if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                    return prim.GetPath().pathString
        fallback = f"{prim_path}/{self.root_body_name}"
        log.warning(
            f"No prim under '{prim_path}' carries UsdPhysics.ArticulationRootAPI; "
            f"falling back to '{fallback}'. If the robot ignores all actions, pass "
            "the real articulation-root prim via --root-prim-name."
        )
        return fallback

    def initialize(self) -> None:
        """Set up articulation handles, DOF reordering, and physics properties.

        Must be called after ``world.reset()`` -- the underlying physics
        views (``dof_names``, gains, ...) do not exist before that.
        """
        self.robot.initialize()
        view = self.robot._articulation_view

        isaac_dof_names = list(self.robot.dof_names)
        missing = set(self.joint_names) - set(isaac_dof_names)
        if missing:
            raise ValueError(
                f"USD articulation is missing DOFs required by the policy: {missing}. "
                f"USD dof_names={isaac_dof_names}"
            )
        if len(isaac_dof_names) != self.num_dofs:
            raise ValueError(
                f"USD articulation has {len(isaac_dof_names)} DOFs but the policy "
                f"expects {self.num_dofs}. Extra DOFs would be left uncontrolled: "
                f"{sorted(set(isaac_dof_names) - set(self.joint_names))}"
            )
        # arr_isaac_order[self._isaac_to_policy] == arr_policy_order
        self._isaac_to_policy = np.array(
            [isaac_dof_names.index(n) for n in self.joint_names], dtype=np.int64
        )

        self._resolve_anchor_link_index(view)
        self._apply_solver_iterations(view)
        self._apply_joint_properties(view)

    def _resolve_anchor_link_index(self, view) -> None:
        """Find the anchor body's index into the articulation's link buffers."""
        if self.anchor_body_name == self.root_body_name:
            # Root pose is queried straight off the articulation; no link lookup.
            self._anchor_link_index = None
            return

        body_names = list(view.body_names or [])
        if self.anchor_body_name not in body_names:
            raise ValueError(
                f"Anchor body '{self.anchor_body_name}' is not a link of the "
                f"articulation. Available links: {body_names}. Check that the USD "
                "preserves MJCF body names."
            )
        self._anchor_link_index = body_names.index(self.anchor_body_name)

    def _apply_solver_iterations(self, view) -> None:
        """Match the PhysX solver iteration counts used during training."""
        props = self.joint_properties
        if props is None:
            return
        pos_iters = props.get("solver_position_iterations")
        vel_iters = props.get("solver_velocity_iterations")
        if pos_iters is not None:
            view.set_solver_position_iteration_counts(
                np.array([int(pos_iters)], dtype=np.int32)
            )
        if vel_iters is not None:
            view.set_solver_velocity_iteration_counts(
                np.array([int(vel_iters)], dtype=np.int32)
            )
        if pos_iters is not None or vel_iters is not None:
            log.info(f"Solver iterations: position={pos_iters} velocity={vel_iters}")

    def _to_isaac_order(self, policy_ordered) -> np.ndarray:
        """Scatter a policy-ordered per-DOF array into Isaac Sim's DOF order."""
        out = np.zeros(self.num_dofs, dtype=np.float32)
        out[self._isaac_to_policy] = np.asarray(policy_ordered, dtype=np.float32)
        return out

    def _apply_joint_properties(self, view) -> None:
        """Apply PD gains, armature, effort and velocity limits in Isaac Sim DOF order."""
        stiffness_isaac = self._to_isaac_order(self.stiffness)
        damping_isaac = self._to_isaac_order(self.damping)

        controller = self.robot.get_articulation_controller()
        controller.set_effort_modes("force")
        controller.switch_control_mode("position")
        # switch_control_mode() restores the USD default gains, so set ours after.
        view.set_gains(stiffness_isaac[None], damping_isaac[None])

        props = self.joint_properties or {}

        # Effort limits: prefer the YAML, fall back to the robot config. The
        # exporter historically wrote null here (it read `.effort` instead of
        # `.effort_limit`), which left the joints with unbounded torque.
        effort_limits = self.effort_limits
        if effort_limits is None:
            effort_limits = props.get("effort_limit")
        if effort_limits is not None:
            view.set_max_efforts(self._to_isaac_order(effort_limits)[None])
            log.info("Applied per-joint effort limits.")
        else:
            log.warning(
                "No effort limits available -- joints run with unbounded torque, "
                "unlike training."
            )

        # Armature (rotor inertia). Critical for robots whose gains are derived
        # from it, e.g. the G1's BeyondMimic gains (kp = armature * w_n**2).
        armature = props.get("armature")
        if armature is not None:
            view.set_armatures(self._to_isaac_order(armature)[None])
            log.info("Applied per-joint armature.")
        else:
            log.warning(
                "No armature available -- joint dynamics will not match training."
            )

        # Joint velocity limits. Training passes these as `velocity_limit_sim`
        # (scene.py); the USD leaves them at the MJCF/PhysX default. Measured
        # peaks on this policy reach 22-34 rad/s against limits of 20-37, so the
        # clamp is active and its absence is a real difference from training.
        velocity_limits = props.get("velocity_limit")
        if velocity_limits is not None:
            view.set_max_joint_velocities(self._to_isaac_order(velocity_limits)[None])
            log.info("Applied per-joint velocity limits.")
        else:
            log.warning(
                "No velocity limits available -- joints run unclamped, unlike training."
            )

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def post_reset(self) -> None:
        self.robot.post_reset()
        # ArticulationView._on_post_reset() re-applies the articulation's *default*
        # (USD-authored) gains -- see isaacsim/core/prims/impl/articulation.py, which
        # calls set_gains(kps=self._default_kps, kds=self._default_kds). Anything
        # initialize() configured is silently reverted here, so re-apply it.
        self._apply_joint_properties(self.robot._articulation_view)
        self.reset_episode()

    def reset_episode(self) -> None:
        """Reset robot state to the first motion frame and clear episode-local filters.

        **Call this only from outside the physics step.** ``set_world_pose`` /
        ``set_joint_positions`` / ``update_articulations_kinematic()`` write the
        articulation state directly; doing that from within a physics callback
        (i.e. part-way through ``world.step()``) is not well-defined in Isaac Sim
        and shows up as a jitter-then-fall at every clip restart. The callback
        therefore only raises ``_pending_reset``; the main loop drains it between
        steps.
        """
        frame0 = self.motion_player.get_state_at_frame(0)
        root_pos = frame0["body_pos"][0]
        root_quat_xyzw = frame0["body_rot"][0]
        root_quat_wxyz = root_quat_xyzw[[3, 0, 1, 2]]
        self.robot.set_world_pose(position=root_pos, orientation=root_quat_wxyz)

        dof_pos_isaac = self._to_isaac_order(frame0["dof_pos"])
        self.robot.set_joint_positions(dof_pos_isaac)

        # Start from the reference *velocities*, not from rest. ProtoMotions'
        # reference-state initialization seeds root and joint velocities from the
        # motion (compute_ref_reset_state), and the policy was trained on that
        # distribution -- dropping a moving clip in at zero velocity is off-
        # distribution for exactly the first few control steps that decide whether
        # the episode stays on its feet.
        self.robot.set_joint_velocities(self._to_isaac_order(frame0["dof_vel"]))
        self.robot.set_linear_velocity(
            np.asarray(frame0["body_vel"][0], dtype=np.float32)
        )
        self.robot.set_angular_velocity(
            np.asarray(frame0["body_ang_vel"][0], dtype=np.float32)
        )

        # Link transforms lag the joint state we just wrote until the kinematics
        # are refreshed. Without this the first _compute_action() -- which is
        # also where the heading offset is latched for the whole episode --
        # would read the pre-reset anchor pose.
        self._refresh_articulation_kinematics()

        self._frame_idx = 0
        self._decimation_counter = 0
        self._prev_actions = None
        self._prev_pd = None
        self._prev_prev_pd = None
        self._ema_prev_targets = None
        self._heading_offset = None
        self._pd_targets_isaac = dof_pos_isaac.copy()
        self._max_ref_err = 0.0

        log.info(
            f"--- Loop {self._loop_idx + 1}"
            + (f"/{self.num_loops}" if self.num_loops < 1_000_000 else "")
            + f" --- root_pos={root_pos.round(3).tolist()}"
        )

    @staticmethod
    def _refresh_articulation_kinematics() -> None:
        """Push freshly-written joint states through FK into the link buffers."""
        try:
            from isaacsim.core.simulation_manager import SimulationManager

            sim_view = None
            for getter in ("get_physics_simulation_view", "get_physics_sim_view"):
                fn = getattr(SimulationManager, getter, None)
                if fn is not None:
                    sim_view = fn()
                    if sim_view is not None:
                        break
            if sim_view is not None:
                sim_view.update_articulations_kinematic()
        except Exception as e:  # pragma: no cover - depends on Isaac Sim version
            log.debug(f"Could not refresh articulation kinematics: {e}")

    # ------------------------------------------------------------------
    # Per-step logic
    # ------------------------------------------------------------------

    def _read_anchor_rot(self, root_quat_xyzw: np.ndarray) -> np.ndarray:
        """Read the anchor body's world orientation (xyzw) from the physics view.

        See the module docstring: this must not go through USD, and
        ``get_link_transforms()`` is already xyzw.
        """
        if self._anchor_link_index is None:
            return root_quat_xyzw

        link_transforms = _to_numpy(
            self.robot._articulation_view._physics_view.get_link_transforms()
        )
        # (count, max_links, 7) with count == 1 -> [x, y, z, qx, qy, qz, qw]
        return link_transforms.reshape(-1, 7)[self._anchor_link_index, 3:7].astype(
            np.float32
        )

    def _read_robot_state(self):
        """Read current DOF state (reordered to policy order) + anchor/root orientation."""
        dof_pos_isaac = np.asarray(self.robot.get_joint_positions())
        dof_vel_isaac = np.asarray(self.robot.get_joint_velocities())
        dof_pos = dof_pos_isaac[self._isaac_to_policy].astype(np.float32)
        dof_vel = dof_vel_isaac[self._isaac_to_policy].astype(np.float32)

        _, root_quat_wxyz = self.robot.get_world_pose()
        root_quat = wxyz_to_xyzw(root_quat_wxyz).astype(np.float32)

        # Angular-velocity frame: `SingleArticulation.get_angular_velocity()` returns
        # the root body's angular velocity in the **world** frame, which is the
        # convention `compute_root_local_ang_vel_np` expects -- it rotates the vector
        # by the inverse root rotation to produce the root-local vector the policy was
        # trained on. So this is passed through as-is, *not* pre-rotated. (Isaac Sim
        # has no body-frame variant of this getter; if a future API returns body-frame
        # velocity, the inverse rotation below must be dropped, not kept.)
        root_ang_vel_world = np.asarray(
            self.robot.get_angular_velocity(), dtype=np.float32
        )
        root_local_ang_vel = compute_root_local_ang_vel_np(
            rigid_body_rot=root_quat[None, :],
            rigid_body_ang_vel=root_ang_vel_world[None, :],
            root_body_index=0,
        )

        anchor_rot = self._read_anchor_rot(root_quat)

        return dof_pos, dof_vel, anchor_rot, root_local_ang_vel

    def _compute_action(self) -> None:

        dof_pos, dof_vel, anchor_rot, root_local_ang_vel = self._read_robot_state()

        # Heading offset: aligns the reference motion's yaw to the robot's yaw at
        # episode start. See test_tracker_mujoco.py's docstring for background.
        if self._heading_offset is None:
            motion_anchor_rot = self.motion_player.get_state_at_frame(0)["body_rot"][
                self.anchor_body_index
            ]
            self._heading_offset = compute_yaw_offset_np(anchor_rot, motion_anchor_rot)

        future_refs = self.motion_player.get_future_references(
            self._frame_idx, self.future_step_indices
        )
        future_refs["body_rot"] = apply_heading_offset_np(
            self._heading_offset, future_refs["body_rot"]
        )

        onnx_inputs = build_onnx_inputs(
            dof_pos=dof_pos,
            dof_vel=dof_vel,
            anchor_rot=anchor_rot,
            root_local_ang_vel=root_local_ang_vel,
            future_refs=future_refs,
            anchor_body_index=self.anchor_body_index,
            onnx_name_to_key=self.onnx_name_to_key,
            num_dofs=self.num_dofs,
            prev_actions=self._prev_actions,
        )
        t0 = time.perf_counter()
        ort_out = self.session.run(self.onnx_out_names, onnx_inputs)
        self.total_ort_ms += (time.perf_counter() - t0) * 1000.0
        pd_targets = (
            ort_out[self.onnx_out_names.index("joint_pos_targets")].squeeze().copy()
        )

        # PD target acceleration clamp -- matches base_simulator._apply_accel_clamp().
        if (
            self.pd_target_max_accel is not None
            and self._prev_pd is not None
            and self._prev_prev_pd is not None
        ):
            delta = pd_targets - self._prev_pd
            prev_delta = self._prev_pd - self._prev_prev_pd
            accel = delta - prev_delta
            clamped_accel = np.clip(
                accel, -self.pd_target_max_accel, self.pd_target_max_accel
            )
            pd_targets = self._prev_pd + prev_delta + clamped_accel

        self._prev_prev_pd = self._prev_pd
        self._prev_pd = pd_targets.copy()

        # EMA action filter -- matches MujocoSimulator._action_filter_alpha.
        if self.action_ema_alpha < 1.0:
            if self._ema_prev_targets is None:
                self._ema_prev_targets = pd_targets.copy()
            pd_targets = (
                self.action_ema_alpha * pd_targets
                + (1.0 - self.action_ema_alpha) * self._ema_prev_targets
            )
            self._ema_prev_targets = pd_targets.copy()

        # `historical.processed_actions` is the actually-commanded position, i.e.
        # after the accel clamp and EMA filter -- not the raw policy output.
        self._prev_actions = pd_targets.copy()

        self._pd_targets_isaac = self._to_isaac_order(pd_targets)

        self._log_progress(dof_pos)
        self._record_trace(dof_pos, dof_vel, anchor_rot)

        self._frame_idx += 1
        self._total_steps += 1
        if self._frame_idx >= self.motion_player.total_frames:
            self._loop_idx += 1
            if self._loop_idx >= self.num_loops:
                self.done = True
            else:
                # Do NOT reset here: this runs inside the physics callback, and
                # writing articulation root/joint state mid-step is undefined in
                # Isaac Sim. Defer to the main loop.
                self._pending_reset = True

    def _log_progress(self, dof_pos: np.ndarray) -> None:
        """Periodically report root height + tracking error (cf. the MuJoCo driver)."""
        if not self.log_every:
            return
        ref_dof_pos = self.motion_player.get_state_at_frame(self._frame_idx)["dof_pos"]
        self._max_ref_err = max(
            self._max_ref_err, float(np.abs(dof_pos - ref_dof_pos).max())
        )
        self.max_ref_err_run = max(self.max_ref_err_run, self._max_ref_err)
        if self._frame_idx % self.log_every:
            return
        root_pos, _ = self.robot.get_world_pose()
        log.info(
            f"  step={self._total_steps:5d}  frame={self._frame_idx:4d}  "
            f"root_h={float(np.asarray(root_pos)[2]):.3f}  "
            f"max_ref_err={self._max_ref_err:.4f}"
        )

    def _record_trace(
        self, dof_pos: np.ndarray, dof_vel: np.ndarray, anchor_rot: np.ndarray
    ) -> None:
        """Append this control step's tracking error to the trace buffer.

        Columns match the cross-simulator comparison table in
        ``docs/isaacsim_g1_tracker_instability_findings.md`` so a run here is
        directly comparable to the IsaacLab and MuJoCo numbers.
        """
        if self.trace is None:
            return
        ref = self.motion_player.get_state_at_frame(self._frame_idx)
        root_pos, _ = self.robot.get_world_pose()
        self.trace.append(
            {
                "loop": self._loop_idx,
                "frame": self._frame_idx,
                "root_h": float(np.asarray(root_pos)[2]),
                "ref_h": float(ref["body_pos"][0][2]),
                "tilt": _tilt_deg_xyzw(anchor_rot),
                "ref_tilt": _tilt_deg_xyzw(ref["body_rot"][self.anchor_body_index]),
                "joint_err": float(np.abs(dof_pos - ref["dof_pos"]).mean()),
                "dof_vel_rms": float(np.sqrt(np.mean(dof_vel**2))),
                "dof_vel_peak": float(np.abs(dof_vel).max()),
            }
        )

    def forward(self, dt: float) -> None:
        """Physics-callback entry point -- called once per physics substep."""
        if self.done:
            return
        if self._decimation_counter % self.decimation == 0:
            self._compute_action()
        self.robot.apply_action(
            ArticulationAction(joint_positions=self._pd_targets_isaac)
        )
        self._decimation_counter += 1

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def log_summary(self, total_sim_ms: float) -> None:
        """Print the end-of-run stats, mirroring test_tracker_mujoco.py's summary.

        ``total_sim_ms`` is wall time spent inside ``world.step()`` -- i.e. all
        ``decimation`` physics substeps of a control step, plus rendering when
        windowed -- so both averages are per *control* step, as in MuJoCo.
        """
        steps = max(self._total_steps, 1)
        log.info(
            f"\n=== Done: {self._total_steps} steps over {self._loop_idx} loop(s) ===\n"
            f"  avg ONNX inference : {self.total_ort_ms / steps:.2f} ms/step\n"
            f"  avg physics        : {total_sim_ms / steps:.2f} ms/step\n"
            f"  max joint ref error: {self.max_ref_err_run:.4f} rad"
        )

    def write_trace(self, path: str) -> None:
        """Dump the per-control-step trace and print its summary."""
        if not self.trace:
            log.warning("No trace recorded -- nothing written.")
            return
        with open(path, "w") as f:
            json.dump(self.trace, f)

        col = {k: np.array([r[k] for r in self.trace]) for k in self.trace[0]}
        log.info(
            f"\n=== Tracking trace ({len(self.trace)} control steps) -> {path} ===\n"
            f"  mean joint err      : {col['joint_err'].mean():.4f} rad\n"
            f"  mean |root_h - ref| : {np.abs(col['root_h'] - col['ref_h']).mean():.4f} m\n"
            f"  mean |tilt - ref|   : {np.abs(col['tilt'] - col['ref_tilt']).mean():.2f} deg\n"
            f"  mean rms dof_vel    : {col['dof_vel_rms'].mean():.3f}\n"
            f"  peak dof_vel        : {col['dof_vel_peak'].max():.1f} rad/s"
        )


# ---------------------------------------------------------------------------
# Scene dressing
# ---------------------------------------------------------------------------


def _configure_physics_scene(world, robot_props: dict) -> None:
    """Match the PhysX scene settings training runs with.

    ``World``'s defaults differ from both ``physx_env.yaml`` and ProtoMotions'
    IsaacLab config in three ways that matter: contact stabilization is off, CCD
    is on, and the bounce threshold is 0 (so every contact is treated as
    bouncing). Solver type, friction offset and correlation distance already
    agree.

    Must be called **after** ``world.reset()``: ``PhysicsContext`` re-applies its
    own cached values when the sim starts playing, so anything written on the
    ``/physicsScene`` prim beforehand is silently dropped.
    """
    ctx = world.get_physics_context()

    # Training (and physx_env.yaml) run with stabilization on and CCD off.
    ctx.enable_stablization(True)  # NB: Isaac Sim spells it this way
    ctx.enable_ccd(False)
    ctx.set_solver_type("TGS")

    bounce = robot_props.get("bounce_threshold_velocity")
    if bounce is not None:
        ctx.set_bounce_threshold(float(bounce))

    log.info(
        "PhysX scene: stabilization=True ccd=False solver=TGS "
        f"bounce_threshold={ctx.get_bounce_threshold():.3f} "
        f"gpu_dynamics={ctx.is_gpu_dynamics_enabled()}"
    )


def setup_lighting_and_camera(target_pos) -> None:
    """Add a key light and frame the camera on the robot's start pose."""
    create_prim(
        "/World/KeyLight",
        "DistantLight",
        orientation=np.array([0.9239, 0.0, 0.3827, 0.0]),  # wxyz, ~45 deg tilt
        attributes={"inputs:intensity": 3000.0, "inputs:angle": 1.0},
    )
    create_prim(
        "/World/FillLight",
        "DomeLight",
        attributes={"inputs:intensity": 300.0},
    )

    target = np.asarray(target_pos, dtype=np.float32)
    set_camera_view(
        eye=target + np.array([2.5, -2.5, 1.2], dtype=np.float32),
        target=target,
        camera_prim_path="/OmniverseKit_Persp",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    onnx_path = str(args.onnx)
    yaml_path = onnx_path.replace(".onnx", ".yaml")
    usd_path = resolve_usd_path(args.usd)

    log.info(f"ONNX: {onnx_path}")
    log.info(f"USD:  {usd_path}")

    with open(yaml_path) as f:
        meta = yaml.safe_load(f)

    control_dt = meta["timing"]["control_dt"]

    # Timing. The exported YAML carries the *MuJoCo* rate (1 kHz / decimation 20)
    # because that is the driver it was written for; IsaacLab trains this policy at
    # 200 Hz / decimation 4 (`g1.py: isaaclab=IsaacLabSimParams(fps=200, decimation=4)`).
    # Prefer the robot config's Isaac rate, and let --physics-dt override both.
    robot_props = (
        load_robot_joint_properties(args.robot, meta["robot"]["joint_names"])
        if args.robot
        else {}
    )
    sim_fps = robot_props.get("sim_fps")
    if args.physics_dt:
        physics_dt = args.physics_dt
        log.info(f"physics_dt={physics_dt}s (from --physics-dt)")
    elif sim_fps:
        physics_dt = 1.0 / float(sim_fps)
        log.info(
            f"physics_dt={physics_dt}s (from the robot config's IsaacLab rate, "
            f"{sim_fps} Hz); the YAML's {meta['timing']['physics_dt']}s is MuJoCo's."
        )
    else:
        physics_dt = meta["timing"]["physics_dt"]
        log.warning(
            f"physics_dt={physics_dt}s taken from the YAML -- this is the MuJoCo "
            "rate. Pass --robot to resolve training's Isaac rate instead."
        )

    # Device: World defaults to the CPU/numpy backend, a different PhysX solver
    # path from the GPU pipeline training uses. --device cuda:0 selects that one.
    world_kwargs = {}
    if args.device:
        world_kwargs = {"device": args.device, "backend": "torch"}
    world = World(
        stage_units_in_meters=1.0,
        physics_dt=physics_dt,
        rendering_dt=control_dt,
        **world_kwargs,
    )
    world.scene.add_default_ground_plane(
        z_position=0.0,
        name="ground_plane",
        prim_path="/World/GroundPlane",
        static_friction=args.ground_friction,
        dynamic_friction=args.ground_friction,
        restitution=0.0,
    )

    policy = TrackerPolicy(
        meta=meta,
        prim_path=args.prim_path,
        usd_path=usd_path,
        onnx_path=onnx_path,
        motion_file=args.motion,
        root_prim_name=args.root_prim_name,
        cache_motion=args.cache_motion,
        action_ema_alpha=args.action_ema_alpha,
        motion_index=args.motion_index,
        robot_name=args.robot or None,
        physics_dt=physics_dt,
    )
    policy.num_loops = (
        args.loops if args.loops is not None else (1 if args.headless else 10_000_000)
    )
    if args.trace_out:
        policy.trace = []

    if not args.headless:
        # Frame the pelvis at its motion-frame-0 pose; the robot starts there.
        setup_lighting_and_camera(
            policy.motion_player.get_state_at_frame(0)["body_pos"][0]
        )

    world.reset()
    # Must come after reset(): PhysicsContext re-applies its own values when the
    # sim starts playing, silently discarding anything written on the stage before.
    _configure_physics_scene(world, robot_props)
    policy.initialize()
    policy.post_reset()
    world.add_physics_callback("tracker_policy_step", callback_fn=policy.forward)

    # Real-time pacing only makes sense when there is something to watch; a
    # headless run is a measurement, so it always goes as fast as it can.
    realtime = not args.no_realtime and not args.headless
    total_sim_ms = 0.0

    while simulation_app.is_running() and not policy.done:
        t0 = time.perf_counter()
        world.step(render=not args.headless)
        step_s = time.perf_counter() - t0
        total_sim_ms += step_s * 1000.0
        # Clip restarts are deferred out of the physics callback -- see
        # TrackerPolicy.reset_episode(). This is the only safe place to run them.
        if policy._pending_reset:
            policy._pending_reset = False
            policy.reset_episode()
        if realtime:
            sleep_time = physics_dt - step_s
            if sleep_time > 0:
                time.sleep(sleep_time)

    policy.log_summary(total_sim_ms)
    if args.trace_out:
        policy.write_trace(args.trace_out)

    simulation_app.close()


if __name__ == "__main__":
    main()
