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

``--control-in-loop`` switches to IsaacLab's shape instead (decide once, then
``decimation`` substeps), which removes a real quarter-control-period phase
error against the motion reference but is less robust over repeated clips --
see :meth:`TrackerPolicy.control_step`.

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

- **Joint properties**: training applies per-joint armature, effort limits,
  velocity limits, Coulomb friction and solver iteration counts from the
  ProtoMotions robot config (see
  ``protomotions/simulator/isaaclab/utils/scene.py``).  The exported
  deployment YAML does not carry all of these, so when a robot name is
  available they are resolved from ``protomotions.robot_configs`` at runtime
  by :func:`load_robot_joint_properties`.  This matters for the G1, whose PD
  gains are *derived from* armature (``kp = armature * w_n**2``), so the
  closed-loop joint dynamics are wrong without it.

- **What training overrides in the USD is not always an unset default.**  The
  G1 asset *explicitly authors*
  ``physxArticulation:enabledSelfCollisions = False``, while training spawns it
  with ``enabled_self_collisions=True`` (``scene.py`` ←
  ``g1.py: self_collisions``).  Inheriting the asset's value therefore made this
  driver solve a different collision problem from training -- limbs passed
  through each other -- which a robust pretrained policy absorbed and a
  locally-trained one did not (it fell after ~84 control steps).
  :meth:`TrackerPolicy._author_articulation_properties` now authors it from the
  robot config.  The lesson generalises: diff against what training *configures*,
  not against what the USD leaves unauthored.  ``--dump-physx-properties``
  prints both sides of that diff.

- **Debugging a policy that works in IsaacLab but not here**: record IsaacLab's
  own trajectory with ``deployment/trace_tracker_isaaclab.py``, check the export
  against it with ``deployment/check_onnx_parity.py``, then replay its actions
  open-loop here with ``--action-tape``/``--init-state`` to separate physics
  from feedback.  Calibrate first -- ``trace_tracker_isaaclab.py --action-tape``
  replaying into IsaacLab must return 0.0, which is what makes a nonzero number
  here meaningful.  Full write-up:
  ``logs/isaacsim_g1_tracker_instability_findings.md``.

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
    p.add_argument(
        "--joint-friction",
        choices=("auto", "on", "off"),
        default="auto",
        help=(
            "Apply the robot config's per-joint Coulomb friction "
            "(control_info[j].friction, 0.1 on every G1 joint). 'auto' (default) "
            "applies it whenever the robot config supplies it. Note training's "
            "PhysX actually holds 0.0 -- IsaacLab drops the actuator's non-'_sim' "
            "friction field -- so 'off' is the training-parity setting; it is not "
            "the default because the extra dissipation measurably helps this "
            "driver stay upright. Pair 'off' with --control-in-loop."
        ),
    )
    p.add_argument(
        "--action-tape",
        type=str,
        default=None,
        help=(
            "Replay a recorded processed_action sequence open-loop instead of "
            "running the policy; the ONNX session is never called. Takes the "
            ".npz written by deployment/trace_tracker_isaaclab.py. Isolates "
            "physics/actuator differences from closed-loop feedback."
        ),
    )
    p.add_argument(
        "--init-state",
        type=str,
        default=None,
        help=(
            "Start from the post-reset state in this init_state.json (written by "
            "deployment/trace_tracker_isaaclab.py) instead of motion frame 0."
        ),
    )
    p.add_argument(
        "--init-z-offset",
        type=float,
        default=0.0,
        help=(
            "Add this to the spawn height. ProtoMotions resets with "
            "env.config.ref_respawn_offset = 0.05 m above the reference; the "
            "driver spawns flush. Use to test that difference empirically."
        ),
    )
    p.add_argument(
        "--onnx-inputs-out",
        type=str,
        default=None,
        help=(
            "Write every control step's assembled ONNX input tensors to this "
            ".npz. Diff against trace_tracker_isaaclab.py's ctx__* arrays to "
            "test observation *assembly* -- check_onnx_parity.py only tests the "
            "graph. Most informative with --init-state, where step 0 starts "
            "from IsaacLab's own post-reset state."
        ),
    )
    p.add_argument(
        "--ground",
        choices=("plane", "trimesh"),
        default="plane",
        help=(
            "Ground surface. 'plane' is an analytic ground plane -- the "
            "deployment-faithful default, since the real robot walks on a floor. "
            "'trimesh' rebuilds ProtoMotions' own terrain mesh, the surface "
            "training and IsaacLab inference actually use, for parity ablation; "
            "it needs --resolved-configs."
        ),
    )
    p.add_argument(
        "--resolved-configs",
        type=str,
        default=None,
        help=(
            "Path to a run's resolved_configs_inference.pt. Only used by "
            "--ground trimesh, which reads the terrain config out of it."
        ),
    )
    p.add_argument(
        "--control-in-loop",
        action="store_true",
        default=False,
        help=(
            "Compute the action in the outer loop and then run `decimation` "
            "physics substeps, the way IsaacLab does, instead of driving from a "
            "physics callback. Removes the quarter-control-period phase offset "
            "against the motion reference clock: worth 13%% of tracking error on "
            "a single clip (0.0903 -> 0.0787 rad), but measurably *less robust* "
            "over repeated clips, which is why it is not the default. See "
            "TrackerPolicy.control_step."
        ),
    )
    p.add_argument(
        "--author-collider-offsets",
        action="store_true",
        default=False,
        help=(
            "Write the robot config's contact_offset/rest_offset onto every "
            "collider. Off by default because training does not actually get "
            "them: IsaacLab requests 0.02/0.0 but its writer cannot author "
            "instance proxies, so PhysX auto-derives both there. Leaving this "
            "off is what matches training; turn it on to ablate the difference."
        ),
    )
    p.add_argument(
        "--dump-physx-properties",
        action="store_true",
        default=False,
        help=(
            "After post_reset(), read every per-joint property back from the "
            "live physics view, diff it element-wise against the robot config, "
            "and exit. The last-resort check when an open-loop replay diverges "
            "and no single ablation explains it."
        ),
    )
    p.add_argument(
        "--tape-divergence-out",
        type=str,
        default=None,
        help=(
            "With --action-tape, write the per-step divergence against the "
            "recorded IsaacLab trajectory as JSON."
        ),
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
from pxr import PhysxSchema, Usd, UsdPhysics, UsdShade  # noqa: E402

from deployment.motion_utils import MotionPlayer  # noqa: E402
from deployment.state_utils import (  # noqa: E402
    apply_heading_offset_np,
    compute_root_local_ang_vel_np,
    compute_yaw_offset_np,
    make_trace_row,
    summarize_trace,
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


def _bound_physics_material(prim):
    """Return the physics material prim bound to ``prim``, or ``None``.

    Physics material bindings live on the ``"physics"`` binding *purpose*
    (``material:binding:physics``), not the default (render) purpose, and USD
    resolves them by walking up the namespace -- a binding on an ancestor
    scope applies to every collider beneath it. ``ComputeBoundMaterial`` does
    that walk; a plain ``GetDirectBinding`` on the shape would miss it and
    report "unbound" for assets that bind at the robot root.

    Note what this can *not* see: PhysX's own compiled-in fallback material,
    used when a shape resolves to no material at all. That value is not in the
    stage, so an empty result here means "falls back to whatever PhysX ships",
    which is the open question this dump exists to settle.
    """
    try:
        binding_api = UsdShade.MaterialBindingAPI(prim)
        material, _rel = binding_api.ComputeBoundMaterial(
            UsdShade.Tokens.physics
            if hasattr(UsdShade.Tokens, "physics")
            else "physics"
        )
    except Exception as e:  # pragma: no cover - USD version dependent
        log.debug(f"{prim.GetPath()}: material binding query failed: {e}")
        return None
    if not material:
        return None
    mat_prim = material.GetPrim()
    if not mat_prim.IsValid() or not mat_prim.HasAPI(UsdPhysics.MaterialAPI):
        return None
    return mat_prim


def _describe_physics_material(mat_prim) -> str:
    """One-line summary of a physics material prim's friction/restitution stack.

    Prints ``authored`` per attribute because an unauthored attribute is the
    whole point of this dump: it means the value shown is a schema default that
    no layer of the stack chose, and that the other simulator may well have
    chosen differently.
    """
    if mat_prim is None:
        return "<no material bound -- PhysX built-in fallback>"
    fields = []
    for attr_name in (
        "physics:staticFriction",
        "physics:dynamicFriction",
        "physics:restitution",
        "physxMaterial:frictionCombineMode",
        "physxMaterial:restitutionCombineMode",
    ):
        attr = mat_prim.GetAttribute(attr_name)
        short = attr_name.split(":")[-1]
        if not attr:
            fields.append(f"{short}=<undeclared>")
            continue
        value = attr.Get()
        mark = "" if attr.IsAuthored() else "*"
        if isinstance(value, float):
            fields.append(f"{short}={value:.4g}{mark}")
        else:
            fields.append(f"{short}={value}{mark}")
    return f"{mat_prim.GetPath()}  " + " ".join(fields)


def _describe_collider_offsets(prim) -> str:
    """Report a collider's contact/rest offsets and whether anyone authored them.

    PhysX **sums** a contacting pair's offsets, so an unauthored offset on
    either shape is auto-derived from that shape's size -- and auto-derivation
    differs between an analytic plane and a triangle mesh. Two sims can pin the
    robot side identically and still disagree about where the floor is.
    """
    fields = []
    for attr_name in ("physxCollision:contactOffset", "physxCollision:restOffset"):
        attr = prim.GetAttribute(attr_name)
        short = attr_name.split(":")[-1]
        if not attr:
            fields.append(f"{short}=<undeclared>")
            continue
        value = attr.Get()
        mark = "" if attr.IsAuthored() else "*(auto)"
        fields.append(f"{short}={value}{mark}")
    return " ".join(fields)


def _find_prims_with_api(stage, root_path: str, api, limit: int = 0) -> list:
    """Collect prims under ``root_path`` that have ``api`` applied.

    Traverses instance proxies: these robot assets are spawned
    ``make_instanceable``, so the colliders are instance proxies that a default
    ``Usd.PrimRange`` walks straight past -- the same trap documented in
    ``_author_body_properties``.
    """
    found = []
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        return found
    for prim in Usd.PrimRange(
        root, Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)
    ):
        if prim.HasAPI(api):
            found.append(prim)
            if limit and len(found) >= limit:
                break
    return found


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
        # Coulomb joint friction; training passes this to ImplicitActuatorCfg.
        "friction": _column("friction"),
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
        # ArticulationRootPropertiesCfg equivalent. The G1 USD authors
        # enabledSelfCollisions=False; training overrides it to True.
        "self_collisions": getattr(asset, "self_collisions", None),
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
        joint_friction_mode: str = "auto",
        action_tape: str | None = None,
        init_state: str | None = None,
        init_z_offset: float = 0.0,
        author_collider_offsets: bool = False,
    ) -> None:
        self.author_collider_offsets = bool(author_collider_offsets)
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
        self._articulation_path = articulation_path
        log.info(f"Articulation root prim: {articulation_path}")
        self._author_drive_gains(prim_path)
        self._author_body_properties(prim_path)
        self._author_articulation_properties(articulation_path)
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
        # Per-control-step ONNX input tensors; None disables (see --onnx-inputs-out).
        self.onnx_input_log: list | None = None
        # Set from inside the physics callback, consumed by the main loop between
        # world.step() calls -- see reset_episode()'s note on why the reset itself
        # must not happen mid-step.
        self._pending_reset = False

        # Run-wide statistics for the end-of-run summary (never reset per episode).
        self.total_ort_ms = 0.0
        self.max_ref_err_run = 0.0

        # Open-loop replay + initial-condition overrides (all inert by default,
        # so the closed-loop path stays byte-for-byte what it was).
        self.joint_friction_mode = joint_friction_mode
        self.init_z_offset = float(init_z_offset)
        self._init_state = self._load_init_state(init_state)
        # Populated by _load_action_tape with the IsaacLab trajectory the replay
        # is scored against; stays empty when running closed-loop.
        self._tape_ref: dict = {}
        self._action_tape = self._load_action_tape(action_tape)
        self.divergence: list | None = None

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
    # Open-loop replay setup
    # ------------------------------------------------------------------

    def _load_init_state(self, path: str | None) -> dict | None:
        """Load an ``init_state.json`` and check its joint order against ours.

        The joint-name assertion is a hard error on purpose. A silent reorder
        here would write every DOF's target onto the wrong joint, and the result
        -- a robot that flails and falls -- looks exactly like the physics bug
        this replay exists to isolate.
        """
        if path is None:
            return None
        with open(path) as f:
            init = json.load(f)
        names = list(init.get("joint_names", []))
        if names != self.joint_names:
            raise ValueError(
                f"--init-state joint order does not match the policy's.\n"
                f"  init_state: {names}\n"
                f"  policy:     {self.joint_names}"
            )
        log.info(
            f"Init state from {path}: root_pos="
            f"{np.round(init['root_pos'], 4).tolist()} "
            f"(recorded respawn offset {init.get('respawn_root_offset')})"
        )
        return init

    def _load_action_tape(self, path: str | None) -> np.ndarray | None:
        """Load the recorded ``processed_action`` sequence for open-loop replay.

        Args:
            path: Path to ``context_isaaclab.npz`` from
                ``deployment/trace_tracker_isaaclab.py``.

        Returns:
            ``[steps, num_dofs]`` float32 array in *policy* joint order, or None.
        """
        if path is None:
            return None
        data = np.load(path, allow_pickle=True)
        if "processed_action" not in data.files:
            raise ValueError(f"{path} has no 'processed_action' array")
        tape = np.asarray(data["processed_action"], dtype=np.float32)
        if tape.ndim != 2 or tape.shape[1] != self.num_dofs:
            raise ValueError(
                f"Action tape has shape {tape.shape}; expected [steps, {self.num_dofs}]"
            )
        if "meta__joint_names" in data.files:
            names = [str(n) for n in data["meta__joint_names"]]
            if names != self.joint_names:
                raise ValueError(
                    f"Action tape joint order does not match the policy's.\n"
                    f"  tape:   {names}\n"
                    f"  policy: {self.joint_names}"
                )
        # The reference trajectory the replay is scored against.
        self._tape_ref = {
            k: np.asarray(data[k])
            for k in ("state__dof_pos", "state__dof_vel", "state__root_pos")
            if k in data.files
        }
        log.info(
            f"Action tape from {path}: {tape.shape[0]} control steps, "
            "policy is out of the loop."
        )
        return tape

    def _replay_tape_step(self) -> None:
        """Record the current state, then apply the next recorded action. No policy.

        **Phase alignment is the whole point of this method's shape.** Isaac
        Sim's physics callback fires *after* the substep it is passed (measured;
        ``SimulationContext.add_physics_callback``'s "called before each physics
        step" docstring is wrong), so a naive replay applies tape action *k* over
        ``[5 + 20k, 25 + 20k]`` ms while IsaacLab applied it over
        ``[20k, 20k + 20]``. That constant one-substep lead is enough on its own
        to make an open-loop replay of a fast clip diverge, which would be read
        as an actuator difference that is not there.

        So the tape is anchored at reset instead: frame 0 is recorded and
        ``tape[0]`` applied *before* any physics runs (see ``reset_episode``),
        and every later call lands on a whole control-step boundary
        (``t = 20k`` ms) -- exactly IsaacLab's read/apply points.

        Deliberately does **not** advance ``_prev_actions`` / ``_prev_pd`` /
        ``_prev_prev_pd`` / ``_ema_prev_targets`` / the heading offset: there is
        no feedback loop to maintain, and touching them would let the filter
        state of a run with no policy diverge from one with a policy for reasons
        unrelated to physics.
        """
        if self._frame_idx >= len(self._action_tape):
            self.done = True
            return

        dof_pos_isaac = _to_numpy(self.robot.get_joint_positions())
        dof_vel_isaac = _to_numpy(self.robot.get_joint_velocities())
        dof_pos = dof_pos_isaac[self._isaac_to_policy].astype(np.float32)
        dof_vel = dof_vel_isaac[self._isaac_to_policy].astype(np.float32)

        if self.divergence is not None and "state__dof_pos" in self._tape_ref:
            ref_dof_pos = self._tape_ref["state__dof_pos"][self._frame_idx]
            delta = np.abs(dof_pos - ref_dof_pos)
            row = {
                "frame": self._frame_idx,
                "max_dof_delta": float(delta.max()),
                "mean_dof_delta": float(delta.mean()),
                "worst_joint": self.joint_names[int(delta.argmax())],
                "dof_vel_rms": float(np.sqrt(np.mean(dof_vel**2))),
            }
            if "state__dof_vel" in self._tape_ref:
                ref_vel = self._tape_ref["state__dof_vel"][self._frame_idx]
                row["ref_dof_vel_rms"] = float(np.sqrt(np.mean(ref_vel**2)))
            if "state__root_pos" in self._tape_ref:
                root_pos, _ = self.robot.get_world_pose()
                row["root_h"] = float(_to_numpy(root_pos)[2])
                row["ref_root_h"] = float(
                    self._tape_ref["state__root_pos"][self._frame_idx][2]
                )
            self.divergence.append(row)

        self._record_trace(
            dof_pos,
            dof_vel,
            self._read_anchor_rot(
                wxyz_to_xyzw(_to_numpy(self.robot.get_world_pose()[1])).astype(
                    np.float32
                )
            ),
        )

        # Apply this frame's action only after its state has been recorded, so
        # the recorded state is the one the action acts on -- IsaacLab's order.
        self._pd_targets_isaac = self._to_isaac_order(
            self._action_tape[self._frame_idx]
        )

        self._frame_idx += 1
        self._total_steps += 1
        if self._frame_idx >= len(self._action_tape):
            self.done = True

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
        # Collider offsets are opt-in because *training does not actually apply
        # them* (E1, measured). IsaacLab asks for contact_offset=0.02 /
        # rest_offset=0.0 (`isaaclab/utils/scene.py:177-180`, and they are in the
        # resolved config), but `modify_collision_properties` is wrapped in
        # `apply_nested`, which cannot author an instance proxy -- and every
        # collider on this asset is one. IsaacLab logs it and moves on:
        #   "Could not perform 'modify_collision_properties' on any prims under
        #    '/World/envs/env_0/Robot' ... The desired attribute exists on an
        #    instanced prim."
        # Read back from a live training-config run, the foot collider's
        # contactOffset/restOffset are *undeclared*, i.e. PhysX auto-derives both.
        #
        # This driver de-instances (below) so its writes succeed -- which means
        # authoring them makes it diverge from training rather than match it. The
        # A11 lesson said to diff against what training configures; this is its
        # sharper form: diff against what training's solver actually receives.
        if self.author_collider_offsets:
            collision_attrs = {
                "physxCollision:contactOffset": props.get("contact_offset"),
                "physxCollision:restOffset": props.get("rest_offset"),
            }
        else:
            collision_attrs = {}
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

    def _author_articulation_properties(self, articulation_path: str) -> None:
        """Apply training's ``ArticulationRootPropertiesCfg`` to the articulation root.

        Only ``enabled_self_collisions`` needs handling here -- the solver
        iteration counts in the same IsaacLab config block are already applied
        through the physics view in :meth:`_apply_solver_iterations`.

        This one is not a case of an unauthored attribute falling back to a
        default that happens to differ. The G1 USD authors
        ``physxArticulation:enabledSelfCollisions = False`` **explicitly**, while
        ``protomotions/simulator/isaaclab/utils/scene.py`` passes
        ``enabled_self_collisions=robot_config.asset.self_collisions``, which is
        ``True`` for the G1. So training simulates leg-against-leg and
        arm-against-torso contact and this driver did not -- the two solve
        different collision problems for any motion where limbs come close.
        """
        props = self.joint_properties or {}
        self_collisions = props.get("self_collisions")
        if self_collisions is None:
            return

        stage = get_current_stage()
        prim = stage.GetPrimAtPath(articulation_path)
        if not prim.IsValid():
            log.warning(
                f"Cannot author articulation properties: '{articulation_path}' is "
                "not a valid prim."
            )
            return

        PhysxSchema.PhysxArticulationAPI.Apply(prim)
        applied = _set_prim_attr(
            prim, "physxArticulation:enabledSelfCollisions", bool(self_collisions)
        )
        log.info(
            f"Articulation self-collisions -> {bool(self_collisions)} "
            f"({'applied' if applied else 'attribute not declared'})."
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
                self._for_view(
                    view, np.array([int(pos_iters)], dtype=np.int32), "int32"
                )
            )
        if vel_iters is not None:
            view.set_solver_velocity_iteration_counts(
                self._for_view(
                    view, np.array([int(vel_iters)], dtype=np.int32), "int32"
                )
            )
        if pos_iters is not None or vel_iters is not None:
            log.info(f"Solver iterations: position={pos_iters} velocity={vel_iters}")

    def _to_isaac_order(self, policy_ordered) -> np.ndarray:
        """Scatter a policy-ordered per-DOF array into Isaac Sim's DOF order."""
        out = np.zeros(self.num_dofs, dtype=np.float32)
        out[self._isaac_to_policy] = np.asarray(policy_ordered, dtype=np.float32)
        return out

    @staticmethod
    def _for_view(view, array, dtype: str = "float32"):
        """Convert a NumPy array to whatever tensor type the view's backend wants.

        ``World(device="cuda:0", backend="torch")`` swaps the view's
        ``_backend_utils`` for the torch implementation, whose helpers call
        ``.cpu()`` on whatever they are handed -- so passing a NumPy array into
        any ``set_*`` raises ``AttributeError: 'numpy.ndarray' object has no
        attribute 'cpu'`` deep inside Isaac Sim. The default CPU/numpy backend
        accepts NumPy directly, which is why the GPU path stayed broken while
        the default one worked.
        """
        backend = getattr(view, "_backend_utils", None)
        convert = getattr(backend, "convert", None)
        if convert is None:
            return array
        return convert(array, device=getattr(view, "_device", None), dtype=dtype)

    def _robot_array(self, array, dtype: str = "float32"):
        """Convert an array for this articulation's backend. See :meth:`_for_view`."""
        return self._for_view(self.robot._articulation_view, array, dtype)

    def _apply_joint_properties(self, view) -> None:
        """Apply PD gains, armature, effort and velocity limits in Isaac Sim DOF order."""
        stiffness_isaac = self._to_isaac_order(self.stiffness)
        damping_isaac = self._to_isaac_order(self.damping)

        controller = self.robot.get_articulation_controller()
        controller.set_effort_modes("force")
        controller.switch_control_mode("position")
        # switch_control_mode() restores the USD default gains, so set ours after.
        view.set_gains(
            self._for_view(view, stiffness_isaac[None]),
            self._for_view(view, damping_isaac[None]),
        )

        props = self.joint_properties or {}

        # Effort limits: prefer the YAML, fall back to the robot config. The
        # exporter historically wrote null here (it read `.effort` instead of
        # `.effort_limit`), which left the joints with unbounded torque.
        effort_limits = self.effort_limits
        if effort_limits is None:
            effort_limits = props.get("effort_limit")
        if effort_limits is not None:
            view.set_max_efforts(
                self._for_view(view, self._to_isaac_order(effort_limits)[None])
            )
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
            view.set_armatures(
                self._for_view(view, self._to_isaac_order(armature)[None])
            )
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
            view.set_max_joint_velocities(
                self._for_view(view, self._to_isaac_order(velocity_limits)[None])
            )
            log.info("Applied per-joint velocity limits.")
        else:
            log.warning(
                "No velocity limits available -- joints run unclamped, unlike training."
            )

        self._apply_joint_friction(view, props)

    def _apply_joint_friction(self, view, props: dict) -> None:
        """Apply the robot config's per-joint Coulomb friction.

        **Training does not actually run with this** -- measured, not inferred.
        ProtoMotions passes ``friction=control_info[j].friction`` (0.1 on every
        G1 joint) into ``ImplicitActuatorCfg``
        (``protomotions/simulator/isaaclab/utils/scene.py``), but IsaacLab only
        forwards its ``*_sim``-suffixed actuator fields to PhysX; plain
        ``friction`` stays in the actuator model, and the implicit actuator never
        applies it because PhysX owns the PD loop. Reading
        ``get_dof_friction_coefficients()`` on both stacks through the same
        tensor API gives **driver 0.1 / IsaacLab 0.0 on all 29 joints** -- the
        one per-joint property that does not agree.

        It is applied by default anyway. Two reasons: real G1 joints do have
        Coulomb friction, so 0.1 is a defensible hardware model and this driver's
        job is to predict hardware; and empirically the extra dissipation keeps
        this driver upright. Over 5 consecutive clips, ``--joint-friction off``
        falls on two of them (root height 0.072 m, tilt 90 deg) where the default
        never falls. ``--joint-friction off`` selects training parity instead.

        Do not use ``off`` without ``--control-in-loop``: on a single clip,
        removing friction while the physics-callback phase offset is still
        present costs 0.0903 -> 0.1045 rad and tilt 2.71 -> 9.96 deg, because the
        friction was damping the instability that phase error causes. With the
        phase fixed the two score the same on one clip (0.0787 vs 0.0790 rad) --
        but see :meth:`control_step` for why that pairing is still not the
        default.

        The read-back is not decoration: ``ArticulationView._on_post_reset()``
        re-applies the articulation's *default* gains, and anything set before it
        runs is silently reverted. Logging what PhysX actually holds is the only
        way to know this landed -- the same trap ``post_reset()`` already
        documents for the gains.
        """
        friction = props.get("friction")
        if self.joint_friction_mode == "off" or friction is None:
            if self.joint_friction_mode == "on":
                log.warning(
                    "--joint-friction on, but the robot config supplies no "
                    "per-joint friction; joints run frictionless, unlike training."
                )
            return

        view.set_friction_coefficients(
            self._for_view(view, self._to_isaac_order(friction)[None])
        )
        try:
            readback = _to_numpy(view.get_friction_coefficients()).reshape(-1)
            log.info(
                f"Applied per-joint friction (PhysX read-back: "
                f"min={readback.min():.4f} max={readback.max():.4f})."
            )
        except Exception as e:  # pragma: no cover - depends on Isaac Sim version
            log.warning(f"Applied per-joint friction, but read-back failed: {e}")

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
        if self._init_state is not None:
            # Replay IsaacLab's own post-reset state, read back from its
            # simulator. Preserves the write order below exactly -- pose, then
            # joint positions, then velocities, then the FK refresh.
            init = self._init_state
            root_pos = np.asarray(init["root_pos"], dtype=np.float32)
            root_quat_xyzw = np.asarray(init["root_rot_xyzw"], dtype=np.float32)
            dof_pos_policy = np.asarray(init["dof_pos"], dtype=np.float32)
            dof_vel_policy = np.asarray(init["dof_vel"], dtype=np.float32)
            root_lin_vel = np.asarray(init["root_lin_vel"], dtype=np.float32)
            root_ang_vel = np.asarray(init["root_ang_vel"], dtype=np.float32)
        else:
            frame0 = self.motion_player.get_state_at_frame(0)
            root_pos = np.asarray(frame0["body_pos"][0], dtype=np.float32)
            root_quat_xyzw = frame0["body_rot"][0]
            dof_pos_policy = frame0["dof_pos"]
            dof_vel_policy = frame0["dof_vel"]
            root_lin_vel = np.asarray(frame0["body_vel"][0], dtype=np.float32)
            root_ang_vel = np.asarray(frame0["body_ang_vel"][0], dtype=np.float32)

        # ProtoMotions resets the robot ref_respawn_offset (0.05 m) above the
        # reference pose; this driver spawns flush. --init-z-offset makes that
        # difference testable instead of assumed.
        if self.init_z_offset:
            root_pos = root_pos.copy()
            root_pos[2] += self.init_z_offset

        root_quat_wxyz = np.asarray(root_quat_xyzw)[[3, 0, 1, 2]]
        self.robot.set_world_pose(
            position=self._robot_array(root_pos),
            orientation=self._robot_array(root_quat_wxyz),
        )

        dof_pos_isaac = self._to_isaac_order(dof_pos_policy)
        self.robot.set_joint_positions(self._robot_array(dof_pos_isaac))

        # Start from the reference *velocities*, not from rest. ProtoMotions'
        # reference-state initialization seeds root and joint velocities from the
        # motion (compute_ref_reset_state), and the policy was trained on that
        # distribution -- dropping a moving clip in at zero velocity is off-
        # distribution for exactly the first few control steps that decide whether
        # the episode stays on its feet.
        self.robot.set_joint_velocities(
            self._robot_array(self._to_isaac_order(dof_vel_policy))
        )
        self.robot.set_linear_velocity(self._robot_array(root_lin_vel))
        self.robot.set_angular_velocity(self._robot_array(root_ang_vel))

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

        # Open-loop replay is anchored here, before any physics runs: record
        # frame 0 from the reset state and load tape[0], so the tape's action k
        # spans the same window IsaacLab applied it over. See _replay_tape_step.
        if self._action_tape is not None:
            self._replay_tape_step()

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
        dof_pos_isaac = _to_numpy(self.robot.get_joint_positions())
        dof_vel_isaac = _to_numpy(self.robot.get_joint_velocities())
        dof_pos = dof_pos_isaac[self._isaac_to_policy].astype(np.float32)
        dof_vel = dof_vel_isaac[self._isaac_to_policy].astype(np.float32)

        _, root_quat_wxyz = self.robot.get_world_pose()
        root_quat_wxyz = _to_numpy(root_quat_wxyz)
        root_quat = wxyz_to_xyzw(root_quat_wxyz).astype(np.float32)

        # Angular-velocity frame: `SingleArticulation.get_angular_velocity()` returns
        # the root body's angular velocity in the **world** frame, which is the
        # convention `compute_root_local_ang_vel_np` expects -- it rotates the vector
        # by the inverse root rotation to produce the root-local vector the policy was
        # trained on. So this is passed through as-is, *not* pre-rotated. (Isaac Sim
        # has no body-frame variant of this getter; if a future API returns body-frame
        # velocity, the inverse rotation below must be dropped, not kept.)
        root_ang_vel_world = np.asarray(
            _to_numpy(self.robot.get_angular_velocity()), dtype=np.float32
        )
        root_local_ang_vel = compute_root_local_ang_vel_np(
            rigid_body_rot=root_quat[None, :],
            rigid_body_ang_vel=root_ang_vel_world[None, :],
            root_body_index=0,
        )

        anchor_rot = self._read_anchor_rot(root_quat)

        return dof_pos, dof_vel, anchor_rot, root_local_ang_vel

    def _compute_action(self) -> None:
        # Never reached under --action-tape: forward() routes the open-loop
        # replay to _replay_tape_step instead, so the ONNX session is never
        # called and no filter state is touched.
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
        # Record the assembled inputs so they can be diffed against IsaacLab's
        # recorded ctx__* arrays. check_onnx_parity.py proves the *graph* is
        # right by feeding it IsaacLab's observations; it says nothing about
        # whether this driver builds those observations the same way from its
        # own state reads. Paired with --init-state, step 0 is the clean test:
        # the physical state is identical by construction there, so any
        # difference in these tensors is assembly, not physics.
        if self.onnx_input_log is not None:
            self.onnx_input_log.append(
                {
                    "frame": int(self._frame_idx),
                    "inputs": {k: v.copy() for k, v in onnx_inputs.items()},
                }
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

    def control_step(self, world, render: bool) -> None:
        """Run one control step in IsaacLab's shape: decide once, then substep.

        Opt in with ``--control-in-loop``; the default is still :meth:`forward`.
        **Read the robustness caveat at the end before making this the default.**

        The default path -- driving from ``world.add_physics_callback`` (see
        :meth:`forward`) -- puts a fixed phase offset between the robot
        and the motion reference. That callback fires *after* its substep, and
        :meth:`forward` tests the decimation counter before incrementing, so
        callback 0 computes reference frame 0 against the state at ``t = 1*dt``
        and every later step inherits the same lag: the driver's state runs 5 ms
        (a quarter of a control period) ahead of the reference clock.

        That offset is harmless for the *controller* -- read-then-apply is
        self-consistent at any phase -- but the motion reference is an exogenous
        clock, so a fixed phase error against it is real tracking error rather
        than a choice of coordinates.

        IsaacLab's loop (``simulator.py:579-589``) computes the observation after
        the last substep and applies the resulting target from the next one, with
        the first observation taken at ``t = 0`` straight out of reset. Mirroring
        it here means the reset also lands naturally between control steps, so
        this path calls :meth:`reset_episode` directly instead of deferring
        through ``_pending_reset``.

        **Robustness caveat -- why this is not the default.** It scores better on
        one clip and is *less robust over many*. Measured on ``g1_walk_box``,
        5 consecutive loops, headless:

        =============================== ======= ==================
        config                          falls   mean joint err
        =============================== ======= ==================
        callback + friction (default)   none    0.0905 rad
        this + ``--joint-friction auto``  loop 3  0.0795 rad
        this + ``--joint-friction off``   loops 1,3  0.0837 rad
        =============================== ======= ==================

        A "fall" is the pelvis dropping to ~0.1 m with ~90 deg tilt. They happen
        near the *end* of a clip (~step 240 of 253), and loop 0 is always clean --
        so a single-loop score, which is what ``--headless`` runs by default,
        cannot see this at all. That is exactly how it got made the default once
        and had to be reverted; score ``--loops 5`` before touching it again.

        Note the loops are not independent even though ``reset_episode`` rewrites
        the full root/joint state: the per-loop errors differ (0.0787, 0.0770,
        0.0773, 0.0868, 0.0776), so something in PhysX -- solver warm-start or
        contact caches -- survives the reset. Whether the instability is inherent
        to removing the phase lag or an artefact of that residue is unresolved.
        """
        if self.done:
            return
        if self._action_tape is not None:
            self._replay_tape_step()
        else:
            self._compute_action()

        action = ArticulationAction(
            joint_positions=self._robot_array(self._pd_targets_isaac)
        )
        for substep in range(self.decimation):
            self.robot.apply_action(action)
            # Render only on the final substep: IsaacLab renders at the control
            # rate, and rendering every substep would quadruple the frame cost
            # for frames nobody asked for.
            world.step(render=render and substep == self.decimation - 1)

        # The clip boundary was reached inside _compute_action, which only raised
        # the flag. Here we are between control steps, which is exactly where
        # reset_episode() requires to be called.
        if self._pending_reset:
            self._pending_reset = False
            self.reset_episode()

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
            f"root_h={float(_to_numpy(root_pos)[2]):.3f}  "
            f"max_ref_err={self._max_ref_err:.4f}"
        )

    def _record_trace(
        self, dof_pos: np.ndarray, dof_vel: np.ndarray, anchor_rot: np.ndarray
    ) -> None:
        """Append this control step's tracking error to the trace buffer.

        Columns come from ``deployment.state_utils.make_trace_row``, shared with
        the MuJoCo and IsaacLab harnesses, so a run here is directly comparable
        to the numbers in ``logs/isaacsim_g1_tracker_instability_findings.md``.
        """
        if self.trace is None:
            return
        ref = self.motion_player.get_state_at_frame(self._frame_idx)
        root_pos, _ = self.robot.get_world_pose()
        self.trace.append(
            make_trace_row(
                loop=self._loop_idx,
                frame=self._frame_idx,
                root_h=float(_to_numpy(root_pos)[2]),
                ref_h=float(ref["body_pos"][0][2]),
                anchor_rot_xyzw=anchor_rot,
                ref_anchor_rot_xyzw=ref["body_rot"][self.anchor_body_index],
                dof_pos=dof_pos,
                ref_dof_pos=ref["dof_pos"],
                dof_vel=dof_vel,
            )
        )

    def forward(self, dt: float) -> None:
        """Physics-callback entry point -- called once per physics substep.

        Note the callback fires *after* the substep it is passed, despite
        ``SimulationContext.add_physics_callback``'s docstring saying otherwise
        (measured). The closed-loop path is unaffected -- it reads state and
        immediately applies the resulting target, so it is self-consistent
        whatever the phase -- but the open-loop replay has to count completed
        substeps to land on IsaacLab's control-step boundaries, hence the
        separate branch.
        """
        if self.done:
            return
        if self._action_tape is not None:
            self._decimation_counter += 1
            if self._decimation_counter % self.decimation == 0:
                self._replay_tape_step()
        else:
            if self._decimation_counter % self.decimation == 0:
                self._compute_action()
            self._decimation_counter += 1
        self.robot.apply_action(
            ArticulationAction(
                joint_positions=self._robot_array(self._pd_targets_isaac)
            )
        )

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

    def dump_physx_properties(self) -> None:
        """Diff every per-joint property PhysX holds against the robot config.

        Must run **after** ``post_reset()``: ``ArticulationView._on_post_reset()``
        re-applies the articulation's USD-authored defaults, so anything read
        before it is not what the solver will actually use.

        Reads are reported in *policy* joint order so a row lines up with the
        joint names the ONNX contract uses.
        """
        view = self.robot._articulation_view
        props = self.joint_properties or {}

        def _getter(name: str):
            """Resolve a view getter lazily; not every setter has a matching one.

            ``set_max_joint_velocities`` exists with no ``get_`` counterpart in
            this Isaac Sim version, so resolving the whole table eagerly makes
            one missing getter abort the entire dump.
            """

            def call():
                fn = getattr(view, name, None)
                if fn is None:
                    raise AttributeError(f"view has no {name}()")
                return fn()

            return call

        readers = {
            "stiffness": (lambda: view.get_gains()[0], np.asarray(self.stiffness)),
            "damping": (lambda: view.get_gains()[1], np.asarray(self.damping)),
            "armature": (_getter("get_armatures"), props.get("armature")),
            "effort_limit": (_getter("get_max_efforts"), props.get("effort_limit")),
            "velocity_limit": (
                _getter("get_max_joint_velocities"),
                props.get("velocity_limit"),
            ),
            "friction": (
                _getter("get_friction_coefficients"),
                props.get("friction"),
            ),
        }

        # Articulation-level settings. IsaacLab authors these through
        # ArticulationRootPropertiesCfg (scene.py); the USD leaves
        # enabledSelfCollisions unauthored, so the driver inherits PhysX's
        # default rather than the configured value.
        stage = get_current_stage()
        root_prim = stage.GetPrimAtPath(self._articulation_path)
        for attr_name in (
            "physxArticulation:enabledSelfCollisions",
            "physxArticulation:solverPositionIterationCount",
            "physxArticulation:solverVelocityIterationCount",
            "physxArticulation:sleepThreshold",
            "physxArticulation:stabilizationThreshold",
        ):
            attr = root_prim.GetAttribute(attr_name)
            if attr:
                log.info(
                    f"  {attr_name:52s} = {attr.Get()} (authored={attr.IsAuthored()})"
                )
            else:
                log.info(f"  {attr_name:52s} = <not declared>")

        log.info("\n=== PhysX read-back vs robot config (policy joint order) ===")
        for name, (reader, expected) in readers.items():
            try:
                actual = _to_numpy(reader()).reshape(-1)[self._isaac_to_policy]
            except Exception as e:  # pragma: no cover - version dependent
                log.warning(f"  {name:15s} read-back failed: {e}")
                continue
            if expected is None:
                log.info(
                    f"  {name:15s} physx=[{actual.min():.4f}, {actual.max():.4f}] "
                    "  (config supplies no value to compare)"
                )
                continue
            expected = np.asarray(expected, dtype=np.float64).reshape(-1)
            delta = np.abs(actual - expected)
            worst = int(delta.argmax())
            verdict = "OK" if delta.max() < 1e-4 else "MISMATCH"
            log.info(
                f"  {name:15s} {verdict:8s} max|d|={delta.max():.6f} "
                f"(worst {self.joint_names[worst]}: physx={actual[worst]:.4f} "
                f"config={expected[worst]:.4f})"
            )

        # Same read as trace_tracker_isaaclab.py's dump, through the same
        # `_physics_view` tensor API and printed in *sim* joint order, so the two
        # logs can be diffed PhysX-against-PhysX. The table above compares PhysX
        # to the robot config, which by construction cannot catch a difference
        # that is shared between the config and this driver's reading of it.
        log.info("\n=== Per-joint PhysX properties (driver, sim joint order) ===")
        physics_view = getattr(view, "_physics_view", None)
        log.info(f"  sim joint order: {list(view.dof_names)}")
        for label, method in (
            ("stiffness", "get_dof_stiffnesses"),
            ("damping", "get_dof_dampings"),
            ("armature", "get_dof_armatures"),
            ("friction", "get_dof_friction_coefficients"),
            ("max_velocity", "get_dof_max_velocities"),
            ("max_effort", "get_dof_max_forces"),
        ):
            fn = getattr(physics_view, method, None) if physics_view else None
            if fn is None:
                log.info(f"  {label:14s} <view has no {method}()>")
                continue
            try:
                vals = _to_numpy(fn()).reshape(-1)
            except Exception as e:  # pragma: no cover - version dependent
                log.warning(f"  {label:14s} read-back failed: {e}")
                continue
            log.info(
                f"  {label:14s} min={vals.min():.4f} max={vals.max():.4f} "
                f"values={np.round(vals, 4).tolist()}"
            )

        self.dump_material_stack()

    def dump_material_stack(self) -> None:
        """Read back both operands of every foot-ground friction pair (E1).

        Effective friction at a footfall is
        ``combine(robot_shape_material, ground_material)``. This driver has
        historically pinned only the ground operand: the G1 USD binds no
        physics material to any of its colliders, and unlike IsaacLab -- which
        spawns a ``RigidBodyMaterialCfg()`` (0.5 / 0.5 / 0.0, ``average``) and
        binds it to the *physics scene prim* as the documented fallback for
        unbound shapes -- ``World(..., set_defaults=True)`` authors no scene
        default at all. The robot side therefore resolves to PhysX's compiled-in
        fallback, whose value is not statically resolvable from the stage.

        ``_physics_view.get_material_properties()`` reads what the solver
        actually holds, so it settles that question directly: it is the same
        read IsaacLab's simulator uses when it applies friction randomization.

        Must run after ``post_reset()`` -- the physics view does not exist, and
        USD-authored defaults are not yet re-applied, before then.
        """
        log.info("\n=== Material stack (foot-ground friction pair) ===")

        stage = get_current_stage()
        view = self.robot._articulation_view

        # --- robot side, straight from the solver ---------------------------
        try:
            mats = _to_numpy(view._physics_view.get_material_properties())
            flat = mats.reshape(-1, mats.shape[-1])
            log.info(
                f"  robot physics-view materials: shape={tuple(mats.shape)} "
                f"(env x shape x [static, dynamic, restitution])"
            )
            for col, label in enumerate(
                ("static_friction", "dynamic_friction", "restitution")
            ):
                if col >= flat.shape[1]:
                    break
                col_vals = flat[:, col]
                uniq = np.unique(np.round(col_vals, 6))
                log.info(
                    f"    {label:18s} min={col_vals.min():.4f} max={col_vals.max():.4f} "
                    f"unique={uniq[:8].tolist()}{' ...' if len(uniq) > 8 else ''}"
                )
        except Exception as e:  # pragma: no cover - version dependent
            log.warning(f"  robot physics-view material read-back failed: {e}")

        # --- robot side, as authored on the stage ---------------------------
        robot_colliders = _find_prims_with_api(
            stage, self._prim_path, UsdPhysics.CollisionAPI
        )
        bound = [p for p in robot_colliders if _bound_physics_material(p) is not None]
        log.info(
            f"  robot colliders: {len(robot_colliders)} total, "
            f"{len(bound)} with a bound physics material"
        )
        # A foot shape is the one that matters; name-match rather than guess an index.
        foot_prims = [
            p
            for p in robot_colliders
            if any(k in p.GetPath().pathString.lower() for k in ("ankle_roll", "foot"))
        ]
        sample = (
            foot_prims[0]
            if foot_prims
            else (robot_colliders[0] if robot_colliders else None)
        )
        if sample is not None:
            log.info(f"  sample foot collider : {sample.GetPath()}")
            log.info(
                f"    material : {_describe_physics_material(_bound_physics_material(sample))}"
            )
            log.info(f"    offsets  : {_describe_collider_offsets(sample)}")

        # --- scene-default material (IsaacLab's mechanism) ------------------
        scene_prims = _find_prims_with_api(stage, "/", UsdPhysics.Scene, limit=4)
        if not scene_prims:  # PhysicsScene is a typed prim, not an applied API
            scene_prims = [
                p for p in stage.Traverse() if p.GetTypeName() == "PhysicsScene"
            ]
        for scene_prim in scene_prims:
            log.info(f"  physics scene prim   : {scene_prim.GetPath()}")
            log.info(
                f"    scene default material : "
                f"{_describe_physics_material(_bound_physics_material(scene_prim))}"
            )

        # --- ground side ----------------------------------------------------
        # Either ground works here: /World/GroundPlane for --ground plane,
        # /World/ground for --ground trimesh (the path ProtoMotions uses).
        ground_colliders = _find_prims_with_api(
            stage, "/World/GroundPlane", UsdPhysics.CollisionAPI
        ) or _find_prims_with_api(stage, "/World/ground", UsdPhysics.CollisionAPI)
        if not ground_colliders:
            log.warning("  no ground collider found")
        for prim in ground_colliders[:2]:
            log.info(
                f"  ground collider      : {prim.GetPath()} ({prim.GetTypeName()})"
            )
            log.info(
                f"    material : {_describe_physics_material(_bound_physics_material(prim))}"
            )
            log.info(f"    offsets  : {_describe_collider_offsets(prim)}")

        # --- where is the floor, really? -----------------------------------
        # The traces show the driver's robot sitting ~1.4 cm above its reference
        # for a whole episode while IsaacLab's sits on it, which is a
        # floor-position difference rather than a transient. Report the ground
        # collider's world z and the resting foot height so that offset is read
        # off directly instead of inferred from tracking error.
        from pxr import UsdGeom

        xform_cache = UsdGeom.XformCache()
        for prim in ground_colliders[:2]:
            tf = xform_cache.GetLocalToWorldTransform(prim)
            log.info(
                f"  ground collider world z: {tf.ExtractTranslation()[2]:.6f} "
                f"({prim.GetPath()})"
            )
        try:
            link_pos = _to_numpy(view._physics_view.get_link_transforms())[0, :, :3]
            body_names = list(view.body_names)
            for name in body_names:
                if "ankle_roll" in name:
                    z = float(link_pos[body_names.index(name)][2])
                    log.info(f"  {name:24s} world z = {z:.6f}")
        except Exception as e:  # pragma: no cover - version dependent
            log.warning(f"  foot link height read-back failed: {e}")

        log.info(
            "  (trailing '*' = value is the schema default, unauthored by any layer; "
            "'*(auto)' = PhysX derives it from shape size)"
        )

    def write_divergence(self, path: str) -> None:
        """Dump the open-loop divergence against the IsaacLab trajectory.

        Reports the first control step at which ``|Δdof_pos|∞`` crosses 0.02,
        0.05 and 0.10 rad, and which joint led. Contact-rich open-loop replay
        always diverges eventually; **when** and on which joint is the signal,
        not whether.
        """
        if not self.divergence:
            log.warning("No divergence recorded -- nothing written.")
            return
        with open(path, "w") as f:
            json.dump(self.divergence, f)

        delta = np.array([r["max_dof_delta"] for r in self.divergence])
        lines = []
        for threshold in (0.02, 0.05, 0.10):
            crossed = np.nonzero(delta > threshold)[0]
            if len(crossed):
                k = int(crossed[0])
                lines.append(
                    f"  first |d dof|inf > {threshold:.2f} rad : step {k:4d} "
                    f"({k * self.control_dt:.2f}s, joint "
                    f"{self.divergence[k]['worst_joint']})"
                )
            else:
                lines.append(f"  first |d dof|inf > {threshold:.2f} rad : never")

        rms = np.array([r["dof_vel_rms"] for r in self.divergence])
        ref_rms = np.array([r.get("ref_dof_vel_rms", np.nan) for r in self.divergence])
        log.info(
            f"\n=== Open-loop divergence ({len(self.divergence)} steps) -> {path} ===\n"
            + "\n".join(lines)
            + f"\n  mean |d dof|inf     : {delta.mean():.4f} rad\n"
            f"  final |d dof|inf    : {delta[-1]:.4f} rad\n"
            f"  mean dof_vel_rms    : driver {rms.mean():.3f}  "
            f"isaaclab {np.nanmean(ref_rms):.3f}"
        )

    def write_onnx_inputs(self, path: str) -> None:
        """Dump the recorded per-step ONNX input tensors as a stacked .npz."""
        if not self.onnx_input_log:
            log.warning("No ONNX inputs recorded -- nothing written.")
            return
        names = list(self.onnx_input_log[0]["inputs"].keys())
        arrays = {
            f"in__{name}": np.concatenate(
                [row["inputs"][name] for row in self.onnx_input_log], axis=0
            )
            for name in names
        }
        arrays["frame"] = np.asarray(
            [row["frame"] for row in self.onnx_input_log], dtype=np.int64
        )
        np.savez_compressed(path, **arrays)
        log.info(
            f"ONNX inputs ({len(self.onnx_input_log)} control steps, "
            f"{len(names)} tensors) -> {path}"
        )

    def write_trace(self, path: str) -> None:
        """Dump the per-control-step trace and print its summary."""
        if not self.trace:
            log.warning("No trace recorded -- nothing written.")
            return
        with open(path, "w") as f:
            json.dump(self.trace, f)

        log.info(
            f"\n=== Tracking trace ({len(self.trace)} control steps) -> {path} ===\n"
            + summarize_trace(self.trace)
        )


# ---------------------------------------------------------------------------
# Scene dressing
# ---------------------------------------------------------------------------


def _configure_physics_scene(world, robot_props: dict) -> None:
    """Match the PhysX scene settings training runs with.

    ``World``'s defaults differ from ProtoMotions' IsaacLab config in two ways
    that matter: CCD is on, and the bounce threshold is 0 (so every contact is
    treated as bouncing). Solver type, friction offset and correlation distance
    already agree.

    **Stabilization is deliberately not touched here.** Every layer of the stack
    leaves ``physxScene:enableStabilization`` off -- the USD schema default is 0,
    ``PhysicsContext`` sets ``False`` on the ``set_defaults=True, sim_params=None``
    path that ``World(...)`` takes, ``isaaclab/sim/simulation_cfg.py`` defaults it
    to ``False``, and ProtoMotions never overrides it. An earlier revision of this
    function turned it *on*, which was a unilateral deviation from training, not a
    match to it. (``PhysicsContext``'s class docstring lists
    ``stabilization_enabled`` among its defaults; the docstring is wrong, the code
    is right -- do not trust it.)

    The ``enable_ccd(False)`` call below is load-bearing for the same reason in
    reverse: the ``set_defaults`` path enables CCD on the CPU pipeline, which
    training does not.

    Must be called **after** ``world.reset()``: ``PhysicsContext`` re-applies its
    own cached values when the sim starts playing, so anything written on the
    ``/physicsScene`` prim beforehand is silently dropped.
    """
    ctx = world.get_physics_context()

    ctx.enable_ccd(False)
    ctx.set_solver_type("TGS")

    bounce = robot_props.get("bounce_threshold_velocity")
    if bounce is not None:
        ctx.set_bounce_threshold(float(bounce))

    log.info(
        "PhysX scene: ccd=False solver=TGS "
        f"stabilization={ctx.is_stablization_enabled()} "
        f"bounce_threshold={ctx.get_bounce_threshold():.3f} "
        f"gpu_dynamics={ctx.is_gpu_dynamics_enabled()}"
    )


def _add_physics_material(
    stage, path: str, static_friction: float, dynamic_friction: float, combine: str
):
    """Author a physics material and return its prim, mirroring IsaacLab's terrain one.

    ``RigidBodyMaterialCfg`` writes both the ``UsdPhysics`` friction values and
    the ``PhysxMaterialAPI`` combine modes; ``add_default_ground_plane`` writes
    only the former and leaves the combine modes at the schema default. They
    agree today by luck -- the schema default happens to be ``average``, which is
    what this terrain config asks for -- but PhysX arbitrates a mismatched pair by
    taking the higher-priority mode (``average < min < multiply < max``), so an
    unauthored mode flips the result silently the moment either side changes.
    Author both explicitly here.
    """
    mat_prim = UsdShade.Material.Define(stage, path).GetPrim()
    UsdPhysics.MaterialAPI.Apply(mat_prim)
    PhysxSchema.PhysxMaterialAPI.Apply(mat_prim)
    _set_prim_attr(mat_prim, "physics:staticFriction", float(static_friction))
    _set_prim_attr(mat_prim, "physics:dynamicFriction", float(dynamic_friction))
    _set_prim_attr(mat_prim, "physics:restitution", 0.0)
    _set_prim_attr(mat_prim, "physxMaterial:frictionCombineMode", combine)
    _set_prim_attr(mat_prim, "physxMaterial:restitutionCombineMode", combine)
    return mat_prim


def add_protomotions_trimesh_ground(
    resolved_configs_path: str, friction: float
) -> None:
    """Spawn ProtoMotions' own flat terrain mesh instead of an analytic plane (E4).

    Training and IsaacLab inference walk on ``/World/ground/terrain/mesh``, a
    triangle mesh built by ``components/terrains/terrain.py``; this driver
    defaults to an analytic ``Plane`` because that is the deployment-faithful
    choice -- the real robot walks on a floor, not on a tessellated height field.
    But box-vs-plane and box-vs-triangle-mesh are *different contact-generation
    code paths* in PhysX (analytic manifold vs PCM triangle clipping), and both
    shapes leave ``contactOffset``/``restOffset`` unauthored, so PhysX
    auto-derives them per shape type. This reproduces training's surface exactly
    so that difference can be measured rather than argued about.

    For this config the terrain is flat, and the mesh optimizer merges coplanar
    regions aggressively: the result is 56 vertices / 84 triangles spanning
    280 x 310 m, all at z = 0. It is recentred on the origin here so the robot's
    spawn (near XY = 0) lands well inside it -- on a flat terrain the XY offset
    is not physically meaningful, but falling off the edge of the mesh would be.
    """
    import torch
    from pxr import UsdGeom

    from protomotions.components.terrains.terrain import Terrain

    resolved = torch.load(resolved_configs_path, map_location="cpu", weights_only=False)
    terrain_config = resolved["terrain"]
    terrain = Terrain(config=terrain_config, num_envs=1, device=torch.device("cpu"))

    vertices = np.asarray(terrain.vertices, dtype=np.float64).copy()
    triangles = np.asarray(terrain.triangles, dtype=np.int32)
    vertices[:, 2] += float(terrain_config.sim_config.height_offset)
    # Recentre on the origin; the driver spawns the robot near XY = 0 while
    # ProtoMotions spawns it in the middle of a 280 x 310 m terrain.
    vertices[:, 0] -= 0.5 * (vertices[:, 0].min() + vertices[:, 0].max())
    vertices[:, 1] -= 0.5 * (vertices[:, 1].min() + vertices[:, 1].max())

    stage = get_current_stage()
    mesh = UsdGeom.Mesh.Define(stage, "/World/ground/terrain/mesh")
    mesh.CreatePointsAttr([tuple(v) for v in vertices])
    mesh.CreateFaceVertexCountsAttr([3] * len(triangles))
    mesh.CreateFaceVertexIndicesAttr(triangles.reshape(-1).tolist())

    mesh_prim = mesh.GetPrim()
    UsdPhysics.CollisionAPI.Apply(mesh_prim)
    PhysxSchema.PhysxCollisionAPI.Apply(mesh_prim)
    # A static triangle mesh collides as a triangle mesh; "none" is the USD token
    # for "do not approximate", which is what the terrain importer relies on.
    mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(mesh_prim)
    mesh_collision.CreateApproximationAttr().Set("none")

    sim_config = terrain_config.sim_config
    combine = getattr(sim_config.combine_mode, "value", sim_config.combine_mode)
    material = _add_physics_material(
        stage,
        "/World/ground/terrain/physicsMaterial",
        static_friction=friction,
        dynamic_friction=friction,
        combine=str(combine),
    )
    binding_api = UsdShade.MaterialBindingAPI.Apply(mesh_prim)
    binding_api.Bind(
        UsdShade.Material(material),
        bindingStrength=UsdShade.Tokens.weakerThanDescendants,
        materialPurpose="physics",
    )

    log.info(
        f"Ground: ProtoMotions trimesh ({len(vertices)} verts, {len(triangles)} tris, "
        f"z={vertices[:, 2].min():.3f}) with friction {friction} / {combine}"
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
    if args.ground == "trimesh":
        if not args.resolved_configs:
            raise SystemExit(
                "--ground trimesh needs --resolved-configs pointing at the run's "
                "resolved_configs_inference.pt (that is where the terrain config lives)"
            )
        add_protomotions_trimesh_ground(args.resolved_configs, args.ground_friction)
    else:
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
        joint_friction_mode=args.joint_friction,
        action_tape=args.action_tape,
        init_state=args.init_state,
        init_z_offset=args.init_z_offset,
        author_collider_offsets=args.author_collider_offsets,
    )
    policy.num_loops = (
        args.loops if args.loops is not None else (1 if args.headless else 10_000_000)
    )
    if args.trace_out:
        policy.trace = []
    if args.onnx_inputs_out:
        policy.onnx_input_log = []
    if args.tape_divergence_out:
        if args.action_tape is None:
            raise SystemExit("--tape-divergence-out requires --action-tape")
        policy.divergence = []

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
    if args.dump_physx_properties:
        policy.dump_physx_properties()
        simulation_app.close()
        return
    if not args.control_in_loop:
        world.add_physics_callback("tracker_policy_step", callback_fn=policy.forward)

    # Real-time pacing only makes sense when there is something to watch; a
    # headless run is a measurement, so it always goes as fast as it can.
    realtime = not args.no_realtime and not args.headless
    total_sim_ms = 0.0

    # Each iteration advances one *control* step in --control-in-loop mode and one
    # *physics* substep otherwise, so the real-time budget differs by decimation.
    loop_period = control_dt if args.control_in_loop else physics_dt

    while simulation_app.is_running() and not policy.done:
        t0 = time.perf_counter()
        if args.control_in_loop:
            policy.control_step(world, render=not args.headless)
        else:
            world.step(render=not args.headless)
        step_s = time.perf_counter() - t0
        total_sim_ms += step_s * 1000.0
        # Clip restarts are deferred out of the physics callback -- see
        # TrackerPolicy.reset_episode(). This is the only safe place to run them.
        # (control_step drains the flag itself, so this is a no-op there.)
        if policy._pending_reset:
            policy._pending_reset = False
            policy.reset_episode()
        if realtime:
            sleep_time = loop_period - step_s
            if sleep_time > 0:
                time.sleep(sleep_time)

    policy.log_summary(total_sim_ms)
    if args.onnx_inputs_out:
        policy.write_onnx_inputs(args.onnx_inputs_out)
    if args.trace_out:
        policy.write_trace(args.trace_out)
    if args.tape_divergence_out:
        policy.write_divergence(args.tape_divergence_out)

    simulation_app.close()


if __name__ == "__main__":
    main()
