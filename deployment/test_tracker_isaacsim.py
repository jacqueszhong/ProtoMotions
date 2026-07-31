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
as the MuJoCo script, so a single ``deployment/export_bm_tracker_onnx.py``
export works for both simulators.

The driver itself lives in ``deployment/isaacsim_tracker.py``, shared with
``deployment/play_policy_isaacsim.py`` -- read its module docstring for the
Isaac Sim specifics (DOF reordering, why body poses must come from the
physics view rather than USD, and which joint properties the deployment YAML
does not carry).

Requirements
------------
- A pre-exported ``unified_pipeline.onnx`` + ``.yaml`` sidecar (see
  ``deployment/export_bm_tracker_onnx.py``).
- A pre-built robot USD asset (see ``protomotions/data/assets/usd/<robot>/``,
  produced offline by ``usd_convert/``). Pass its path via ``--usd``.

Usage
-----
::

    python deployment/test_tracker_isaacsim.py \
        --onnx data/pretrained_models/motion_tracker/g1-bones-deploy/compiled_models/unified_pipeline.onnx \
        --motion data/motion_for_trackers/g1_random_subset_tiny.pt \
        --usd protomotions/data/assets/usd/g1_holo_compat/g1_holo_compat.usda \
        --robot g1
"""

from __future__ import annotations

import argparse
import logging
import os
import sys


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
        required=True,
        help=(
            "Path to the robot USD asset, e.g. "
            "protomotions/data/assets/usd/g1_holo_compat/g1_holo_compat.usda"
        ),
    )
    p.add_argument(
        "--robot",
        default=None,
        help=(
            "Robot name for protomotions.robot_configs (e.g. 'g1'). Supplies the "
            "per-joint armature, effort limits and solver iteration counts that "
            "the deployment YAML does not carry. Strongly recommended."
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
            "Physics timestep. Defaults to the YAML metadata, which carries MuJoCo's "
            "1 kHz rate rather than the backend the policy trained on. control_dt is "
            "preserved; decimation is re-derived."
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
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# SimulationApp must be created before importing any other omni/isaacsim module.
from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": args.headless})

import yaml  # noqa: E402
from isaacsim.core.api import World  # noqa: E402

from deployment.isaacsim_tracker import TrackerPolicy, resolve_usd_path  # noqa: E402

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")


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

    physics_dt = args.physics_dt or meta["timing"]["physics_dt"]
    control_dt = meta["timing"]["control_dt"]

    world = World(
        stage_units_in_meters=1.0, physics_dt=physics_dt, rendering_dt=control_dt
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
        robot_name=args.robot,
        physics_dt=physics_dt,
    )
    policy.num_loops = (
        args.loops if args.loops is not None else (1 if args.headless else 10_000_000)
    )

    world.reset()
    policy.initialize()
    policy.post_reset()
    world.add_physics_callback("tracker_policy_step", callback_fn=policy.forward)

    while simulation_app.is_running() and not policy.done:
        world.step(render=not args.headless)

    simulation_app.close()


if __name__ == "__main__":
    main()
