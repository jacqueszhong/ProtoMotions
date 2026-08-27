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

This driver uses IsaacLab's shape instead (decide once, then ``decimation``
substeps), which removes a real quarter-control-period phase error against the
motion reference. ``--no-control-in-loop`` restores the callback shape -- see
:meth:`TrackerPolicy.control_step`.

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
  here with ``--action-tape``/``--init-state`` to separate physics from feedback.
  Prefer ``--resync-state`` over a bare tape replay: an open-loop replay's
  divergence at step *k* is the accumulated sum of *k* one-step errors and cannot
  be attributed to any of them, while ``--resync-state`` restarts every step from
  IsaacLab's recorded state and so yields N independent one-step experiments.
  Calibrate first -- ``trace_tracker_isaaclab.py --action-tape`` replaying into
  IsaacLab must return 0.0, which is what makes a nonzero number here meaningful.
  Full write-up: ``logs/isaacsim_g1_tracker_instability_findings.md`` plus the
  round-by-round logs in ``~/dev_nv/logs/isaacsim_tracker_parity_*``.

  **Resolved in round 4 -- the cause was the drive's velocity target.**
  Isaac Sim's ``Articulation.set_joint_velocities`` writes the joint velocity to
  ``set_dof_velocity_targets`` as well as to ``set_dof_velocities``, and nothing
  in this driver ever wrote a velocity target again, so every episode ran with
  PhysX damping toward the *reset* velocity instead of toward zero. See
  :meth:`TrackerPolicy._zero_drive_velocity_targets`. Fixing it took the
  one-step resync error from 0.0291 to 0.00291 rad, the joint-velocity ratio
  from 1.441 to 0.991, and closed-loop tracking on ``g1_walk_box`` from
  0.1009 rad with 4 falls in 5 loops to **0.0454 rad with none** -- against
  IsaacLab's own 0.0437 on the same clip.

  **Settled by rounds 1-3, and still true -- do not re-litigate**: every
  per-joint property (stiffness, damping, armature, friction, max velocity, max
  effort) and every per-link rigid-body property (mass, inertia, COM, damping,
  max linear/angular velocity, depenetration velocity, sleep/stabilization
  thresholds, gravity, retainAccelerations) reads back **identical**
  PhysX-to-PhysX in identical sim order, ``body_names`` included --
  ``--dump-physx-properties`` against ``trace_tracker_isaaclab.py
  --dump-material-stack`` diffs to nothing. In particular ``maxAngularVelocity``
  is 1000 deg/s on both, so the suspected velocity clip is shared, not a
  difference. Contact was exonerated too: the resync divergence was *lowest* in
  double stance. The lesson those three rounds paid for is that no amount of
  property diffing finds a difference in a *value nobody thought to read* --
  what found it was ``--drive-probe``, which puts the robot out of contact and
  isolates the drive's stiffness and damping terms from each other.

  **Round 5, the SOMA sitting demo -- two causes, and neither works alone.**
  ``soma_sit_1`` held its posture under IsaacLab and curled up here. The gap was
  27.4 deg of mean torso tilt over 149 control steps; it is now **0.57 deg**, and
  it took both of:

  1. *The CPU solver is not training's solver.* ``--device`` unset leaves
     ``physxScene:enableGPUDynamics=False`` with the MBP broadphase; training
     runs GPU dynamics with the GPU broadphase. Per control step from IsaacLab's
     own state (``--resync-state``) that is worth mean |d root_quat| **4.51 deg
     on CPU vs 0.057 deg on cuda:0** and |d root_pos| 5.3 mm vs 0.25 mm; the
     contact-free ``--drive-probe`` splits the same way (4.70 deg vs 0.070 deg),
     so it is the articulation solver, not contact generation. This is the value
     the scene-prim dump was added to catch -- see
     :func:`deployment.physx_probe.dump_physics_scene`, and note that the earlier
     rounds' whole-prim comparison had never been done.
  2. *The spawn lift.* ProtoMotions resets at ``env.config.ref_respawn_offset``
     (0.05 m) above the reference and applies the same offset to the **scene**
     (``BaseEnv.move_reset_robot_obj_states_to_respawn_position``); this driver
     spawned flush. ``--init-z-offset`` now defaults to ``auto`` and reads the
     value out of ``--resolved-configs``, lifting robot and objects together.

  They are not additive, which is why round 3 measured each alone and called both
  negative: on CPU the lift is *worse* than flush (37.97 deg vs 27.43 deg), and on
  GPU without the lift the robot slides off the chair (end root_h 0.176 m vs
  IsaacLab's 0.552 m). Together: 0.57 deg mean tilt, 0.8 mm mean root height,
  0.0058 rad mean joint error, stable over 5 loops. The same pair also improves
  the standing box demo that already looked fine (mean |d tilt| 3.60 -> 0.96 deg).

  The **defaults are unchanged** -- ``--device`` stays CPU because that is the
  deployment target -- so a run that takes the CPU path now says so in its log
  (:func:`_warn_if_diverging_from_training`) instead of only on screen.

  Still open, and small: a constant ~0.066 rad offset on the wrist DOFs, present
  in every ``--drive-probe`` case including the contact-free ones and unchanged
  by the solver device. It moves the root by 0.25 mm and 0.057 deg per control
  step, i.e. it is not what the sitting failure was about.

- **``--scenes-file`` spawns a SceneLib scene's objects.**  Without it this
  driver authors robot + ground + lights and nothing else, so a policy trained
  against a scene -- ``examples/experiments/mimic/g1_pick_box.py``, whose G1
  picks up a box along a reference trajectory -- had nothing to reach for.  With
  it, the scene's objects become real PhysX rigid bodies, are teleported onto
  their SceneLib reference pose at every clip restart, and are scored in
  ``--trace-out`` (``obj_pos_err``, ``obj_rot_err_deg``).  Points worth knowing:

  - *The policy does not observe them.*  ``g1_pick_box.py`` fixes the box pose
    per motion via ``Scene.humanoid_motion_id`` and the exporters carry no
    ``scene_obs`` channel, so objects here are physics and scoring only -- the
    ONNX input contract is untouched.
  - *Two unrelated meanings of "static".*  ``ObjectOptions.fix_base_link`` is
    the physical one and maps to ``physics:kinematicEnabled``, matching
    IsaacLab's ``RigidBodyPropertiesCfg(kinematic_enabled=...)``.
    ``SceneLib._is_static_object`` (``not obj.has_motion()``) is a data
    property that only gates the z respawn lift in ``get_scene_pose``.  A
    single-frame object is still an ordinary body that falls.
  - *A kinematic object needs its USD transform authored, not just its physics
    pose.*  PhysX writes poses back out to USD/Fabric only for dynamic actors;
    a kinematic one is assumed to be driven *from* USD, so a physics-view-only
    write leaves it colliding in the right place and **rendering at its spawn
    pose**.  Every box scene predating the sit task was ``fix_base_link=False``
    and so never hit it.  :meth:`SceneObjects._write_kinematic_xforms` writes
    both.  Same asymmetry for velocities: PhysX errors out on
    ``setLinearVelocity`` for a kinematic body, so ``reset`` zeroes only the
    dynamic subset.
  - *Collider offsets are authored on objects but not on the robot*, which is
    not the contradiction it looks like -- see
    :func:`_author_scene_object_physics`.  In short: the robot's colliders are
    instance proxies IsaacLab's writer cannot reach, so training's request
    never lands and matching training means not authoring; freshly authored
    object prims are not instance proxies, so IsaacLab's request *does* land
    there.
  - *``--resync-state`` is allowed only for an all-kinematic scene.*  The action
    tape has no object channel, so resyncing the robot while a *dynamic* object
    drifts would quietly invalidate that mode's one-step-experiment claim and the
    combination exits.  ``fix_base_link`` objects cannot drift, so a scene made
    only of those (the sitting chair) is accepted -- see
    :func:`_reject_resync_with_dynamic_objects`.  Plain ``--action-tape`` warns
    and runs.

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

- **``world.step(render=True)`` steps physics more than once.**  With a non-zero
  ``rendering_dt`` it delegates the frame to Kit's own loop, which advances
  ``int(rendering_dt / physics_dt)`` substeps -- 4 here.  It is a *frame* step,
  not a physics step.  Never call it; use ``world.step(render=False)`` plus
  :func:`_render`, which is physics-free.  This is the one difference between a
  windowed and a ``--headless`` run of the same command, and it used to make the
  windowed run fall within a couple of control steps.

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
  produced offline by ``usd_convert/``). Taken from the YAML's
  ``robot.usd_path``; override with ``--usd``.

Robots and observation families
-------------------------------
Nothing here is robot-specific: the driver reads the robot's identity, USD,
body/joint names and gains out of the exported YAML, and its physics
parameters out of ``protomotions.robot_configs`` under the YAML's
``robot.robot_name``. Two observation families are supported, chosen per run
from the semantic keys in ``_runtime.onnx_name_to_in_key``:

- **reduced coordinates** (BeyondMimic; the G1 deploy tracker) -- joint state
  plus the anchor body's rotation, and an action history of *processed* PD
  targets. Only the root and anchor poses are ever read from the sim.
- **max coordinates** (the SOMA trackers) -- every body's world pose and
  velocity plus ``ground_heights``, matched against full-body reference poses,
  with an action history of *raw* pre-tanh policy outputs. Needs the
  articulation's link buffers and a body-order remap; see
  :meth:`TrackerPolicy._read_body_state` and :func:`build_onnx_inputs`.

Two joint encodings are likewise handled: single-axis revolute joints (G1, one
prim per DOF) and 3-DOF D6 joints (soma23, one prim per body carrying
``rotX``/``rotY``/``rotZ`` drives). See :meth:`TrackerPolicy._author_drive_gains`.

Usage
-----
::

    python deployment/test_tracker_isaacsim.py --robot g1 \
        --onnx data/pretrained_models/motion_tracker/g1-bones-deploy/compiled_models/unified_pipeline.onnx \
        --motion data/motion_for_trackers/g1_bones_seed_mini.pt

``--robot`` is only needed for YAMLs predating the ``robot.robot_name`` field
(the committed G1 export is one); a freshly exported YAML carries it::

    python deployment/export_bm_tracker_onnx_isaacsim.py \
        --checkpoint data/pretrained_models/motion_tracker/soma-bones/last.ckpt
    python deployment/test_tracker_isaacsim.py \
        --onnx data/pretrained_models/motion_tracker/soma-bones/compiled_models/unified_pipeline.onnx \
        --motion data/motion_for_trackers/soma23_bones_seed_mini.pt \
        --headless --loops 5 --trace-out /tmp/soma_trace.json

With a scene (objects spawn, reset per loop, and are scored in the trace)::

    python deployment/test_tracker_isaacsim.py \
        --onnx <pick_box_export>/unified_pipeline.onnx \
        --motion <pickup_motion>.pt \
        --scenes-file /tmp/box.pt --loops 3 --trace-out /tmp/box_trace.json

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
        default=None,
        help=(
            "Path to the robot USD asset (absolute, or relative to the repo root, "
            "or to protomotions/data/assets). Default (unset) takes the YAML "
            "metadata's robot.usd_path, falling back to --robot's robot config."
        ),
    )
    p.add_argument(
        "--robot",
        default=None,
        help=(
            "Robot name for protomotions.robot_configs (e.g. 'g1', 'soma23'). "
            "Supplies the per-joint armature, effort limits, solver iteration "
            "counts and physics rate that the deployment YAML does not carry. "
            "Default (unset) takes the YAML metadata's robot.robot_name; pass "
            "'' to skip it entirely, at the cost of joint dynamics that no "
            "longer match training."
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
        "--scenes-file",
        type=str,
        default=None,
        help=(
            "Path to a ProtoMotions scenes .pt (data/scripts/create_box_scene.py). "
            "Spawns the scene's objects as real PhysX rigid bodies, resets them "
            "from the SceneLib reference at every clip restart, and scores their "
            "pose error in --trace-out. Same flag name as inference_agent.py; "
            "'none'/'null' disables. Without it no ProtoMotions module is "
            "imported at all."
        ),
    )
    p.add_argument(
        "--scenes-asset-root",
        type=str,
        default=None,
        help=(
            "Root for resolving relative mesh paths inside the scenes file. "
            "Defaults to SceneLib's own guess, the *grandparent* of the scenes "
            "file -- which is asymmetric with the save side (paths are stored "
            "relative to the file's parent), so pass this when a mesh scene "
            "fails to load."
        ),
    )
    p.add_argument(
        "--scene-index",
        type=int,
        default=None,
        help=(
            "Which scene in the file to spawn. Default (unset) picks the scene "
            "whose humanoid_motion_id equals --motion-index, falling back to 0 "
            "with a warning when nothing is paired."
        ),
    )
    p.add_argument(
        "--scene-object-z-offset",
        type=float,
        default=0.0,
        help=(
            "Add this to the objects' spawn height, the object analogue of "
            "--init-z-offset. Passed to SceneLib.get_scene_pose(respawn_offset=), "
            "so like ProtoMotions' env.config.ref_object_respawn_offset it lifts "
            "only objects that carry a motion trajectory. Defaults to 0.0, "
            "matching this driver's flush-spawn stance for the robot."
        ),
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
        "--renderer",
        choices=(
            "raytracedlighting",
            "realtimepathtracing",
            "pathtracing",
            "minimal",
        ),
        default="realtimepathtracing",
        help=(
            "RTX render mode for windowed runs; the default is Kit's own. None "
            "of these touch physics -- _render() is physics-free -- but on a "
            "shared GPU the choice costs far more than the frame itself. "
            "Measured on soma_sit_1 (GB10, --loops 2 --no-realtime), ms per "
            "control step: realtimepathtracing 43.1, raytracedlighting 42.6, "
            "**minimal 29.9**. Only the last fits soma23's 33.3 ms control "
            "period, i.e. only the last actually runs in real time. Note where "
            "the difference lands: with either ray-traced mode the *state read* "
            "costs 9.2 ms instead of 0.9 ms, because every `.cpu()` on a "
            "physics-view tensor is a stream sync that now queues behind the "
            "renderer's work on the same device. Resolution and antialiasing "
            "are nearly irrelevant next to that (960x540 with AA off still "
            "costs 40.0 ms). 'minimal' is flat-shaded and much plainer to look "
            "at; that is the trade. Ignored under --headless."
        ),
    )
    p.add_argument(
        "--render-resolution",
        type=str,
        default=None,
        metavar="WxH",
        help=(
            "Viewport render resolution for windowed runs, e.g. '960x540'. "
            "Default (unset) keeps Kit's 1280x720. Purely visual; halving it is "
            "the cheapest frame-rate lever there is."
        ),
    )
    p.add_argument(
        "--anti-aliasing",
        type=int,
        default=None,
        choices=(0, 1, 2, 3, 4),
        help=(
            "RTX antialiasing mode: 0 off, 1 TAA, 2 FXAA, 3 DLSS, 4 RTXAA. "
            "Default (unset) keeps Kit's own choice."
        ),
    )
    p.add_argument(
        "--profile",
        action="store_true",
        default=False,
        help=(
            "Accumulate a per-phase wall-clock breakdown of the control loop "
            "(state read, observation assembly, ONNX, drive write, physics "
            "substeps, render) and print it with the end-of-run summary. The "
            "timers are guarded, so an unprofiled run pays two no-op method "
            "calls per phase and nothing else."
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
            "(`physxScene:enableGPUDynamics=False`, MBP broadphase); training "
            "runs GPU dynamics with the GPU broadphase. Those are different "
            "solvers, and on a contact-rich quasi-static clip the difference is "
            "not small. Measured on soma_sit_1 against the IsaacLab run of the "
            "same checkpoint, per control step from IsaacLab's own state "
            "(--resync-state): mean |d root_quat| 4.51 deg on CPU vs 0.057 deg "
            "on cuda:0, mean |d root_pos| 5.3 mm vs 0.25 mm. Contact-free "
            "(--drive-probe) the same split shows up as |d root_quat| 4.70 deg "
            "vs 0.070 deg over one control step, so it is the articulation "
            "solver rather than contact generation. "
            "**Pair it with the spawn lift**: cuda:0 *alone* is not a fix "
            "(closed-loop mean |d tilt| 27.4 -> 18.5 deg but the robot slides off "
            "the chair, end root_h 0.176 m vs 0.552 m), and --init-z-offset alone "
            "is worse than nothing on CPU (37.97 deg). Together they track: 0.57 "
            "deg mean tilt error, 0.8 mm mean root height error over 149 steps. "
            "The default stays CPU because that is the deployment target; the "
            "startup log names the divergence on every run that takes it."
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
        default="off",
        help=(
            "Apply the robot config's per-joint Coulomb friction "
            "(control_info[j].friction, 0.1 on every G1 joint). 'auto' applies it "
            "whenever the robot config supplies it; 'off' (default) matches "
            "training, whose PhysX actually holds 0.0 -- IsaacLab drops the "
            "actuator's non-'_sim' friction field. 'auto' was the default until "
            "round 4 because the extra dissipation kept this driver upright; that "
            "was a symptom, not a hardware argument (PhysX was applying no joint "
            "damping at all -- see "
            "TrackerPolicy._zero_drive_velocity_targets), and with the drive fixed "
            "'off' is both faithful and better: 0.0454 vs 0.0486 rad over 5 loops, "
            "no falls either way. Use 'auto' to model the real G1's Coulomb "
            "friction when predicting hardware rather than training."
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
        type=str,
        default="auto",
        metavar="METRES|auto",
        help=(
            "Lift the whole spawn -- robot *and* scene objects -- by this much, "
            "reproducing ProtoMotions' env.config.ref_respawn_offset (0.05 m by "
            "default), which training applies to both "
            "(BaseEnv.move_reset_robot_obj_states_to_respawn_position). 'auto' "
            "reads the value out of --resolved-configs and falls back to 0.0 "
            "(flush) with a warning when those are not supplied. This is not "
            "cosmetic on a seated clip: measured on soma_sit_1 against the "
            "IsaacLab run of the same checkpoint, GPU PhysX with the lift tracks "
            "to 0.57 deg mean tilt error, and the same run spawned flush falls "
            "off the chair (end root_h 0.176 m vs 0.552 m). It needs the GPU "
            "solver to help: on CPU PhysX the lift alone is *worse* than flush "
            "(37.97 deg vs 27.43 deg). See --device."
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
            "it needs --resolved-configs. Round 2 measured this as null; "
            "re-measured in round 3 on --control-in-loop --joint-friction off it "
            "cost +17%% (0.0801 -> 0.0939 rad). Treat that number as stale: it "
            "was measured while the drive velocity target was poisoned (round 4, "
            "TrackerPolicy._zero_drive_velocity_targets), i.e. on a driver with "
            "no joint damping and hypersensitive to contact-solve quality. The "
            "sign may not survive re-measurement; 'plane' remains the "
            "deployment-faithful default either way."
        ),
    )
    p.add_argument(
        "--self-collisions",
        choices=("auto", "on", "off"),
        default="auto",
        help=(
            "Articulation self-collision. 'auto' takes the robot config's value "
            "(what training spawns with); 'on'/'off' force it. A forced value is "
            "an ablation instrument, not a tuning knob: the sitting pose folds "
            "thighs against the torso and arms against the legs, so self-contact "
            "carries real force there, and forcing it off is how you test "
            "whether the configured value actually reached PhysX."
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
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Compute the action in the outer loop and then run `decimation` "
            "physics substeps, the way IsaacLab does, instead of driving from a "
            "physics callback. Removes the quarter-control-period phase offset "
            "against the motion reference clock. Default since round 4: it was "
            "held back because it was less robust over repeated clips, which the "
            "round-4 drive fix (TrackerPolicy._zero_drive_velocity_targets) "
            "removed -- it now scores 0.0454 rad with no falls over 5 loops "
            "against the callback path's 0.0537, with IsaacLab itself at 0.0437. "
            "--no-control-in-loop restores the callback path. See "
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
            "off is what matches training; turn it on to ablate the difference. "
            "Round 2 measured this as null on the phase-broken control loop; "
            "round 3 re-measured it on --control-in-loop --joint-friction off as "
            "worth -3%% (0.0801 -> 0.0777 rad). Treat that number as stale: it "
            "predates the round-4 drive fix "
            "(TrackerPolicy._zero_drive_velocity_targets), which removed the "
            "contact-solve hypersensitivity that made a 3%% effect readable."
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
        "--drive-probe",
        default=None,
        help=(
            "Replay a drive-probe spec written by trace_tracker_isaaclab.py "
            "--drive-probe-out, then exit (see deployment/drive_probe.py). Each "
            "case teleports the robot out of contact, writes a fixed state and a "
            "fixed PD target, and records the joint response after every physics "
            "substep -- isolating the articulation/drive integration from "
            "gravity, contact and the controller. Requires --drive-probe-out."
        ),
    )
    p.add_argument(
        "--drive-probe-out",
        default=None,
        help="Where to write this driver's drive-probe response .npz.",
    )
    p.add_argument(
        "--scene-solver-iterations",
        choices=("training", "default"),
        default="training",
        help=(
            "PhysX *scene* max position/velocity solver iteration caps. "
            "'training' uses the robot config's values (8/4 on the G1), matching "
            "what ProtoMotions' IsaacLab config sets; 'default' leaves World's "
            "USD schema defaults of 255/255. The articulation asks for 8/4 either "
            "way, so this only changes the caps -- notably for the contact solve. "
            "Ablatable because it is a parity fix whose dynamical effect must be "
            "measured, not assumed."
        ),
    )
    p.add_argument(
        "--resync-state",
        action="store_true",
        default=False,
        help=(
            "With --action-tape, overwrite the full robot state from IsaacLab's "
            "recording at the top of every control step, so each step becomes an "
            "independent one-step experiment instead of an accumulating open-loop "
            "replay. Answers 'given IsaacLab's exact state at step k and its exact "
            "action, does one control step land where IsaacLab landed?' -- which "
            "plain --action-tape cannot, because its divergence is a sum over all "
            "previous steps. Needs a tape recorded with the sim__* state group."
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

def _app_config(parsed) -> dict:
    """Kit launch config. Everything here is renderer-side and physics-free.

    Only what the CLI actually asked for is set, so Kit's own defaults stay in
    force everywhere else. ``--renderer`` matters more than its name suggests on
    a GPU shared with PhysX; its help text carries the measurements.
    """
    config = {"headless": parsed.headless, "renderer": parsed.renderer}
    if parsed.render_resolution is not None:
        try:
            width, height = (
                int(v) for v in parsed.render_resolution.lower().split("x")
            )
        except ValueError:
            raise SystemExit(
                f"--render-resolution expects WxH, e.g. '960x540' (got "
                f"{parsed.render_resolution!r})"
            )
        config["width"], config["height"] = width, height
    if parsed.anti_aliasing is not None:
        config["anti_aliasing"] = parsed.anti_aliasing
    return config


simulation_app = SimulationApp(_app_config(args))

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
from isaacsim.core.utils.stage import get_current_stage  # noqa: E402
from isaacsim.core.utils.viewports import set_camera_view  # noqa: E402
from pxr import PhysxSchema, Usd, UsdPhysics, UsdShade  # noqa: E402

from deployment import physx_probe  # noqa: E402
from deployment import scene_utils  # noqa: E402
from deployment.physx_probe import CONTACT_FORCE_THRESHOLD_N  # noqa: E402
from deployment.motion_utils import MotionPlayer  # noqa: E402
from deployment.obs_assembly import ActionHistory, build_onnx_inputs  # noqa: E402
from deployment.state_utils import (  # noqa: E402
    apply_heading_offset_np,
    apply_heading_offset_to_positions_np,
    compute_root_local_ang_vel_np,
    compute_yaw_offset_np,
    make_trace_row,
    quat_angle_deg_xyzw,
    quat_rotate_np,
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


def _render(world) -> None:
    """Draw a frame **without** advancing physics.

    Use this instead of ``world.step(render=True)``. That call is not "step once
    and also draw": with a non-zero ``rendering_dt`` it takes neither of the
    single-step branches in ``SimulationContext.step`` and falls through to a bare
    ``self._app.update()`` (``isaacsim/core/api/simulation_context/simulation_context.py``),
    which hands the frame to Kit's own loop -- and Kit steps physics by a whole
    *rendering* period, i.e. ``int(rendering_dt / physics_dt)`` substeps. This
    driver builds ``World`` with ``rendering_dt=control_dt``, so that is 4.

    Measured on a bare ``World`` at this driver's timing (physics_dt 5 ms,
    rendering_dt 20 ms), counting ``current_time_step_index`` over 4 calls::

        step(render=False)              d_time=0.020s   4 substeps    4 physics callbacks
        step(render=True)               d_time=0.080s  16 substeps   16 physics callbacks
        step(render=False) + render()   d_time=0.020s   4 substeps    4 physics callbacks
        render() alone                  d_time=0.000s   0 substeps    0 physics callbacks

    ``render()`` is physics-free because it brackets its ``app.update()`` with
    ``/app/player/playSimulations = False``; the ``step(render=True)`` fall-through
    does not. This is the whole reason a windowed run diverged from the identical
    ``--headless`` command, which never passes ``render=True`` at all.
    """
    world.render()


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
        # Feet, for the contact-height probe. Same list training instruments with
        # ContactSensorCfg (`isaaclab/utils/scene.py`), so "foot" means the same
        # bodies in both harnesses. Note this is set by the *experiment* file,
        # so a bare RobotConfig can leave it empty -- hence the naming-map
        # fallback below, which _build_foot_probe prefers over substring guesses.
        "contact_bodies": list(getattr(config, "contact_bodies", None) or []),
        "all_left_foot_bodies": list(
            (getattr(config, "common_naming_to_robot_body_names", None) or {}).get(
                "all_left_foot_bodies", []
            )
        ),
        "all_right_foot_bodies": list(
            (getattr(config, "common_naming_to_robot_body_names", None) or {}).get(
                "all_right_foot_bodies", []
            )
        ),
    }


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
        resync_state: bool = False,
        control_in_loop: bool = True,
        self_collisions: str = "auto",
        scene_objects=None,
        profile: bool = False,
    ) -> None:
        # --profile: per-phase wall-clock accounting of the control loop. Off by
        # default, and the timers are guarded rather than always-on, so a normal
        # run does not pay for a clock read it will never print.
        self.profile = bool(profile)
        self._phase_ms: dict[str, float] = {}
        # None = take the robot config's value; True/False = forced ablation.
        self.self_collisions_override = (
            None if self_collisions == "auto" else (self_collisions == "on")
        )
        self.author_collider_offsets = bool(author_collider_offsets)
        self.resync_state = bool(resync_state)
        # Which of the two control shapes main() drives this policy with. Only
        # the open-loop tape cares: the two paths need *different* anchoring at
        # reset -- see reset_episode's note.
        self.control_in_loop = bool(control_in_loop)
        # Spawned SceneLib objects (--scenes-file), or None. Physics and scoring
        # only: the ONNX graph carries no scene channel, so the policy does not
        # observe them -- see the module docstring.
        self.scene_objects = scene_objects
        robot_meta = meta["robot"]
        timing = meta["timing"]
        motion_meta = meta["motion"]
        control = meta["control"]
        runtime = meta["_runtime"]

        self.anchor_body_index = robot_meta["anchor_body_index"]
        self.root_body_name = robot_meta.get("root_body_name") or "pelvis"
        # `anchor_body_name` is present-but-null for robots that anchor on their
        # own root (soma23: RobotConfig leaves it None and resolves the index to
        # 0). `.get(key, default)` does not fire on an explicit null, so read it
        # with `or` -- otherwise `body_names.index(None)` raises in
        # `_resolve_anchor_link_index`.
        self.anchor_body_name = (
            robot_meta.get("anchor_body_name") or self.root_body_name
        )
        self.joint_names = list(robot_meta["joint_names"])
        # Body order the policy's max-coordinate observations are expressed in.
        # Absent only in YAMLs predating the body_names field.
        self.body_names = list(robot_meta.get("body_names") or [])
        self.num_dofs = robot_meta["num_dofs"]
        self.num_bodies = robot_meta.get("num_bodies", len(self.body_names))
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

        # Which observation family is this? Decided by the semantic keys the
        # contract asks for, not by robot name -- the same robot can be trained
        # either way. See build_onnx_inputs.
        required_keys = set(self.onnx_name_to_key.values())
        self._needs_body_state = any(
            k.startswith("current.rigid_body_") for k in required_keys
        )
        # `historical.actions` is the raw pre-tanh policy output, fed back from
        # the ONNX `actions` head; `historical.processed_actions` is the
        # commanded PD target, fed back after the accel clamp and EMA. Getting
        # these two crossed produces a plausible-looking but wrong observation.
        self._needs_raw_actions = "historical.actions" in required_keys
        self._raw_action_steps = 1
        if self._needs_raw_actions:
            raw_name = next(
                n for n, k in self.onnx_name_to_key.items() if k == "historical.actions"
            )
            raw_shape = next(
                i.shape for i in self.session.get_inputs() if i.name == raw_name
            )
            # [batch, history_steps, num_dofs]; batch is dynamic, the rest concrete.
            if len(raw_shape) == 3 and isinstance(raw_shape[1], int):
                self._raw_action_steps = int(raw_shape[1])
        log.info(
            f"Observation family: {'max-coords' if self._needs_body_state else 'reduced-coords'}"
            f"; action history: "
            + (
                f"raw x{self._raw_action_steps}"
                if self._needs_raw_actions
                else "processed x1"
            )
        )

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
        # Worst per-field mismatch between what --resync-state wrote and what
        # PhysX reported back; see _verify_resync_write.
        self._resync_write_error: dict = {}

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

        # Foot contact-height probe and optional contact-force view, both built
        # in initialize() once the physics view exists.
        self._foot_probe = None
        self._foot_contact_view = None
        self._foot_contact_names: list = []

        # Episode-local filter/history state -- reset every episode in reset_episode().
        self._isaac_to_policy: np.ndarray | None = None
        self._isaac_to_policy_body: np.ndarray | None = None
        self._anchor_link_index: int | None = None
        # Both action histories carry ProtoMotions' one-step lag: the action
        # just computed goes in, but the *observed* window starts one step
        # further back. See ActionHistory -- getting this wrong feeds the policy
        # a fresher action than it ever saw in training.
        self._prev_actions = ActionHistory(1, self.num_dofs)
        # Raw (pre-tanh) policy outputs, for policies whose action-history input
        # is `historical.actions` rather than `historical.processed_actions`.
        # Depth comes from the ONNX input shape.
        self._raw_action_history = ActionHistory(self._raw_action_steps, self.num_dofs)
        self._prev_pd: np.ndarray | None = None
        self._prev_prev_pd: np.ndarray | None = None
        self._ema_prev_targets: np.ndarray | None = None
        self._heading_offset: np.ndarray | None = None
        # Root positions the max-coords reference realignment pivots around.
        self._motion_pivot: np.ndarray | None = None
        self._robot_pivot: np.ndarray | None = None
        self._pd_targets_isaac: np.ndarray | None = None
        # Device-side mirrors of the two drive commands. Both are written on
        # every physics substep, so both are converted once and reused: the
        # position target whenever _set_pd_targets latches a new one, the zero
        # velocity target never. See _apply_pd_targets.
        self._pd_target_tensor_cache = None
        self._zero_vel_tensor = None
        self._all_env_indices = None
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
        # The reference trajectory the replay is scored against. `state__*` is
        # IsaacLab's *context* (what its policy saw); `sim__*` is what its PhysX
        # held, which is what --resync-state has to write back. Both are loaded
        # when present so an old tape still scores, and the resync path checks
        # for the group it needs rather than silently replaying garbage.
        self._tape_ref = {
            k: np.asarray(data[k])
            for k in (
                "state__dof_pos",
                "state__dof_vel",
                "state__root_pos",
                "sim__root_pos",
                "sim__root_rot",
                "sim__root_lin_vel",
                "sim__root_ang_vel",
                "sim__dof_pos",
                "sim__dof_vel",
                "sim__link_pos",
                "sim__link_quat",
                "sim__foot_contact_force",
            )
            if k in data.files
        }
        if "meta__contact_body_names" in data.files:
            self._tape_ref["contact_body_names"] = [
                str(n) for n in data["meta__contact_body_names"]
            ]
        log.info(
            f"Action tape from {path}: {tape.shape[0]} control steps, "
            "policy is out of the loop."
        )
        if self.resync_state:
            missing = [
                k
                for k in (
                    "sim__root_pos",
                    "sim__root_rot",
                    "sim__root_lin_vel",
                    "sim__root_ang_vel",
                    "sim__dof_pos",
                    "sim__dof_vel",
                )
                if k not in self._tape_ref
            ]
            if missing:
                raise SystemExit(
                    "--resync-state needs the sim__* state group in the tape, but "
                    f"{path} is missing {missing}. Re-record it with a current "
                    "deployment/trace_tracker_isaaclab.py."
                )
            log.info(
                "--resync-state: every control step starts from IsaacLab's own "
                "PhysX state, so each step is an independent one-step experiment."
            )
        return tape

    def _tape_step(self) -> None:
        """Advance the open-loop replay by one control step, resynced or not."""
        if self.resync_state:
            self._resync_tape_step()
        else:
            self._replay_tape_step()

    def _resync_tape_step(self) -> None:
        """One control step as an *independent* one-step experiment.

        ``--action-tape`` alone cannot answer the remaining question. It is
        open-loop from a single initial condition, so by step *k* its divergence
        is the accumulated sum of *k* one-step errors plus the trajectory drift
        they caused -- a single number that cannot be attributed to any step, and
        that grows even between two identical simulators fed slightly different
        floating-point rounding. Contact-rich open-loop replay always diverges
        eventually; that it does says nothing.

        This method instead runs N independent experiments. Per control step:

        1. **Score** the state PhysX currently holds against IsaacLab's state at
           this step. That state is the *outcome* of applying IsaacLab's action
           ``k-1`` to IsaacLab's state ``k-1`` for one control period, because
           step 2 below put the robot exactly there -- so the difference is the
           one-step error of step ``k-1``, with no history in it.
        2. **Overwrite** the full state from IsaacLab's recording, clearing that
           error before it can propagate.
        3. **Apply** IsaacLab's action for this step and let the caller run
           ``decimation`` substeps.

        Step 1 is skipped at frame 0, where there is no preceding step to score.

        Deliberately does **not** touch ``_prev_actions`` / ``_prev_pd`` /
        ``_prev_prev_pd`` / ``_ema_prev_targets`` / the heading offset, for the
        same reason :meth:`_replay_tape_step` documents: there is no feedback
        loop to maintain, and letting the filter state drift would introduce a
        difference that has nothing to do with physics.
        """
        if self._frame_idx >= len(self._action_tape):
            self.done = True
            return

        frame = self._frame_idx
        ref = self._tape_ref

        # --- 1. score the previous step's one-step outcome -------------------
        dof_pos = _to_numpy(self.robot.get_joint_positions())[
            self._isaac_to_policy
        ].astype(np.float32)
        dof_vel = _to_numpy(self.robot.get_joint_velocities())[
            self._isaac_to_policy
        ].astype(np.float32)
        root_pos, root_quat_wxyz = self.robot.get_world_pose()
        root_pos = _to_numpy(root_pos).astype(np.float32)
        root_quat = wxyz_to_xyzw(_to_numpy(root_quat_wxyz)).astype(np.float32)

        # The trace is recorded from the *pre-overwrite* state: that is this
        # driver's own one-step prediction. Recording after the overwrite would
        # trace IsaacLab's trajectory back to itself and always report zero.
        self._record_trace(dof_pos, dof_vel, self._read_anchor_rot(root_quat))

        if self.divergence is not None and frame > 0:
            self.divergence.append(
                self._resync_divergence_row(
                    frame, dof_pos, dof_vel, root_pos, root_quat
                )
            )

        # --- 2. overwrite with IsaacLab's state at this step ------------------
        self._write_robot_state(
            root_pos=ref["sim__root_pos"][frame],
            root_quat_xyzw=ref["sim__root_rot"][frame],
            dof_pos_policy=ref["sim__dof_pos"][frame],
            dof_vel_policy=ref["sim__dof_vel"][frame],
            root_lin_vel=ref["sim__root_lin_vel"][frame],
            root_ang_vel=ref["sim__root_ang_vel"][frame],
        )
        self._verify_resync_write(frame)

        # --- 3. apply IsaacLab's action for this step -------------------------
        self._set_pd_targets(self._action_tape[frame])

        self._frame_idx += 1
        self._total_steps += 1
        if self._frame_idx >= len(self._action_tape):
            self.done = True

    def _verify_resync_write(self, frame: int) -> None:
        """Confirm PhysX actually holds the state the resync just wrote.

        Not optional bookkeeping -- without it the whole probe is unfalsifiable.
        If ``set_joint_velocities`` silently failed to reach the solver, every
        control step would start from the wrong velocity and the measured
        divergence would be ``|Δv| · dt``: largest on the fastest joints,
        negligible on the root pose (which *is* written successfully), and
        unrelated to contact. That is *precisely* the signature this probe
        produces, so "the write landed" and "the physics differs" predict the
        same numbers and only a read-back can separate them.

        Accumulates the worst mismatch over the run and reports it once, in
        :meth:`log_summary`, rather than logging 250 lines.
        """
        ref = self._tape_ref
        dof_pos = _to_numpy(self.robot.get_joint_positions())[
            self._isaac_to_policy
        ].astype(np.float64)
        dof_vel = _to_numpy(self.robot.get_joint_velocities())[
            self._isaac_to_policy
        ].astype(np.float64)
        root_pos, _ = self.robot.get_world_pose()
        lin_vel = _to_numpy(self.robot.get_linear_velocity()).astype(np.float64)
        ang_vel = _to_numpy(self.robot.get_angular_velocity()).astype(np.float64)

        for label, actual, wanted in (
            ("dof_pos", dof_pos, ref["sim__dof_pos"][frame]),
            ("dof_vel", dof_vel, ref["sim__dof_vel"][frame]),
            (
                "root_pos",
                _to_numpy(root_pos).astype(np.float64),
                ref["sim__root_pos"][frame],
            ),
            ("root_lin_vel", lin_vel, ref["sim__root_lin_vel"][frame]),
            ("root_ang_vel", ang_vel, ref["sim__root_ang_vel"][frame]),
        ):
            error = float(np.abs(actual - np.asarray(wanted, dtype=np.float64)).max())
            if error > self._resync_write_error.get(label, -1.0):
                self._resync_write_error[label] = error

    def _resync_divergence_row(
        self,
        frame: int,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        root_pos: np.ndarray,
        root_quat: np.ndarray,
    ) -> dict:
        """Build one row of per-step, history-free divergence.

        Labelled with **IsaacLab's** foot-contact state rather than this
        driver's. The label has to be independent of the thing under test: if the
        contact solve is what differs, classifying by the driver's own contact
        state would sort steps by the very quantity in question. Since the resync
        put the robot in IsaacLab's exact state for this step, IsaacLab's contact
        reading is the correct description of that state for both stacks.

        Args:
            frame: Control-step index being scored.
            dof_pos: Driver joint positions ``[num_dofs]``, policy order.
            dof_vel: Driver joint velocities ``[num_dofs]``, policy order.
            root_pos: Driver root position ``[3]``.
            root_quat: Driver root orientation ``[4]`` (xyzw).

        Returns:
            One JSON-serialisable divergence row.
        """
        ref = self._tape_ref
        delta_pos = np.abs(dof_pos - ref["sim__dof_pos"][frame])
        delta_vel = np.abs(dof_vel - ref["sim__dof_vel"][frame])
        ref_quat = np.asarray(ref["sim__root_rot"][frame], dtype=np.float64)

        # Angle of the relative rotation, via |dot| so that q and -q (the same
        # orientation) do not read as 180 degrees apart.
        dot = float(np.clip(abs(np.dot(root_quat.astype(np.float64), ref_quat)), 0, 1))
        row = {
            "frame": int(frame),
            "max_dof_delta": float(delta_pos.max()),
            "mean_dof_delta": float(delta_pos.mean()),
            "worst_joint": self.joint_names[int(delta_pos.argmax())],
            "max_dof_vel_delta": float(delta_vel.max()),
            "worst_vel_joint": self.joint_names[int(delta_vel.argmax())],
            "root_pos_delta": float(
                np.linalg.norm(root_pos - ref["sim__root_pos"][frame])
            ),
            "root_quat_delta_deg": float(np.degrees(2.0 * np.arccos(dot))),
            "dof_vel_rms": float(np.sqrt(np.mean(dof_vel**2))),
            "ref_dof_vel_rms": float(
                np.sqrt(np.mean(np.asarray(ref["sim__dof_vel"][frame]) ** 2))
            ),
            "root_h": float(root_pos[2]),
            "ref_root_h": float(ref["sim__root_pos"][frame][2]),
        }

        forces = ref.get("sim__foot_contact_force")
        if forces is not None:
            magnitudes = np.asarray(forces[frame], dtype=np.float64)
            row["foot_contact"] = int(np.sum(magnitudes > CONTACT_FORCE_THRESHOLD_N))
            row["foot_force_max"] = float(np.nanmax(magnitudes))

        # Full per-joint and per-link deltas. The scalar summaries above rank
        # steps; only the vectors say *where* a step went wrong -- whether one
        # joint carries the error (a drive/property difference) or a whole limb
        # translates (contact). 66 + 23 floats a step is cheap next to being
        # unable to tell those apart.
        row["dof_delta"] = (dof_pos - ref["sim__dof_pos"][frame]).tolist()
        row["dof_vel_delta"] = (dof_vel - ref["sim__dof_vel"][frame]).tolist()
        if "sim__link_pos" in ref:
            link_tf = self._read_link_transforms()
            if link_tf is not None:
                # Both sides are raw ``get_link_transforms()`` output, i.e. PhysX's
                # own link order -- *not* the policy's body order. Do not reorder
                # one of them: ``sim__link_pos`` is recorded unmapped on the
                # IsaacLab side for exactly this comparison.
                row["link_pos_delta"] = np.linalg.norm(
                    link_tf[:, 0:3] - ref["sim__link_pos"][frame], axis=-1
                ).tolist()
        return row

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

        So on that path the tape is anchored at reset: frame 0 is recorded and
        ``tape[0]`` applied *before* any physics runs (see ``reset_episode``),
        and every later call lands on a whole control-step boundary
        (``t = 20k`` ms) -- exactly IsaacLab's read/apply points.

        ``control_step`` (the default) needs no anchor: it already calls this
        method at the top of each control step, before its substeps, which *is*
        IsaacLab's read/apply order. ``reset_episode`` therefore skips the anchor
        there; anchoring both ways was a one-control-step shift of the entire
        replay, with ``tape[0]`` never reaching physics at all.

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
        self._set_pd_targets(self._action_tape[self._frame_idx])

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

        **Two joint encodings.** A single-axis revolute joint (the G1) is one prim
        per DOF, named after the DOF, carrying one ``"angular"`` drive. A D6 joint
        (soma23) is one prim per *body*, named after the body (``Spine1``), carrying
        three rotational drives ``rotX``/``rotY``/``rotZ`` whose DOF names live in
        the custom tokens ``mjcf:rotX:name`` etc. (``Spine1_x``). Matching only the
        prim name against ``joint_names`` finds nothing on such an asset, so the
        whole pass silently no-ops -- hence the hard failure at the end.
        """
        stage = get_current_stage()
        deg = math.pi / 180.0  # Nm/rad -> Nm/deg, matching set_gains(save_to_usd=True)
        by_name = {
            name: (float(kp) * deg, float(kd) * deg)
            for name, kp, kd in zip(self.joint_names, self.stiffness, self.damping)
        }
        applied = 0
        unmatched: list = []
        for prim in Usd.PrimRange(stage.GetPrimAtPath(prim_path)):
            if not prim.HasAPI(UsdPhysics.DriveAPI):
                continue
            # Which drive tokens does this prim actually carry?
            tokens = [
                schema.split(":", 1)[1]
                for schema in prim.GetAppliedSchemas()
                if schema.startswith("PhysicsDriveAPI:")
            ]
            matched_here = False
            for token in tokens:
                # Single-axis: the prim itself is the DOF. D6: read the DOF name
                # from the converter's mjcf:<token>:name token, falling back to
                # the <body>_<axis> convention the MJCF uses.
                if token == "angular":
                    dof_name = prim.GetName()
                else:
                    attr = prim.GetAttribute(f"mjcf:{token}:name")
                    dof_name = attr.Get() if attr and attr.HasValue() else None
                    if dof_name is None and token.startswith("rot"):
                        dof_name = f"{prim.GetName()}_{token[3:].lower()}"
                gains = by_name.get(dof_name)
                if gains is None:
                    continue
                drive = UsdPhysics.DriveAPI.Get(prim, token)
                if not drive:
                    continue
                drive.CreateStiffnessAttr().Set(gains[0])
                drive.CreateDampingAttr().Set(gains[1])
                applied += 1
                matched_here = True
            if not matched_here:
                unmatched.append(prim.GetName())

        log.info(
            f"Authored PD gains on {applied}/{len(by_name)} USD joint drives "
            "(converted Nm/rad -> Nm/deg)."
        )
        if applied == 0:
            raise RuntimeError(
                "Authored PD gains on 0 USD joint drives. The policy's joint "
                "names match no drive on this asset, so every physics step "
                "before the physics view exists would run at the stage's own "
                f"gains. Policy joints: {self.joint_names[:5]}... ; prims with a "
                f"drive API: {unmatched[:5]}..."
            )
        if applied < len(by_name):
            log.warning(
                f"{len(by_name) - applied} policy joints had no matching USD "
                "drive; those run on the stage's authored gains until "
                "post_reset() applies set_gains()."
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
        if self.self_collisions_override is not None:
            self_collisions = self.self_collisions_override
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

        # Same gather for *links*. The articulation's link order is PhysX's own
        # (breadth-first from the root), while max-coordinate observations are
        # in the MJCF's kinematic order -- for soma23 the two differ from index 2
        # onward (sim: Hips, Spine1, RightLeg, ... ; MJCF: Hips, Spine1, Spine2, ...).
        # Only needed by policies that observe per-body state; reduced-coords
        # policies never touch it.
        self._isaac_to_policy_body = None
        if self.body_names:
            isaac_body_names = list(view.body_names or [])
            missing_bodies = set(self.body_names) - set(isaac_body_names)
            if missing_bodies:
                raise ValueError(
                    "USD articulation is missing bodies required by the policy: "
                    f"{sorted(missing_bodies)}. USD body_names={isaac_body_names}"
                )
            if len(isaac_body_names) != len(self.body_names):
                raise ValueError(
                    f"USD articulation has {len(isaac_body_names)} links but the "
                    f"policy expects {len(self.body_names)}. Extra links: "
                    f"{sorted(set(isaac_body_names) - set(self.body_names))}"
                )
            self._isaac_to_policy_body = np.array(
                [isaac_body_names.index(n) for n in self.body_names], dtype=np.int64
            )

        self._resolve_anchor_link_index(view)
        self._apply_solver_iterations(view)
        self._apply_joint_properties(view)
        self._build_foot_probe(view)
        if self.scene_objects is not None:
            self.scene_objects.initialize()

    def _build_foot_probe(self, view) -> None:
        """Resolve the feet's collider geometry so contact height can be measured.

        The link origin is not the contact point. Each G1 ``*_ankle_roll_link``
        carries seven collision capsules whose lowest surface sits 0.035 m below
        the link origin when the foot is flat -- 2.5x the pelvis-height anomaly
        under investigation -- so a dump that reports link origins (which this
        one did) cannot tell "the robot stands higher" from "the robot stands
        differently". :class:`deployment.physx_probe.FootProbe` reads the real
        geometry off the stage once, here, and evaluates it from the link
        transforms every control step.

        Built from the same ``contact_bodies`` list training instruments with
        contact sensors, so "foot" means the same links in both harnesses.
        """
        props = self.joint_properties or {}
        body_names = list(view.body_names or [])
        feet = [n for n in (props.get("contact_bodies") or []) if n in body_names]
        if not feet:
            # `contact_bodies` is set by the *experiment* file, not the robot
            # config, so a bare RobotConfig leaves it None (soma23 does). The
            # common-naming map is the same source the experiment resolves
            # through, so prefer it over guessing from substrings.
            for key in (
                props.get("all_left_foot_bodies") or [],
                props.get("all_right_foot_bodies") or [],
            ):
                feet.extend(n for n in key if n in body_names and n not in feet)
            if feet:
                log.info(f"Feet from the robot config's foot-body naming map: {feet}.")
        if not feet:
            # Last resort. "toe" matters for rigs that split the foot into an
            # ankle plus a toe link (soma23: LeftFoot + LeftToeBase); missing it
            # measures contact height against the wrong body.
            feet = [
                n
                for n in body_names
                if any(t in n.lower() for t in ("ankle_roll", "foot", "toe"))
            ]
            if feet:
                log.warning(
                    "Robot config supplies no contact_bodies; falling back to "
                    f"name-matched feet {feet}."
                )
        if not feet:
            log.warning(
                "No foot links identified -- the trace's foot_z column will be "
                "omitted and the height decomposition unavailable."
            )
            return

        stage = get_current_stage()
        link_paths = physx_probe.resolve_link_prim_paths(stage, self._prim_path, feet)
        self._foot_probe = physx_probe.FootProbe(
            stage,
            link_paths=link_paths,
            link_indices={n: body_names.index(n) for n in feet},
        )
        self._foot_contact_names = feet
        if self._foot_probe.missing:
            log.warning(
                f"No collider geometry resolved for {self._foot_probe.missing}; "
                "their contact height will read as nan."
            )
        log.info(
            f"Foot probe: {len(link_paths)} feet resolved\n"
            + self._foot_probe.describe()
        )

    def _read_link_transforms(self) -> np.ndarray | None:
        """Link poses in sim link order, ``[num_links, 7]`` as ``x y z qx qy qz qw``.

        Straight off the physics view, for the reason the module docstring gives
        at length: USD xforms are never authored for articulation links during
        simulation, and these assets are instanceable, so a stage read returns
        the spawn pose forever. Note ``get_link_transforms()`` is already xyzw --
        do not run it through :func:`wxyz_to_xyzw`.
        """
        view = self.robot._articulation_view
        physics_view = getattr(view, "_physics_view", None)
        if physics_view is None:
            return None
        try:
            return _to_numpy(physics_view.get_link_transforms()).reshape(-1, 7)
        except Exception as e:  # pragma: no cover - version dependent
            log.debug(f"link transform read-back failed: {e}")
            return None

    def _read_link_velocities(self) -> np.ndarray | None:
        """Link velocities in sim link order, ``[num_links, 6]`` as ``vx vy vz wx wy wz``.

        World frame, **linear first**. That split is not documented anywhere in
        the Isaac Sim API surface and it is the opposite of MuJoCo's ``cvel``
        (angular first), so it was measured rather than assumed: writing a known
        root linear/angular velocity and reading the buffer back gives
        ``first3 == linear`` exactly (see the round-5 probe notes in
        ``docs/`` / ``agent_logs/``). Getting it backwards is silent -- the
        policy just sees a nonsense body-velocity observation.
        """
        view = self.robot._articulation_view
        physics_view = getattr(view, "_physics_view", None)
        if physics_view is None:
            return None
        try:
            return _to_numpy(physics_view.get_link_velocities()).reshape(-1, 6)
        except Exception as e:  # pragma: no cover - version dependent
            log.debug(f"link velocity read-back failed: {e}")
            return None

    def _read_body_state(self):
        """Full per-body state in **policy** body order, for max-coords observations.

        Returns ``(pos [B,3], rot [B,4] xyzw, vel [B,3], ang_vel [B,3])``, all in
        the world frame -- which is what ``compute_humanoid_max_coords_observations``
        and ``build_max_coords_target_poses`` expect; they do their own
        root-relative, heading-inverted normalization internally.

        Raises rather than returning ``None``: a policy that needs these cannot
        run a single step without them, so a soft failure would only hide the
        cause behind a wrong-looking trajectory.
        """
        if self._isaac_to_policy_body is None:
            raise RuntimeError(
                "This policy observes per-body state but the YAML carried no "
                "robot.body_names, so no link reorder map could be built. "
                "Re-export with deployment/export_bm_tracker_onnx_isaacsim.py."
            )
        link_tf = self._read_link_transforms()
        link_vel = self._read_link_velocities()
        if link_tf is None or link_vel is None:
            raise RuntimeError(
                "Could not read the articulation's link buffers "
                "(get_link_transforms / get_link_velocities). They only exist "
                "after world.reset(); check the initialization order."
            )
        idx = self._isaac_to_policy_body
        body_pos = link_tf[idx, 0:3].astype(np.float32)
        body_rot = link_tf[idx, 3:7].astype(np.float32)  # already xyzw
        body_vel = link_vel[idx, 0:3].astype(np.float32)
        body_ang_vel = link_vel[idx, 3:6].astype(np.float32)
        return body_pos, body_rot, body_vel, body_ang_vel

    def _alignment_is_identity(self) -> bool:
        """True when realigning the reference into the robot's frame is a no-op.

        The robot is normally spawned at motion frame 0, so the yaw offset is
        identity and the two pivots coincide; skipping the transform then keeps
        the reference bit-exact rather than paying float32 round-trip error on
        every body, every step. Becomes False under ``--init-state`` or any
        start pose that is not the motion's own.
        """
        if self._heading_offset is None:
            return True
        # Yaw-only quaternion, so w == 1 (up to sign) means no rotation.
        if abs(abs(float(self._heading_offset[3])) - 1.0) > 1e-6:
            return False
        if self._motion_pivot is None or self._robot_pivot is None:
            return True
        # Horizontal only -- the z components are equal by construction, since
        # the realignment deliberately leaves height alone.
        return bool(np.abs(self._robot_pivot[:2] - self._motion_pivot[:2]).max() <= 1e-6)

    def _ground_height_under_root(self) -> float:
        """Terrain height beneath the root, for ``root_height_obs``.

        Zero on the analytic ground plane (``--ground plane``, the default),
        which is also what training sees on flat terrain -- ProtoMotions sets
        ``env.skip_correct_terrain_height_on_flat = True``. Under
        ``--ground trimesh`` the mesh is built around z=0 as well, so zero
        remains correct until a non-flat terrain is actually requested.
        """
        return 0.0

    def _lowest_foot_z(self) -> float | None:
        """World z of the lowest foot collider point, or ``None`` if unavailable."""
        if self._foot_probe is None:
            return None
        link_tf = self._read_link_transforms()
        if link_tf is None:
            return None
        return self._foot_probe.lowest(link_tf[:, :3], link_tf[:, 3:7])

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
        # Isaac Sim's set_effort_modes() hardcodes the USD drive instance name
        # ("angular" for every rotational DOF), and its `HasAPI(DriveAPI)` guard
        # passes on any instance -- so on D6 joints that author drive:rotX/rotY/rotZ
        # (soma23) it writes to an undefined attribute and raises "Empty typeName".
        # Those assets already declare `type = "force"` per axis, so skipping is a
        # no-op; the G1's revolute `angular` drives still take the real call.
        try:
            controller.set_effort_modes("force")
        except Exception:
            log.info(
                "set_effort_modes skipped: per-axis joint drives are already force-mode."
            )
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

        It is applied by default anyway, but **the empirical half of that
        rationale expired in round 4**. Real G1 joints do have Coulomb friction,
        so 0.1 remains a defensible hardware model and this driver's job is
        partly to predict hardware. The other reason -- that the extra
        dissipation kept the driver upright, where ``off`` fell on 2 of 5 clips
        -- was a symptom, not a hardware argument: PhysX was applying no joint
        damping at all (see :meth:`_zero_drive_velocity_targets`), and 0.1 rad.s
        of friction was standing in for it. With the drive fixed,
        ``--joint-friction off`` no longer falls and is now the *better* score as
        well as the training-faithful one: 0.0454 vs 0.0486 rad over 5 loops of
        ``g1_walk_box`` with ``--control-in-loop``.

        Historic caution, now explained: ``off`` without ``--control-in-loop``
        used to cost 0.0903 -> 0.1045 rad and tilt 2.71 -> 9.96 deg, because the
        friction was damping the instability the phase error caused on top of the
        missing joint damping.

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

        dof_pos_isaac = self._write_robot_state(
            root_pos=root_pos,
            root_quat_xyzw=root_quat_xyzw,
            dof_pos_policy=dof_pos_policy,
            dof_vel_policy=dof_vel_policy,
            root_lin_vel=root_lin_vel,
            root_ang_vel=root_ang_vel,
        )

        # Objects land in the same "outside the physics step" window the robot
        # write above requires, and at the same clip time (frame 0). --init-state
        # moves the robot to IsaacLab's absolute spawn, so the objects take the
        # same translation -- ProtoMotions applies one offset to both.
        if self.scene_objects is not None:
            if self._init_state is not None:
                root_offset = self._init_state.get("respawn_root_offset")
            elif self.init_z_offset:
                # ProtoMotions' respawn lift moves the robot *and the scene*
                # together (`BaseEnv.move_reset_robot_obj_states_to_respawn_position`
                # adds the same `respawn_root_offset` to both), so lifting only
                # the robot would push it 5 cm out of the seat it is fitted to --
                # the opposite of matching training. Note this is not
                # `--scene-object-z-offset`, which mirrors ProtoMotions'
                # *separate* `ref_object_respawn_offset` and, like it, reaches
                # only objects that carry a trajectory; a static chair is exactly
                # the case that one skips.
                root_offset = (0.0, 0.0, self.init_z_offset)
            else:
                root_offset = None
            self.scene_objects.reset(motion_time=0.0, root_offset=root_offset)

        self._frame_idx = 0
        self._decimation_counter = 0
        self._prev_actions.reset()
        self._prev_pd = None
        self._prev_prev_pd = None
        self._ema_prev_targets = None
        self._heading_offset = None
        self._motion_pivot = None
        self._robot_pivot = None
        # ProtoMotions zero-fills the action history on reset
        # (env.py `_reset_state_history`: "Zero actions for historical reset").
        self._raw_action_history.reset()
        self._set_pd_targets_isaac(dof_pos_isaac.copy())
        self._max_ref_err = 0.0

        # Open-loop replay anchoring, and the two control shapes need opposite
        # treatment here -- getting this wrong shifts the whole replay by one
        # control step, which reads as a physics divergence that is not there.
        #
        # ``forward()`` (callback path) fires *after* each substep and only calls
        # into the tape on every ``decimation``-th one, so nothing would read
        # frame 0 or apply ``tape[0]`` before the first four substeps run. It
        # needs the anchor.
        #
        # ``control_step()`` (the default) already opens every control step with
        # ``_tape_step()`` *before* its substeps -- IsaacLab's own order. Calling
        # the anchor as well would make the loop's first call re-read the same
        # untouched reset state as frame 1 and apply ``tape[1]``, so ``tape[0]``
        # would never reach physics and every recorded frame ``k`` would hold the
        # state after ``k-1`` control steps. The tell is ``dof_vel`` at frame 1
        # being bit-identical to frame 0.
        if self._action_tape is not None and not self.control_in_loop:
            self._tape_step()

        log.info(
            f"--- Loop {self._loop_idx + 1}"
            + (f"/{self.num_loops}" if self.num_loops < 1_000_000 else "")
            + f" --- root_pos={root_pos.round(3).tolist()}"
        )

    def _write_robot_state(
        self,
        root_pos: np.ndarray,
        root_quat_xyzw: np.ndarray,
        dof_pos_policy: np.ndarray,
        dof_vel_policy: np.ndarray,
        root_lin_vel: np.ndarray,
        root_ang_vel: np.ndarray,
    ) -> np.ndarray:
        """Write a complete robot state into PhysX, in the one validated order.

        Extracted from :meth:`reset_episode` so ``--resync-state`` reuses the
        *same* write sequence rather than a second, plausible-looking one. Order
        is load-bearing and was established empirically: pose, then joint
        positions, then joint velocities, then root velocities, then the FK
        refresh. Link transforms lag the joint state until that refresh, so
        anything reading the anchor pose before it -- the heading offset latch,
        the foot-height probe -- would see the *previous* step's kinematics.

        Velocities are written, not left at rest: ProtoMotions' reference-state
        initialization seeds root and joint velocities from the motion
        (``compute_ref_reset_state``) and the policy was trained on that
        distribution, so dropping a moving clip in at zero velocity is
        off-distribution for exactly the first control steps that decide whether
        the episode stays upright.

        **Call only from outside the physics step.** See
        :meth:`reset_episode`'s note; this is why ``--resync-state`` requires
        ``--control-in-loop``.

        Args:
            root_pos: World root position ``[3]``.
            root_quat_xyzw: World root orientation ``[4]`` (xyzw).
            dof_pos_policy: Joint positions ``[num_dofs]`` in policy order.
            dof_vel_policy: Joint velocities ``[num_dofs]`` in policy order.
            root_lin_vel: World root linear velocity ``[3]``.
            root_ang_vel: World root angular velocity ``[3]``.

        Returns:
            The joint positions that were written, in Isaac Sim DOF order.
        """
        root_quat_wxyz = np.asarray(root_quat_xyzw)[[3, 0, 1, 2]]
        self.robot.set_world_pose(
            position=self._robot_array(np.asarray(root_pos, dtype=np.float32)),
            orientation=self._robot_array(root_quat_wxyz.astype(np.float32)),
        )

        dof_pos_isaac = self._to_isaac_order(dof_pos_policy)
        self.robot.set_joint_positions(self._robot_array(dof_pos_isaac))
        self.robot.set_joint_velocities(
            self._robot_array(self._to_isaac_order(dof_vel_policy))
        )
        self.robot.set_linear_velocity(
            self._robot_array(np.asarray(root_lin_vel, dtype=np.float32))
        )
        self.robot.set_angular_velocity(
            self._robot_array(np.asarray(root_ang_vel, dtype=np.float32))
        )

        self._zero_drive_velocity_targets()
        self._refresh_articulation_kinematics()
        return dof_pos_isaac

    def _zero_drive_velocity_targets(self) -> None:
        """Undo the velocity *target* that ``set_joint_velocities`` writes as a side effect.

        **This was the parity bug.** Isaac Sim's
        ``Articulation.set_joint_velocities`` (``core/prims/impl/articulation.py``)
        does *two* writes, not one::

            self._physics_view.set_dof_velocities(new_dof_vel, indices)
            self._physics_view.set_dof_velocity_targets(new_dof_vel, indices)

        so writing the joint state also makes the PD drive's velocity target equal
        the written velocity. PhysX then solves
        ``kp*(qTarget - q) + kd*(vTarget - v)``, and with ``vTarget = v_reset`` the
        damping term is offset by a constant ``kd * v_reset`` for the rest of the
        episode -- nothing ever writes a velocity target again, because
        ``ArticulationController.apply_action`` passes ``joint_velocities=None``
        straight through and leaves the stale value in place.

        IsaacLab never does this: ``write_joint_state_to_sim`` writes velocities
        only, and ``ImplicitActuator`` re-writes ``joint_vel_target = 0`` on every
        substep. So training damps toward zero and this driver damped toward its
        reset velocity.

        Measured with ``--drive-probe`` (contact-free, one control step from an
        identical state with an identical target): the driver read back
        ``max|vTarget| = 9.998 rad/s`` where IsaacLab read back exactly 0, its
        joints kept 40-65% more speed on the ``damp_*`` cases, and the
        ``gravity_only`` and ``step_*`` cases -- which carry no damping term --
        matched to within 3%. That is the whole of the "under-damped by 44%"
        residual the resync probe measured in round 3, and it also explains why
        0.1 rad.s of *joint friction* propped this driver up: it was substituting
        for the joint damping PhysX was not applying.
        """
        view = self.robot._articulation_view
        setter = getattr(view, "set_joint_velocity_targets", None)
        if setter is None:  # pragma: no cover - version dependent
            log.warning(
                "ArticulationView has no set_joint_velocity_targets(); the drive "
                "velocity target may stay at the last written joint velocity."
            )
            return
        # Allocated and converted once: the value never changes, and this runs
        # on every physics substep, where a fresh np.zeros plus a pageable
        # host-to-device copy is pure overhead.
        if self._zero_vel_tensor is None:
            self._zero_vel_tensor = self._for_view(
                view, np.zeros((1, self.num_dofs), dtype=np.float32)
            )
        physics_view = self._all_dof_write_view(view)
        if physics_view is not None:
            physics_view.set_dof_velocity_targets(
                self._zero_vel_tensor, self._all_env_indices
            )
        else:
            setter(self._zero_vel_tensor)

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
    # Phase timing (--profile)
    # ------------------------------------------------------------------

    def _tick(self) -> float:
        """Start a phase timer, or return 0.0 when not profiling."""
        return time.perf_counter() if self.profile else 0.0

    def _tock(self, name: str, t0: float) -> None:
        """Charge the elapsed time since ``t0`` to phase ``name``.

        The buckets partition one control step. They are wall clock, not CPU
        time, which is the point: most of what this loop spends on the GPU
        pipeline is spent *waiting* -- a ``.cpu()`` on a physics-view tensor is
        a ``cudaStreamSynchronize``, and a profiler that only counted CPU work
        would report it as free.
        """
        if self.profile:
            self._phase_ms[name] = (
                self._phase_ms.get(name, 0.0) + (time.perf_counter() - t0) * 1000.0
            )

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
        _t = self._tick()
        dof_pos, dof_vel, anchor_rot, root_local_ang_vel = self._read_robot_state()
        self._tock("state_read", _t)

        # Heading offset: aligns the reference motion's yaw to the robot's yaw at
        # episode start. See test_tracker_mujoco.py's docstring for background.
        if self._heading_offset is None:
            frame0 = self.motion_player.get_state_at_frame(0)
            motion_anchor_rot = frame0["body_rot"][self.anchor_body_index]
            self._heading_offset = compute_yaw_offset_np(anchor_rot, motion_anchor_rot)
            # Pivots for the position half of the realignment (max-coords only).
            self._motion_pivot = np.asarray(
                frame0["body_pos"][0], dtype=np.float32
            ).copy()
            robot_pivot = np.asarray(
                _to_numpy(self.robot.get_world_pose()[0]), dtype=np.float32
            ).copy()
            # **Horizontal only.** The realignment exists to absorb an arbitrary
            # startup heading and ground position; the *vertical* offset between
            # robot and reference is physically meaningful and must survive.
            # ProtoMotions resets the robot `env.config.ref_respawn_offset`
            # (0.05 m) above the reference, so the policy is trained seeing a
            # target 5 cm below itself at t=0. Cancelling that -- which taking
            # the robot's own z as the pivot does -- silently changes the very
            # first observation. Measured against IsaacLab's recorded context
            # under --init-state: it showed up as exactly 0.05 m on
            # `mimic.future_pos` and nothing else.
            robot_pivot[2] = self._motion_pivot[2]
            self._robot_pivot = robot_pivot

        _t = self._tick()
        future_refs = self.motion_player.get_future_references(
            self._frame_idx, self.future_step_indices
        )
        future_refs["body_rot"] = apply_heading_offset_np(
            self._heading_offset, future_refs["body_rot"]
        )
        self._tock("motion_ref", _t)

        body_state = None
        if self._needs_body_state:
            _t = self._tick()
            body_state = self._read_body_state()
            self._tock("state_read", _t)
            # Max-coordinate target poses difference the reference *positions*
            # against the live root in world coordinates, so a rotation-only
            # realignment is not enough -- the reference has to move as a rigid
            # body. Skipped entirely when the transform is the identity (the
            # robot spawned at motion frame 0), which keeps the normal path
            # bit-exact instead of paying float32 rounding for a no-op.
            if not self._alignment_is_identity():
                for key in ("body_pos",):
                    future_refs[key] = apply_heading_offset_to_positions_np(
                        self._heading_offset,
                        future_refs[key],
                        self._motion_pivot,
                        self._robot_pivot,
                    )
                # Velocities are free vectors: rotate, do not translate.
                for key in ("body_vel", "body_ang_vel"):
                    future_refs[key] = quat_rotate_np(
                        self._heading_offset, future_refs[key]
                    )

        _t = self._tick()
        onnx_inputs = build_onnx_inputs(
            dof_pos=dof_pos,
            dof_vel=dof_vel,
            anchor_rot=anchor_rot,
            root_local_ang_vel=root_local_ang_vel,
            future_refs=future_refs,
            anchor_body_index=self.anchor_body_index,
            onnx_name_to_key=self.onnx_name_to_key,
            num_dofs=self.num_dofs,
            prev_actions=self._prev_actions.view()[0],
            body_state=body_state,
            raw_action_history=(
                self._raw_action_history.view() if self._needs_raw_actions else None
            ),
            ground_height=self._ground_height_under_root(),
            num_bodies=self.num_bodies,
        )
        self._tock("obs_build", _t)
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
        self._tock("onnx", t0)
        _t = self._tick()
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
        self._prev_actions.push(pd_targets)

        # `historical.actions` is the other thing: the raw pre-tanh network
        # output, straight off the `actions` head, unfiltered. Both histories
        # are pushed here, at the end of the control step, mirroring
        # `BaseEnv.post_physics_step`; ActionHistory owns the `actions[:, 1:]`
        # view that keeps the observed window one step behind this push.
        if self._needs_raw_actions:
            raw = (
                ort_out[self.onnx_out_names.index("actions")]
                .squeeze()
                .astype(np.float32)
            )
            self._raw_action_history.push(raw)

        self._set_pd_targets(pd_targets)

        self._log_progress(dof_pos)
        self._record_trace(dof_pos, dof_vel, anchor_rot)
        self._tock("post", _t)

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

        The default since round 4; ``--no-control-in-loop`` selects
        :meth:`forward` instead. **Read the robustness table at the end before
        changing that back.**

        The callback path -- driving from ``world.add_physics_callback`` (see
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

        **Robustness.** Measured on ``g1_walk_box``, 5 consecutive loops,
        headless, ``--device cuda:0``. The round-3 column is what these configs
        scored while the drive velocity target was poisoned (see
        :meth:`_zero_drive_velocity_targets`); the round-4 column is the same
        commands after the fix, with training's 8/4 scene solver iterations
        throughout:

        ====================================== ======================= ================
        config                                 round 3                 round 4
        ====================================== ======================= ================
        callback + friction (old default)      0.0866, no fall         0.0537, no fall
        this + friction                        (not measured)          0.0486, no fall
        this + no friction (**new default**)   0.1009, **4 of 5 fell** **0.0454, no fall**
        ====================================== ======================= ================

        IsaacLab's own score on this clip is **0.0437 rad**, so the last row is
        parity to 4%. Note what changed qualitatively: before the fix, the
        training-faithful pairing was the *worst* and needed a third,
        anti-faithful flag (``--scene-solver-iterations default``, 255/255 where
        training sets 8/4) to reach zero falls. After it, the training-faithful
        pairing is simply the best, and that crutch is no longer needed. That is
        what finally made both flags defaults in round 4.

        A "fall" is the pelvis dropping to ~0.1 m with ~90 deg tilt. They happened
        near the *end* of a clip (~step 240 of 253), so a single-loop score --
        what ``--headless`` runs by default -- could not see them at all. That is
        how this got made the default once and had to be reverted; score
        ``--loops 5`` before touching it again.

        These numbers were all measured ``--headless``. A *windowed* run of this
        path used to fall within a couple of control steps, because rendering was
        requested through ``world.step(render=True)`` -- which quietly ran four
        extra physics substeps per control step (35 ms of physics per 20 ms of
        reference time). Only this path was affected: the callback path's
        decimation counter counts physics callbacks, so it stayed self-consistent
        through the extra substeps. Rendering now goes through :func:`_render`,
        and windowed reproduces headless to 4 s.f. (0.0454 rad / 0.0086 m /
        1.92 deg over 5 loops, either way).

        Note the loops are not independent even though ``reset_episode`` rewrites
        the full root/joint state: the per-loop errors still differ slightly, so
        something in PhysX -- solver warm-start or contact caches -- survives the
        reset.
        """
        if self.done:
            return
        if self._action_tape is not None:
            self._tape_step()
        else:
            self._compute_action()

        for _ in range(self.decimation):
            _t = self._tick()
            self._apply_pd_targets()
            self._tock("drive_write", _t)
            # Never `world.step(render=True)` here -- it does not step physics
            # once, it runs a whole Kit app update. See _render() for why.
            _t = self._tick()
            world.step(render=False)
            self._tock("physics_step", _t)
        # Render once per control step, after the substeps: IsaacLab renders at
        # the control rate, and rendering every substep would quadruple the frame
        # cost for frames nobody asked for.
        if render:
            _t = self._tick()
            _render(world)
            self._tock("render", _t)

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

        ``foot_z`` is added when the foot probe resolved (Isaac-family only; the
        MuJoCo harness has no equivalent), which turns the trace's ``root_h``
        column from a bare height into a decomposition: ``pelvis - foot`` is
        posture, ``foot - ground`` is contact.

        ``obj_pos_err``/``obj_rot_err_deg`` are added with ``--scenes-file``.
        """
        if self.trace is None:
            return
        ref = self.motion_player.get_state_at_frame(self._frame_idx)
        root_pos, _ = self.robot.get_world_pose()
        obj_pos_err, obj_rot_err_deg = self._object_pose_errors()
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
                foot_z=self._lowest_foot_z(),
                obj_pos_err=obj_pos_err,
                obj_rot_err_deg=obj_rot_err_deg,
            )
        )

    def _object_pose_errors(self):
        """Scene-object pose error against the SceneLib reference at this frame.

        The object-side analogue of ``joint_err``: how far the objects are from
        where the scene's reference trajectory says they should be *now*. Nothing
        drives them there -- they are written once at reset and then left to
        physics, exactly as in IsaacLab -- so this measures the manipulation, not
        a tracking controller.

        Returns:
            ``(mean position error [m], mean orientation error [deg])``, or
            ``(None, None)`` without ``--scenes-file``, which drops the columns.
        """
        if self.scene_objects is None:
            return None, None
        ref_pos, ref_quat = self.scene_objects.reference_pose(
            self._frame_idx * self.control_dt
        )
        pos, quat = self.scene_objects.measured_pose()
        pos_err = float(np.linalg.norm(pos - ref_pos, axis=-1).mean())
        rot_err = float(
            np.mean(
                [quat_angle_deg_xyzw(quat[i], ref_quat[i]) for i in range(len(quat))]
            )
        )
        return pos_err, rot_err

    def forward(self, dt: float) -> None:
        """Physics-callback entry point -- called once per physics substep.

        Note the callback fires *after* the substep it is passed, despite
        ``SimulationContext.add_physics_callback``'s docstring saying otherwise
        (measured). The closed-loop path is unaffected -- it reads state and
        immediately applies the resulting target, so it is self-consistent
        whatever the phase -- but the open-loop replay has to count completed
        substeps to land on IsaacLab's control-step boundaries, hence the
        separate branch.

        ``--resync-state`` is deliberately *not* reachable from here: it writes
        articulation root and joint state, which is undefined from inside a
        physics callback (see :meth:`reset_episode`). ``main()`` rejects the
        combination rather than letting it produce plausible-looking numbers.
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
        self._apply_pd_targets()

    def _set_pd_targets(self, policy_ordered: np.ndarray) -> None:
        """Latch a new PD target, in policy order, and invalidate its device copy.

        Every write to ``_pd_targets_isaac`` goes through here so the cached
        ``(1, num_dofs)`` backend tensor :meth:`_pd_target_tensor` hands to the
        drive can never go stale.

        The NaN check is what stands in for the per-DOF scan
        ``ArticulationController.apply_action`` used to run on our behalf (see
        :meth:`_apply_pd_targets`). Two differences, both deliberate: it is one
        vectorised test on host memory instead of ``num_dofs`` device
        round-trips, and it *raises* where the scan silently substituted the
        previously-applied target. A NaN out of the policy means the run is
        already broken; papering over it produces a plausible-looking trajectory
        that is impossible to explain later. It fires before PhysX is told
        anything, so nothing partial reaches the sim.
        """
        self._set_pd_targets_isaac(self._to_isaac_order(policy_ordered))

    def _set_pd_targets_isaac(self, targets: np.ndarray) -> None:
        """:meth:`_set_pd_targets` for an array that is already in Isaac DOF order."""
        if not np.isfinite(targets).all():
            raise RuntimeError(
                "Non-finite PD target at frame "
                f"{self._frame_idx}: "
                f"{np.flatnonzero(~np.isfinite(targets)).tolist()} "
                "(Isaac DOF indices). The policy output, the accel clamp or the "
                "EMA filter produced NaN/inf; nothing was written to PhysX."
            )
        self._pd_targets_isaac = targets
        self._pd_target_tensor_cache = None

    def _pd_target_tensor(self):
        """The current PD targets as a ``(1, num_dofs)`` tensor of the view's backend.

        Cached, because the target only changes once per *control* step while
        the drive is commanded once per *physics* substep -- so the host-to-device
        convert happens once per control step instead of ``decimation`` times.
        Built lazily rather than in ``__init__``: :meth:`_for_view` needs the
        articulation view's ``_backend_utils``, which only exists after
        :meth:`initialize`.
        """
        if self._pd_target_tensor_cache is None:
            self._pd_target_tensor_cache = self._robot_array(
                self._pd_targets_isaac[None, :]
            )
        return self._pd_target_tensor_cache

    def _apply_pd_targets(self) -> None:
        """Command the drive exactly as IsaacLab does: position target *and* zero velocity target.

        The **zero velocity target is not decoration**. IsaacLab's
        ``ImplicitActuator.write_data_to_sim`` writes ``joint_vel_target`` (which
        ProtoMotions leaves at 0) to PhysX on every substep, so training's drive
        always damps toward zero. ``ArticulationController.apply_action`` skips
        that write when ``joint_velocities`` is ``None``, leaving whatever
        ``set_joint_velocities`` last stamped on the target -- see
        :meth:`_zero_drive_velocity_targets` for the measurement that caught it.

        The zero could never have gone through an ``ArticulationAction`` anyway:
        ``ArticulationController.apply_action`` NaN-scans ``joint_velocities``
        with a bare ``np.isnan(joint_velocities[0][i])`` (positions get a
        ``_backend_utils.to_numpy`` first, velocities do not), which raises on
        the GPU backend -- the only backend that matches training.
        ``view.set_joint_velocity_targets`` is what that call would have reached
        anyway.

        Re-asserted every substep rather than only at reset, so nothing that
        writes joint velocities mid-episode can silently reintroduce the bias.

        **The position target now goes the same way, bypassing**
        ``SingleArticulation.apply_action`` **and its controller.** That is a
        performance fix, not a behavioural one:
        ``ArticulationController.apply_action`` scans the position target for
        NaN *one DOF at a time*, and each test is
        ``_backend_utils.to_numpy(joint_positions[0][i])`` -- i.e. ``.cpu()`` on
        a 0-dim CUDA tensor, which is a ``cudaStreamSynchronize`` plus a 4-byte
        readback. At soma23's 66 DOFs and decimation 4 that is **264
        synchronising round-trips per control step**, on top of a
        ``get_applied_action()`` (three physics-view reads) whose result the scan
        is the only consumer of. Everything the controller does after the scan
        reduces to ``_physics_view.set_dof_position_targets(...)``, which
        ``view.set_joint_position_targets`` reaches directly. Same buffer, same
        float32 values, same point in the loop -- verified bit-identical over 5
        loops of ``soma_sit_1`` against ``--trace-out``/``--onnx-inputs-out``.
        The NaN guard the scan provided moved to :meth:`_set_pd_targets`, where
        it costs one vectorised host-side test.
        """
        view = self.robot._articulation_view
        physics_view = self._all_dof_write_view(view)
        if physics_view is not None:
            physics_view.set_dof_position_targets(
                self._pd_target_tensor(), self._all_env_indices
            )
        else:
            view.set_joint_position_targets(self._pd_target_tensor())
        self._zero_drive_velocity_targets()

    def _all_dof_write_view(self, view):
        """The physics view, when this driver can write every DOF of it directly.

        ``Articulation.set_joint_*_targets`` reads the current target buffer,
        scatters the new values into it, and writes it back -- a
        read-modify-write that only earns its keep for a partial update. This
        driver always commands *every* DOF of the *only* articulation in the
        view, so the scattered result is exactly the ``(1, num_dofs)`` tensor
        going in, and the read and the scatter are both dead work on a call that
        runs ``decimation`` times per control step.

        Returns ``None`` -- and the caller falls back to the public setter --
        whenever that reasoning does not hold or the private handle is missing,
        so a future Isaac Sim can only cost speed here, never correctness.
        """
        if not getattr(view, "_is_initialized", False):
            return None
        physics_view = getattr(view, "_physics_view", None)
        if physics_view is None or view.count != 1 or view.num_dof != self.num_dofs:
            return None
        if self._all_env_indices is None:
            self._all_env_indices = view._backend_utils.resolve_indices(
                None, view.count, view._device
            )
        return physics_view

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
        message = (
            f"\n=== Done: {self._total_steps} steps over {self._loop_idx} loop(s) ===\n"
            f"  avg ONNX inference : {self.total_ort_ms / steps:.2f} ms/step\n"
            f"  avg physics        : {total_sim_ms / steps:.2f} ms/step\n"
            f"  max joint ref error: {self.max_ref_err_run:.4f} rad"
        )
        if self.profile and self._phase_ms:
            # Ordered as the control step runs them, not by size, so the table
            # reads as a walk through control_step(). `accounted` will fall a
            # little short of `avg physics` -- the difference is the loop
            # scaffolding outside any bucket.
            order = (
                "state_read",
                "motion_ref",
                "obs_build",
                "onnx",
                "post",
                "drive_write",
                "physics_step",
                "render",
            )
            rows = [k for k in order if k in self._phase_ms]
            rows += [k for k in sorted(self._phase_ms) if k not in order]
            width = max(len(k) for k in rows)
            accounted = sum(self._phase_ms[k] for k in rows)
            message += "\n  phase breakdown (ms/step):"
            for key in rows:
                per_step = self._phase_ms[key] / steps
                share = 100.0 * self._phase_ms[key] / max(accounted, 1e-9)
                message += f"\n    {key:<{width}}  {per_step:7.2f}  ({share:4.1f}%)"
            message += f"\n    {'accounted':<{width}}  {accounted / steps:7.2f}"
        if self._resync_write_error:
            worst = " ".join(
                f"{k}={v:.3e}" for k, v in sorted(self._resync_write_error.items())
            )
            message += (
                f"\n  resync write-back error (worst over the run): {worst}\n"
                "    Anything but ~0 here invalidates the divergence numbers: it "
                "would mean the state PhysX started each step from was not the "
                "state IsaacLab recorded."
            )
        log.info(message)

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
        # `get_dof_limits` is [num_envs, num_dofs, 2] rather than per-DOF scalar,
        # so it is printed separately below. It is here because it was the one
        # solver-visible per-joint quantity no dump on either side reported, and
        # a limit is a *force*: a pose outside it is pushed back with no contact
        # and no drive torque involved.
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

        limits = None
        if physics_view is not None:
            try:
                limits = _to_numpy(physics_view.get_dof_limits()).reshape(-1, 2)
            except Exception as e:  # pragma: no cover - version dependent
                log.warning(f"  dof_limits read-back failed: {e}")
        if limits is not None:
            log.info("  dof_limits (sim joint order, [lower, upper] rad):")
            for name, (lower, upper) in zip(view.dof_names, limits):
                log.info(f"    {name:18s} [{lower:+.6f}, {upper:+.6f}]")

        log.info("\n=== PhysX scene prim (driver) ===")
        physx_probe.dump_physics_scene(get_current_stage(), log.info)

        self.dump_collision_filtering()
        self.dump_material_stack()

    def dump_collision_filtering(self) -> None:
        """Print the articulation's self-collision filter set.

        Delegates to :func:`deployment.physx_probe.dump_filtered_pairs`, which
        the IsaacLab dump calls too, so the two logs come out of one reader --
        see that function for why the relation matters.
        """
        stage = get_current_stage()
        root = stage.GetPrimAtPath(self._prim_path)
        log.info("\n=== Self-collision filtered pairs (driver) ===")
        physx_probe.dump_filtered_pairs(root, log.info)

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

        log.info(
            "  (trailing '*' = value is the schema default, unauthored by any layer; "
            "'*(auto)' = PhysX derives it from shape size)"
        )

        self.dump_height_decomposition(ground_colliders)
        self.dump_link_properties()

    def dump_height_decomposition(self, ground_colliders: list) -> None:
        """Report pelvis / lowest-foot-collider / ground heights and their gaps.

        The measurement that decides the contact hypothesis. The driver's pelvis
        sits ~1.4 cm above its reference for whole episodes where IsaacLab's sits
        on it, unmoved by ground geometry, collider offsets or friction. That is
        one number with two incompatible explanations, and splitting it settles
        which:

        - **pelvis - lowest foot collider** is *posture*: it is fixed by the joint
          angles alone. If this agrees between the two stacks, the robot is
          standing the same way and the disagreement is about where the floor is.
        - **lowest foot collider - ground z** is *contact*: how far the foot rests
          above (or into) the collision surface. If this differs, the gap is in
          the contact solve -- offsets, penetration, patch friction -- and not in
          the ONNX graph, the observation assembly or the policy.

        Reports link origins alongside, since every earlier dump reported only
        those and the two logs need to stay comparable.
        """
        from pxr import UsdGeom

        log.info("\n=== Height decomposition (driver) ===")
        link_tf = self._read_link_transforms()
        if link_tf is None:
            log.warning("  link transforms unavailable -- cannot decompose height.")
            return

        root_z = float(link_tf[0, 2])
        log.info(f"  pelvis link origin world z   : {root_z:.6f}")

        if self._foot_probe is None:
            log.warning(
                "  no foot probe -- reporting link origins only, which cannot "
                "separate posture from contact."
            )
        else:
            body_names = list(self.robot._articulation_view.body_names or [])
            for name in sorted(self._foot_probe.geometry):
                index = body_names.index(name)
                tips = self._foot_probe.geometry[name]
                log.info(f"  {name}:")
                log.info(
                    f"    link origin world z        : {float(link_tf[index, 2]):.6f}"
                )
                log.info(
                    f"    lowest collider world z    : "
                    f"{physx_probe.lowest_tip_z(link_tf[index, :3], link_tf[index, 3:7], tips):.6f}"
                )

            lowest = self._foot_probe.lowest(link_tf[:, :3], link_tf[:, 3:7])
            log.info(f"  lowest foot collider world z : {lowest:.6f}")
            log.info(
                f"  pelvis - lowest collider     : {root_z - lowest:.6f}   <- posture"
            )

            xform_cache = UsdGeom.XformCache()
            for prim in ground_colliders[:1]:
                ground_z = float(
                    xform_cache.GetLocalToWorldTransform(prim).ExtractTranslation()[2]
                )
                log.info(f"  ground prim {prim.GetPath()} world z = {ground_z:.6f}")
                log.info(
                    f"    lowest collider - ground   : {lowest - ground_z:.6f}   <- contact"
                )

    def dump_link_properties(self) -> None:
        """Print every per-link rigid-body property in sim link order.

        Closes the one diff that was never done PhysX-to-PhysX. Both stacks
        author ``physxRigidBody:*``, but through different traversals: this
        driver's :meth:`_author_body_properties` walks ``Usd.PrimRange`` *without*
        ``TraverseInstanceProxies``, while IsaacLab's ``modify_rigid_body_properties``
        is wrapped in ``apply_nested``, which skips instanced prims -- the same
        mechanism that already silently dropped the collider offsets. Two
        different skip rules over one instanceable asset can reach two different
        sets of links, and every rigid-body property had so far been compared
        USD-side or config-side only.

        The lead candidate inside this gap is ``maxAngularVelocity = 1000``,
        which both stacks write verbatim in **deg/s** -- a hard clip at
        17.45 rad/s on every link. A hard velocity clip is velocity-ranked and
        leg-concentrated by construction, which is exactly the shape of the
        residual divergence. It is *intended* to be equal on both sides; it has
        never been *read back* as equal.

        The reads deliberately traverse instance proxies (via
        :func:`deployment.physx_probe.resolve_link_prim_paths`), so the dump sees
        links the authoring walk may have missed. That asymmetry is the point.
        """
        view = self.robot._articulation_view
        physics_view = getattr(view, "_physics_view", None)
        body_names = list(view.body_names or [])

        def _view_read(method: str):
            fn = getattr(physics_view, method, None) if physics_view else None
            if fn is None:
                return None
            try:
                return _to_numpy(fn()).reshape(len(body_names), -1)
            except Exception as e:  # pragma: no cover - version dependent
                log.warning(f"  {method}() read-back failed: {e}")
                return None

        physx_probe.dump_link_properties(
            log.info,
            get_current_stage(),
            body_names=body_names,
            link_prim_paths=physx_probe.resolve_link_prim_paths(
                get_current_stage(), self._prim_path, body_names
            ),
            masses=_view_read("get_masses"),
            inertias=_view_read("get_inertias"),
            coms=_view_read("get_coms"),
        )

    def run_drive_probe(self, world, spec_path: str, out_path: str) -> None:
        """Replay the paired drive-response probe (see ``deployment/drive_probe.py``).

        The spec comes from the IsaacLab harness, so both stacks run *the same*
        cases rather than two independently derived sets. Each case teleports the
        robot 5 m into the air -- no contact at all -- writes a fixed state and a
        fixed PD target, then takes ``decimation`` substeps recording the joint
        response after every one.

        This is the experiment three rounds of property diffing could not
        substitute for. Every previous measurement was taken with the robot
        standing, so drive torque, gravity, contact and multibody coupling summed
        into one number; here only gravity and the drive act, gravity acts on
        verified-identical inertias from an identical state, and the
        ``gravity_only`` case makes even *that* falsifiable.

        Uses :meth:`_write_robot_state` verbatim -- the same write order the
        resync probe already verified lands in PhysX -- and ``apply_action`` in
        the same per-substep pattern as :meth:`control_step`, so nothing about
        the driver's own control path is bypassed except the policy.
        """
        from deployment import drive_probe

        spec = drive_probe.load_spec(spec_path)
        spec_joints = [str(x) for x in spec["spec__joint_names"]]
        if spec_joints != list(self.joint_names):
            raise SystemExit(
                "drive-probe spec joint names do not match this policy's:\n"
                f"  spec  : {spec_joints}\n  driver: {list(self.joint_names)}"
            )
        substeps = int(spec["spec__substeps"])
        if substeps != self.decimation:
            log.warning(
                f"drive-probe spec was recorded with decimation={substeps} but this "
                f"driver runs {self.decimation}; replaying the spec's {substeps}."
            )
        dt = float(spec["spec__physics_dt"])
        if abs(dt - self.physics_dt) > 1e-9:
            log.warning(
                f"drive-probe spec physics_dt={dt} but this driver runs "
                f"{self.physics_dt}; the comparison is not apples-to-apples."
            )

        recorder = drive_probe.ProbeRecorder(spec, stack="isaacsim-driver")
        view = self.robot._articulation_view
        physics_view = getattr(view, "_physics_view", None)

        def _sample(case: int) -> None:
            root_pos, root_quat_wxyz = self.robot.get_world_pose()
            recorder.sample(
                case,
                dof_pos=_to_numpy(self.robot.get_joint_positions())[
                    self._isaac_to_policy
                ],
                dof_vel=_to_numpy(self.robot.get_joint_velocities())[
                    self._isaac_to_policy
                ],
                root_pos=_to_numpy(root_pos),
                root_quat_xyzw=wxyz_to_xyzw(_to_numpy(root_quat_wxyz)),
            )

        target_readback, vel_target_readback, actuation_readback = [], [], []
        for case in range(recorder.num_cases):
            self._write_robot_state(
                root_pos=spec["spec__root_pos"][case],
                root_quat_xyzw=spec["spec__root_quat_xyzw"][case],
                dof_pos_policy=spec["spec__dof_pos"][case],
                dof_vel_policy=spec["spec__dof_vel"][case],
                root_lin_vel=spec["spec__root_lin_vel"][case],
                root_ang_vel=spec["spec__root_ang_vel"][case],
            )
            self._set_pd_targets(spec["spec__target"][case])
            self._apply_pd_targets()
            _sample(case)
            target_readback.append(
                self._probe_dof_read(physics_view, "get_dof_position_targets")
            )
            vel_target_readback.append(
                self._probe_dof_read(physics_view, "get_dof_velocity_targets")
            )
            # Mirror of the IsaacLab read: the explicit joint force PhysX holds,
            # separate from the implicit drive. See the note in
            # trace_tracker_isaaclab.run_drive_probe.
            actuation_readback.append(
                self._probe_dof_read(physics_view, "get_dof_actuation_forces")
            )

            for _ in range(substeps):
                self._apply_pd_targets()
                world.step(render=False)
                _sample(case)

        recorder.write(
            out_path,
            actuation_readback=actuation_readback,
            target_readback=target_readback,
            vel_target_readback=vel_target_readback,
            stiffness=self._probe_dof_read(physics_view, "get_dof_stiffnesses"),
            damping=self._probe_dof_read(physics_view, "get_dof_dampings"),
            armature=self._probe_dof_read(physics_view, "get_dof_armatures"),
            friction=self._probe_dof_read(
                physics_view, "get_dof_friction_coefficients"
            ),
            max_force=self._probe_dof_read(physics_view, "get_dof_max_forces"),
            max_velocity=self._probe_dof_read(physics_view, "get_dof_max_velocities"),
        )
        log.info(
            f"Drive probe (isaacsim-driver, {recorder.num_cases} cases x "
            f"{substeps} substeps) -> {out_path}"
        )

    def _probe_dof_read(self, physics_view, method: str):
        """Read one per-DOF array off the physics view, in policy order."""
        fn = getattr(physics_view, method, None) if physics_view else None
        if fn is None:
            log.warning(f"drive probe: view has no {method}()")
            return None
        try:
            return _to_numpy(fn()).reshape(-1)[self._isaac_to_policy]
        except Exception as e:  # pragma: no cover - version dependent
            log.warning(f"drive probe: {method}() read-back failed: {e}")
            return None

    def write_divergence(self, path: str) -> None:
        """Dump the divergence against the IsaacLab trajectory.

        Two report shapes, because the two experiments answer different
        questions:

        **Accumulating** (plain ``--action-tape``): reports the first control
        step at which ``|Δdof_pos|∞`` crosses 0.02, 0.05 and 0.10 rad, and which
        joint led. Contact-rich open-loop replay always diverges eventually;
        **when** and on which joint is the signal, not whether.

        **Resynced** (``--resync-state``): every row is an independent one-step
        error, so the distribution is meaningful where the accumulating one's is
        not -- and it can be *split by stance phase*, which is the measurement
        that separates the two remaining hypotheses. Divergence concentrated in
        stance implicates the contact solve; divergence equally large in flight
        exonerates contact and points at the articulation solve or drive
        integration.
        """
        if not self.divergence:
            log.warning("No divergence recorded -- nothing written.")
            return
        with open(path, "w") as f:
            json.dump(self.divergence, f)

        if self.resync_state:
            log.info(self._format_resync_divergence(path))
            return

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

    def _format_resync_divergence(self, path: str) -> str:
        """Summarise the per-step resync divergence, split by stance phase."""
        rows = self.divergence or []
        dof = np.array([r["max_dof_delta"] for r in rows])
        vel = np.array([r["max_dof_vel_delta"] for r in rows])
        root = np.array([r["root_pos_delta"] for r in rows])
        quat = np.array([r["root_quat_delta_deg"] for r in rows])

        # Which joints carry the one-step error, counted over steps rather than
        # averaged: a single joint leading 200 of 250 steps is a mechanism, the
        # same total spread over 29 joints is numerical noise.
        worst = {}
        for row in rows:
            worst[row["worst_joint"]] = worst.get(row["worst_joint"], 0) + 1
        leaders = sorted(worst.items(), key=lambda kv: -kv[1])[:5]

        lines = [
            f"\n=== Per-step resync divergence ({len(rows)} one-step experiments) "
            f"-> {path} ===",
            "  Each row is one control step from IsaacLab's exact state with "
            "IsaacLab's exact action;",
            "  no history accumulates, so these are directly comparable across steps.",
            f"  |d dof_pos|inf   mean={dof.mean():.5f} p50={np.median(dof):.5f} "
            f"p95={np.percentile(dof, 95):.5f} max={dof.max():.5f} rad",
            f"  |d dof_vel|inf   mean={vel.mean():.4f} p95={np.percentile(vel, 95):.4f} "
            f"max={vel.max():.4f} rad/s",
            f"  |d root_pos|     mean={root.mean():.6f} max={root.max():.6f} m",
            f"  |d root_quat|    mean={quat.mean():.4f} max={quat.max():.4f} deg",
            "  worst-joint tally: "
            + ", ".join(f"{name} x{count}" for name, count in leaders),
        ]

        contacts = np.array([r.get("foot_contact", -1) for r in rows])
        if (contacts >= 0).all():
            lines.append("  --- split by IsaacLab's foot-contact state ---")
            for label, mask in (
                ("flight (0 feet)", contacts == 0),
                ("single stance", contacts == 1),
                ("double stance", contacts == 2),
            ):
                if not mask.any():
                    lines.append(f"  {label:16s} n=0")
                    continue
                lines.append(
                    f"  {label:16s} n={int(mask.sum()):4d}  "
                    f"|d dof|inf mean={dof[mask].mean():.5f} "
                    f"p95={np.percentile(dof[mask], 95):.5f}  "
                    f"|d root_pos| mean={root[mask].mean():.6f}"
                )
            flight, stance = contacts == 0, contacts > 0
            if flight.any() and stance.any():
                ratio = dof[stance].mean() / max(dof[flight].mean(), 1e-12)
                lines.append(
                    f"  stance/flight |d dof|inf ratio = {ratio:.2f}  "
                    "(>>1 implicates the contact solve; ~1 exonerates it and "
                    "points at the articulation solve)"
                )
        else:
            lines.append(
                "  (no contact labels in the tape -- re-record it to split by "
                "stance phase)"
            )
        return "\n".join(lines)

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


def _scene_iteration_attrs(world) -> dict:
    """Return the PhysX scene's solver iteration-count attributes, if declared."""
    ctx = world.get_physics_context()
    prim = get_current_stage().GetPrimAtPath(ctx.prim_path)
    attrs = {}
    for name in (
        "physxScene:maxPositionIterationCount",
        "physxScene:maxVelocityIterationCount",
    ):
        attr = prim.GetAttribute(name)
        if attr:
            attrs[name] = attr
    return attrs


def _author_physics_scene_iterations(world, robot_props: dict) -> None:
    """Cap the PhysX *scene* solver iterations at training's values.

    ``World`` leaves ``physxScene:maxPositionIterationCount`` /
    ``maxVelocityIterationCount`` at the USD schema defaults of **255 / 255**,
    while ProtoMotions' IsaacLab config sets them to the robot's
    ``num_position_iterations`` / ``num_velocity_iterations`` -- **8 / 4** on the
    G1 (``simulator/isaaclab/simulator.py``, ``PhysxCfg``). These are *caps*, and
    the articulation asks for 8/4 itself
    (:meth:`TrackerPolicy._apply_solver_iterations`), so for the articulation
    solve the two are equivalent. They are not necessarily equivalent for the
    **contact** solve, which is the part of the pipeline still under suspicion --
    and matching them costs one attribute write.

    **Written before ``world.reset()``, unlike the rest of the scene config.**
    ``PhysicsContext`` re-applies its own cached values when the sim starts
    playing, which is why :func:`_configure_physics_scene` must run *after*
    reset; but these two attributes are not in that cached set (``PhysicsContext``
    has no setter for them at all), so the reliable moment to author them is
    before PhysX parses the scene at play time. A post-reset USD write would read
    back as 8/4 whether or not PhysX ever consumed it, which is precisely the
    kind of unfalsifiable check this investigation has already been burned by.
    """
    if args.scene_solver_iterations != "training":
        log.info(
            "PhysX scene solver iteration caps left at World's defaults "
            "(--scene-solver-iterations default); training uses "
            f"{robot_props.get('solver_position_iterations')}/"
            f"{robot_props.get('solver_velocity_iterations')}."
        )
        return

    pos_iters = robot_props.get("solver_position_iterations")
    vel_iters = robot_props.get("solver_velocity_iterations")
    if pos_iters is None and vel_iters is None:
        return

    attrs = _scene_iteration_attrs(world)
    wanted = {
        "physxScene:maxPositionIterationCount": pos_iters,
        "physxScene:maxVelocityIterationCount": vel_iters,
    }
    for name, value in wanted.items():
        if value is None:
            continue
        attr = attrs.get(name)
        if attr is None:
            log.warning(f"PhysX scene has no '{name}' to set.")
            continue
        log.info(
            f"PhysX scene {name.split(':')[-1]}: {attr.Get()} -> {int(value)} "
            "(training's value)"
        )
        attr.Set(int(value))


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

    iterations = {
        name.split(":")[-1]: attr.Get()
        for name, attr in _scene_iteration_attrs(world).items()
    }
    log.info(
        "PhysX scene: ccd=False solver=TGS "
        f"stabilization={ctx.is_stablization_enabled()} "
        f"bounce_threshold={ctx.get_bounce_threshold():.3f} "
        f"gpu_dynamics={ctx.is_gpu_dynamics_enabled()} "
        f"iterations={iterations}"
    )
    _warn_if_diverging_from_training(ctx)


def _warn_if_diverging_from_training(ctx) -> None:
    """Name every live setting that differs from what ProtoMotions trained with.

    Each of these is a deliberate default, and each has now been measured as a
    real behavioural difference rather than a theoretical one, so a run that
    takes them should say so in its own log instead of only on screen. Round 3
    of this investigation spent four runs re-deriving that the demo was in a
    divergent configuration; a line in the log is cheaper.

    Only settings that are *live* are reported -- pass the matching flag and the
    warning goes away, which is what makes it worth reading.

    Args:
        ctx: The live ``PhysicsContext``, read after ``world.reset()``.
    """
    divergences = []
    if not ctx.is_gpu_dynamics_enabled():
        divergences.append(
            "PhysX runs on the CPU (dynamics off the GPU, MBP broadphase); "
            "training runs GPU dynamics with the GPU broadphase "
            "(`physxScene:enableGPUDynamics=True`, measured on both stacks). "
            "These are different solvers, not the same solver on different "
            "hardware: on soma_sit_1 the per-control-step root-orientation error "
            "against IsaacLab is 4.51 deg on the CPU and 0.057 deg on the GPU. "
            "Pass --device cuda:0 to match training."
        )
    for message in divergences:
        log.warning(f"DIVERGES FROM TRAINING: {message}")


def _add_physics_material(
    stage,
    path: str,
    static_friction: float,
    dynamic_friction: float,
    combine: str,
    restitution: float = 0.0,
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

    Args:
        stage: USD stage to author on.
        path: Prim path for the material.
        static_friction: Static friction coefficient.
        dynamic_friction: Dynamic friction coefficient.
        combine: Friction/restitution combine mode token, e.g. ``"average"``.
        restitution: Restitution coefficient. Defaults to 0.0, which is what
            every ground caller wants; scene objects pass their
            ``ObjectOptions.restitution`` through here.
    """
    mat_prim = UsdShade.Material.Define(stage, path).GetPrim()
    UsdPhysics.MaterialAPI.Apply(mat_prim)
    PhysxSchema.PhysxMaterialAPI.Apply(mat_prim)
    _set_prim_attr(mat_prim, "physics:staticFriction", float(static_friction))
    _set_prim_attr(mat_prim, "physics:dynamicFriction", float(dynamic_friction))
    _set_prim_attr(mat_prim, "physics:restitution", float(restitution))
    _set_prim_attr(mat_prim, "physxMaterial:frictionCombineMode", combine)
    _set_prim_attr(mat_prim, "physxMaterial:restitutionCombineMode", combine)
    return mat_prim


def add_protomotions_trimesh_ground(
    resolved_configs_path: str, friction: float, center_xy=None
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
    280 x 310 m, all at z = 0. It is recentred on ``center_xy`` (the origin by
    default) so the robot's spawn lands well inside it -- on a flat terrain the XY
    offset is not physically meaningful, but falling off the edge of the mesh
    would be.

    ``center_xy`` exists because of a real trap. The driver normally spawns near
    XY = 0, but ``--init-state``/``--resync-state`` write IsaacLab's *absolute*
    root position, and ProtoMotions spawns its robot in the middle of that
    280 x 310 m terrain -- i.e. up to ~140 m out, right at the edge of a
    mesh centred on the origin. Recentring on the actual spawn keeps the geometry
    relationship identical (the terrain is flat) while removing any chance of the
    robot walking off the collision mesh and "falling" for a reason that has
    nothing to do with physics parity.

    **Measured null on the seated task (round 3).** Against the IsaacLab
    reference for ``soma_sit_1`` (149 control steps), swapping the analytic
    plane for this mesh moves nothing on the CPU solver: closed-loop mean
    |Δ tilt| 27.43° → 27.38°, |Δ root_h| 0.0225 → 0.0230 m, |Δ foot_z| 0.0624 →
    0.0591 m; open-loop mean |Δ dof|∞ 0.6260 → 0.6246 rad. The
    contact-generation difference this function exists to test is real, but it
    does not explain the sit gap, so the plane stays the default. It is *not*
    null in combination with GPU dynamics -- see the ``--device`` note.

    Args:
        resolved_configs_path: Path to a run's ``resolved_configs_inference.pt``.
        friction: Static and dynamic friction for the terrain material.
        center_xy: Optional ``[2]`` XY to centre the mesh on. Defaults to the
            origin.
    """
    import torch
    from pxr import UsdGeom

    from deployment.scene_utils import load_resolved_configs
    from protomotions.components.terrains.terrain import Terrain

    # Lenient unpickling: a resolved-configs file also pickles the agent config,
    # whose module chain reaches ``lightning``, which the deployment interpreter
    # does not carry. Only ``terrain`` is read here. See load_resolved_configs.
    resolved = load_resolved_configs(resolved_configs_path)
    terrain_config = resolved["terrain"]
    terrain = Terrain(config=terrain_config, num_envs=1, device=torch.device("cpu"))

    vertices = np.asarray(terrain.vertices, dtype=np.float64).copy()
    triangles = np.asarray(terrain.triangles, dtype=np.int32)
    vertices[:, 2] += float(terrain_config.sim_config.height_offset)
    # Recentre on the spawn; ProtoMotions builds this mesh around the middle of a
    # 280 x 310 m terrain, and the driver's robot may sit anywhere from XY = 0
    # (its own default) to IsaacLab's absolute spawn (--init-state).
    target = (
        np.zeros(2) if center_xy is None else np.asarray(center_xy, dtype=np.float64)
    )
    for axis in (0, 1):
        vertices[:, axis] += target[axis] - 0.5 * (
            vertices[:, axis].min() + vertices[:, axis].max()
        )

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
        f"z={vertices[:, 2].min():.3f}) with friction {friction} / {combine}, "
        f"centred on XY={np.round(target, 3).tolist()}"
    )


# ---------------------------------------------------------------------------
# Scene objects (--scenes-file)
# ---------------------------------------------------------------------------

#: Per-kind fallback colours, copied from
#: ``IsaacLabSimulator._preprocess_object_playground`` so an object whose
#: ``ObjectOptions.color`` is unset looks the same in both stacks.
_OBJECT_COLOR_DEFAULTS = {
    "box": (0.8, 0.3, 0.3),
    "sphere": (0.3, 0.3, 0.8),
    "cylinder": (0.3, 0.8, 0.3),
    "mesh": (0.2, 0.7, 0.3),
}

#: Collider offsets IsaacLab spawns every scene object with
#: (``CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0)``).
_OBJECT_CONTACT_OFFSET = 0.002
_OBJECT_REST_OFFSET = 0.0

#: PhysX's own default friction, used when ``ObjectOptions`` requests none --
#: see :func:`_author_scene_object_physics` on why not the wrappers' default.
_PHYSX_DEFAULT_FRICTION = 0.5


def _author_scene_object_physics(
    stage, prim_path: str, spec, mesh_collision_approximation=None
) -> None:
    """Write one scene object's PhysX properties onto an already-created prim.

    Split out of :func:`add_scene_objects` so primitives and meshes go through
    the *same* property path -- the part that has to mirror IsaacLab is the
    physics, not the geometry.

    Three decisions worth stating:

    - **Collider offsets are authored unconditionally**, unlike the robot's
      (``--author-collider-offsets`` defaults to off). That is not a
      contradiction: the robot default exists because its colliders are
      *instance proxies* that IsaacLab's writer cannot reach, so training's
      0.02/0.0 request never lands and matching training means not authoring
      either. These object prims are freshly authored and not instance proxies,
      so IsaacLab's request *does* land on them there -- authoring is what
      matches.

    - **Mass, not the wrapper's guess.** ``DynamicCuboid``/``DynamicSphere``/
      ``DynamicCylinder`` default to ``mass=0.02`` kg when none is passed, which
      would silently outrank the scene's density. Author both fields explicitly:
      ``physics:mass`` when the scene names a mass, otherwise ``0.0`` (the
      schema's "derive from density") plus ``physics:density`` -- the same split
      as ``IsaacLabSimulator._mass_props_from_options``.

    - **The material is explicit.** IsaacLab passes no physics material for
      scene objects, so they land on PhysX's built-in default; the Isaac Sim
      convenience wrappers, in contrast, invent a 0.2/1.0 material. Neither is a
      value anyone chose. Author one: the scene's own friction/restitution when
      ``ObjectOptions`` supplies them (a deliberate divergence -- IsaacLab drops
      those fields, and for a pickup task grip friction is exactly the number
      that matters), otherwise PhysX's 0.5/0.5/0.0.

    Args:
        stage: The USD stage.
        prim_path: Path of the object's root prim.
        spec: The :class:`deployment.scene_utils.SceneObjectSpec` to apply.
        mesh_collision_approximation: ``SceneLibConfig.mesh_collision_approximation``,
            applied to mesh colliders when set.
    """
    prim = stage.GetPrimAtPath(prim_path)

    # fix_base_link is the *physical* static flag; SceneLib's `_is_static_object`
    # (= no motion data) is a different property and must not be used here.
    # IsaacLab: RigidBodyPropertiesCfg(kinematic_enabled=object_options.fix_base_link).
    _set_prim_attr(prim, "physics:kinematicEnabled", bool(spec.fix_base_link))

    UsdPhysics.MassAPI.Apply(prim)
    if spec.mass is not None:
        _set_prim_attr(prim, "physics:mass", float(spec.mass))
        _set_prim_attr(prim, "physics:density", 0.0)
    else:
        _set_prim_attr(prim, "physics:mass", 0.0)
        _set_prim_attr(prim, "physics:density", float(spec.density or 0.0))

    colliders = _find_prims_with_api(stage, prim_path, UsdPhysics.CollisionAPI)
    for collider in colliders:
        PhysxSchema.PhysxCollisionAPI.Apply(collider)
        _set_prim_attr(collider, "physxCollision:contactOffset", _OBJECT_CONTACT_OFFSET)
        _set_prim_attr(collider, "physxCollision:restOffset", _OBJECT_REST_OFFSET)
        if spec.kind == "mesh" and mesh_collision_approximation is not None:
            mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(collider)
            mesh_collision.CreateApproximationAttr().Set(
                str(mesh_collision_approximation)
            )

    static_friction = (
        spec.static_friction
        if spec.static_friction is not None
        else _PHYSX_DEFAULT_FRICTION
    )
    dynamic_friction = (
        spec.dynamic_friction
        if spec.dynamic_friction is not None
        else _PHYSX_DEFAULT_FRICTION
    )
    material = _add_physics_material(
        stage,
        f"{prim_path}/physicsMaterial",
        static_friction=static_friction,
        dynamic_friction=dynamic_friction,
        combine="average",
        restitution=spec.restitution if spec.restitution is not None else 0.0,
    )
    # Bound on the "physics" purpose and on the object root, so it resolves for
    # every collider beneath it -- the same shape as the trimesh ground's bind.
    binding_api = UsdShade.MaterialBindingAPI.Apply(prim)
    binding_api.Bind(
        UsdShade.Material(material),
        bindingStrength=UsdShade.Tokens.weakerThanDescendants,
        materialPurpose="physics",
    )

    log.info(
        f"  {prim_path}: {spec.kind} "
        + (f"mass={spec.mass}" if spec.mass is not None else f"density={spec.density}")
        + f" kinematic={bool(spec.fix_base_link)} "
        f"friction={static_friction}/{dynamic_friction} "
        f"restitution={spec.restitution if spec.restitution is not None else 0.0} "
        f"({len(colliders)} collider(s))"
    )


def _add_mesh_scene_object(stage, prim_path: str, spec, color) -> None:
    """Reference a mesh asset in and give it rigid-body + collider APIs.

    The primitive kinds get all of this from ``isaacsim.core.api.objects``;
    meshes have no such wrapper, so the same three things are done by hand:
    reference the asset, apply ``RigidBodyAPI`` to its root, and apply
    ``CollisionAPI`` to every geometry prim underneath (a referenced asset's
    meshes are descendants, not the root itself).

    Args:
        stage: The USD stage.
        prim_path: Where to reference the asset in.
        spec: The mesh :class:`~deployment.scene_utils.SceneObjectSpec`.
        color: RGB display colour applied to any gprim lacking its own.
    """
    from pxr import Gf, UsdGeom

    asset_path = spec.usd_path
    if not os.path.exists(asset_path):
        raise FileNotFoundError(
            f"Scene mesh asset not found: {asset_path}. Relative paths inside a "
            f"scenes file resolve against --scenes-asset-root (default: the "
            f"grandparent of the scenes file)."
        )
    add_reference_to_stage(usd_path=asset_path, prim_path=prim_path)
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Referencing {asset_path} at {prim_path} produced no prim")

    # XformCommonAPI rather than AddScaleOp: the referenced asset may already
    # carry xform ops, and this keeps the op order valid instead of appending a
    # second, possibly out-of-order, scale.
    UsdGeom.XformCommonAPI(prim).SetScale(Gf.Vec3f(*[float(s) for s in spec.scale]))

    UsdPhysics.RigidBodyAPI.Apply(prim)
    geom_count = 0
    for child in Usd.PrimRange(
        prim, Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)
    ):
        if not child.IsA(UsdGeom.Gprim):
            continue
        geom_count += 1
        UsdPhysics.CollisionAPI.Apply(child)
        gprim = UsdGeom.Gprim(child)
        if not gprim.GetDisplayColorAttr().IsAuthored():
            gprim.CreateDisplayColorAttr([Gf.Vec3f(*[float(c) for c in color])])
    if geom_count == 0:
        raise ValueError(
            f"{asset_path} contains no geometry prims under {prim_path}; nothing "
            f"to collide with."
        )


def add_scene_objects(
    specs, root_path: str = "/World/Scene", mesh_collision_approximation=None
) -> list:
    """Spawn a SceneLib scene's objects onto the stage as PhysX rigid bodies.

    The scene half of ``--scenes-file``: turns the descriptions from
    :func:`deployment.scene_utils.scene_object_specs` into prims. Follows the
    precedent set by :func:`add_protomotions_trimesh_ground` -- hand-authored USD
    rather than any ProtoMotions spawner, and no ProtoMotions import in this
    function at all, so a run without ``--scenes-file`` never touches the
    package.

    Objects are spawned at the origin; their real poses are written at every
    reset by :class:`SceneObjects`, exactly as IsaacLab spawns at the origin and
    positions in ``reset_envs``.

    **Must be called before ``world.reset()``** -- these are stage edits, and
    PhysX only reads the stage when the sim starts playing.

    Args:
        specs: List of :class:`~deployment.scene_utils.SceneObjectSpec`.
        root_path: Scope to author the objects under.
        mesh_collision_approximation: ``SceneLibConfig.mesh_collision_approximation``,
            forwarded to mesh colliders.

    Returns:
        The created prim paths, in spec order -- which is SceneLib's object
        order, so the caller can index pose tensors with it positionally.

    Raises:
        ValueError: On an unknown object kind.
    """
    from isaacsim.core.api.objects import DynamicCuboid, DynamicCylinder, DynamicSphere

    stage = get_current_stage()
    log.info(f"Scene: spawning {len(specs)} object(s) under {root_path}")

    prim_paths = []
    for index, spec in enumerate(specs):
        prim_path = f"{root_path}/Object_{index}"
        name = f"scene_object_{index}"
        color = np.asarray(
            spec.color
            if spec.color is not None
            else _OBJECT_COLOR_DEFAULTS.get(spec.kind, (0.5, 0.5, 0.5)),
            dtype=np.float32,
        )

        if spec.kind == "box":
            # size=1.0 + scale=(w, d, h) is how IsaacLab's CuboidCfg expresses a
            # box too: a unit cube carrying its dimensions in the xform scale.
            DynamicCuboid(
                prim_path=prim_path,
                name=name,
                size=1.0,
                scale=np.asarray(spec.size, dtype=np.float32),
                color=color,
            )
        elif spec.kind == "sphere":
            DynamicSphere(
                prim_path=prim_path, name=name, radius=float(spec.radius), color=color
            )
        elif spec.kind == "cylinder":
            DynamicCylinder(
                prim_path=prim_path,
                name=name,
                radius=float(spec.radius),
                height=float(spec.height),
                color=color,
            )
        elif spec.kind == "mesh":
            _add_mesh_scene_object(stage, prim_path, spec, color)
        else:
            raise ValueError(f"Unsupported scene object kind: {spec.kind}")

        _author_scene_object_physics(
            stage,
            prim_path,
            spec,
            mesh_collision_approximation=mesh_collision_approximation,
        )
        prim_paths.append(prim_path)

    return prim_paths


class SceneObjects:
    """Runtime handle on the spawned scene objects: reset them, read them back.

    Deliberately thin. Like IsaacLab, objects are written once per episode and
    then left entirely to physics -- there is no per-substep work here, because
    a driver that nudged the box every step would be measuring its own writes
    instead of the policy's manipulation.

    All reads go through the initialized :class:`isaacsim.core.prims.RigidPrim`
    view rather than a USD stage read. The module docstring's instance-proxy
    trap does not literally apply to these freshly authored prims, but the rule
    it justifies -- physics view, never USD -- does, and the failure mode is
    identical: a stage read returns the spawn pose forever, so a box being
    carried across the room would trace as a perfectly stationary object.
    """

    def __init__(
        self, scene_lib, prim_paths: list, specs=None, z_offset: float = 0.0
    ) -> None:
        """Wrap the spawned object prims.

        Args:
            scene_lib: Single-scene ``SceneLib`` from
                :func:`deployment.scene_utils.build_scene_lib`.
            prim_paths: Prim paths from :func:`add_scene_objects`, in SceneLib
                object order.
            specs: The :class:`~deployment.scene_utils.SceneObjectSpec` list the
                prims were built from, same order. Only ``fix_base_link`` is
                read, to tell kinematic bodies from dynamic ones -- see
                :meth:`reset`. ``None`` treats every object as dynamic, which is
                what every pre-chair scene was.
            z_offset: ``respawn_offset`` for ``get_scene_pose``
                (``--scene-object-z-offset``).
        """
        from isaacsim.core.prims import RigidPrim

        self.scene_lib = scene_lib
        self.prim_paths = list(prim_paths)
        self.z_offset = float(z_offset)
        self.num_objects = len(self.prim_paths)
        self.kinematic = (
            [bool(spec.fix_base_link) for spec in specs]
            if specs is not None
            else [False] * self.num_objects
        )
        self._dynamic_indices = [i for i, kin in enumerate(self.kinematic) if not kin]
        # Latched by reset(); applied by reference_pose() so the trace scores
        # against the same frame the objects were written in.
        self.root_offset = None
        # An explicit path list, not a regex: a regex view is ordered by string
        # match, which stops agreeing with SceneLib's object order at ten
        # objects ("Object_10" sorts before "Object_2").
        self.view = RigidPrim(prim_paths_expr=self.prim_paths, name="scene_objects")

    def initialize(self) -> None:
        """Create the physics view. Call after ``world.reset()``, like the robot's."""
        self.view.initialize()
        log.info(
            f"Scene objects initialized: {self.num_objects} rigid bodies "
            f"({', '.join(self.prim_paths)})"
        )

    def reference_pose(self, motion_time: float):
        """Reference object poses at ``motion_time``, in world coordinates.

        Includes :attr:`root_offset`, so this is directly comparable to
        :meth:`measured_pose` -- which is what the trace needs.

        Args:
            motion_time: Time into the clip, seconds. The object trajectory and
                the motion share a clock -- that is the whole meaning of
                ``Scene.humanoid_motion_id``.

        Returns:
            ``(pos [N, 3], quat_xyzw [N, 4])`` NumPy arrays.
        """
        import torch

        state = self.scene_lib.get_scene_pose(
            torch.tensor([0], dtype=torch.long),
            torch.tensor([float(motion_time)], dtype=torch.float),
            respawn_offset=self.z_offset,
        )
        pos = state.root_pos[0].cpu().numpy().astype(np.float32)
        quat_xyzw = state.root_rot[0].cpu().numpy().astype(np.float32)
        if self.root_offset is not None:
            pos = pos + self.root_offset
        return pos, quat_xyzw

    def reset(self, motion_time: float = 0.0, root_offset=None) -> None:
        """Teleport the objects onto their reference pose and zero their velocity.

        **Call only from outside the physics step**, for the same reason
        :meth:`TrackerPolicy.reset_episode` gives for the robot.

        Args:
            motion_time: Time into the clip to take the reference from.
            root_offset: Optional ``[3]`` world translation added to every
                object, carrying ``--init-state``'s ``respawn_root_offset``.
                ProtoMotions applies that same offset to robot *and* objects
                (``BaseEnv.move_reset_robot_obj_states_to_respawn_position``);
                without it the robot would spawn at IsaacLab's absolute position
                and the objects tens of metres away at the motion's own origin.
        """
        if root_offset is not None:
            self.root_offset = np.asarray(root_offset, dtype=np.float32).reshape(1, 3)
        pos, quat_xyzw = self.reference_pose(motion_time)

        # get/set_world_poses speak wxyz, unlike the articulation's
        # get_link_transforms() (already xyzw). Convert at the boundary.
        quat_wxyz = np.asarray(quat_xyzw)[:, [3, 0, 1, 2]].astype(np.float32)
        self.view.set_world_poses(
            positions=TrackerPolicy._for_view(self.view, pos.astype(np.float32)),
            orientations=TrackerPolicy._for_view(self.view, quat_wxyz),
        )
        # The physics write above is enough for a *dynamic* body: PhysX pushes
        # its pose back out to USD/Fabric every step, so the render transform
        # follows. A **kinematic** actor gets no such writeback -- PhysX assumes
        # it is driven *from* USD -- so the prim would keep rendering at its
        # spawn pose forever while colliding in the right place. Measured on a
        # kinematic/dynamic cuboid pair: the kinematic one read (-0.006, 0.099,
        # 0.381) from the physics view and (0, 0, 0) from both USD and Fabric.
        # Author the transform for those, which is also how a kinematic body is
        # meant to be positioned in the first place.
        self._write_kinematic_xforms(pos, quat_wxyz)

        # Velocities are a dynamic-body concept: PhysX rejects
        # setLinearVelocity/setAngularVelocity on a kinematic actor with an
        # error per call. Write only the bodies that can actually carry one.
        if self._dynamic_indices:
            self.view.set_velocities(
                TrackerPolicy._for_view(
                    self.view,
                    np.zeros((len(self._dynamic_indices), 6), dtype=np.float32),
                ),
                indices=TrackerPolicy._for_view(
                    self.view,
                    np.asarray(self._dynamic_indices, dtype=np.int32),
                    "int32",
                ),
            )
        log.info(
            f"    scene objects reset to t={motion_time:.3f}s, "
            f"pos={np.round(pos, 3).tolist()}"
        )

    def _write_kinematic_xforms(self, pos, quat_wxyz) -> None:
        """Author the USD transform of every kinematic object prim.

        The render-side half of :meth:`reset`. Writes the ops the spawner
        already authored (``DynamicCuboid`` and friends lay down
        ``translate``/``orient``/``scale``, in that order) rather than appending
        new ones, which would invalidate the op order.

        Args:
            pos: ``[N, 3]`` world positions, in prim order.
            quat_wxyz: ``[N, 4]`` world orientations in **wxyz** -- the same
                array handed to ``set_world_poses``, not the xyzw one
                ``get_link_transforms`` returns.
        """
        from pxr import Gf, UsdGeom

        stage = get_current_stage()
        for index, prim_path in enumerate(self.prim_paths):
            if not self.kinematic[index]:
                continue
            prim = stage.GetPrimAtPath(prim_path)
            translate = [float(v) for v in pos[index]]
            w, x, y, z = (float(v) for v in quat_wxyz[index])
            for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
                op_type = op.GetOpType()
                double = op.GetPrecision() == UsdGeom.XformOp.PrecisionDouble
                if op_type == UsdGeom.XformOp.TypeTranslate:
                    op.Set(Gf.Vec3d(*translate) if double else Gf.Vec3f(*translate))
                elif op_type == UsdGeom.XformOp.TypeOrient:
                    op.Set(
                        Gf.Quatd(w, Gf.Vec3d(x, y, z))
                        if double
                        else Gf.Quatf(w, Gf.Vec3f(x, y, z))
                    )

    def measured_pose(self):
        """Current object poses off the physics view.

        Returns:
            ``(pos [N, 3], quat_xyzw [N, 4])`` NumPy arrays.
        """
        pos, quat_wxyz = self.view.get_world_poses()
        pos = _to_numpy(pos).reshape(-1, 3)
        quat_wxyz = _to_numpy(quat_wxyz).reshape(-1, 4)
        return pos, wxyz_to_xyzw(quat_wxyz)


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


def _resolve_scenes_file(raw: str | None) -> str | None:
    """Normalise ``--scenes-file``, matching ``inference_agent.py``'s handling."""
    if raw is None:
        return None
    return None if raw.lower() in ("none", "null") else raw


def _reject_resync_with_dynamic_objects(specs, scenes_file: str) -> None:
    """Exit if ``--resync-state`` would leave a *movable* object off the resync.

    ``--resync-state`` rests on one claim: every control step starts from
    IsaacLab's exact recorded state. The action tape has no object channel, so
    anything in the scene that physics can move is outside that guarantee --
    resyncing the robot each step while the object keeps whatever pose it drifted
    to means step *k* no longer starts from IsaacLab's state, and the "N
    independent one-step experiments" reading is silently void.

    A ``fix_base_link`` object cannot drift: it is spawned kinematic, nothing
    writes it after ``SceneObjects.reset``, and PhysX never integrates it. For a
    scene made only of those -- the sitting chair is two kinematic boxes -- the
    missing channel carries no information and the probe stays interpretable, so
    the combination is allowed. That matters: the sit task is the one that needs
    the probe, and a blanket rejection put its sharpest instrument out of reach.

    Args:
        specs: :class:`~deployment.scene_utils.SceneObjectSpec` list for the scene.
        scenes_file: Path, for the error message.

    Raises:
        SystemExit: If any object is dynamic.
    """
    dynamic = [i for i, spec in enumerate(specs) if not spec.fix_base_link]
    if dynamic:
        raise SystemExit(
            f"--resync-state cannot be combined with {scenes_file}: object(s) "
            f"{dynamic} are dynamic (fix_base_link=False) and the action tape "
            "records no object state, so they would diverge from the recording "
            "while the robot is resynced to it, and the one-step divergence "
            "would no longer be interpretable. Scenes whose objects are all "
            "kinematic are allowed."
        )
    log.info(
        f"--resync-state with {len(specs)} kinematic scene object(s): allowed, "
        "they cannot drift from the recording."
    )


def main() -> None:
    if args.drive_probe and not args.drive_probe_out:
        raise SystemExit("--drive-probe requires --drive-probe-out")
    scenes_file = _resolve_scenes_file(args.scenes_file)
    if scenes_file is not None:
        # The --resync-state / --scenes-file check needs the object specs, so it
        # lives where they are built (see _reject_resync_with_dynamic_objects).
        if args.action_tape is not None:
            log.warning(
                "--action-tape with --scenes-file: the replay drives the robot "
                "open-loop but nothing controls the objects, so they only match "
                "the recording for as long as the robot's trajectory does. Valid "
                "as a diagnostic; not a parity measurement."
            )
    if args.resync_state:
        if args.action_tape is None:
            raise SystemExit("--resync-state requires --action-tape")
        if not args.control_in_loop:
            raise SystemExit(
                "--resync-state requires --control-in-loop: it writes articulation "
                "root and joint state, which is undefined from inside a physics "
                "callback (see TrackerPolicy.reset_episode)."
            )

    onnx_path = str(args.onnx)
    yaml_path = onnx_path.replace(".onnx", ".yaml")

    with open(yaml_path) as f:
        meta = yaml.safe_load(f)

    # The USD is resolved *after* the YAML is read so the contract can supply it.
    # Precedence: --usd > the YAML's robot.usd_path (written by
    # export_bm_tracker_onnx_isaacsim.py) > the robot config's own asset. Only
    # the last needs --robot, so a YAML from the current exporter is
    # self-describing and the driver has no per-robot default to get wrong.
    # Robot name: needed for armature, solver iterations and the physics rate,
    # none of which the YAML carries. Default it from the contract rather than
    # from a hardcoded "g1", which would silently apply G1 joint dynamics to a
    # SOMA run. YAMLs predating robot_name still work via --robot.
    robot_name = args.robot or meta.get("robot", {}).get("robot_name")
    if robot_name:
        source = "--robot" if args.robot else "YAML robot.robot_name"
        log.info(f"Robot config: {robot_name} (from {source})")
    else:
        log.warning(
            "No robot name: the YAML carries no robot.robot_name and --robot was "
            "not passed. Armature, velocity limits, solver iteration counts and "
            "the physics rate cannot be matched to training. Pass --robot, or "
            "re-export with deployment/export_bm_tracker_onnx_isaacsim.py."
        )

    usd_source = "--usd"
    usd_arg = args.usd
    if usd_arg is None:
        usd_arg = meta.get("robot", {}).get("usd_path")
        usd_source = "YAML robot.usd_path"
    if usd_arg is None and robot_name:
        usd_arg = getattr(
            build_robot_config(robot_name).asset, "usd_asset_file_name", None
        )
        usd_source = f"robot config '{robot_name}'"
    if usd_arg is None:
        raise SystemExit(
            "Cannot determine the robot USD. The YAML carries no robot.usd_path "
            "(re-export with deployment/export_bm_tracker_onnx_isaacsim.py), and "
            "neither --usd nor --robot was given."
        )
    usd_path = resolve_usd_path(usd_arg)

    log.info(f"ONNX: {onnx_path}")
    log.info(f"USD:  {usd_path}  (from {usd_source})")

    control_dt = meta["timing"]["control_dt"]

    # Timing. A YAML from export_bm_tracker_onnx.py carries the *MuJoCo* rate,
    # 1 kHz / decimation 20; one from export_bm_tracker_onnx_isaacsim.py already
    # carries the Isaac rate the policy trained at (g1 200 Hz / 4, soma23
    # 120 Hz / 4). The robot config is authoritative for both, so prefer it;
    # --physics-dt overrides everything.
    robot_props = (
        load_robot_joint_properties(robot_name, meta["robot"]["joint_names"])
        if robot_name
        else {}
    )
    sim_fps = robot_props.get("sim_fps")
    if args.physics_dt:
        physics_dt = args.physics_dt
        log.info(f"physics_dt={physics_dt}s (from --physics-dt)")
    elif sim_fps:
        physics_dt = 1.0 / float(sim_fps)
        yaml_dt = meta["timing"]["physics_dt"]
        agrees = abs(yaml_dt - physics_dt) < 1e-9
        log.info(
            f"physics_dt={physics_dt}s (from the robot config's IsaacLab rate, "
            f"{sim_fps} Hz); the YAML says {yaml_dt}s"
            + (" -- they agree." if agrees else " (an older MuJoCo-rate YAML).")
        )
    else:
        physics_dt = meta["timing"]["physics_dt"]
        log.warning(
            f"physics_dt={physics_dt}s taken from the YAML. Without a robot "
            "config this cannot be checked against training's Isaac rate, and a "
            "YAML from the MuJoCo exporter would put it 20x too fine. "
            "Pass --robot."
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
    _author_physics_scene_iterations(world, robot_props)

    if args.ground == "trimesh":
        if not args.resolved_configs:
            raise SystemExit(
                "--ground trimesh needs --resolved-configs pointing at the run's "
                "resolved_configs_inference.pt (that is where the terrain config lives)"
            )
        # Centre the mesh on wherever the robot will actually spawn -- with
        # --init-state that is IsaacLab's absolute position, tens of metres out.
        center_xy = None
        if args.init_state:
            with open(args.init_state) as f:
                center_xy = json.load(f)["root_pos"][:2]
        add_protomotions_trimesh_ground(
            args.resolved_configs, args.ground_friction, center_xy=center_xy
        )
    else:
        world.scene.add_default_ground_plane(
            z_position=0.0,
            name="ground_plane",
            prim_path="/World/GroundPlane",
            static_friction=args.ground_friction,
            dynamic_friction=args.ground_friction,
            restitution=0.0,
        )

    # Scene objects, before the robot: a bad scenes file should fail here rather
    # than after Kit has spent time referencing and authoring the robot asset.
    # Also before world.reset(), because PhysX only reads the stage on play.
    scene_objects = None
    if scenes_file is not None:
        scene_index = scene_utils.resolve_scene_index(
            scenes_file, args.motion_index, explicit=args.scene_index
        )
        scene_lib = scene_utils.build_scene_lib(
            scenes_file, scene_index, asset_root=args.scenes_asset_root
        )
        specs = scene_utils.scene_object_specs(scene_lib)
        if args.resync_state:
            _reject_resync_with_dynamic_objects(specs, scenes_file)
        if not specs:
            # A scene with no objects is a valid SceneLib state and nothing to
            # spawn; an empty RigidPrim view would fail on initialize() instead.
            log.warning(
                f"Scene {scene_index} of {scenes_file} holds no objects; running "
                "without scene objects."
            )
        else:
            prim_paths = add_scene_objects(
                specs,
                mesh_collision_approximation=scene_lib.config.mesh_collision_approximation,
            )
            scene_objects = SceneObjects(
                scene_lib,
                prim_paths,
                specs=specs,
                z_offset=args.scene_object_z_offset,
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
        robot_name=robot_name or None,
        physics_dt=physics_dt,
        joint_friction_mode=args.joint_friction,
        action_tape=args.action_tape,
        init_state=args.init_state,
        init_z_offset=scene_utils.resolve_init_z_offset(
            args.init_z_offset, args.resolved_configs
        ),
        author_collider_offsets=args.author_collider_offsets,
        resync_state=args.resync_state,
        control_in_loop=args.control_in_loop,
        self_collisions=args.self_collisions,
        scene_objects=scene_objects,
        profile=args.profile,
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
    if args.drive_probe:
        policy.run_drive_probe(world, args.drive_probe, args.drive_probe_out)
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

    substep_idx = 0
    while simulation_app.is_running() and not policy.done:
        t0 = time.perf_counter()
        if args.control_in_loop:
            policy.control_step(world, render=not args.headless)
        else:
            # render=False always -- see _render(). The callback path drives one
            # substep per iteration, so draw every `decimation`-th one to keep the
            # frame rate at the control rate, as control_step does.
            world.step(render=False)
            substep_idx += 1
            if not args.headless and substep_idx % policy.decimation == 0:
                _render(world)
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
