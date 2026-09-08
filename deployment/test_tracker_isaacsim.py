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
"""Standalone Isaac Sim CLI for tracker ONNX policies.

A thin wrapper: parse flags, boot a :class:`SimulationApp`, hand the flags to
:func:`deployment.tracker_runtime.build_runtime`, and own the stepping loop.

**All the driver logic, and the deployment-contract reference documentation,
live in** ``deployment/tracker_runtime.py`` -- read that module's docstring
first. It is importable without booting an app, which is what lets the same
driver run inside a live Isaac Sim via ``deployment/exts/protomotions.tracker``
(see ``deployment/tracker_session.py``).

What stays here, and only here:

- the 40-odd CLI flags and their help text (the user-facing contract);
- ``SimulationApp`` creation and the renderer/headless decisions, which in an
  extension the host app has already made;
- ``logging.basicConfig(force=True)``, which reconfigures the *root* logger and
  so must never run inside Kit;
- the diagnostic modes that own their own stepping: ``--control-in-loop``,
  ``--action-tape``, ``--resync-state``, ``--drive-probe``,
  ``--dump-physx-properties``, ``--trace-out``, ``--onnx-inputs-out``,
  ``--profile``;
- the ``while`` loop and its real-time pacing. Inside Kit the timeline drives
  physics instead.

Usage::

    python deployment/test_tracker_isaacsim.py --robot g1 \
        --onnx <export>/unified_pipeline.onnx \
        --motion data/motion_for_trackers/g1_bones_seed_mini.pt
"""

from __future__ import annotations

import argparse
import logging
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


# Safe now that the app is up: tracker_runtime imports isaacsim.* and pxr at
# module scope.
from deployment.tracker_runtime import (  # noqa: E402
    TrackerConfigError,
    TrackerRuntimeConfig,
    build_runtime,
)

# `force=True` is load-bearing: SimulationApp has already installed Kit's
# `_CarbLogHandler` on the root logger by this point, and a plain basicConfig()
# is a documented no-op once the root logger has any handler. Without the
# override every line below silently goes to Kit's log file instead of stderr.
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s", force=True)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _validate_flags(parsed) -> None:
    """Reject flag combinations that cannot work, before anything is built.

    The ``--resync-state`` / ``--scenes-file`` check is not here: it needs the
    scene object specs, so it lives where those are built (see
    ``tracker_runtime._reject_resync_with_dynamic_objects``).
    """
    if parsed.drive_probe and not parsed.drive_probe_out:
        raise SystemExit("--drive-probe requires --drive-probe-out")
    if parsed.tape_divergence_out and parsed.action_tape is None:
        raise SystemExit("--tape-divergence-out requires --action-tape")
    if parsed.resync_state:
        if parsed.action_tape is None:
            raise SystemExit("--resync-state requires --action-tape")
        if not parsed.control_in_loop:
            raise SystemExit(
                "--resync-state requires --control-in-loop: it writes articulation "
                "root and joint state, which is undefined from inside a physics "
                "callback (see TrackerPolicy.reset_episode)."
            )
    if parsed.scenes_file and parsed.action_tape is not None:
        log.warning(
            "--action-tape with --scenes-file: the replay drives the robot "
            "open-loop but nothing controls the objects, so they only match "
            "the recording for as long as the robot's trajectory does. Valid "
            "as a diagnostic; not a parity measurement."
        )


def _runtime_config(parsed) -> TrackerRuntimeConfig:
    """Map the parsed flags onto a :class:`TrackerRuntimeConfig`.

    Only the CLI-only defaults are resolved here; every other field maps across
    by name.
    """
    cfg = TrackerRuntimeConfig.from_namespace(parsed)
    # A headless run is a measurement, so it plays the clip once unless told
    # otherwise; a windowed run loops until the user quits.
    cfg.loops = (
        parsed.loops
        if parsed.loops is not None
        else (1 if parsed.headless else 10_000_000)
    )
    # This script owns the whole stage, so it lights and frames it. An extension
    # runs inside someone else's stage and does neither by default.
    cfg.setup_lighting = not parsed.headless
    cfg.move_camera = not parsed.headless
    return cfg


def main() -> None:
    _validate_flags(args)
    runtime = build_runtime(_runtime_config(args))
    policy = runtime.policy

    # Output channels are opt-in: a None list disables recording entirely.
    if args.trace_out:
        policy.trace = []
    if args.onnx_inputs_out:
        policy.onnx_input_log = []
    if args.tape_divergence_out:
        policy.divergence = []

    runtime.initialize()

    # Diagnostics that replace the run rather than observe it.
    if args.dump_physx_properties:
        policy.dump_physx_properties()
        simulation_app.close()
        return
    if args.drive_probe:
        policy.run_drive_probe(runtime.world, args.drive_probe, args.drive_probe_out)
        simulation_app.close()
        return

    if not args.control_in_loop:
        runtime.add_physics_callback()

    # Real-time pacing only makes sense when there is something to watch; a
    # headless run is a measurement, so it always goes as fast as it can.
    realtime = not args.no_realtime and not args.headless
    while simulation_app.is_running() and not runtime.done:
        step_s = runtime.step_once(render=not args.headless)
        if realtime:
            sleep_time = runtime.loop_period - step_s
            if sleep_time > 0:
                time.sleep(sleep_time)

    policy.log_summary(runtime.total_sim_ms)
    if args.onnx_inputs_out:
        policy.write_onnx_inputs(args.onnx_inputs_out)
    if args.trace_out:
        policy.write_trace(args.trace_out)
    if args.tape_divergence_out:
        policy.write_divergence(args.tape_divergence_out)

    runtime.teardown()
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except TrackerConfigError as e:
        # Same shell behaviour as the SystemExit these used to be; the runtime
        # raises a plain exception so the code is also usable from a Kit callback.
        raise SystemExit(str(e))
