# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create a scenes file containing a static chair, for use with ``--scenes-file``.

The chair is two pinned boxes in one scene -- a seat slab and a reclined
backrest -- so it needs no mesh asset: IsaacLab spawns ``CuboidCfg``s and
IsaacGym generates URDFs. Both carry ``fix_base_link=True``, which is the whole
point: a chair that can be shoved is not a chair.

This is the mirror image of ``create_box_scene.py``. There, a single-frame object
is a bug (object-tracking rewards pay it for not moving). Here it is the
requirement, and the consequence is that the sitting experiment must carry *no*
object rewards at all -- see ``examples/experiments/mimic/soma_sit_chair.py``.

By default every dimension is fitted to the motion the chair is paired with:

    # Fit a chair under motion 0 of a seated clip.
    python data/scripts/create_chair_scene.py --output soma_sit_chair.pt \
        --motion-file soma_sit_motion.pt --motion-id 0

    # One scene per motion in the file.
    python data/scripts/create_chair_scene.py --output chairs.pt \
        --motion-file soma_sit_motions.pt --all-motions

Then check what you built before spending a training run on it:

    python data/scripts/create_chair_scene.py --inspect soma_sit_chair.pt \
        --motion-file soma_sit_motion.pt

The fit that matters is the seat height. It is taken from the *collision*
geometry, not the body origins: on SOMA the thighs are capsules of radius 0.06
and the pelvis a sphere of radius 0.08, and the thigh undersides sit 4-6 cm
below the pelvis underside -- so the thighs, not the buttocks, carry the weight.
``seat_top`` is the lowest of those surfaces over the whole clip, which puts the
reference pose exactly at rest on the seat.

THE BACKREST IS FITTED THE SAME WAY, AND FOR THE SAME REASON. Left at a fixed
recline and a fixed distance behind the hips it is decorative: on the SOMA sit
clip the shipped 12 deg / 0.18 m backrest leaves ``Spine2`` 10.1 cm and ``Chest``
8.0 cm short of the front face, so the back never reaches it. Labelling back
contact against an unreachable backrest makes ``contact_match_rew`` a permanent
penalty with no achievable fix. So ``--back-angle`` and ``--back-offset`` both
default to a fit against the clip:

* the recline is the common tangent to the mean ``Spine2`` and ``Chest``
  collision spheres, so both bodies touch rather than only the larger one;
* the offset then slides that plane back until it is tangent to the rearmost
  torso surface over the whole clip.

Fitting the recline is decisive, not a refinement: at a fixed 12 deg, fitting
only the offset reaches the ``Chest`` and leaves ``Spine2`` labelled 0/150, which
makes the reward *penalise* the one body it was added for.

Tilting the backrest to touch both back bodies swings its lower half forward, so
the fit is clamped by a pelvis check: if the plane comes within
``BACK_MIN_BODY_CLEARANCE`` of the ``Hips`` or ``Spine1`` spheres the recline is
bisected back towards vertical. On an upright clip the raw tangent would pass
straight through the pelvis sphere, which at reset would shove the character off
its reference and trip the tracking-error termination.

One consequence worth knowing: because the plane is tangent to the *rearmost*
frame, the typical frame sits ~1.5 cm off it. A policy tracking the reference
perfectly therefore registers no simulated contact on most frames while the label
says 1, so the term nudges the torso ~1.5 cm further back than the reference.
That is intended -- it is what makes the back load the chair -- and it is far
inside what ``gt_coef`` notices, but it will show as a small steady ``gt_error``.

Facing is derived from the hips->knees vector, not from the root quaternion:
SOMA's rest orientation makes the root yaw read +28.8 deg on one seated clip and
-178.6 deg on another that faces the same way.

Chair poses live in the same world frame as the motion -- the packaged ``gts``
body positions -- and both are shifted together by the respawn offsets at reset.
"""

import argparse
import math
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import torch

from protomotions.components.scene_lib import (
    BoxSceneObject,
    ObjectOptions,
    Scene,
    SceneLib,
)
from protomotions.utils.rotations import quat_rotate

# SOMA seated support chain. The thigh is the capsule between <side>Leg (hip
# joint) and <side>Shin (knee); the pelvis is the sphere at Hips.
DEFAULT_PELVIS_BODY = "Hips"
DEFAULT_THIGH_BODIES = [("LeftLeg", "LeftShin"), ("RightLeg", "RightShin")]

# Collision radii from protomotions/data/assets/mjcf/soma23_humanoid.xml.
DEFAULT_THIGH_RADIUS = 0.06
DEFAULT_PELVIS_RADIUS = 0.08

# Samples along each thigh capsule axis when looking for its lowest point.
CAPSULE_SAMPLES = 21

# Bodies the backrest is fitted to touch. Spine2 + Chest cover the
# mid-back-to-shoulder-blade band a real backrest rests against. Spine1 is
# deliberately absent: at r=0.06 it sits too far forward for a backrest to reach
# without driving the plane into the pelvis, and a label the simulator never
# registers is a permanent penalty rather than a target.
DEFAULT_BACK_BODIES = ["Spine2", "Chest"]

# Bodies the backrest must stay clear of. Reclining the plane to touch the back
# swings its lower half forward, straight towards these two.
DEFAULT_BACK_CLEARANCE_BODIES = [DEFAULT_PELVIS_BODY, "Spine1"]

# Torso collision spheres from the MJCF, as (radius, local offset). SOMA's local
# axes are +x left, +z up and -y FORWARD, so every one of these bulges forward of
# its joint -- which is why the offsets must be rotated by the body's world
# orientation rather than added to the body origin.
DEFAULT_BACK_SPHERES = {
    "Spine2": (0.07, (0.0, -0.04, 0.06)),
    "Chest": (0.11, (0.0, -0.04, 0.12)),
    DEFAULT_PELVIS_BODY: (DEFAULT_PELVIS_RADIUS, (0.0, -0.03, 0.00)),
    "Spine1": (0.06, (0.0, -0.03, 0.05)),
}

# Recline used when the backrest cannot be fitted (no motion, --no-back-contacts,
# or a degenerate tangent). This is the value the script shipped with.
BACK_ANGLE_FALLBACK = 12.0

# A fitted recline outside this range is not a chair.
BACK_ANGLE_MIN = 0.0
BACK_ANGLE_MAX = 35.0

# Closest the backrest plane may come to the pelvis/lumbar spheres before the
# recline is bisected back towards vertical.
BACK_MIN_BODY_CLEARANCE = 0.005

# A back body further than this from the plane is not "slightly off", it is a
# body the backrest never reaches -- worth a louder warning than a tolerance miss.
BACK_NEVER_RESTS_GAP = 0.10


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a scenes file with a static two-box chair fitted to a motion"
    )
    parser.add_argument(
        "--inspect",
        type=str,
        default=None,
        metavar="SCENES_PT",
        help="Report on an existing scenes file instead of creating one.",
    )
    parser.add_argument("--output", type=str, help="Output .pt path")
    parser.add_argument(
        "--motion-file",
        type=str,
        default=None,
        help="Packaged MotionLib .pt to fit the chair to, and to cross-check "
        "against under --inspect.",
    )
    parser.add_argument(
        "--motion-id",
        type=int,
        action="append",
        default=None,
        help="Motion index this scene is paired with; repeat for one scene per "
        "motion. Pairing is what binds the chair to the right clip. Default: 0.",
    )
    parser.add_argument(
        "--all-motions",
        action="store_true",
        help="Emit one fitted scene per motion in --motion-file.",
    )
    parser.add_argument(
        "--robot-name",
        type=str,
        default="soma23",
        help="Robot whose body names the support bodies refer to.",
    )

    geometry = parser.add_argument_group(
        "chair geometry (all default to the motion fit)"
    )
    geometry.add_argument(
        "--seat-top",
        type=float,
        default=None,
        help="Height (m) of the seat's top surface. Default: fitted to the lowest "
        "thigh/pelvis collision surface in the clip.",
    )
    geometry.add_argument(
        "--seat-clearance",
        type=float,
        default=0.0,
        help="Lower the fitted seat by this much (m). Positive drops the seat, "
        "leaving the reference pose hovering; negative pushes it up into the "
        "thighs. Ignored when --seat-top is given.",
    )
    geometry.add_argument(
        "--seat-width", type=float, default=0.60, help="Seat size across the body (m)"
    )
    geometry.add_argument(
        "--seat-depth", type=float, default=0.55, help="Seat size front-to-back (m)"
    )
    geometry.add_argument(
        "--seat-thickness", type=float, default=0.06, help="Seat slab thickness (m)"
    )
    geometry.add_argument(
        "--seat-back-offset",
        type=float,
        default=0.18,
        help="How much seat is behind the hips (m). The backrest is placed by "
        "--back-offset, which is fitted separately: the value that puts the "
        "backrest against the back is far too small to leave the pelvis "
        "supported, so one number cannot drive both.",
    )
    geometry.add_argument(
        "--back-width",
        type=float,
        default=0.60,
        help="Backrest size across the body (m)",
    )
    geometry.add_argument(
        "--back-height",
        type=float,
        default=0.45,
        help="Backrest size up from the seat (m)",
    )
    geometry.add_argument(
        "--back-thickness", type=float, default=0.06, help="Backrest thickness (m)"
    )
    geometry.add_argument(
        "--back-angle",
        type=float,
        default=None,
        help="Backrest recline from vertical (degrees); positive tips the top "
        f"backwards. Default: fitted as the common tangent to the {' and '.join(DEFAULT_BACK_BODIES)} "
        f"spheres, clamped to [{BACK_ANGLE_MIN:.0f}, {BACK_ANGLE_MAX:.0f}] and "
        "reduced further if it would reach the pelvis.",
    )
    geometry.add_argument(
        "--back-offset",
        type=float,
        default=None,
        help="How far behind the hips the backrest's front face sits (m). "
        "Default: fitted so the face is tangent to the rearmost torso surface in "
        "the clip. This is NOT --seat-back-offset: the fitted value (~0.035 on "
        "the SOMA sit clip) would leave the pelvis hanging off the back of the "
        "seat if it drove the seat too.",
    )
    geometry.add_argument(
        "--back-clearance",
        type=float,
        default=0.0,
        help="Move the fitted backrest back by this much (m). Positive leaves the "
        "reference back hovering in front of it; negative pushes it into the "
        "torso. The mirror of --seat-clearance. Ignored when --back-offset is given.",
    )
    geometry.add_argument(
        "--yaw",
        type=float,
        default=None,
        help="Chair heading (degrees). Default: fitted from the hips->knees vector.",
    )
    geometry.add_argument(
        "--friction",
        type=float,
        default=1.0,
        help="Chair static/dynamic friction. Robot friction randomization does not "
        "touch scene objects, so how much the robot slides on the seat rides on this.",
    )

    fit = parser.add_argument_group("support-body fit")
    fit.add_argument(
        "--thigh-radius",
        type=float,
        default=DEFAULT_THIGH_RADIUS,
        help="Thigh capsule collision radius (m).",
    )
    fit.add_argument(
        "--pelvis-radius",
        type=float,
        default=DEFAULT_PELVIS_RADIUS,
        help="Pelvis sphere collision radius (m).",
    )

    relabel = parser.add_argument_group("reference contact relabelling")
    relabel.add_argument(
        "--relabel-contacts",
        type=str,
        default=None,
        metavar="OUT_PT",
        help="Write a copy of --motion-file with the thigh bodies marked as in "
        "contact wherever they rest on the fitted seat, and the back bodies "
        "wherever they rest on the fitted backrest. Needed before adding either "
        "to contact_bodies: the packaged labels come from ground-plane contact "
        "detection and mark only the feet, so contact_match_rew would otherwise "
        "penalise sitting. Never overwrites the input.",
    )
    relabel.add_argument(
        "--contact-tol",
        type=float,
        default=0.02,
        help="Thigh surface must be within this distance (m) of the seat top to "
        "count as resting on it.",
    )
    relabel.add_argument(
        "--back-contact-tol",
        type=float,
        default=0.03,
        help="Back surface must be within this distance (m) of the backrest face "
        "to count as resting on it. Looser than --contact-tol on purpose: the "
        "seat pins the thighs, but the torso sways a couple of centimetres over "
        "the clip, and a tighter value flickers the label on and off for no "
        "reason the policy can act on.",
    )
    relabel.add_argument(
        "--no-back-contacts",
        action="store_true",
        help="Skip the backrest fit and the back relabelling entirely, restoring "
        "the pre-backrest behaviour: a decorative backrest at "
        f"{BACK_ANGLE_FALLBACK:.0f} deg and --seat-back-offset behind the hips. "
        "Pair with dropping "
        f"{'/'.join(DEFAULT_BACK_BODIES)} from contact_bodies.",
    )

    args = parser.parse_args()
    if args.inspect is None and not args.output:
        parser.error("--output is required unless --inspect is given")
    if args.inspect is None and not args.motion_file and args.seat_top is None:
        parser.error("--motion-file is required unless --seat-top is given")
    if args.all_motions and args.motion_id:
        parser.error("--all-motions and --motion-id are mutually exclusive")
    if args.relabel_contacts and not args.motion_file:
        parser.error("--relabel-contacts requires --motion-file")
    return args


def load_packaged_motion(motion_file: str, motion_id: int):
    """Slice one motion out of a packaged MotionLib .pt.

    Forward kinematics is already baked in (``gts``/``grs`` hold per-body world
    poses), so this is a pure lookup -- same access pattern as
    ``create_box_scene.py``.

    ``grs`` is needed as well as ``gts`` because the torso collision spheres are
    offset forward of their body origins: placing them takes the body's world
    rotation, not just its position.

    Args:
        motion_file: Path to the packaged MotionLib .pt.
        motion_id: Index of the motion to slice.

    Returns:
        Tuple of (body_pos [frames, bodies, 3], body_rot [frames, bodies, 4]
        XYZW, dt, num_motions).
    """
    data = torch.load(motion_file, map_location="cpu", weights_only=False)

    num_motions = data["motion_num_frames"].shape[0]
    if not 0 <= motion_id < num_motions:
        raise ValueError(
            f"--motion-id {motion_id} out of range; file has {num_motions} motion(s)"
        )

    start = int(data["length_starts"][motion_id])
    num_frames = int(data["motion_num_frames"][motion_id])
    dt = float(data["motion_dt"][motion_id])
    frames = slice(start, start + num_frames)

    return data["gts"][frames], data["grs"][frames], dt, num_motions


def resolve_support_indices(robot_name: str):
    """Map the seated support bodies to indices on the robot.

    Args:
        robot_name: Robot config name.

    Returns:
        Tuple of (pelvis index, list of (hip index, knee index) thigh pairs,
        {torso sphere body name: index} covering the back and clearance bodies).
    """
    from protomotions.robot_configs.factory import robot_config

    body_names = robot_config(robot_name).kinematic_info.body_names

    needed = [DEFAULT_PELVIS_BODY] + [b for pair in DEFAULT_THIGH_BODIES for b in pair]
    needed += list(DEFAULT_BACK_SPHERES)
    missing = [b for b in needed if b not in body_names]
    if missing:
        raise ValueError(
            f"Seated support bodies not found on robot '{robot_name}': {missing}.\n"
            f"This script is shaped for SOMA. Available: {body_names}"
        )

    pelvis_index = body_names.index(DEFAULT_PELVIS_BODY)
    thigh_indices = [
        (body_names.index(hip), body_names.index(knee))
        for hip, knee in DEFAULT_THIGH_BODIES
    ]
    sphere_indices = {name: body_names.index(name) for name in DEFAULT_BACK_SPHERES}
    return pelvis_index, thigh_indices, sphere_indices


def thigh_surface_heights(
    body_pos: torch.Tensor,
    thigh_indices: Sequence[Tuple[int, int]],
    thigh_radius: float,
) -> torch.Tensor:
    """Lowest point of each thigh capsule, per frame.

    The capsule axis runs from the hip body origin to the knee body origin, so
    sampling that segment and subtracting the radius gives the underside without
    needing the body rotations.

    Args:
        body_pos: Per-body world positions [frames, bodies, 3].
        thigh_indices: (hip index, knee index) pairs.
        thigh_radius: Capsule collision radius (m).

    Returns:
        Underside heights [frames, num_thighs].
    """
    weights = torch.linspace(0.0, 1.0, CAPSULE_SAMPLES).view(-1, 1)
    per_thigh = []
    for hip_idx, knee_idx in thigh_indices:
        hip_z = body_pos[:, hip_idx, 2].unsqueeze(0)
        knee_z = body_pos[:, knee_idx, 2].unsqueeze(0)
        axis_z = hip_z * (1.0 - weights) + knee_z * weights
        per_thigh.append(axis_z.min(dim=0).values - thigh_radius)
    return torch.stack(per_thigh, dim=1)


def fit_chair_to_motion(
    body_pos: torch.Tensor,
    pelvis_index: int,
    thigh_indices: Sequence[Tuple[int, int]],
    thigh_radius: float,
    pelvis_radius: float,
):
    """Derive seat height, heading and hip location from a seated clip.

    Args:
        body_pos: Per-body world positions [frames, bodies, 3].
        pelvis_index: Index of the pelvis body.
        thigh_indices: (hip index, knee index) pairs.
        thigh_radius: Thigh capsule collision radius (m).
        pelvis_radius: Pelvis sphere collision radius (m).

    Returns:
        Tuple of (seat_top, yaw_radians, hips_xy [2], thigh_surface [frames, thighs]).
    """
    thigh_surface = thigh_surface_heights(body_pos, thigh_indices, thigh_radius)
    pelvis_surface = body_pos[:, pelvis_index, 2] - pelvis_radius
    seat_top = float(min(thigh_surface.min(), pelvis_surface.min()))

    hips_xy = body_pos[:, pelvis_index, :2].mean(dim=0)
    knee_xy = (
        torch.stack([body_pos[:, knee_idx, :2] for _, knee_idx in thigh_indices])
        .mean(dim=0)
        .mean(dim=0)
    )

    forward = knee_xy - hips_xy
    if float(forward.norm()) < 1e-6:
        raise ValueError(
            "Hips and knees are vertically aligned -- cannot infer a facing "
            "direction. Pass --yaw explicitly."
        )
    forward = forward / forward.norm()

    # Canonical chair faces +y, so the yaw taking (0, 1) to `forward` is
    # atan2(-fx, fy).
    yaw = math.atan2(-float(forward[0]), float(forward[1]))
    return seat_top, yaw, hips_xy, thigh_surface


def yaw_rotate(vec: Sequence[float], yaw: float) -> Tuple[float, float, float]:
    """Rotate a vector about the world z axis.

    Args:
        vec: (x, y, z) in the canonical chair frame.
        yaw: Rotation angle (radians).

    Returns:
        Rotated (x, y, z).
    """
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    return (
        vec[0] * cos_y - vec[1] * sin_y,
        vec[0] * sin_y + vec[1] * cos_y,
        vec[2],
    )


def quat_multiply(lhs: Sequence[float], rhs: Sequence[float]) -> Tuple[float, ...]:
    """Hamilton product of two XYZW quaternions.

    Args:
        lhs: Left quaternion (applied second).
        rhs: Right quaternion (applied first).

    Returns:
        Product quaternion, XYZW.
    """
    lx, ly, lz, lw = lhs
    rx, ry, rz, rw = rhs
    return (
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    )


class BackrestFit(NamedTuple):
    """The fitted backrest, and everything needed to argue it is right.

    Attributes:
        angle: Recline from vertical (radians).
        offset: Distance behind the hips of the front face (m).
        plane_const: Chair-frame constant of the front face plane.
        gaps: Back body name -> per-frame gap to the front face (m).
        clearances: Clearance body name -> per-frame gap to the front face (m).
        warnings: Human-readable problems with the fit, empty when it is clean.
    """

    angle: float
    offset: float
    plane_const: float
    gaps: Dict[str, torch.Tensor]
    clearances: Dict[str, torch.Tensor]
    warnings: List[str]


class BackrestPlane(NamedTuple):
    """The backrest's front face, read back off the box that was actually built.

    Deriving this from the ``BoxSceneObject`` rather than from a second run of
    the fit is the point: relabelling and ``--inspect`` then describe the
    geometry that got saved, not the geometry that was intended.

    Attributes:
        normal: World unit normal pointing out of the front face.
        const: Face is the plane ``p . normal == const``.
        bottom_centre: World centre of the slab's bottom face.
        up: World direction of the slab's local +z, up its height.
        lateral: World direction of the slab's local +x, across its width.
        height: Slab height (m).
        half_width: Half the slab width (m).
    """

    normal: torch.Tensor
    const: float
    bottom_centre: torch.Tensor
    up: torch.Tensor
    lateral: torch.Tensor
    height: float
    half_width: float


def sphere_centres_world(
    body_pos: torch.Tensor,
    body_rot: torch.Tensor,
    index: int,
    local_offset: Sequence[float],
) -> torch.Tensor:
    """World centre of one body-local collision sphere, per frame.

    The torso spheres are offset forward of their body origins, so the offset has
    to be carried by the body's world rotation -- adding it to the origin would
    put the sphere in front of the chest on a clip that leans back.

    Args:
        body_pos: Per-body world positions [frames, bodies, 3].
        body_rot: Per-body world rotations [frames, bodies, 4], XYZW.
        index: Body index.
        local_offset: Sphere centre in the body's local frame.

    Returns:
        Sphere centres [frames, 3].
    """
    offset = torch.as_tensor(local_offset, dtype=torch.float32).expand(
        body_pos.shape[0], 3
    )
    rotation = body_rot[:, index, :].to(torch.float32)
    position = body_pos[:, index, :].to(torch.float32)
    return position + quat_rotate(rotation, offset, w_last=True)


def chair_frame(centres_w: torch.Tensor, yaw: float, hips_xy) -> torch.Tensor:
    """Express world points in the canonical chair frame.

    The chair frame is the one ``build_chair_objects`` builds in before applying
    the heading: x is lateral, y is forward of the hips, z is world height. The
    backrest fit is closed-form in this frame and merely ugly in world.

    Args:
        centres_w: World points [frames, 3].
        yaw: Chair heading (radians).
        hips_xy: Mean hip position in world xy.

    Returns:
        Chair-frame points [frames, 3].
    """
    forward = torch.tensor([-math.sin(yaw), math.cos(yaw)])
    lateral = torch.tensor([math.cos(yaw), math.sin(yaw)])
    origin = torch.as_tensor(hips_xy, dtype=torch.float32).reshape(2)
    rel = centres_w[:, :2].to(torch.float32) - origin
    return torch.stack([rel @ lateral, rel @ forward, centres_w[:, 2]], dim=1)


def _plane_const(
    offset: float, angle: float, thickness: float, seat_top: float
) -> float:
    """Chair-frame constant of the backrest's front face.

    The face is ``{p : p . n == C}`` with ``n = (0, cos a, sin a)``. This is the
    same arithmetic ``build_chair_objects`` does, collapsed: the half-height
    terms cancel, which is why the recline can be fitted without ever knowing
    ``--back-height``.
    """
    half_t = thickness / 2.0
    return (
        -offset * math.cos(angle)
        - half_t * math.cos(angle)
        + half_t
        + seat_top * math.sin(angle)
    )


def _offset_for_plane_const(
    plane_const: float, angle: float, thickness: float, seat_top: float
) -> float:
    """Invert :func:`_plane_const` for the offset that places a given face."""
    half_t = thickness / 2.0
    return (
        -plane_const + half_t - half_t * math.cos(angle) + seat_top * math.sin(angle)
    ) / math.cos(angle)


def _fit_recline(centres: Dict[str, Tuple[torch.Tensor, float]]):
    """Recline whose plane is the common tangent to the two mean back spheres.

    Fitting to the tangent rather than to one body is what gets both bodies onto
    the backrest: a plane fitted to the ``Chest`` alone (the larger sphere) sits
    the ``Spine2`` a centimetre clear of it, and an unreachable label is a
    permanent penalty.

    With ``d`` the vector from the upper to the lower sphere centre, ``dr`` their
    radius difference and ``psi = atan2(dz, dy)``, tangency is
    ``cos(a - psi) == dr / |d|``, so ``a = psi +- acos(dr / |d|)``. The lower
    sphere is always below the upper one, so ``psi`` is negative and only the
    ``+`` root can come out facing forwards -- no case analysis, just a check.

    Args:
        centres: Body name -> (chair-frame centres [frames, 3], radius).

    Returns:
        Tuple of (recline in radians or None if degenerate, warning or None).
    """
    if len(DEFAULT_BACK_BODIES) != 2:
        return None, (
            f"the common-tangent fit needs exactly 2 back bodies, got "
            f"{len(DEFAULT_BACK_BODIES)}"
        )

    lower, upper = sorted(
        DEFAULT_BACK_BODIES, key=lambda name: float(centres[name][0][:, 2].mean())
    )
    mean_lower, radius_lower = centres[lower][0].mean(dim=0), centres[lower][1]
    mean_upper, radius_upper = centres[upper][0].mean(dim=0), centres[upper][1]

    delta_y = float(mean_lower[1] - mean_upper[1])
    delta_z = float(mean_lower[2] - mean_upper[2])
    delta_r = radius_lower - radius_upper
    span = math.hypot(delta_y, delta_z)
    if span < 1e-6 or abs(delta_r) >= span:
        return None, (
            f"{lower} and {upper} admit no common tangent plane (separation "
            f"{span:.4f}m vs radius difference {abs(delta_r):.4f}m)"
        )

    angle = math.atan2(delta_z, delta_y) + math.acos(delta_r / span)
    angle = math.atan2(math.sin(angle), math.cos(angle))
    if math.cos(angle) <= 0.0:
        return None, (
            f"the common tangent to {lower} and {upper} faces backwards "
            f"({math.degrees(angle):+.1f}deg)"
        )
    return angle, None


def fit_backrest_to_motion(
    body_pos: torch.Tensor,
    body_rot: torch.Tensor,
    sphere_indices: Dict[str, int],
    seat_top: float,
    yaw: float,
    hips_xy,
    back_thickness: float,
    angle_deg: Optional[float] = None,
    offset: Optional[float] = None,
    back_clearance: float = 0.0,
) -> BackrestFit:
    """Fit the backrest's recline and distance behind the hips to a seated clip.

    Three steps, in this order, because each one constrains the next:

    1. The recline is the common tangent to the mean back spheres, clamped to
       ``[BACK_ANGLE_MIN, BACK_ANGLE_MAX]``.
    2. If that recline brings the plane within ``BACK_MIN_BODY_CLEARANCE`` of the
       pelvis or lumbar spheres it is bisected back towards vertical. Reclining
       swings the slab's lower half forward, so this clearance falls
       monotonically as the angle grows; on an upright clip the raw tangent
       passes clean through the pelvis sphere.
    3. The offset slides the plane back until it is tangent to the rearmost torso
       surface *over the whole clip* -- a minimum, not a quantile. A quantile
       would put the backrest inside the reference torso at reset, which is the
       one failure the tracking-error termination turns into a dead run.

    Args:
        body_pos: Per-body world positions [frames, bodies, 3].
        body_rot: Per-body world rotations [frames, bodies, 4], XYZW.
        sphere_indices: Torso sphere body name -> body index.
        seat_top: Height of the seat's top surface (m).
        yaw: Chair heading (radians).
        hips_xy: Mean hip position in world xy.
        back_thickness: Backrest slab thickness (m).
        angle_deg: Explicit recline (degrees); fitted when None.
        offset: Explicit distance behind the hips (m); fitted when None.
        back_clearance: Extra distance to push the fitted backrest back (m).

    Returns:
        The fitted :class:`BackrestFit`.
    """
    warnings: List[str] = []
    centres = {
        name: (
            chair_frame(
                sphere_centres_world(body_pos, body_rot, sphere_indices[name], local),
                yaw,
                hips_xy,
            ),
            radius,
        )
        for name, (radius, local) in DEFAULT_BACK_SPHERES.items()
    }

    def normal(angle: float) -> torch.Tensor:
        return torch.tensor([0.0, math.cos(angle), math.sin(angle)])

    def tangent_const(angle: float) -> float:
        """Plane constant that just touches the rearmost back surface."""
        axis = normal(angle)
        return min(
            float((centres[name][0] @ axis - centres[name][1]).min())
            for name in DEFAULT_BACK_BODIES
        )

    def pelvis_clearance(angle: float) -> float:
        axis = normal(angle)
        const = tangent_const(angle)
        return min(
            float((centres[name][0] @ axis - centres[name][1] - const).min())
            for name in DEFAULT_BACK_CLEARANCE_BODIES
        )

    if angle_deg is None:
        angle, tangent_warning = _fit_recline(centres)
        if angle is None:
            warnings.append(
                f"{tangent_warning}; falling back to --back-angle "
                f"{BACK_ANGLE_FALLBACK:.1f}deg"
            )
            angle = math.radians(BACK_ANGLE_FALLBACK)
        clamped = min(
            max(angle, math.radians(BACK_ANGLE_MIN)), math.radians(BACK_ANGLE_MAX)
        )
        if abs(clamped - angle) > 1e-9:
            warnings.append(
                f"fitted recline {math.degrees(angle):+.2f}deg clamped to "
                f"{math.degrees(clamped):+.2f}deg"
            )
        angle = clamped

        if pelvis_clearance(angle) < BACK_MIN_BODY_CLEARANCE:
            upright = math.radians(BACK_ANGLE_MIN)
            if pelvis_clearance(upright) < BACK_MIN_BODY_CLEARANCE:
                warnings.append(
                    f"even at {BACK_ANGLE_MIN:.0f}deg the fitted backrest comes "
                    f"within {pelvis_clearance(upright):+.4f}m of "
                    f"{'/'.join(DEFAULT_BACK_CLEARANCE_BODIES)}. This clip is too "
                    "upright for a backrest to touch the back without reaching "
                    "the pelvis -- pass --back-offset explicitly, or generate "
                    "with --no-back-contacts."
                )
                angle = upright
            else:
                requested = angle
                low, high = upright, angle
                for _ in range(10):
                    mid = 0.5 * (low + high)
                    if pelvis_clearance(mid) >= BACK_MIN_BODY_CLEARANCE:
                        low = mid
                    else:
                        high = mid
                angle = low
                warnings.append(
                    f"fitted recline reduced from {math.degrees(requested):+.2f}deg "
                    f"to {math.degrees(angle):+.2f}deg to keep the backrest clear "
                    f"of {'/'.join(DEFAULT_BACK_CLEARANCE_BODIES)}"
                )
    else:
        angle = math.radians(angle_deg)

    if offset is None:
        offset = (
            _offset_for_plane_const(
                tangent_const(angle), angle, back_thickness, seat_top
            )
            + back_clearance
        )

    plane_const = _plane_const(offset, angle, back_thickness, seat_top)
    axis = normal(angle)
    gaps = {
        name: centres[name][0] @ axis - centres[name][1] - plane_const
        for name in DEFAULT_BACK_BODIES
    }
    clearances = {
        name: centres[name][0] @ axis - centres[name][1] - plane_const
        for name in DEFAULT_BACK_CLEARANCE_BODIES
    }

    for name, clearance in clearances.items():
        if float(clearance.min()) < 0.0:
            warnings.append(
                f"the backrest is {-float(clearance.min()):.4f}m inside the "
                f"{name} collision sphere. The robot will be pushed off its "
                "reference at reset."
            )
    for name, gap in gaps.items():
        if float(gap.min()) > BACK_NEVER_RESTS_GAP:
            warnings.append(
                f"{name} never comes closer than {float(gap.min()):.4f}m to the "
                "backrest -- it can never register contact, so labelling it "
                "makes contact_match_rew a permanent penalty."
            )

    return BackrestFit(angle, offset, plane_const, gaps, clearances, warnings)


def backrest_plane(backrest: BoxSceneObject) -> BackrestPlane:
    """Read the front-face plane off a built backrest box.

    Args:
        backrest: The backrest object, as built or as loaded from a scenes file.

    Returns:
        The :class:`BackrestPlane` of its front face.
    """
    rotation = torch.as_tensor(backrest.rotation, dtype=torch.float32).reshape(-1, 4)[
        :1
    ]
    translation = torch.as_tensor(backrest.translation, dtype=torch.float32).reshape(
        -1, 3
    )[0]

    def axis(vec: Sequence[float]) -> torch.Tensor:
        return quat_rotate(
            rotation, torch.tensor([vec], dtype=torch.float32), w_last=True
        )[0]

    # The box's local +y is its front face -- see BoxSceneObject.compute_pointcloud,
    # where max_y is labelled "front".
    normal = axis((0.0, 1.0, 0.0))
    up = axis((0.0, 0.0, 1.0))
    lateral = axis((1.0, 0.0, 0.0))
    return BackrestPlane(
        normal=normal,
        const=float(translation @ normal) + backrest.depth / 2.0,
        bottom_centre=translation - up * (backrest.height / 2.0),
        up=up,
        lateral=lateral,
        height=backrest.height,
        half_width=backrest.width / 2.0,
    )


def backrest_contact(plane: BackrestPlane, centres_w: torch.Tensor, radius: float):
    """Where a collision sphere sits relative to the backrest slab.

    The plane is infinite; the slab is not. ``|gap| ~ 0`` only says the sphere
    touches the *plane*, which is also true of a shoulder floating past the top
    edge, so the contact point has to be checked against the slab's extent too.

    Args:
        plane: The backrest's front face.
        centres_w: Sphere centres in world [frames, 3].
        radius: Sphere radius (m).

    Returns:
        Tuple of (gap [frames], height up the slab [frames], lateral offset from
        the slab's centreline [frames], on-slab mask [frames]).
    """
    rel = centres_w - plane.bottom_centre
    gap = centres_w @ plane.normal - plane.const - radius
    up = rel @ plane.up
    lateral = (rel @ plane.lateral).abs()
    on_slab = (up >= 0.0) & (up <= plane.height) & (lateral <= plane.half_width)
    return gap, up, lateral, on_slab


def build_chair_objects(
    args, seat_top: float, yaw: float, hips_xy, back_angle: float, back_offset: float
) -> List[BoxSceneObject]:
    """Build the seat and backrest boxes for one scene.

    The canonical chair frame faces +y, so ``width`` is lateral and ``depth`` is
    front-to-back on both boxes, and the recline is a rotation about the lateral
    (x) axis before the heading yaw is applied.

    The seat and the backrest are placed by *different* numbers.
    ``--seat-back-offset`` sets how much seat is behind the hips; ``back_offset``
    sets where the backrest's front face goes, and is fitted an order of
    magnitude smaller. Driving both from one flag would either float the
    backrest out of reach of the back or leave the pelvis hanging off the seat.

    Args:
        args: Parsed CLI arguments.
        seat_top: Height of the seat's top surface (m).
        yaw: Chair heading (radians).
        hips_xy: Mean hip position in world xy.
        back_angle: Backrest recline from vertical (radians).
        back_offset: How far behind the hips the backrest's front face sits (m).

    Returns:
        [seat, backrest] -- the object order the experiment's object_index refers to.
    """
    options = ObjectOptions(
        fix_base_link=True,
        density=1000.0,
        static_friction=args.friction,
        dynamic_friction=args.friction,
        max_angular_velocity=100.0,
    )

    hips_x, hips_y = float(hips_xy[0]), float(hips_xy[1])

    # Seat spans from --seat-back-offset behind the hips to --seat-depth ahead
    # of that, so its centre sits forward of the hips.
    seat_shift = args.seat_depth / 2.0 - args.seat_back_offset
    seat_offset = yaw_rotate((0.0, seat_shift, 0.0), yaw)
    seat = BoxSceneObject(
        width=args.seat_width,
        depth=args.seat_depth,
        height=args.seat_thickness,
        translation=(
            hips_x + seat_offset[0],
            hips_y + seat_offset[1],
            seat_top - args.seat_thickness / 2.0,
        ),
        rotation=(0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)),
        options=options,
    )

    # Backrest: place its bottom edge on the seat, behind the hips, then tilt.
    # Rotating about +x by +angle sends local +z towards -y, i.e. backwards.
    angle = back_angle
    bottom_shift = -(back_offset + args.back_thickness / 2.0)
    bottom_offset = yaw_rotate((0.0, bottom_shift, 0.0), yaw)

    half_h = args.back_height / 2.0
    centre_from_bottom = yaw_rotate(
        (0.0, -half_h * math.sin(angle), half_h * math.cos(angle)), yaw
    )

    tilt_quat = (math.sin(angle / 2.0), 0.0, 0.0, math.cos(angle / 2.0))
    yaw_quat = (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))

    backrest = BoxSceneObject(
        width=args.back_width,
        depth=args.back_thickness,
        height=args.back_height,
        translation=(
            hips_x + bottom_offset[0] + centre_from_bottom[0],
            hips_y + bottom_offset[1] + centre_from_bottom[1],
            seat_top + centre_from_bottom[2],
        ),
        rotation=quat_multiply(yaw_quat, tilt_quat),
        options=options,
    )

    return [seat, backrest]


class MotionFit(NamedTuple):
    """What one motion's fitted chair needs to relabel that motion's contacts.

    Attributes:
        seat_top: Height of the seat's top surface (m).
        backrest: The backrest box as built, or None when the backrest was not
            fitted to the clip and back contacts must be left alone.
    """

    seat_top: float
    backrest: Optional[BoxSceneObject]


def relabel_chair_contacts(
    motion_file: str,
    output_file: str,
    fits: Dict[int, MotionFit],
    thigh_indices: Sequence[Tuple[int, int]],
    thigh_radius: float,
    contact_tol: float,
    sphere_indices: Dict[str, int],
    back_contact_tol: float,
) -> None:
    """Write a copy of a motion with thigh-on-seat and back-on-backrest contacts marked.

    The packaged labels come from ground-plane contact detection, so a seated
    clip marks only the feet. Adding the thighs or the back to ``contact_bodies``
    without this makes ``contact_match_rew`` penalise the very behaviour we want.

    Contacts are indexed by absolute body index and OR-ed into the existing
    labels, so the foot columns the packager wrote survive untouched and the
    result stays binary -- which ``MotionLib.smooth_contacts`` requires.

    Back contact is scored against the plane of the backrest that was actually
    built, and requires the contact point to lie on the slab as well as on the
    plane: a shoulder sailing past the top edge is not leaning on anything.

    Args:
        motion_file: Packaged MotionLib .pt to copy.
        output_file: Where to write the relabelled copy.
        fits: Motion index -> fitted seat top and backrest.
        thigh_indices: (hip index, knee index) pairs.
        thigh_radius: Thigh capsule collision radius (m).
        contact_tol: Max gap (m) that still counts as resting on the seat.
        sphere_indices: Torso sphere body name -> body index.
        back_contact_tol: Max gap (m) that still counts as resting on the backrest.
    """
    if Path(output_file).resolve() == Path(motion_file).resolve():
        raise ValueError("--relabel-contacts must not overwrite --motion-file")

    data = torch.load(motion_file, map_location="cpu", weights_only=False)
    if "contacts" not in data:
        raise ValueError(f"{motion_file} has no 'contacts' field to relabel")

    contacts = data["contacts"].clone()
    marked = 0
    back_marked = 0
    for motion_id, fit in sorted(fits.items()):
        start = int(data["length_starts"][motion_id])
        num_frames = int(data["motion_num_frames"][motion_id])
        frames = slice(start, start + num_frames)

        surface = thigh_surface_heights(
            data["gts"][frames], thigh_indices, thigh_radius
        )
        resting = (surface - fit.seat_top).abs() <= contact_tol
        for thigh_slot, (hip_idx, _) in enumerate(thigh_indices):
            contacts[frames, hip_idx] = torch.maximum(
                contacts[frames, hip_idx],
                resting[:, thigh_slot].to(contacts.dtype),
            )
            marked += int(resting[:, thigh_slot].sum())

        if fit.backrest is None:
            continue

        plane = backrest_plane(fit.backrest)
        for name in DEFAULT_BACK_BODIES:
            radius, local = DEFAULT_BACK_SPHERES[name]
            centres_w = sphere_centres_world(
                data["gts"][frames],
                data["grs"][frames],
                sphere_indices[name],
                local,
            )
            gap, up, _, on_slab = backrest_contact(plane, centres_w, radius)
            leaning = (gap.abs() <= back_contact_tol) & on_slab
            body_idx = sphere_indices[name]
            contacts[frames, body_idx] = torch.maximum(
                contacts[frames, body_idx], leaning.to(contacts.dtype)
            )
            back_marked += int(leaning.sum())

            print(
                f"  motion {motion_id} {name}: {int(leaning.sum())}/{num_frames} "
                f"frames leaning (gap {float(gap.min()):+.4f} to "
                f"{float(gap.max()):+.4f}m, tolerance {back_contact_tol:.3f}m)"
            )
            if float(gap.min()) > back_contact_tol:
                print(
                    f"    WARNING: {name} never reaches the backrest. Labelling "
                    "it would be fine here -- nothing is labelled -- but with "
                    f"{name} in contact_bodies contact_match_rew will penalise "
                    "it on every frame it does touch. Refit, or drop it."
                )
            elif float(gap.max()) > back_contact_tol:
                print(
                    f"    WARNING: {name} swings up to {float(gap.max()):.4f}m "
                    "off the backrest, so the label flickers. Raise "
                    "--back-contact-tol if that is not what you want."
                )
            if not bool(on_slab.all()):
                print(
                    f"    WARNING: on {int((~on_slab).sum())} frame(s) the {name} "
                    f"contact point is off the slab (height up the slab "
                    f"{float(up.min()):.3f} to {float(up.max()):.3f}m of "
                    f"{plane.height:.3f}m). Raise --back-height."
                )

    data["contacts"] = contacts
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, output_file)
    print(
        f"Wrote {output_file} with {marked} thigh-frame and {back_marked} "
        f"back-frame contacts marked (tolerances {contact_tol:.3f}m seat, "
        f"{back_contact_tol:.3f}m back)"
    )


def inspect_scenes(
    scenes_file: str,
    motion_file: Optional[str],
    robot_name: str = "soma23",
    thigh_radius: float = DEFAULT_THIGH_RADIUS,
    pelvis_radius: float = DEFAULT_PELVIS_RADIUS,
    back_contact_tol: float = 0.03,
) -> None:
    """Print what a chair scenes file actually contains, and flag silent traps.

    Args:
        scenes_file: Path to the scenes .pt.
        motion_file: Optional packaged MotionLib .pt to cross-check the fit.
        robot_name: Robot whose body names the support bodies refer to.
        thigh_radius: Thigh capsule collision radius (m).
        pelvis_radius: Pelvis sphere collision radius (m).
        back_contact_tol: Gap (m) within which the back counts as leaning.
    """
    scenes = SceneLib._load_scenes_from_file(scenes_file, device="cpu")
    print(f"{scenes_file}: {len(scenes)} scene(s)")

    motion_data = None
    thigh_indices = None
    sphere_indices = None
    if motion_file is not None:
        motion_data = torch.load(motion_file, map_location="cpu", weights_only=False)
        print(f"{motion_file}: {motion_data['motion_num_frames'].shape[0]} motion(s)")
        try:
            _, thigh_indices, sphere_indices = resolve_support_indices(robot_name)
        except ValueError:
            thigh_indices = None
            sphere_indices = None

    for scene_idx, scene in enumerate(scenes):
        print(f"\nscene {scene_idx}: humanoid_motion_id={scene.humanoid_motion_id}")
        if scene.humanoid_motion_id == -1:
            print(
                "  WARNING: unpaired scene. Motions are sampled independently, so "
                "the chair need not be under the clip being tracked."
            )
        if len(scene.objects) != 2:
            print(
                f"  WARNING: {len(scene.objects)} object(s); a chair from this "
                "script is [seat, backrest]."
            )

        for obj_idx, obj in enumerate(scene.objects):
            role = ["seat", "backrest"][obj_idx] if obj_idx < 2 else f"object {obj_idx}"
            frames = obj.translation.shape[0]
            translation = obj.translation[0].tolist()
            print(
                f"  object {obj_idx} ({role}, {type(obj).__name__}): "
                f"size=({obj.width:.3f}, {obj.depth:.3f}, {obj.height:.3f}) "
                f"translation=({translation[0]:+.3f}, {translation[1]:+.3f}, "
                f"{translation[2]:.3f}) frames={frames} "
                f"fix_base_link={obj.options.fix_base_link}"
            )
            if not obj.options.fix_base_link:
                print(
                    "    WARNING: fix_base_link=False -- this part of the chair is "
                    "dynamic and will be pushed away when the robot leans on it."
                )
            if frames > 1:
                print(
                    "    WARNING: multi-frame object. A chair should be static; a "
                    "moving one drags its reference around under the robot."
                )

        seat = scene.objects[0]
        seat_top = float(seat.translation[0][2]) + seat.height / 2.0
        print(f"  seat top = {seat_top:.4f}m")

        if motion_data is not None and thigh_indices and scene.humanoid_motion_id >= 0:
            motion_id = scene.humanoid_motion_id
            start = int(motion_data["length_starts"][motion_id])
            num_frames = int(motion_data["motion_num_frames"][motion_id])
            surface = thigh_surface_heights(
                motion_data["gts"][start : start + num_frames],
                thigh_indices,
                thigh_radius,
            )
            gap = float(surface.min()) - seat_top
            print(f"  lowest reference thigh surface = {float(surface.min()):.4f}m")
            print(f"  clearance (thigh - seat) = {gap:+.4f}m")
            if gap > 0.01:
                print(
                    "    WARNING: the reference thighs float above the seat. The "
                    "policy can hold the pose without ever loading the chair."
                )
            if gap < -0.02:
                print(
                    "    WARNING: the seat is inside the reference thighs. The "
                    "robot will be pushed off its reference at reset."
                )

            if sphere_indices is not None and len(scene.objects) > 1:
                inspect_backrest(
                    scene.objects[1],
                    motion_data,
                    motion_id,
                    seat_top,
                    thigh_indices,
                    sphere_indices,
                    thigh_radius,
                    pelvis_radius,
                    back_contact_tol,
                )


def inspect_backrest(
    backrest: BoxSceneObject,
    motion_data: dict,
    motion_id: int,
    seat_top: float,
    thigh_indices: Sequence[Tuple[int, int]],
    sphere_indices: Dict[str, int],
    thigh_radius: float,
    pelvis_radius: float,
    back_contact_tol: float,
) -> None:
    """Report how the built backrest sits against its reference clip.

    Everything here is derived from the box that was saved, not from a re-run of
    the fit, so it describes the chair training will actually load.

    Args:
        backrest: The backrest object from the scene.
        motion_data: Loaded packaged MotionLib dict.
        motion_id: Motion this scene is paired with.
        seat_top: Height of the seat's top surface (m).
        thigh_indices: (hip index, knee index) pairs.
        sphere_indices: Torso sphere body name -> body index.
        thigh_radius: Thigh capsule collision radius (m).
        pelvis_radius: Pelvis sphere collision radius (m).
        back_contact_tol: Gap (m) within which the back counts as leaning.
    """
    start = int(motion_data["length_starts"][motion_id])
    num_frames = int(motion_data["motion_num_frames"][motion_id])
    frames = slice(start, start + num_frames)
    body_pos = motion_data["gts"][frames]
    body_rot = motion_data["grs"][frames]

    plane = backrest_plane(backrest)
    # The normal's z component is sin(recline) whatever the heading is.
    angle = math.asin(max(-1.0, min(1.0, float(plane.normal[2]))))

    _, yaw, hips_xy, _ = fit_chair_to_motion(
        body_pos,
        sphere_indices[DEFAULT_PELVIS_BODY],
        thigh_indices,
        thigh_radius,
        pelvis_radius,
    )
    face_point = torch.as_tensor(backrest.translation, dtype=torch.float32).reshape(
        -1, 3
    )[0] + plane.normal * (backrest.depth / 2.0)
    face_chair = chair_frame(face_point.reshape(1, 3), yaw, hips_xy)[0]
    plane_const = float(
        face_chair @ torch.tensor([0.0, math.cos(angle), math.sin(angle)])
    )
    offset = _offset_for_plane_const(plane_const, angle, backrest.depth, seat_top)

    slab_top = float((plane.bottom_centre + plane.up * plane.height)[2])
    print(
        f"  backrest: angle={math.degrees(angle):+.2f}deg offset={offset:.4f}m "
        f"slab top={slab_top:.4f}m"
    )

    contacts = motion_data.get("contacts")
    for name in list(DEFAULT_BACK_BODIES) + list(DEFAULT_BACK_CLEARANCE_BODIES):
        radius, local = DEFAULT_BACK_SPHERES[name]
        centres_w = sphere_centres_world(
            body_pos, body_rot, sphere_indices[name], local
        )
        gap, up, _, on_slab = backrest_contact(plane, centres_w, radius)
        role = "back" if name in DEFAULT_BACK_BODIES else "clearance"
        print(
            f"    {name} ({role}): gap min={float(gap.min()):+.4f} "
            f"mean={float(gap.mean()):+.4f} max={float(gap.max()):+.4f}m, "
            f"contact point {float(up.mean()) / plane.height:.2f} up the slab"
        )

        if name in DEFAULT_BACK_CLEARANCE_BODIES:
            if float(gap.min()) < 0.0:
                print(
                    f"      WARNING: the backrest is inside the {name} sphere. "
                    "The robot will be pushed off its reference at reset."
                )
            continue

        leaning = (gap.abs() <= back_contact_tol) & on_slab
        print(
            f"      {int(leaning.sum())}/{num_frames} frames within "
            f"{back_contact_tol:.3f}m and on the slab"
        )
        if float(gap.min()) > back_contact_tol:
            print(
                f"      WARNING: {name} never reaches the backrest. With it in "
                "contact_bodies, contact_match_rew has no achievable target."
            )
        if not bool(on_slab.all()):
            print(
                f"      WARNING: on {int((~on_slab).sum())} frame(s) the contact "
                "point is off the slab. Raise --back-height."
            )

        # The likeliest operator error: a correctly fitted chair paired with the
        # motion file that was never relabelled to match it.
        if contacts is not None and int(leaning.sum()) > 0:
            labelled = int(contacts[frames, sphere_indices[name]].sum())
            if labelled == 0:
                print(
                    f"      WARNING: the geometry says {name} rests on the "
                    f"backrest for {int(leaning.sum())} frame(s), but this motion "
                    "file labels it as never in contact. This is the "
                    "un-relabelled file -- train against the --relabel-contacts "
                    "output instead, or contact_match_rew will penalise leaning."
                )


def main():
    args = parse_args()

    if args.inspect:
        inspect_scenes(
            args.inspect,
            args.motion_file,
            robot_name=args.robot_name,
            thigh_radius=args.thigh_radius,
            pelvis_radius=args.pelvis_radius,
            back_contact_tol=args.back_contact_tol,
        )
        return

    pelvis_index, thigh_indices, sphere_indices = resolve_support_indices(
        args.robot_name
    )

    if args.all_motions:
        *_, num_motions = load_packaged_motion(args.motion_file, 0)
        motion_ids = list(range(num_motions))
    else:
        motion_ids = args.motion_id or [0]

    scenes = []
    fits: Dict[int, MotionFit] = {}
    for motion_id in motion_ids:
        body_pos = body_rot = None
        if args.motion_file:
            body_pos, body_rot, _, _ = load_packaged_motion(args.motion_file, motion_id)
            fitted_top, fitted_yaw, hips_xy, _ = fit_chair_to_motion(
                body_pos,
                pelvis_index,
                thigh_indices,
                args.thigh_radius,
                args.pelvis_radius,
            )
        else:
            fitted_top, fitted_yaw, hips_xy = 0.45, 0.0, torch.zeros(2)

        seat_top = (
            args.seat_top
            if args.seat_top is not None
            else fitted_top - args.seat_clearance
        )
        yaw = math.radians(args.yaw) if args.yaw is not None else fitted_yaw

        # The backrest can only be fitted against a clip. Without one -- or with
        # --no-back-contacts -- fall back to the decorative backrest this script
        # shipped with, driven by --seat-back-offset, and leave back contacts
        # alone.
        fit_back = body_rot is not None and not args.no_back_contacts
        if fit_back:
            back_fit = fit_backrest_to_motion(
                body_pos,
                body_rot,
                sphere_indices,
                seat_top=seat_top,
                yaw=yaw,
                hips_xy=hips_xy,
                back_thickness=args.back_thickness,
                angle_deg=args.back_angle,
                offset=args.back_offset,
                back_clearance=args.back_clearance,
            )
            back_angle, back_offset = back_fit.angle, back_fit.offset
            for warning in back_fit.warnings:
                print(f"  WARNING: {warning}")
        else:
            back_fit = None
            back_angle = math.radians(
                args.back_angle if args.back_angle is not None else BACK_ANGLE_FALLBACK
            )
            back_offset = (
                args.back_offset
                if args.back_offset is not None
                else args.seat_back_offset
            )

        objects = build_chair_objects(
            args, seat_top, yaw, hips_xy, back_angle, back_offset
        )
        scenes.append(Scene(objects=objects, humanoid_motion_id=motion_id))
        fits[motion_id] = MotionFit(
            seat_top=seat_top, backrest=objects[1] if fit_back else None
        )
        print(
            f"motion {motion_id}: seat_top={seat_top:.4f}m "
            f"yaw={math.degrees(yaw):+.1f}deg "
            f"hips=({float(hips_xy[0]):+.3f}, {float(hips_xy[1]):+.3f}) "
            f"back_angle={math.degrees(back_angle):+.2f}deg "
            f"back_offset={back_offset:.4f}m"
        )
        if back_fit is not None:
            for name, gap in back_fit.gaps.items():
                print(
                    f"  {name}: gap min={float(gap.min()):+.4f} "
                    f"mean={float(gap.mean()):+.4f} max={float(gap.max()):+.4f}m"
                )
            for name, clearance in back_fit.clearances.items():
                print(f"  {name} clearance: min={float(clearance.min()):+.4f}m")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    SceneLib.save_scenes_to_file(scenes, args.output)
    print(f"Saved {len(scenes)} chair scene(s) to {args.output}")

    if args.relabel_contacts:
        relabel_chair_contacts(
            motion_file=args.motion_file,
            output_file=args.relabel_contacts,
            fits=fits,
            thigh_indices=thigh_indices,
            thigh_radius=args.thigh_radius,
            contact_tol=args.contact_tol,
            sphere_indices=sphere_indices,
            back_contact_tol=args.back_contact_tol,
        )


if __name__ == "__main__":
    main()
