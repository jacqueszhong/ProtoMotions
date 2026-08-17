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
"""Shared PhysX/USD read-back probes for the Isaac Sim <-> IsaacLab parity work.

Both ``deployment/test_tracker_isaacsim.py`` (standalone Isaac Sim driver) and
``deployment/trace_tracker_isaaclab.py`` (IsaacLab ground truth) drive PhysX, and
the whole point of their dumps is that the two logs diff by eye.  That only holds
if the *same code* produces both, which is why these helpers live here instead of
being written twice -- four rounds of this investigation have been decided by
reading a value back, and two subtly different readers would have wasted one of
them.

Two probes, matching the two instrumentation gaps:

:func:`collect_contact_geometry` / :func:`lowest_tip_z`
    Where the robot's feet actually are.  The link origin is *not* the contact
    point: on the G1 each ``*_ankle_roll_link`` carries seven collision capsules
    at local ``z = -0.025`` with radius ``0.01``, so the lowest contact point sits
    0.035 m below the link origin when the foot is flat -- and further when it is
    not.  Reporting link origins (which the earlier dumps did) cannot distinguish
    "the robot stands 1.4 cm higher" from "the robot stands in a different pose".

:func:`dump_link_properties`
    Every per-link rigid-body property, in sim link order, read from the physics
    view where the tensor API exposes it and from the composed stage otherwise.
    This closes the one diff that was never done PhysX-to-PhysX: the two stacks
    author ``physxRigidBody:*`` through **different traversals** -- the driver
    walks ``Usd.PrimRange`` without ``TraverseInstanceProxies``, IsaacLab wraps
    its writer in ``apply_nested``, which skips instanced prims -- and two
    different skip rules over the same instanceable asset can reach two different
    sets of links.  The reads here deliberately traverse instance proxies, so the
    dump sees links the *authoring* walk may have missed.  That asymmetry is the
    measurement, not a bug in the dump.

``pxr`` is imported lazily inside each function: the IsaacLab harness must not
import USD above its ``AppLauncher`` line, and this module is imported from both
sides of that boundary.
"""

from __future__ import annotations

import numpy as np

#: Contact-force magnitude above which a foot counts as in contact, in newtons.
#: Matches ``Simulator.get_binary_body_contacts``' default, which is what the
#: training environment's terminations see -- so "stance" means the same thing
#: here as it does to the policy that was trained. The G1 weighs ~35 kg, so a
#: real footfall is three to four orders of magnitude above this; the threshold
#: exists to reject solver noise, not to discriminate light contacts.
CONTACT_FORCE_THRESHOLD_N = 0.01

#: Per-link rigid-body attributes read off the composed stage, in the order they
#: are printed.  ``physxRigidBody:*`` is what both stacks author (the driver in
#: ``_author_body_properties``, IsaacLab in ``modify_rigid_body_properties``), so
#: a disagreement here is a disagreement about what the solver was handed.
LINK_STAGE_ATTRS = (
    "physxRigidBody:linearDamping",
    "physxRigidBody:angularDamping",
    "physxRigidBody:maxLinearVelocity",
    "physxRigidBody:maxAngularVelocity",
    "physxRigidBody:maxDepenetrationVelocity",
    "physxRigidBody:sleepThreshold",
    "physxRigidBody:stabilizationThreshold",
    "physxRigidBody:disableGravity",
    "physxRigidBody:retainAccelerations",
)


# ---------------------------------------------------------------------------
# Quaternion helper
# ---------------------------------------------------------------------------


def quat_rotate_np(q_xyzw: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vectors ``v`` by quaternion ``q_xyzw`` (forward rotation).

    Uses the cross-product form rather than building a rotation matrix, so there
    is no transpose convention to get wrong -- the sign errors in this file would
    show up as feet a few centimetres in the wrong place, which is exactly the
    magnitude of the effect being measured.

    Args:
        q_xyzw: Unit quaternion ``[4]`` in ProtoMotions' xyzw order.
        v: Vectors ``[..., 3]``.

    Returns:
        Rotated vectors, same shape as ``v``.
    """
    q = np.asarray(q_xyzw, dtype=np.float64).reshape(4)
    v = np.asarray(v, dtype=np.float64)
    q_vec, q_w = q[:3], q[3]
    t = 2.0 * np.cross(np.broadcast_to(q_vec, v.shape), v)
    return v + q_w * t + np.cross(np.broadcast_to(q_vec, t.shape), t)


# ---------------------------------------------------------------------------
# Contact geometry (Step 1)
# ---------------------------------------------------------------------------


def collect_contact_geometry(stage, link_prim_path: str) -> list:
    """Collect a link's collider extremities, expressed in the link's own frame.

    Every supported collider is reduced to a set of *bounding spheres* -- a
    (centre, radius) pair per extremity -- because the lowest point of a sphere
    is ``centre_z - radius`` regardless of orientation, so the runtime query in
    :func:`lowest_tip_z` needs no per-shape branching. A capsule becomes its two
    end-cap centres, a sphere itself, a box its eight corners (radius 0).

    This is exact for spheres and capsules (the G1's feet are seven capsules
    each) and exact for boxes too, since a box's lowest point is always a corner.

    Traverses instance proxies: these assets are spawned ``make_instanceable``,
    so a default ``Usd.PrimRange`` walks straight past every collider.

    Args:
        stage: The composed USD stage.
        link_prim_path: Path of the rigid-body link prim (e.g.
            ``/World/Robot/pelvis/left_ankle_roll_link``).

    Returns:
        List of ``(centre_in_link_frame [3], radius)`` tuples. Empty if the link
        prim is invalid or carries no colliders.
    """
    from pxr import Usd, UsdGeom, UsdPhysics

    link = stage.GetPrimAtPath(link_prim_path)
    if not link.IsValid():
        return []

    cache = UsdGeom.XformCache()
    link_inv = cache.GetLocalToWorldTransform(link).GetInverse()

    tips: list = []
    for prim in Usd.PrimRange(
        link, Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)
    ):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        # Collider-local -> link-local. Constant for the life of the sim: the
        # collider is rigidly attached, so this is resolved once at setup and
        # reused every control step.
        rel = cache.GetLocalToWorldTransform(prim) * link_inv
        type_name = prim.GetTypeName()

        local_points: list = []
        radius = 0.0
        if type_name == "Capsule":
            capsule = UsdGeom.Capsule(prim)
            radius = float(capsule.GetRadiusAttr().Get() or 0.0)
            half = 0.5 * float(capsule.GetHeightAttr().Get() or 0.0)
            axis = str(capsule.GetAxisAttr().Get() or "Z").upper()
            unit = {"X": (1.0, 0.0, 0.0), "Y": (0.0, 1.0, 0.0)}.get(
                axis, (0.0, 0.0, 1.0)
            )
            local_points = [
                tuple(sign * half * c for c in unit) for sign in (-1.0, 1.0)
            ]
        elif type_name == "Sphere":
            radius = float(UsdGeom.Sphere(prim).GetRadiusAttr().Get() or 0.0)
            local_points = [(0.0, 0.0, 0.0)]
        elif type_name == "Cube":
            half = 0.5 * float(UsdGeom.Cube(prim).GetSizeAttr().Get() or 0.0)
            local_points = [
                (sx * half, sy * half, sz * half)
                for sx in (-1.0, 1.0)
                for sy in (-1.0, 1.0)
                for sz in (-1.0, 1.0)
            ]
        else:
            # Meshes and cylinders: fall back to the prim origin. Reported rather
            # than silently skipped -- a robot whose feet are meshes needs a real
            # implementation here, and a wrong height is worse than a missing one.
            local_points = [(0.0, 0.0, 0.0)]

        for point in local_points:
            centre = np.asarray(rel.Transform(point), dtype=np.float64)
            tips.append((centre, radius))
    return tips


def lowest_tip_z(link_pos, link_quat_xyzw, tips: list) -> float:
    """World z of the lowest collider extremity of one link.

    Args:
        link_pos: Link origin in world coordinates ``[3]``.
        link_quat_xyzw: Link orientation ``[4]`` (xyzw).
        tips: Output of :func:`collect_contact_geometry` for that link.

    Returns:
        Lowest world z, or ``nan`` if the link has no collider geometry.
    """
    if not tips:
        return float("nan")
    centres = np.stack([t[0] for t in tips])
    radii = np.asarray([t[1] for t in tips], dtype=np.float64)
    world = np.asarray(link_pos, dtype=np.float64)[None, :] + quat_rotate_np(
        link_quat_xyzw, centres
    )
    return float(np.min(world[:, 2] - radii))


class FootProbe:
    """Resolves the lowest foot contact point per control step.

    Built once, after the physics view exists, from the composed stage; then
    queried every control step with the link transforms the physics view already
    reports. Holds no simulator handles, so it works identically on both stacks
    -- the caller supplies the transforms.

    Args:
        stage: Composed USD stage.
        link_paths: Mapping ``body_name -> link prim path`` for the feet.
        link_indices: Mapping ``body_name -> index into the link-transform
            buffer`` (i.e. the position of that body in ``body_names``).
    """

    def __init__(self, stage, link_paths: dict, link_indices: dict) -> None:
        self.link_indices = dict(link_indices)
        self.geometry = {
            name: collect_contact_geometry(stage, path)
            for name, path in link_paths.items()
        }
        self.missing = [name for name, tips in self.geometry.items() if not tips]

    def describe(self) -> str:
        """One line per foot: collider count and rest-pose drop below the origin."""
        lines = []
        for name, tips in sorted(self.geometry.items()):
            if not tips:
                lines.append(f"  {name:24s} <no collider geometry found>")
                continue
            drop = lowest_tip_z(np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]), tips)
            lines.append(
                f"  {name:24s} {len(tips):3d} collider extremities, "
                f"lowest is {-drop:.4f} m below the link origin at rest"
            )
        return "\n".join(lines)

    def lowest(self, link_pos, link_quat_xyzw) -> float:
        """Lowest contact point over all probed feet.

        Args:
            link_pos: World positions of every link ``[num_links, 3]``.
            link_quat_xyzw: World orientations of every link ``[num_links, 4]``
                in xyzw order.

        Returns:
            Lowest world z over all feet, or ``nan`` if none could be resolved.
        """
        heights = []
        for name, tips in self.geometry.items():
            index = self.link_indices.get(name)
            if index is None or not tips:
                continue
            heights.append(lowest_tip_z(link_pos[index], link_quat_xyzw[index], tips))
        if not heights:
            return float("nan")
        return float(np.nanmin(heights))


# ---------------------------------------------------------------------------
# Link properties (Step 2)
# ---------------------------------------------------------------------------


def _format_value(value) -> str:
    if value is None:
        return "None"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def dump_link_properties(
    emit,
    stage,
    body_names: list,
    link_prim_paths: dict,
    masses=None,
    inertias=None,
    coms=None,
) -> None:
    """Print every per-link rigid-body property in sim link order.

    The dump is deliberately verbose and deliberately ordered by
    ``body_names``: a row-for-row diff of two of these logs is the whole
    deliverable, and any reordering or summarising would hide exactly the
    per-link disagreement it exists to find.

    ``masses``/``inertias``/``coms`` come from the physics view (the tensor API
    exposes them on both stacks); everything in :data:`LINK_STAGE_ATTRS` is read
    off the composed stage, because PhysX's tensor API does not expose per-body
    damping or velocity clamps at all. Each stage value is marked ``*`` when no
    layer authored it -- an unauthored value is a schema default that *neither*
    stack chose, and therefore the one most likely to differ between them.

    Args:
        emit: Single-argument printer (``print`` on the IsaacLab side, where
            Kit's ``dictConfig`` swallows ``log.info``; ``log.info`` in the
            driver).
        stage: Composed USD stage.
        body_names: Link names in sim order.
        link_prim_paths: Mapping ``body_name -> prim path``.
        masses: Optional ``[num_links]`` from the physics view.
        inertias: Optional ``[num_links, 9]`` or ``[num_links, 3]``.
        coms: Optional ``[num_links, 3]``.
    """
    emit(f"\n=== Per-link properties (sim link order, {len(body_names)} links) ===")
    emit(f"  body_names: {list(body_names)}")

    def _row(array, index):
        if array is None:
            return None
        array = np.asarray(array)
        flat = array.reshape(array.shape[0], -1) if array.ndim > 1 else array
        if index >= flat.shape[0]:
            return None
        return flat[index]

    for index, name in enumerate(body_names):
        path = link_prim_paths.get(name)
        emit(f"  [{index:2d}] {name}   prim={path}")

        mass = _row(masses, index)
        com = _row(coms, index)
        inertia = _row(inertias, index)
        if mass is not None:
            emit(f"       mass    = {float(np.asarray(mass).reshape(-1)[0]):.6f}")
        if com is not None:
            emit(f"       com     = {np.round(np.asarray(com), 6).tolist()}")
        if inertia is not None:
            emit(f"       inertia = {np.round(np.asarray(inertia), 6).tolist()}")

        prim = stage.GetPrimAtPath(path) if path else None
        if prim is None or not prim.IsValid():
            emit("       <prim not found on the stage -- stage attrs unavailable>")
            continue
        for attr_name in LINK_STAGE_ATTRS:
            attr = prim.GetAttribute(attr_name)
            short = attr_name.split(":")[-1]
            if not attr:
                emit(f"       {short:28s} = <not declared>")
                continue
            mark = "" if attr.IsAuthored() else "  *"
            emit(f"       {short:28s} = {_format_value(attr.Get())}{mark}")

    emit(
        "  (trailing '*' = unauthored by any layer, i.e. a schema default neither "
        "stack chose)"
    )


def resolve_link_prim_paths(stage, root_path: str, body_names: list) -> dict:
    """Map each sim link name to its rigid-body prim path under ``root_path``.

    Matches by prim *name* against ``UsdPhysics.RigidBodyAPI`` holders rather
    than assuming a path shape: the driver's links live at
    ``/World/Robot/pelvis/<body>`` and IsaacLab's at
    ``/World/envs/env_0/Robot/pelvis/<body>``, and the articulation root is a
    doubled-name prim (``pelvis/pelvis``) on both.

    Traverses instance proxies for the reason documented at module level.

    Args:
        stage: Composed USD stage.
        root_path: Robot root prim path.
        body_names: Link names in sim order.

    Returns:
        Mapping ``body_name -> prim path``; names with no match are absent.
    """
    from pxr import Usd, UsdPhysics

    wanted = set(body_names)
    found: dict = {}
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        return found
    for prim in Usd.PrimRange(
        root, Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)
    ):
        name = prim.GetName()
        if name in wanted and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            # First match wins. On the G1 the only ambiguity is the root: the
            # outer `.../pelvis` is a plain Xform holding the body subtree and
            # carries no RigidBodyAPI, so the `RigidBodyAPI` test already
            # selects the inner `.../pelvis/pelvis`, which is both the pelvis
            # rigid body and the articulation root.
            found.setdefault(name, prim.GetPath().pathString)
    return found
