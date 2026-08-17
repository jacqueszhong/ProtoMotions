# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SceneLib access for the standalone Isaac Sim deployment driver.

``deployment/test_tracker_isaacsim.py`` authors its own stage -- robot USD,
ground, lights -- so a policy trained against a *scene* (the motivating case is
``examples/experiments/mimic/g1_pick_box.py``, whose G1 picks up a box along a
reference trajectory) had nothing to interact with there.  This module turns a
ProtoMotions scenes ``.pt`` into a simulator-agnostic description the driver can
spawn, plus the ``SceneLib`` instance that supplies the objects' reference poses
at reset.

It lives outside the driver on purpose: that file creates ``SimulationApp`` at
import time, so nothing in it can be unit-tested.  Everything here is pure
Python + torch -- no Isaac Sim, no USD -- and is covered by
``protomotions/tests/test_deployment_scene_utils.py``.

Two unrelated notions of "static"
---------------------------------
ProtoMotions overloads the word, and conflating the two is the obvious way to
get object spawning wrong:

``ObjectOptions.fix_base_link``
    The physical one.  ``True`` means the body never moves -- IsaacLab maps it
    to ``RigidBodyPropertiesCfg(kinematic_enabled=...)`` and this driver maps it
    to ``physics:kinematicEnabled``.  It is what :attr:`SceneObjectSpec.fix_base_link`
    carries.

``SceneLib._is_static_object`` (``not obj.has_motion()``)
    A data property: the object has a single stored frame rather than a
    trajectory.  Its *only* effect is gating the z respawn lift inside
    ``get_scene_pose(respawn_offset=...)``.  A single-frame object can still be
    a perfectly ordinary dynamic rigid body that falls under gravity.

They are independent: ``create_box_scene.py`` writes a multi-frame box with
``fix_base_link=False`` (dynamic, moving reference), but a fixed table would be
single-frame *and* ``fix_base_link=True``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

__all__ = [
    "SceneObjectSpec",
    "resolve_scene_index",
    "build_scene_lib",
    "scene_object_specs",
]


@dataclass
class SceneObjectSpec:
    """USD-agnostic description of one scene object, ready for stage authoring.

    Flattens a :class:`~protomotions.components.scene_lib.SceneObject` into the
    handful of numbers a spawner needs, so the driver's stage-authoring code
    never imports ProtoMotions types.

    Attributes:
        kind: One of ``"box"``, ``"sphere"``, ``"cylinder"``, ``"mesh"``.
        size: ``(width, depth, height)`` for ``kind == "box"``, else ``None``.
        radius: Radius for spheres and cylinders, else ``None``.
        height: Height for cylinders, else ``None``.
        usd_path: Absolute asset path for ``kind == "mesh"``, else ``None``.
        scale: Mesh scale factor ``(x, y, z)``; ``(1, 1, 1)`` for primitives.
        fix_base_link: ``True`` spawns the body kinematic (immovable). This is
            the *physical* sense of static -- see the module docstring.
        mass: Explicit mass in kg, or ``None`` when density governs.
        density: Density in kg/m^3, or ``None`` when ``mass`` is set. Exactly
            one of the two is always set (``ObjectOptions.__post_init__``
            defaults density when neither is given).
        static_friction: Static friction, or ``None`` to leave PhysX's default.
        dynamic_friction: Dynamic friction, or ``None``.
        restitution: Restitution, or ``None``.
        color: RGB in 0-1, or ``None`` to use the per-kind default.
    """

    kind: str
    size: Optional[Tuple[float, float, float]] = None
    radius: Optional[float] = None
    height: Optional[float] = None
    usd_path: Optional[str] = None
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    fix_base_link: bool = False
    mass: Optional[float] = None
    density: Optional[float] = None
    static_friction: Optional[float] = None
    dynamic_friction: Optional[float] = None
    restitution: Optional[float] = None
    color: Optional[Tuple[float, float, float]] = None


def _load_raw_scenes(scenes_file: str) -> List[dict]:
    """Read the serialized scene dicts straight out of a scenes ``.pt``.

    Deliberately *not* a ``SceneLib`` build: resolving which scene to load only
    needs the ``humanoid_motion_id`` field, and constructing a throwaway
    ``SceneLib`` to read it would sample pointclouds and load mesh files for
    every scene in the library first.  The format is the one written by
    ``SceneLib._serialize_scenes_for_storage_static``.

    Args:
        scenes_file: Path to the scenes ``.pt``.

    Returns:
        The file's ``original_scenes`` list.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If it is not a scenes file (no ``original_scenes`` key).
    """
    import torch

    if not os.path.exists(scenes_file):
        raise FileNotFoundError(f"Scenes file not found: {scenes_file}")
    data = torch.load(scenes_file, map_location="cpu", weights_only=False)
    if not isinstance(data, dict) or "original_scenes" not in data:
        raise ValueError(
            f"{scenes_file} is not a ProtoMotions scenes file (no 'original_scenes' "
            f"key; got {type(data).__name__}). Build one with "
            f"data/scripts/create_box_scene.py."
        )
    return data["original_scenes"]


def resolve_scene_index(
    scenes_file: str, motion_index: int, explicit: Optional[int] = None
) -> int:
    """Pick which scene in the library goes with the clip being played.

    Scenes carry ``humanoid_motion_id``, the motion they were authored against
    (``create_box_scene.py --motion-id``); a box trajectory only means anything
    next to *its* clip.  ProtoMotions pairs them per-env through
    ``SceneLib.get_per_env_humanoid_motion_ids_tensor``; this driver plays one
    clip, so the pairing collapses to a single lookup.

    Args:
        scenes_file: Path to the scenes ``.pt``.
        motion_index: The clip index the driver is playing (``--motion-index``).
        explicit: Explicit override (``--scene-index``); returned as-is after a
            range check.

    Returns:
        Index into the file's scene list.

    Raises:
        FileNotFoundError: If the scenes file does not exist.
        ValueError: If the file holds no scenes, or ``explicit`` is out of range.
    """
    scenes = _load_raw_scenes(scenes_file)
    if not scenes:
        raise ValueError(f"{scenes_file} contains no scenes")

    if explicit is not None:
        if not 0 <= explicit < len(scenes):
            raise ValueError(
                f"--scene-index {explicit} out of range: {scenes_file} has "
                f"{len(scenes)} scene(s)"
            )
        log.info(f"Scene index {explicit} (explicit --scene-index)")
        return explicit

    motion_ids = [int(s.get("humanoid_motion_id", -1)) for s in scenes]
    for idx, motion_id in enumerate(motion_ids):
        if motion_id == int(motion_index):
            log.info(
                f"Scene index {idx} (paired with motion {motion_index} via "
                f"humanoid_motion_id)"
            )
            return idx

    # -1 means "universal": the scene was authored without a specific clip in
    # mind, so falling back to it is expected rather than a mismatch.
    log.warning(
        f"No scene in {scenes_file} has humanoid_motion_id == {motion_index} "
        f"(found {motion_ids}); falling back to scene 0. Pass --scene-index to "
        f"choose deliberately."
    )
    return 0


def build_scene_lib(
    scenes_file: str, scene_index: int, asset_root: Optional[str] = None
):
    """Build a single-env, single-scene ``SceneLib`` for the driver.

    Three choices here are load-bearing:

    ``terrain=None``
        Makes this work at all for a one-env driver. ``_assign_scene_offsets``
        only computes a terrain-layout offset when a terrain is supplied;
        without one the scene keeps its stored offset (``(0, 0)`` for anything
        ``create_box_scene.py`` writes), so ``get_scene_pose`` returns poses in
        the same world frame the driver spawns the robot in. With a terrain the
        scene would be shifted tens of metres out to its slot in the object
        playground, and the robot would reach for nothing.

    ``scene_indices=[scene_index]``
        Pre-filters *before* replication/subsetting, so no other scene's meshes
        are ever touched.

    ``pointcloud_samples_per_object=None``
        The driver scores object poses; it never feeds object observations to
        the policy (the ONNX graph has no scene channel), so sampling
        pointclouds would be pure startup cost.

    Args:
        scenes_file: Path to the scenes ``.pt``.
        scene_index: Which scene to keep, from :func:`resolve_scene_index`.
        asset_root: Root for resolving relative mesh paths. ``None`` keeps
            ``SceneLib``'s own default, the *grandparent* of the scenes file --
            asymmetric with the save side, which stores paths relative to the
            file's parent, so pass this explicitly for mesh scenes that fail to
            load.

    Returns:
        A ``SceneLib`` holding exactly one scene, on the CPU.
    """
    from protomotions.components.scene_lib import SceneLib, SceneLibConfig

    config = SceneLibConfig(
        scene_file=scenes_file,
        asset_root=asset_root,
        scene_indices=[int(scene_index)],
        pointcloud_samples_per_object=None,
    )
    scene_lib = SceneLib(config, num_envs=1, device="cpu", terrain=None)
    log.info(
        f"SceneLib: {scene_lib.num_scenes()} scene, "
        f"{scene_lib.num_objects_per_scene} object(s), "
        f"offset={scene_lib.scene_offsets[0] if scene_lib.scene_offsets else None}"
    )
    return scene_lib


def scene_object_specs(scene_lib) -> List[SceneObjectSpec]:
    """Flatten the scene's objects into :class:`SceneObjectSpec` descriptions.

    Reads ``scene_lib.scenes[0]`` -- the driver builds exactly one scene, so the
    replicated list has one entry and it is the one to spawn.

    Args:
        scene_lib: A ``SceneLib`` from :func:`build_scene_lib`.

    Returns:
        One spec per object, in the scene's object order (which is the order
        every ``SceneLib`` pose tensor is indexed by, so the driver can zip them
        positionally).

    Raises:
        ValueError: On an object subclass with no spawn mapping, mirroring
            ``IsaacLabSimulator._preprocess_object_playground``.
    """
    from protomotions.components.scene_lib import (
        BoxSceneObject,
        CylinderSceneObject,
        MeshSceneObject,
        SphereSceneObject,
    )

    if not scene_lib.scenes:
        return []

    specs: List[SceneObjectSpec] = []
    for obj in scene_lib.scenes[0].objects:
        options = obj.options
        common: Dict[str, Any] = dict(
            # The *physical* static flag. `fix_base_link` defaults to None in
            # ObjectOptions, which means "not requested" -- i.e. dynamic.
            fix_base_link=bool(options.fix_base_link),
            mass=options.mass,
            density=options.density,
            static_friction=options.static_friction,
            dynamic_friction=options.dynamic_friction,
            restitution=options.restitution,
            color=tuple(options.color) if options.color is not None else None,
        )

        if isinstance(obj, BoxSceneObject):
            spec = SceneObjectSpec(
                kind="box",
                size=(float(obj.width), float(obj.depth), float(obj.height)),
                **common,
            )
        elif isinstance(obj, SphereSceneObject):
            spec = SceneObjectSpec(kind="sphere", radius=float(obj.radius), **common)
        elif isinstance(obj, CylinderSceneObject):
            spec = SceneObjectSpec(
                kind="cylinder",
                radius=float(obj.radius),
                height=float(obj.height),
                **common,
            )
        elif isinstance(obj, MeshSceneObject):
            spec = SceneObjectSpec(
                kind="mesh",
                usd_path=str(obj.object_path),
                scale=(
                    float(obj.scale[0]),
                    float(obj.scale[1]),
                    float(obj.scale[2]),
                ),
                **common,
            )
        else:
            raise ValueError(f"Unsupported object type: {type(obj)}")
        specs.append(spec)
    return specs
