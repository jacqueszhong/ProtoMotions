# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``deployment.scene_utils``, the Isaac Sim driver's SceneLib bridge.

Simulator-free by construction: everything under test is pure Python + torch,
which is precisely why it lives outside ``deployment/test_tracker_isaacsim.py``
(that module boots ``SimulationApp`` at import time and cannot be imported here).

The trace-column assertions cover the other half of the change -- the optional
object columns in ``deployment.state_utils``, which every driver shares.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch

from deployment.scene_utils import (
    build_scene_lib,
    resolve_scene_index,
    scene_object_specs,
)
from deployment.state_utils import (
    make_trace_row,
    quat_angle_deg_xyzw,
    summarize_trace,
)
from protomotions.components.scene_lib import (
    BoxSceneObject,
    CylinderSceneObject,
    ObjectOptions,
    PrimitiveSceneObject,
    Scene,
    SceneLib,
)

IDENTITY_QUAT = (0.0, 0.0, 0.0, 1.0)


def _write_scenes(tmp_path) -> str:
    """Write a two-scene library: an unpaired scene, then one paired with motion 1.

    Scene order matters for the tests: the paired scene is *not* first, so a
    resolver that ignored ``humanoid_motion_id`` and returned 0 would still look
    right on the fallback case and wrong here.
    """
    unpaired = Scene(
        objects=[
            BoxSceneObject(
                width=0.4,
                depth=0.3,
                height=0.2,
                translation=(1.0, 2.0, 0.5),
                rotation=IDENTITY_QUAT,
                options=ObjectOptions(mass=2.0),
            )
        ],
        humanoid_motion_id=-1,
    )
    paired = Scene(
        objects=[
            BoxSceneObject(
                width=0.5,
                depth=0.25,
                height=0.15,
                translation=(0.3, -0.2, 0.8),
                rotation=IDENTITY_QUAT,
                options=ObjectOptions(
                    fix_base_link=True,
                    mass=1.5,
                    static_friction=0.9,
                    dynamic_friction=0.8,
                    restitution=0.1,
                    color=(0.1, 0.2, 0.3),
                ),
            )
        ],
        humanoid_motion_id=1,
    )
    path = str(tmp_path / "scenes" / "boxes.pt")
    SceneLib.save_scenes_to_file([unpaired, paired], path)
    return path


def test_resolve_scene_index_pairs_by_humanoid_motion_id(tmp_path):
    """The scene authored for clip 1 is the one clip 1 gets."""
    scenes_file = _write_scenes(tmp_path)
    assert resolve_scene_index(scenes_file, motion_index=1) == 1


def test_resolve_scene_index_falls_back_to_zero(tmp_path):
    """An unpaired clip still gets a scene rather than an error."""
    scenes_file = _write_scenes(tmp_path)
    assert resolve_scene_index(scenes_file, motion_index=7) == 0


def test_resolve_scene_index_honours_explicit_override(tmp_path):
    scenes_file = _write_scenes(tmp_path)
    assert resolve_scene_index(scenes_file, motion_index=1, explicit=0) == 0


def test_resolve_scene_index_rejects_out_of_range_override(tmp_path):
    scenes_file = _write_scenes(tmp_path)
    with pytest.raises(ValueError, match="out of range"):
        resolve_scene_index(scenes_file, motion_index=0, explicit=5)


def test_resolve_scene_index_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_scene_index(str(tmp_path / "nope.pt"), motion_index=0)


def test_build_scene_lib_is_single_scene_at_the_origin(tmp_path):
    """One scene, one env, and no terrain offset.

    The zero offset is the load-bearing part: the driver spawns its robot in the
    motion's own frame, so an offset scene would put the objects somewhere the
    robot never reaches.
    """
    scenes_file = _write_scenes(tmp_path)
    scene_lib = build_scene_lib(scenes_file, scene_index=1)

    assert scene_lib.num_scenes() == 1
    assert scene_lib.num_objects_per_scene == 1
    assert scene_lib.scene_offsets == [(0.0, 0.0)]
    assert scene_lib.scenes[0].humanoid_motion_id == 1


def test_build_scene_lib_pose_matches_the_authored_translation(tmp_path):
    """``get_scene_pose`` returns the scene's own coordinates, unshifted."""
    scenes_file = _write_scenes(tmp_path)
    scene_lib = build_scene_lib(scenes_file, scene_index=1)

    state = scene_lib.get_scene_pose(
        torch.tensor([0]), torch.tensor([0.0]), respawn_offset=0.0
    )
    assert state.root_pos.shape == (1, 1, 3)
    assert state.root_rot.shape == (1, 1, 4)
    np.testing.assert_allclose(
        state.root_pos[0, 0].numpy(), np.array([0.3, -0.2, 0.8]), atol=1e-6
    )


def test_respawn_offset_skips_objects_without_motion(tmp_path):
    """``respawn_offset`` lifts only objects that carry a trajectory.

    This is ``SceneLib._is_static_object`` -- "has no motion data" -- and not
    ``fix_base_link``. Both boxes here are single-frame, so neither is lifted
    even though only one is ``fix_base_link=True``: conflating the two flags
    would show up right here.
    """
    scenes_file = _write_scenes(tmp_path)
    scene_lib = build_scene_lib(scenes_file, scene_index=1)

    lifted = scene_lib.get_scene_pose(
        torch.tensor([0]), torch.tensor([0.0]), respawn_offset=0.5
    )
    assert float(lifted.root_pos[0, 0, 2]) == pytest.approx(0.8)


def test_scene_object_specs_reports_box_geometry_and_options(tmp_path):
    scenes_file = _write_scenes(tmp_path)
    scene_lib = build_scene_lib(scenes_file, scene_index=1)

    (spec,) = scene_object_specs(scene_lib)
    assert spec.kind == "box"
    assert spec.size == pytest.approx((0.5, 0.25, 0.15))
    assert spec.fix_base_link is True
    assert spec.mass == pytest.approx(1.5)
    # ObjectOptions forbids mass and density together, so density stays unset.
    assert spec.density is None
    assert spec.static_friction == pytest.approx(0.9)
    assert spec.dynamic_friction == pytest.approx(0.8)
    assert spec.restitution == pytest.approx(0.1)
    assert spec.color == pytest.approx((0.1, 0.2, 0.3))


def test_scene_object_specs_defaults_dynamic_and_density(tmp_path):
    """``fix_base_link=None`` means "not requested", i.e. an ordinary rigid body."""
    scenes_file = _write_scenes(tmp_path)
    scene_lib = build_scene_lib(scenes_file, scene_index=0)

    (spec,) = scene_object_specs(scene_lib)
    assert spec.fix_base_link is False
    assert spec.size == pytest.approx((0.4, 0.3, 0.2))


def test_scene_object_specs_handles_cylinders(tmp_path):
    cylinder = CylinderSceneObject(
        radius=0.15,
        height=0.6,
        translation=(0.0, 0.0, 0.3),
        rotation=IDENTITY_QUAT,
        options=ObjectOptions(density=500.0),
    )
    path = str(tmp_path / "scenes" / "cyl.pt")
    SceneLib.save_scenes_to_file([Scene(objects=[cylinder])], path)

    (spec,) = scene_object_specs(build_scene_lib(path, scene_index=0))
    assert spec.kind == "cylinder"
    assert spec.radius == pytest.approx(0.15)
    assert spec.height == pytest.approx(0.6)
    assert spec.mass is None
    assert spec.density == pytest.approx(500.0)


def test_scene_object_specs_rejects_unknown_object_type():
    """An unmapped subclass fails loudly, mirroring IsaacLab's spawner."""

    @dataclass
    class ConeSceneObject(PrimitiveSceneObject):
        radius: float = 0.2

        def calculate_dimensions(self):
            return (-0.2, 0.2, -0.2, 0.2, -0.5, 0.5)

    cone = ConeSceneObject(translation=(0.0, 0.0, 0.0), rotation=IDENTITY_QUAT)

    class _FakeSceneLib:
        scenes = [Scene(objects=[cone])]

    with pytest.raises(ValueError, match="Unsupported object type"):
        scene_object_specs(_FakeSceneLib())


# ---------------------------------------------------------------------------
# Trace schema (deployment.state_utils)
# ---------------------------------------------------------------------------


def _row(**extra):
    return make_trace_row(
        loop=0,
        frame=0,
        root_h=0.8,
        ref_h=0.8,
        anchor_rot_xyzw=np.array(IDENTITY_QUAT),
        ref_anchor_rot_xyzw=np.array(IDENTITY_QUAT),
        dof_pos=np.zeros(3),
        ref_dof_pos=np.zeros(3),
        dof_vel=np.zeros(3),
        **extra,
    )


def test_object_columns_are_absent_without_objects():
    """The shared schema is unchanged for harnesses with no scene."""
    row = _row()
    assert "obj_pos_err" not in row
    assert "obj_rot_err_deg" not in row


def test_object_columns_appear_when_supplied():
    row = _row(obj_pos_err=0.02, obj_rot_err_deg=3.5)
    assert row["obj_pos_err"] == pytest.approx(0.02)
    assert row["obj_rot_err_deg"] == pytest.approx(3.5)


def test_summarize_trace_reports_objects_only_when_every_row_has_them():
    with_objects = [_row(obj_pos_err=0.02, obj_rot_err_deg=3.5) for _ in range(3)]
    assert "mean obj pos err" in summarize_trace(with_objects)

    # A partially populated column would average two different definitions.
    mixed = with_objects[:2] + [_row()]
    assert "mean obj pos err" not in summarize_trace(mixed)
    assert "mean joint err" in summarize_trace(mixed)


def test_quat_angle_deg_xyzw_identity_and_known_rotation():
    identity = np.array(IDENTITY_QUAT)
    assert quat_angle_deg_xyzw(identity, identity) == pytest.approx(0.0, abs=1e-4)

    half = np.radians(90.0) / 2.0
    yaw90 = np.array([0.0, 0.0, np.sin(half), np.cos(half)])
    assert quat_angle_deg_xyzw(identity, yaw90) == pytest.approx(90.0, abs=1e-3)


def test_quat_angle_deg_xyzw_is_double_cover_safe():
    """``q`` and ``-q`` are the same rotation; the metric must agree."""
    q = np.array([0.1, 0.2, 0.3, 0.927])
    q = q / np.linalg.norm(q)
    assert quat_angle_deg_xyzw(q, -q) == pytest.approx(0.0, abs=1e-3)
