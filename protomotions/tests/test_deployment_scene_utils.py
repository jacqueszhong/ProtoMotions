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

import importlib
import sys
from dataclasses import dataclass

import numpy as np
import pytest
import torch

from deployment.scene_utils import (
    build_scene_lib,
    load_resolved_configs,
    resolve_init_z_offset,
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


# ---------------------------------------------------------------------------
# load_resolved_configs
# ---------------------------------------------------------------------------
#
# A resolved-configs .pt pickles the agent config too, whose module chain
# reaches ``lightning``. The deployment interpreter carries only runtime
# dependencies, so a plain ``torch.load`` dies there and takes ``--ground
# trimesh`` down with it even though only ``terrain`` is ever read. These tests
# pin the leniency that fixes it, standing in for that split with a module that
# is importable at save time and gone at load time.


def _save_with_throwaway_module(tmp_path, module_name, key="gone", attrs=None):
    """``torch.save`` a payload whose class then becomes unimportable.

    Args:
        tmp_path: pytest tmp dir; doubles as the throwaway module's home.
        module_name: Unique per test -- the module is removed from ``sys.modules``
            afterwards, so reusing a name across tests would let one test's class
            stay importable for the next.
        key: Which config key the unpicklable object is stored under.
        attrs: Extra attributes to set on it, for callers that read a field off
            the stub rather than just checking that it survived.
    """
    (tmp_path / f"{module_name}.py").write_text(
        "class Vanishing:\n    def __init__(self, tag):\n        self.tag = tag\n"
    )
    sys.path.insert(0, str(tmp_path))
    try:
        module = importlib.import_module(module_name)
        obj = module.Vanishing("kept")
        for name, value in (attrs or {}).items():
            setattr(obj, name, value)
        payload = {key: obj, "kept": {"terrain": 1.5}}
        out = tmp_path / "resolved_configs_inference.pt"
        torch.save(payload, out)
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop(module_name, None)
    return out


def test_load_resolved_configs_survives_an_unimportable_class(tmp_path):
    path = _save_with_throwaway_module(tmp_path, "vanishing_cfg_survives")

    with pytest.raises(ModuleNotFoundError):
        torch.load(path, map_location="cpu", weights_only=False)

    assert load_resolved_configs(str(path))["kept"] == {"terrain": 1.5}


def test_load_resolved_configs_keeps_the_pickled_state_on_the_stub(tmp_path):
    path = _save_with_throwaway_module(tmp_path, "vanishing_cfg_state")

    stub = load_resolved_configs(str(path))["gone"]

    # A placeholder, not the real class -- but not lossy: what was pickled is
    # still readable, so a caller that does reach into it can say what it lost.
    assert type(stub).__name__ == "Vanishing"
    assert stub.tag == "kept"
    assert "unavailable" in repr(stub)


def test_load_resolved_configs_leaves_importable_entries_as_real_types(tmp_path):
    """Only the unreachable subtrees degrade; everything else round-trips."""
    path = tmp_path / "resolved_configs.pt"
    torch.save({"terrain": ObjectOptions(static_friction=0.75), "n": 3}, path)

    resolved = load_resolved_configs(str(path))

    assert isinstance(resolved["terrain"], ObjectOptions)
    assert resolved["terrain"].static_friction == 0.75
    assert resolved["n"] == 3


def test_load_resolved_configs_rejects_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_resolved_configs(str(tmp_path / "nope.pt"))


# ---------------------------------------------------------------------------
# resolve_init_z_offset
#
# The spawn lift is the one number that decides whether a seated clip starts in
# contact with its seat or 5 cm above it, and it lives only in the run's
# resolved configs -- never in the ONNX export -- so the "auto" path has to keep
# working through the lenient unpickler and has to be loud when it cannot.
# ---------------------------------------------------------------------------


@dataclass
class _EnvCfg:
    ref_respawn_offset: float = 0.05


def test_resolve_init_z_offset_reads_the_training_value(tmp_path):
    path = tmp_path / "resolved_configs_inference.pt"
    torch.save({"env": _EnvCfg(0.05), "terrain": None}, path)

    assert resolve_init_z_offset("auto", str(path)) == pytest.approx(0.05)


def test_resolve_init_z_offset_survives_an_unimportable_env_config(tmp_path):
    """The env config's module chain is absent from the deployment interpreter.

    The stub still carries the pickled ``__dict__``, so the offset is readable
    even though the class is not -- which is the whole reason ``auto`` can work
    from ``isaacsim/python.sh``.
    """
    path = _save_with_throwaway_module(
        tmp_path,
        "vanishing_env_cfg",
        key="env",
        attrs={"ref_respawn_offset": 0.07},
    )

    assert resolve_init_z_offset("auto", str(path)) == pytest.approx(0.07)


def test_resolve_init_z_offset_falls_back_to_flush_without_configs(caplog):
    with caplog.at_level("WARNING"):
        assert resolve_init_z_offset("auto", None) == 0.0
    assert "--resolved-configs" in caplog.text


def test_resolve_init_z_offset_falls_back_when_the_key_is_absent(tmp_path, caplog):
    path = tmp_path / "resolved_configs.pt"
    torch.save({"terrain": None}, path)

    with caplog.at_level("WARNING"):
        assert resolve_init_z_offset("auto", str(path)) == 0.0
    assert "ref_respawn_offset" in caplog.text


def test_resolve_init_z_offset_passes_an_explicit_value_through():
    assert resolve_init_z_offset("0.05", "/does/not/exist.pt") == pytest.approx(0.05)
    assert resolve_init_z_offset(0.0, None) == 0.0


def test_resolve_init_z_offset_rejects_nonsense():
    with pytest.raises(ValueError):
        resolve_init_z_offset("high", None)
