# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``deployment.obs_assembly`` and the reference-realignment helpers.

Simulator-free by construction, like ``test_deployment_scene_utils.py``:
``deployment/test_tracker_isaacsim.py`` boots ``SimulationApp`` at import time
and cannot be imported here, which is exactly why the ONNX feed-dict assembly
lives in its own module.

Two observation families share one assembly table -- BeyondMimic-style reduced
coordinates (the G1 deploy tracker) and max coordinates (the SOMA trackers) --
selected purely by the semantic keys the exported contract asks for. These
tests pin that dispatch, the shapes, and the hard failure on an unsatisfiable
contract.
"""

from __future__ import annotations

import numpy as np
import pytest

from deployment.obs_assembly import build_onnx_inputs
from deployment.state_utils import (
    apply_heading_offset_to_positions_np,
    compute_yaw_offset_np,
    quat_rotate_np,
)

NUM_DOFS = 66
NUM_BODIES = 23
NUM_FUTURE = 1

# The 10 live ONNX inputs of the SOMA max-coordinate trackers, exactly as
# `_runtime.onnx_name_to_in_key` records them.
SOMA_CONTRACT = {
    "current_rigid_body_pos": "current.rigid_body_pos",
    "current_rigid_body_rot": "current.rigid_body_rot",
    "current_rigid_body_vel": "current.rigid_body_vel",
    "current_rigid_body_ang_vel": "current.rigid_body_ang_vel",
    "ground_heights": "ground_heights",
    "historical_actions": "historical.actions",
    "mimic_future_pos": "mimic.future_pos",
    "mimic_future_rot": "mimic.future_rot",
    "mimic_future_vel": "mimic.future_vel",
    "mimic_future_ang_vel": "mimic.future_ang_vel",
}

# The BeyondMimic G1 contract, for the same table.
G1_CONTRACT = {
    "current_dof_pos": "current.dof_pos",
    "current_dof_vel": "current.dof_vel",
    "current_anchor_rot": "current.anchor_rot",
    "current_root_local_ang_vel": "current.root_local_ang_vel",
    "historical_processed_actions": "historical.processed_actions",
    "mimic_future_anchor_rot": "mimic.future_anchor_rot",
    "mimic_future_dof_pos": "mimic.future_dof_pos",
    "mimic_future_dof_vel": "mimic.future_dof_vel",
}


def _future_refs(num_bodies=NUM_BODIES, num_dofs=NUM_DOFS, steps=NUM_FUTURE):
    rng = np.random.default_rng(0)
    return {
        "body_pos": rng.normal(size=(steps, num_bodies, 3)).astype(np.float32),
        "body_rot": rng.normal(size=(steps, num_bodies, 4)).astype(np.float32),
        "body_vel": rng.normal(size=(steps, num_bodies, 3)).astype(np.float32),
        "body_ang_vel": rng.normal(size=(steps, num_bodies, 3)).astype(np.float32),
        "dof_pos": rng.normal(size=(steps, num_dofs)).astype(np.float32),
        "dof_vel": rng.normal(size=(steps, num_dofs)).astype(np.float32),
    }


def _body_state(num_bodies=NUM_BODIES):
    rng = np.random.default_rng(1)
    return (
        rng.normal(size=(num_bodies, 3)).astype(np.float32),
        rng.normal(size=(num_bodies, 4)).astype(np.float32),
        rng.normal(size=(num_bodies, 3)).astype(np.float32),
        rng.normal(size=(num_bodies, 3)).astype(np.float32),
    )


def _common_kwargs():
    return dict(
        dof_pos=np.zeros(NUM_DOFS, dtype=np.float32),
        dof_vel=np.zeros(NUM_DOFS, dtype=np.float32),
        anchor_rot=np.array([0, 0, 0, 1], dtype=np.float32),
        root_local_ang_vel=np.zeros(3, dtype=np.float32),
        future_refs=_future_refs(),
        anchor_body_index=0,
        num_dofs=NUM_DOFS,
    )


class TestMaxCoordsContract:
    """The SOMA family: per-body world state plus full-body references."""

    def test_produces_every_requested_input_with_the_right_shape(self):
        raw_history = np.zeros((1, NUM_DOFS), dtype=np.float32)
        out = build_onnx_inputs(
            onnx_name_to_key=SOMA_CONTRACT,
            body_state=_body_state(),
            raw_action_history=raw_history,
            num_bodies=NUM_BODIES,
            **_common_kwargs(),
        )
        assert set(out) == set(SOMA_CONTRACT)
        assert out["current_rigid_body_pos"].shape == (1, NUM_BODIES, 3)
        assert out["current_rigid_body_rot"].shape == (1, NUM_BODIES, 4)
        assert out["current_rigid_body_vel"].shape == (1, NUM_BODIES, 3)
        assert out["current_rigid_body_ang_vel"].shape == (1, NUM_BODIES, 3)
        assert out["ground_heights"].shape == (1,)
        assert out["historical_actions"].shape == (1, 1, NUM_DOFS)
        assert out["mimic_future_pos"].shape == (1, NUM_FUTURE, NUM_BODIES, 3)
        assert out["mimic_future_rot"].shape == (1, NUM_FUTURE, NUM_BODIES, 4)
        assert all(v.dtype == np.float32 for v in out.values())

    def test_body_state_is_passed_through_unmodified(self):
        pos, rot, vel, ang_vel = _body_state()
        out = build_onnx_inputs(
            onnx_name_to_key=SOMA_CONTRACT,
            body_state=(pos, rot, vel, ang_vel),
            raw_action_history=np.zeros((1, NUM_DOFS), dtype=np.float32),
            num_bodies=NUM_BODIES,
            **_common_kwargs(),
        )
        # The obs kernels do their own root-relative normalization, so the
        # driver must hand over raw world-frame state.
        np.testing.assert_array_equal(out["current_rigid_body_pos"][0], pos)
        np.testing.assert_array_equal(out["current_rigid_body_ang_vel"][0], ang_vel)

    def test_ground_height_is_carried_through(self):
        out = build_onnx_inputs(
            onnx_name_to_key=SOMA_CONTRACT,
            body_state=_body_state(),
            raw_action_history=np.zeros((1, NUM_DOFS), dtype=np.float32),
            num_bodies=NUM_BODIES,
            ground_height=0.25,
            **_common_kwargs(),
        )
        assert out["ground_heights"] == pytest.approx(0.25)

    def test_multi_step_action_history_keeps_its_depth(self):
        history = np.arange(3 * NUM_DOFS, dtype=np.float32).reshape(3, NUM_DOFS)
        out = build_onnx_inputs(
            onnx_name_to_key=SOMA_CONTRACT,
            body_state=_body_state(),
            raw_action_history=history,
            num_bodies=NUM_BODIES,
            **_common_kwargs(),
        )
        assert out["historical_actions"].shape == (1, 3, NUM_DOFS)
        np.testing.assert_array_equal(out["historical_actions"][0], history)


class TestReducedCoordsContract:
    """The BeyondMimic family must keep working with no per-body state at all."""

    def test_assembles_without_body_state(self):
        prev = np.full(NUM_DOFS, 0.5, dtype=np.float32)
        kwargs = _common_kwargs()
        kwargs["num_dofs"] = NUM_DOFS
        out = build_onnx_inputs(
            onnx_name_to_key=G1_CONTRACT, prev_actions=prev, **kwargs
        )
        assert set(out) == set(G1_CONTRACT)
        assert out["current_dof_pos"].shape == (1, NUM_DOFS)
        assert out["historical_processed_actions"].shape == (1, 1, NUM_DOFS)
        np.testing.assert_array_equal(out["historical_processed_actions"][0, 0], prev)

    def test_anchor_rot_is_sliced_from_the_reference_body_rotations(self):
        kwargs = _common_kwargs()
        kwargs["anchor_body_index"] = 16
        out = build_onnx_inputs(onnx_name_to_key=G1_CONTRACT, **kwargs)
        np.testing.assert_array_equal(
            out["mimic_future_anchor_rot"][0], kwargs["future_refs"]["body_rot"][:, 16]
        )

    def test_missing_prev_actions_defaults_to_zeros(self):
        out = build_onnx_inputs(onnx_name_to_key=G1_CONTRACT, **_common_kwargs())
        assert not out["historical_processed_actions"].any()


class TestUnsatisfiableContract:
    """A key the driver cannot build must fail, not warn.

    onnxruntime is happy to run with an incomplete feed dict; the policy then
    sees garbage and the run looks merely bad rather than broken.
    """

    def test_raises_naming_the_missing_key(self):
        contract = dict(SOMA_CONTRACT)
        contract["some_new_input"] = "current.not_a_real_field"
        with pytest.raises(KeyError) as excinfo:
            build_onnx_inputs(
                onnx_name_to_key=contract,
                body_state=_body_state(),
                raw_action_history=np.zeros((1, NUM_DOFS), dtype=np.float32),
                num_bodies=NUM_BODIES,
                **_common_kwargs(),
            )
        assert "current.not_a_real_field" in str(excinfo.value)

    def test_max_coords_contract_without_body_state_raises(self):
        with pytest.raises(KeyError) as excinfo:
            build_onnx_inputs(
                onnx_name_to_key=SOMA_CONTRACT,
                raw_action_history=np.zeros((1, NUM_DOFS), dtype=np.float32),
                **_common_kwargs(),
            )
        assert "current.rigid_body_pos" in str(excinfo.value)


class TestReferenceRealignment:
    """Rigid realignment of the reference into the robot's frame.

    Rotation-only alignment is enough for reduced coordinates, which read the
    anchor's orientation and nothing else. Max-coordinate target poses
    difference reference *positions* against the live root in world frame, so
    the reference has to be moved as a rigid body.
    """

    def test_identity_offset_with_coincident_pivots_is_a_no_op(self):
        rng = np.random.default_rng(3)
        pos = rng.normal(size=(2, NUM_BODIES, 3)).astype(np.float32)
        pivot = pos[0, 0]
        identity = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        out = apply_heading_offset_to_positions_np(identity, pos, pivot, pivot)
        np.testing.assert_allclose(out, pos, atol=1e-6)

    def test_motion_pivot_maps_onto_the_robot_pivot(self):
        # 90 deg about z.
        q = np.array([0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)], dtype=np.float32)
        motion_pivot = np.array([1.0, 2.0, 0.9], dtype=np.float32)
        robot_pivot = np.array([-3.0, 0.5, 0.9], dtype=np.float32)
        out = apply_heading_offset_to_positions_np(
            q, motion_pivot[None, None, :], motion_pivot, robot_pivot
        )
        np.testing.assert_allclose(out[0, 0], robot_pivot, atol=1e-6)

    def test_realignment_is_rigid(self):
        rng = np.random.default_rng(4)
        q = np.array([0.0, 0.0, np.sin(0.35), np.cos(0.35)], dtype=np.float32)
        pos = rng.normal(size=(3, NUM_BODIES, 3)).astype(np.float32)
        motion_pivot = np.array([1.0, 2.0, 0.9], dtype=np.float32)
        robot_pivot = np.array([-3.0, 0.5, 0.9], dtype=np.float32)
        out = apply_heading_offset_to_positions_np(q, pos, motion_pivot, robot_pivot)
        before = np.linalg.norm(pos[:, 1:] - pos[:, :-1], axis=-1)
        after = np.linalg.norm(out[:, 1:] - out[:, :-1], axis=-1)
        np.testing.assert_allclose(before, after, atol=1e-5)

    def test_yaw_offset_composes_back_to_identity(self):
        # compute_yaw_offset_np(robot, motion) must map motion onto robot's yaw.
        robot = np.array([0.0, 0.0, np.sin(0.6), np.cos(0.6)], dtype=np.float32)
        motion = np.array([0.0, 0.0, np.sin(-0.2), np.cos(-0.2)], dtype=np.float32)
        offset = compute_yaw_offset_np(robot, motion)
        forward = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        np.testing.assert_allclose(
            quat_rotate_np(offset, quat_rotate_np(motion, forward)),
            quat_rotate_np(robot, forward),
            atol=1e-5,
        )

    def test_quat_rotate_np_matches_a_known_rotation(self):
        # 90 deg about z sends +x to +y.
        q = np.array([0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)], dtype=np.float64)
        out = quat_rotate_np(q, np.array([[1.0, 0.0, 0.0]]))
        np.testing.assert_allclose(out[0], [0.0, 1.0, 0.0], atol=1e-7)
