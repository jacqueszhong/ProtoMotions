# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Assembly of the ONNX feed dict from live robot state and motion futures.

Lives outside ``deployment/test_tracker_isaacsim.py`` for the same reason
``scene_utils`` and ``state_utils`` do: that module boots ``SimulationApp`` at
import time, so nothing in it can be unit-tested. Everything here is pure NumPy.

The one function is shared by both observation families the Isaac Sim driver
supports; see its docstring.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


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
    body_state: tuple | None = None,
    raw_action_history: np.ndarray | None = None,
    ground_height: float = 0.0,
    num_bodies: int | None = None,
) -> dict:
    """Assemble the ONNX input dict from live robot state + motion futures.

    Two observation families share this one table, selected purely by which
    semantic keys the exported YAML's ``_runtime.onnx_name_to_in_key`` asks for:

    * **reduced coordinates** (BeyondMimic, G1) -- joint state plus the anchor
      body's rotation. Cheap: only the root and anchor poses are ever read.
    * **max coordinates** (the SOMA trackers) -- every body's world pose and
      velocity, plus the same for the reference motion. Needs ``body_state``,
      which the caller reads from the articulation's link buffers.

    ``anchor_rot`` and ``root_local_ang_vel`` are passed in pre-computed rather
    than derived from a full per-body rotation array, so the reduced-coords path
    stays cheap even though the max-coords one exists.

    Args:
        body_state: ``(pos, rot, vel, ang_vel)`` in policy body order, world
            frame, from :meth:`TrackerPolicy._read_body_state`. Required only
            when the contract asks for ``current.rigid_body_*``.
        raw_action_history: ``[history_steps, num_dofs]`` of raw pre-tanh policy
            outputs, newest first. Required only for ``historical.actions``.
        ground_height: World z of the ground under the root, for
            ``root_height_obs``. Zero on the analytic ground plane.

    Raises:
        KeyError: if the contract asks for a key this function cannot supply.
            Previously a warning -- but a missing input is not a degraded run,
            it is onnxruntime being handed an incomplete feed dict.
    """
    if prev_actions is None:
        prev_actions = np.zeros(num_dofs, dtype=np.float32)

    future_anchor_rot = future_refs["body_rot"][:, anchor_body_index, :]  # [nsteps, 4]

    key_to_array: dict[str, np.ndarray] = {
        # --- reduced coordinates ---
        "current.dof_pos": dof_pos[None],
        "current.dof_vel": dof_vel[None],
        "current.anchor_rot": anchor_rot[None],
        "current.root_local_ang_vel": root_local_ang_vel[None],
        "historical.processed_actions": prev_actions[None, None],
        "mimic.future_anchor_rot": future_anchor_rot[None],
        "mimic.future_dof_pos": future_refs["dof_pos"][None],
        "mimic.future_dof_vel": future_refs["dof_vel"][None],
        # --- shared / max coordinates ---
        "mimic.future_rot": future_refs["body_rot"][None],
        "mimic.future_pos": future_refs["body_pos"][None],
        "mimic.future_vel": future_refs["body_vel"][None],
        "mimic.future_ang_vel": future_refs["body_ang_vel"][None],
        "ground_heights": np.asarray([ground_height], dtype=np.float32),
    }

    if body_state is not None:
        body_pos, body_rot, body_vel, body_ang_vel = body_state
        key_to_array["current.rigid_body_pos"] = body_pos[None]
        key_to_array["current.rigid_body_rot"] = body_rot[None]
        key_to_array["current.rigid_body_vel"] = body_vel[None]
        key_to_array["current.rigid_body_ang_vel"] = body_ang_vel[None]
        if num_bodies is None:
            num_bodies = body_pos.shape[0]
    if num_bodies is not None:
        # observe_contacts is off in every shipped config, so this is normally
        # constant-folded away and never appears as an ONNX input. Supply it
        # anyway for the configs that do observe it.
        key_to_array["body_contacts"] = np.zeros((1, num_bodies), dtype=np.float32)

    if raw_action_history is not None:
        key_to_array["historical.actions"] = raw_action_history[None]

    onnx_inputs: dict[str, np.ndarray] = {}
    missing: list = []
    for onnx_name, sem_key in onnx_name_to_key.items():
        if sem_key in key_to_array:
            onnx_inputs[onnx_name] = np.ascontiguousarray(
                key_to_array[sem_key], dtype=np.float32
            )
        else:
            missing.append((onnx_name, sem_key))
    if missing:
        raise KeyError(
            "The exported model asks for ONNX inputs this driver cannot build: "
            + ", ".join(f"{n} (key={k})" for n, k in missing)
            + ". Running the session without them would silently feed garbage. "
            "Add them to build_onnx_inputs' table."
        )
    return onnx_inputs


class ActionHistory:
    """Ring buffer of past actions, with ProtoMotions' one-step lag built in.

    Reproduces ``StateHistoryBuffer`` (``protomotions/envs/obs/state_history_buffer.py``)
    and the view the observations are actually built from. Those are two
    separate things, and conflating them is a silent off-by-one:

    ``rotate_and_update`` runs in ``BaseEnv.post_physics_step``, *after* the
    physics substeps, and writes the action that was just applied into index 0.
    ``HistoricalView.actions`` is then ``buffer.actions[:, 1:]`` -- so the
    observation deliberately **excludes** the most recent action and starts one
    step further back. Measured on a recorded IsaacLab trace: the
    ``previous_actions`` observation the policy consumes at control step ``s``
    is the action from step ``s - 2``, not ``s - 1``.

    A driver that keeps only ``steps`` slots and feeds all of them is therefore
    always one control step too fresh. Keeping ``steps + 1`` and feeding
    ``[1:]`` is the whole point of this class.

    Attributes:
        steps: Number of history steps the policy observes.
        num_dofs: Width of one action.
    """

    def __init__(self, steps: int, num_dofs: int) -> None:
        """Allocate a zero-filled buffer.

        Args:
            steps: History steps the exported contract asks for (the ONNX
                input's middle dimension).
            num_dofs: Action width.
        """
        self.steps = int(steps)
        self.num_dofs = int(num_dofs)
        # steps + 1: index 0 mirrors ProtoMotions' "current" slot, which the
        # observation view skips.
        self._buffer = np.zeros((self.steps + 1, self.num_dofs), dtype=np.float32)

    def reset(self) -> None:
        """Zero-fill, matching ``BaseEnv._reset_state_history(..., actions=None)``."""
        self._buffer[:] = 0.0

    def push(self, action: np.ndarray) -> None:
        """Insert the action just applied at index 0, dropping the oldest.

        Args:
            action: ``[num_dofs]`` action for the control step that just ran.
        """
        self._buffer = np.roll(self._buffer, 1, axis=0)
        self._buffer[0] = np.asarray(action, dtype=np.float32).reshape(self.num_dofs)

    def view(self) -> np.ndarray:
        """The observed history: ``[steps, num_dofs]``, newest first.

        Skips index 0 exactly as ``HistoricalView.actions`` skips the buffer's
        current slot.
        """
        return self._buffer[1 : self.steps + 1]
