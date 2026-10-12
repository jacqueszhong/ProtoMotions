# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
import queue

import pytest
import torch

from protomotions.simulator.mujoco.keyboard import (
    MujocoKeyboardWindow,
    _printable_key,
)
from protomotions.simulator.mujoco.simulator import MujocoSimulator


def test_mujoco_rejects_body_mass_randomization_before_initialization():
    config = SimpleNamespace(
        domain_randomization=SimpleNamespace(body_mass=object()),
    )

    with pytest.raises(
        NotImplementedError,
        match="MuJoCo does not support body-mass domain randomization",
    ):
        MujocoSimulator(
            config=config,
            robot_config=None,
            terrain=None,
            device=torch.device("cpu"),
            scene_lib=None,
        )


def _window_with_events(*events):
    window = object.__new__(MujocoKeyboardWindow)
    window._events = queue.Queue()
    window._process = SimpleNamespace(is_alive=lambda: True)
    for event in events:
        window._events.put(event)
    return window


def test_keyboard_window_drains_key_transitions_without_gui():
    window = _window_with_events(("w", True), ("w", False))

    assert window.drain_events() == [("w", True), ("w", False)]
    assert window.drain_events() == []


@pytest.mark.parametrize(
    "char, keysym, expected",
    [("z", "z", "z"), ("", "Z", "z"), ("", "Shift_L", "")],
)
def test_keyboard_window_uses_keysym_when_tk_release_has_no_char(
    char, keysym, expected
):
    assert _printable_key(char, keysym) == expected


def test_simulator_forwards_keyboard_events_to_user_interface():
    simulator = object.__new__(MujocoSimulator)
    simulator._keyboard_window = SimpleNamespace(
        drain_events=lambda: [("w", True), ("w", False)]
    )
    forwarded = []
    simulator.user_interface = SimpleNamespace(
        handle_key_event=lambda key, *, pressed: forwarded.append((key, pressed))
    )

    simulator._poll_keyboard_events()

    assert forwarded == [("w", True), ("w", False)]


@pytest.mark.parametrize(
    "event, message",
    [
        (("__window_error__", "tk failed"), "tk failed"),
        (("__window_closed__", False), "was closed"),
    ],
)
def test_keyboard_window_reports_child_failures(event, message):
    window = _window_with_events(event)

    with pytest.raises(RuntimeError, match=message):
        window.drain_events()
