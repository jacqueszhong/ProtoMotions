# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for MuJoCo viewer-camera video capture."""

from contextlib import nullcontext
from types import SimpleNamespace

import matplotlib.pyplot as plt
import mujoco
import numpy as np

from protomotions.simulator.mujoco.simulator import MujocoSimulator


def test_viewport_capture_uses_live_viewer_camera(monkeypatch, tmp_path):
    calls = []

    class FakeRenderer:
        def __init__(self, model, *, height, width):
            assert (height, width) == (480, 640)

        def update_scene(self, data, *, camera=-1):
            calls.append((data, camera))

        def render(self):
            return np.zeros((2, 2, 3), dtype=np.uint8)

        def close(self):
            pass

    monkeypatch.setattr(mujoco, "Renderer", FakeRenderer)
    monkeypatch.setattr(plt, "imsave", lambda path, pixels: None)

    camera = object()
    data = object()
    simulator = object.__new__(MujocoSimulator)
    simulator.model = object()
    simulator.data = data
    simulator.viewer = SimpleNamespace(cam=camera, lock=lambda: nullcontext())
    simulator._viewer_initialized = True

    simulator._write_viewport_to_file(str(tmp_path / "frame.png"))

    assert calls == [(data, camera)]


def test_viewport_capture_falls_back_without_viewer(monkeypatch, tmp_path):
    calls = []

    class FakeRenderer:
        def __init__(self, model, *, height, width):
            pass

        def update_scene(self, data, *, camera=-1):
            calls.append((data, camera))

        def render(self):
            return np.zeros((2, 2, 3), dtype=np.uint8)

        def close(self):
            pass

    monkeypatch.setattr(mujoco, "Renderer", FakeRenderer)
    monkeypatch.setattr(plt, "imsave", lambda path, pixels: None)

    data = object()
    simulator = object.__new__(MujocoSimulator)
    simulator.model = object()
    simulator.data = data
    simulator.viewer = None
    simulator._viewer_initialized = False

    simulator._write_viewport_to_file(str(tmp_path / "frame.png"))

    assert calls == [(data, -1)]
