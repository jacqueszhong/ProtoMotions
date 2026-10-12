# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deployment utilities for ProtoMotions policies.

This package provides a minimal deployment stack for running trained policies
outside of the full ProtoMotions training framework:

    deployment/
        state_utils.py   -- derive anchor_rot / root_local_ang_vel from raw sim state
        motion_utils.py  -- MotionPlayer: interpolation + 50fps caching
        obs_assembly.py  -- build_onnx_inputs / ActionHistory
        scene_utils.py   -- resolved-config + SceneLib loading for scene tasks
        physx_probe.py   -- PhysX property/contact introspection helpers
        export_tracker_onnx.py -- ONNX export for standard tracker policies
        export_bm_tracker_onnx.py -- ONNX export for BeyondMimic tracker policies
        test_tracker_mujoco.py -- standalone MuJoCo inference loop

Isaac Sim tracker driver
------------------------
Split into three layers so the same driver runs standalone *and* inside an
already-running Isaac Sim::

    deployment/
        tracker_runtime.py  -- the whole driver: TrackerPolicy, stage/physics
                               authoring, TrackerRuntimeConfig/TrackerRuntime/
                               build_runtime. No argparse, no SimulationApp, no
                               logging.basicConfig, no stepping loop, so it is
                               importable inside a live Kit app. Its module
                               docstring is the deployment-contract reference.
        tracker_session.py  -- TrackerSession: Kit timeline + app-update
                               subscription + teardown. Kit-only.
        test_tracker_isaacsim.py -- the standalone CLI. Owns SimulationApp, the
                               root logger, the diagnostic modes and the loop.
        exts/protomotions.tracker/ -- the Kit extension (library API only); see
                               its docs/Overview.md.

Typical workflow
----------------
1. Export the ONNX model once::

    python deployment/export_tracker_onnx.py \\
        --checkpoint results/my_exp/last.ckpt \\
        --output deployment/models/

   Use ``export_bm_tracker_onnx.py`` instead for BeyondMimic checkpoints.

2. Test with MuJoCo (first run also caches the motion at 50fps)::

    python deployment/test_tracker_mujoco.py \\
        --onnx  deployment/models/unified_pipeline.onnx \\
        --motion data/motions/walk.pt \\
        --cache-motion  # write walk.50fps.pt next to walk.pt

3. Subsequent test runs are PyTorch-computation-free::

    python deployment/test_tracker_mujoco.py \\
        --onnx  deployment/models/unified_pipeline.onnx \\
        --motion data/motions/walk.50fps.pt
"""
