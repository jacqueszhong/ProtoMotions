# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for body-mass domain-randomization inertia scaling.

Newton's body-mass randomization scales each randomized body's inertia by the
same ratio as its mass. Newton stores a full 3x3 inertia matrix per body, so
scaling with a plainly-unsqueezed ``[num_envs, 1]`` ratio raised
``RuntimeError: The size of tensor a (3) must match the size of tensor b
(<num_envs>) at non-singleton dimension 1`` at simulator init. The shared
``scale_inertia_for_mass_change`` helper broadcasts correctly for both the
full-matrix and principal-moment layouts; these tests pin that behavior without
needing Newton or a GPU.
"""

import pytest
import torch

from protomotions.simulator.base_simulator.utils import scale_inertia_for_mass_change


def test_scales_full_3x3_inertia_matrix():
    """The 3x3 layout is the shape that used to crash."""
    num_envs = 8
    inertia = torch.randn(num_envs, 3, 3)
    mass_ratio = torch.rand(num_envs) + 0.5  # in (0.5, 1.5), strictly positive

    scaled = scale_inertia_for_mass_change(inertia, mass_ratio)

    assert scaled.shape == inertia.shape
    # Every component of a given env's matrix scales by that env's ratio.
    torch.testing.assert_close(scaled, inertia * mass_ratio[:, None, None])


def test_scales_principal_moment_inertia():
    """A 3-vector (diagonal) inertia layout still broadcasts correctly."""
    num_envs = 8
    inertia = torch.randn(num_envs, 3)
    mass_ratio = torch.rand(num_envs) + 0.5

    scaled = scale_inertia_for_mass_change(inertia, mass_ratio)

    torch.testing.assert_close(scaled, inertia * mass_ratio[:, None])


def test_inertia_is_linear_in_mass():
    """Doubling the mass ratio doubles the inertia (physical invariant)."""
    inertia = torch.randn(4, 3, 3)
    ones = torch.ones(4)

    torch.testing.assert_close(scale_inertia_for_mass_change(inertia, ones), inertia)
    torch.testing.assert_close(
        scale_inertia_for_mass_change(inertia, 2.0 * ones), 2.0 * inertia
    )


def test_matches_newton_per_body_indexing():
    """Mirror Newton's usage: scale one body's inertia in a batched buffer."""
    num_envs, num_bodies = 6, 5
    link_idx = 2
    inertia = torch.randn(num_envs, num_bodies, 3, 3)
    original = inertia.clone()
    mass_ratio = torch.rand(num_envs) + 0.5

    inertia[:, link_idx] = scale_inertia_for_mass_change(
        inertia[:, link_idx], mass_ratio
    )

    # The targeted body scaled by its ratio; all other bodies untouched.
    torch.testing.assert_close(
        inertia[:, link_idx], original[:, link_idx] * mass_ratio[:, None, None]
    )
    other = [b for b in range(num_bodies) if b != link_idx]
    torch.testing.assert_close(inertia[:, other], original[:, other])


def test_naive_unsqueeze_would_fail_on_3x3():
    """Document the original bug: a single unsqueeze cannot broadcast a matrix."""
    inertia = torch.randn(8, 3, 3)
    mass_ratio = torch.rand(8) + 0.5
    with pytest.raises(RuntimeError):
        _ = inertia * mass_ratio.unsqueeze(-1)
