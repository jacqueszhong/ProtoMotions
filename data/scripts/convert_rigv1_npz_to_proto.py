# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Convert rigv1 npz motion files to ProtoMotions format.

Usage:
    python data/scripts/convert_rigv1_npz_to_proto.py <input_dir> <output_dir> [options]

Example:
    python data/scripts/convert_rigv1_npz_to_proto.py \
            examples/data/rigv1/ \
            examples/data/rigv1/proto \
            --input-fps 30 \
            --output-fps 30
"""
from typing import Any, Dict, Optional
import typer
import os
from pathlib import Path
import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation as R

from protomotions.components.pose_lib import (
    compute_angular_velocity,
    compute_joint_rot_mats_from_global_mats,
    extract_kinematic_info,
    extract_qpos_from_transforms,
    fk_from_transforms_with_velocities,
)
from protomotions.simulator.base_simulator.simulator_state import RobotState
from protomotions.utils.rotations import (
    matrix_to_quaternion,
    quat_mul,
    quat_rotate,
    quaternion_to_matrix,
)

from contact_detection import compute_contact_labels_from_pos_and_vel
from keypoint_utils import extract_keypoints_from_motion, get_keypoint_indices
from motion_filter import passes_exclude_motion_filter

app = typer.Typer(pretty_exceptions_enable=True)


def gen_yaml_one_motion_default(
    output_motion_path: Path,
    fps: int,
    idx: int,
    additional_fields: Optional[Dict[str, Any]] = None,
):
    result = {
        "file": str(output_motion_path),
        "fps": fps,
        "weight": 1.0,
        "idx": idx,
    }
    if additional_fields is not None:
        result.update(additional_fields)
    return result


def create_motion_from_rigv1_data(
    global_rot_mats: torch.Tensor,  # [T, 27, 3, 3]
    root_pos: torch.Tensor,  # [T, 3]
    kinematic_info,
    fps: int,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> RobotState:
    """
    Create a RobotState motion from Rigv1 global rotation matrices and root position.

    Args:
        global_rot_mats: Global rotation matrices [T, 27, 3, 3]
        root_pos: Root position [T, 3]
        kinematic_info: Kinematic information for the robot
        fps: Frame rate
        device: Torch device
        dtype: Torch dtype

    Returns:
        RobotState motion object
    """
    # Convert to torch tensors if not already
    if not isinstance(global_rot_mats, torch.Tensor):
        global_rot_mats = torch.from_numpy(global_rot_mats)
    if not isinstance(root_pos, torch.Tensor):
        root_pos = torch.from_numpy(root_pos)

    global_rot_mats = global_rot_mats.to(device, dtype)
    root_pos = root_pos.to(device, dtype)

    # Convert to quaternions and filter joints
    global_quat = matrix_to_quaternion(global_rot_mats, w_last=True)
    # for removing 4 dead joints in the Rigv1 raw data
    joints_in_xml_order_idx = [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        13,
        14,
        15,
        16,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
    ]
    global_quat = global_quat[:, joints_in_xml_order_idx]
    assert global_quat.shape[1] == 23

    # Apply coordinate system transformations
    # rot1 transforms motion rotations represented with a y-up character to a z-up character
    # but did not change the motion itself
    # rot2 changes the motion from y-up to z-up
    rot1 = R.from_euler("xyz", np.array([-np.pi / 2, 0, 0]), degrees=False)
    rot2 = R.from_euler("xyz", np.array([-np.pi / 2, np.pi, 0]), degrees=False)

    rot1_quat = (
        torch.from_numpy(rot1.as_quat())
        .to(device, dtype)
        .expand(global_quat.shape[0], -1)
    )
    rot2_quat = (
        torch.from_numpy(rot2.as_quat())
        .to(device, dtype)
        .expand(global_quat.shape[0], -1)
    )

    for i in range(0, 23):
        global_quat[:, i, :] = quat_mul(global_quat[:, i, :], rot1_quat, w_last=True)
        global_quat[:, i, :] = quat_mul(rot2_quat, global_quat[:, i, :], w_last=True)

    root_pos = quat_rotate(rot2_quat, root_pos, w_last=True)

    # compute vels of local rotation
    local_rot_mats = compute_joint_rot_mats_from_global_mats(
        kinematic_info=kinematic_info,
        global_rot_mats=quaternion_to_matrix(global_quat, w_last=True),
    )

    motion = fk_from_transforms_with_velocities(
        kinematic_info=kinematic_info,
        root_pos=root_pos,
        joint_rot_mats=local_rot_mats,
        fps=fps,
        compute_velocities=True,
        velocity_max_horizon=3,  # Use multi-horizon minimum for noise-filtered velocities
    )
    # caching local rotation to disk file, in case anyone needs it later
    motion.local_rigid_body_rot = matrix_to_quaternion(local_rot_mats, w_last=True)

    # calc dof pos and dof vel
    qpos = extract_qpos_from_transforms(
        kinematic_info=kinematic_info,
        root_pos=root_pos,
        joint_rot_mats=local_rot_mats,
        multi_dof_decomposition_method="exp_map",
    )
    motion.dof_pos = qpos[:, 7:]  # (T, 22 * 3)

    local_angular_vels = compute_angular_velocity(
        batched_robot_rot_mats=local_rot_mats[:, 1:, :, :],
        fps=fps,
    )
    assert local_angular_vels.shape[1] == 22  # (T, 22, 3)
    # because we know all joints are 3 dof exp_map joints...
    motion.dof_vel = local_angular_vels.reshape(-1, 22 * 3)

    # compute contacts using position and velocity thresholds
    motion.rigid_body_contacts = compute_contact_labels_from_pos_and_vel(
        positions=motion.rigid_body_pos,
        velocity=motion.rigid_body_vel,
        vel_thres=0.15,
        height_thresh=0.1,
    ).to(torch.bool)

    return motion


@app.command()
def main(
    input_dir: Path,
    output_dir: Path,
    input_fps: int = 120,
    output_fps: int = 30,
    # Motion filter options
    ignore_motion_filter: bool = False,
    min_height_threshold: float = -0.05,
    max_velocity_threshold: float = 15.0,
    max_dof_vel_threshold: float = 40.0,
    duration_height_filter: float = 0.2,
    duration_height_seconds: float = 1.0,
    # Output options
    yaml_output_name: Optional[str] = None,
    extract_keypoints: bool = False,
    keypoints_output_path: Optional[Path] = None,
    force_remake: bool = False,
):
    """Convert rigv1 npz motion files to ProtoMotions format."""
    if yaml_output_name is not None:
        yaml_output = output_dir / yaml_output_name
    else:
        yaml_output = None

    device = torch.device("cpu")
    dtype = torch.float32

    kinematic_info = extract_kinematic_info(
        "protomotions/data/assets/mjcf/rigv1_humanoid.xml"
    )
    print("kinematic_info", kinematic_info)
    assert kinematic_info.num_bodies == 23
    assert kinematic_info.nq == 22 * 3 + 7

    if extract_keypoints and keypoints_output_path is None:
        raise typer.Exit(
            "Error: --keypoints-output-path must be provided when --extract-keypoints is enabled."
        )

    if extract_keypoints:
        os.makedirs(keypoints_output_path, exist_ok=True)
        print(f"Keypoints will be saved to: {keypoints_output_path}")
        conceptual_keypoint_names, _, keypoint_indices_in_mjcf = get_keypoint_indices(
            kinematic_info
        )

    output_motions_yaml = []
    output_yaml_idx = 0

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_fps % output_fps != 0:
        raise ValueError(
            f"input_fps ({input_fps}) must be divisible by output_fps ({output_fps})"
        )

    # Process all .npz files in input_dir (flat, no recursion)
    npz_files = sorted(input_dir.glob("*.npz"))
    print(f"Found {len(npz_files)} npz files in {input_dir}")

    for npz_file in npz_files:
        file_name = npz_file.stem + ".motion"
        output_file = output_dir / file_name

        if not force_remake and output_file.exists():
            print(f"Skipping {file_name} because it already exists")
            continue

        print(f"Processing {npz_file}")

        data = np.load(npz_file, allow_pickle=True)

        # Check required fields
        if "global_rot_mats" not in data or "root_positions" not in data:
            raise ValueError(f"{npz_file} doesn't contain required fields 'global_rot_mats' and 'root_positions'")

        # Load and squeeze batch dimension (assume first dim is size 1)
        global_rot_mats = data["global_rot_mats"]
        root_pos = data["root_positions"]

        if global_rot_mats.ndim == 5:
            global_rot_mats = global_rot_mats[0]  # (1, T, 27, 3, 3) -> (T, 27, 3, 3)
        if root_pos.ndim == 3:
            root_pos = root_pos[0]  # (1, T, 3) -> (T, 3)

        print(f"  global_rot_mats: {global_rot_mats.shape}")
        print(f"  root_pos: {root_pos.shape}")

        # Validate shapes
        if (
            global_rot_mats.ndim != 4
            or global_rot_mats.shape[1] != 27
            or global_rot_mats.shape[2:] != (3, 3)
        ):
            print(
                f"Skipping {npz_file} because global_rot_mats has wrong shape: "
                f"{global_rot_mats.shape}, expected: [T, 27, 3, 3]"
            )
            continue

        if root_pos.ndim != 2 or root_pos.shape[1] != 3:
            print(
                f"Skipping {npz_file} because root_pos has wrong shape: "
                f"{root_pos.shape}, expected: [T, 3]"
            )
            continue

        if global_rot_mats.shape[0] != root_pos.shape[0]:
            print(
                f"Skipping {npz_file} because global_rot_mats and root_pos "
                "have different sequence lengths"
            )
            continue


        downsample_factor = input_fps // output_fps
        global_rot_mats = global_rot_mats[::downsample_factor]
        root_pos = root_pos[::downsample_factor]

        # Create motion using helper function
        motion = create_motion_from_rigv1_data(
            global_rot_mats=global_rot_mats,
            root_pos=root_pos,
            kinematic_info=kinematic_info,
            fps=output_fps,
            device=device,
            dtype=dtype,
        )

        # Apply motion filtering
        if not ignore_motion_filter and not passes_exclude_motion_filter(
            motion,
            min_height_threshold=min_height_threshold,
            max_velocity_threshold=max_velocity_threshold,
            max_dof_vel_threshold=max_dof_vel_threshold,
            duration_height_filter=duration_height_filter,
            duration_height_seconds=duration_height_seconds,
        ):
            print(f"Skipping {npz_file.name} because it does not pass motion filter")
            continue

        # Extract and save keypoints for pyroki retargeting if enabled
        if extract_keypoints:
            keypoint_output_file = keypoints_output_path / f"{npz_file.stem}_keypoints.npy"

            if not force_remake and keypoint_output_file.exists():
                print(f"Skipping keypoint extraction for {npz_file.name}, file already exists.")
            else:
                keypoint_data = extract_keypoints_from_motion(
                    all_body_positions=motion.rigid_body_pos,
                    all_body_rotations_quat=motion.rigid_body_rot,
                    keypoint_indices_in_mjcf=keypoint_indices_in_mjcf,
                    conceptual_keypoint_names=conceptual_keypoint_names,
                    device=device,
                    flat_feet=True,
                    aux_points=True,
                    contacts=motion.rigid_body_contacts,
                    kinematic_info=kinematic_info,
                )
                keypoint_data_to_save = {
                    "positions": keypoint_data["positions"].cpu().numpy(),
                    "orientations": keypoint_data["orientations"].cpu().numpy(),
                    "left_foot_contacts": keypoint_data["left_foot_contacts"],
                    "right_foot_contacts": keypoint_data["right_foot_contacts"],
                }
                np.save(str(keypoint_output_file), keypoint_data_to_save)
                print(f"Saved keypoints to {keypoint_output_file}")

        # Save motion
        torch.save(motion.to_dict(), str(output_file))
        print(f"Saved to {output_file}")

        if yaml_output is not None:
            output_motions_yaml.append(
                gen_yaml_one_motion_default(file_name, output_fps, output_yaml_idx)
            )
            output_yaml_idx += 1

    if yaml_output is not None:
        with open(yaml_output, "w") as f:
            yaml.dump({"motions": output_motions_yaml}, f)
        print(f"Saved motions list to {yaml_output}")


if __name__ == "__main__":
    app()

