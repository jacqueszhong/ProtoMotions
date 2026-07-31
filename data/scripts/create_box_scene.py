# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create a scenes file containing a single box, for use with ``--scenes-file``.

A box with one pose is a prop: object-tracking rewards score it against
``scene_lib.get_scene_pose(env_ids, motion_times)``, which interpolates the
object's own motion frames, so a single-frame box has a *constant* reference and
the reward pays it for staying still. Picking such a box up scores zero. Give the
box a trajectory whenever anything is meant to track it.

Three ways to produce one:

    # 1. Static prop (obstacle, table) -- no reference trajectory.
    python data/scripts/create_box_scene.py --output data/scenes/box.pt

    # 2. Derived from a motion: box rests, is carried by the hands between two
    #    times, then stays where it was released.
    python data/scripts/create_box_scene.py --output data/scenes/box.pt \
        --motion-file motions.pt --motion-id 0 \
        --grasp-start 1.2 --grasp-end 3.4

    # 3. From an authored trajectory: [N, 3] translations (+ optional [N, 4]
    #    XYZW rotations) in a .npy or .pt file.
    python data/scripts/create_box_scene.py --output data/scenes/box.pt \
        --trajectory-file box_traj.npy --fps 50

Then check what you built before spending a training run on it:

    python data/scripts/create_box_scene.py --inspect data/scenes/box.pt \
        --motion-file motions.pt

Train with it:

    python protomotions/train_agent.py --robot-name g1 --simulator isaaclab \
        --experiment-path examples/experiments/mimic/g1_pick_box.py \
        --experiment-name box_test --motion-file motions.pt \
        --scenes-file data/scenes/box.pt

Box poses live in the same world frame as the motion -- the packaged ``gts`` body
positions -- and both are shifted together by the respawn offsets at reset.
"""

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import torch

from protomotions.components.scene_lib import (
    BoxSceneObject,
    ObjectOptions,
    Scene,
    SceneLib,
)

# G1 hands. The box is carried at the midpoint between them.
DEFAULT_CARRY_BODIES = ["left_rubber_hand", "right_rubber_hand"]

IDENTITY_QUAT = (0.0, 0.0, 0.0, 1.0)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a scenes file with one box, optionally carrying a trajectory"
    )
    parser.add_argument(
        "--inspect",
        type=str,
        default=None,
        metavar="SCENES_PT",
        help="Report on an existing scenes file instead of creating one.",
    )
    parser.add_argument("--output", type=str, help="Output .pt path")
    parser.add_argument("--width", type=float, default=0.25, help="Box size along x (m)")
    parser.add_argument("--depth", type=float, default=0.35, help="Box size along y (m)")
    parser.add_argument("--height", type=float, default=0.35, help="Box size along z (m)")

    parser.add_argument(
        "--translation",
        type=float,
        nargs=3,
        default=(0.4, 0.0, 0.18),
        metavar=("X", "Y", "Z"),
        help="Box rest position (m). Z is the center, so a resting box sits at height/2.",
    )
    parser.add_argument(
        "--fixed",
        action="store_true",
        help="Pin the box in place (kinematic -- it cannot be picked up). "
        "Omit to let it fall under gravity.",
    )
    parser.add_argument("--mass", type=float, default=3.0, help="Box mass (kg)")
    parser.add_argument(
        "--friction",
        type=float,
        default=1.0,
        help="Box static/dynamic friction. Robot friction randomization does not "
        "touch scene objects, so grasp reliability rides on this value.",
    )
    parser.add_argument(
        "--motion-id",
        type=int,
        default=-1,
        help="Motion index this scene is paired with (-1 for any). Pairing is what "
        "binds the box trajectory to the right motion.",
    )

    trajectory = parser.add_argument_group("box trajectory")
    trajectory.add_argument(
        "--motion-file",
        type=str,
        default=None,
        help="Packaged MotionLib .pt to derive the carry trajectory from, and to "
        "cross-check against under --inspect.",
    )
    trajectory.add_argument(
        "--grasp-start",
        type=float,
        default=None,
        help="Time (s) the hands take the box. Before it, the box rests at "
        "--translation. Defaults to the start of the motion.",
    )
    trajectory.add_argument(
        "--grasp-end",
        type=float,
        default=None,
        help="Time (s) the box is released. After it, the box holds its last "
        "carried pose. Defaults to the end of the motion.",
    )
    trajectory.add_argument(
        "--carry-body",
        action="append",
        default=None,
        help=f"Body carrying the box; repeat for a midpoint. Default: {DEFAULT_CARRY_BODIES}.",
    )
    trajectory.add_argument(
        "--robot-name",
        type=str,
        default="g1",
        help="Robot whose body names --carry-body refers to.",
    )
    trajectory.add_argument(
        "--trajectory-file",
        type=str,
        default=None,
        help="Authored trajectory: .npy or .pt holding [N, 3] translations, or a "
        "dict with 'translation' and optional 'rotation' [N, 4] XYZW.",
    )
    trajectory.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Frames per second of --trajectory-file. Must match the motion's fps.",
    )

    args = parser.parse_args()
    if args.inspect is None and not args.output:
        parser.error("--output is required unless --inspect is given")
    if args.trajectory_file and args.motion_file:
        parser.error("--trajectory-file and --motion-file are mutually exclusive")
    if args.trajectory_file and args.fps is None:
        parser.error("--fps is required with --trajectory-file")
    return args


def load_packaged_motion(motion_file: str, motion_id: int):
    """Slice one motion out of a packaged MotionLib .pt.

    Forward kinematics is already baked in (``gts``/``grs`` hold per-body world
    poses), so this is a pure lookup -- same access pattern as
    ``extract_body_trajectories_to_csv.py``.

    Args:
        motion_file: Path to the packaged MotionLib .pt.
        motion_id: Index of the motion to slice.

    Returns:
        Tuple of (body_pos [frames, bodies, 3], body_rot [frames, bodies, 4], dt).
    """
    data = torch.load(motion_file, map_location="cpu", weights_only=False)

    num_motions = data["motion_num_frames"].shape[0]
    if not 0 <= motion_id < num_motions:
        raise ValueError(
            f"--motion-id {motion_id} out of range; file has {num_motions} motion(s)"
        )

    start = int(data["length_starts"][motion_id])
    num_frames = int(data["motion_num_frames"][motion_id])
    dt = float(data["motion_dt"][motion_id])
    frames = slice(start, start + num_frames)

    return data["gts"][frames], data["grs"][frames], dt


def build_carry_trajectory(
    body_pos: torch.Tensor,
    dt: float,
    carry_indices: List[int],
    rest_translation: Tuple[float, float, float],
    grasp_start: Optional[float],
    grasp_end: Optional[float],
) -> torch.Tensor:
    """Box translations for rest -> carry -> placed.

    The box keeps whatever offset it had from the carry point at the moment of
    the grasp, so the reference stays consistent with where the box actually sits
    when the hands arrive, rather than snapping into them.

    Args:
        body_pos: World body positions [frames, bodies, 3].
        dt: Seconds per frame.
        carry_indices: Body indices whose midpoint carries the box.
        rest_translation: Box center before the grasp.
        grasp_start: Grasp time in seconds, or None for the motion start.
        grasp_end: Release time in seconds, or None for the motion end.

    Returns:
        Box translations [frames, 3].
    """
    num_frames = body_pos.shape[0]
    duration = num_frames * dt

    if grasp_start is not None and not 0 <= grasp_start <= duration:
        raise ValueError(
            f"--grasp-start {grasp_start}s is outside the motion (0-{duration:.2f}s)"
        )
    if grasp_end is not None and not 0 <= grasp_end <= duration:
        raise ValueError(
            f"--grasp-end {grasp_end}s is outside the motion (0-{duration:.2f}s)"
        )

    start_frame = 0 if grasp_start is None else int(round(grasp_start / dt))
    end_frame = num_frames - 1 if grasp_end is None else int(round(grasp_end / dt))
    start_frame = max(0, min(start_frame, num_frames - 1))
    end_frame = max(start_frame, min(end_frame, num_frames - 1))

    carry_point = body_pos[:, carry_indices, :].mean(dim=1)  # [frames, 3]

    rest = torch.tensor(rest_translation, dtype=torch.float)
    translations = rest.unsqueeze(0).repeat(num_frames, 1)

    grasp_offset = rest - carry_point[start_frame]
    carried = carry_point[start_frame : end_frame + 1] + grasp_offset
    translations[start_frame : end_frame + 1] = carried

    # A released box stays where it was put.
    translations[end_frame + 1 :] = carried[-1]

    return translations


def load_authored_trajectory(
    trajectory_file: str,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Load [N, 3] translations and optional [N, 4] rotations from a file.

    Args:
        trajectory_file: Path to a .npy or .pt file.

    Returns:
        Tuple of (translations [N, 3], rotations [N, 4] or None).
    """
    path = Path(trajectory_file)
    if path.suffix == ".npy":
        import numpy as np

        data = torch.from_numpy(np.load(path)).float()
    else:
        data = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(data, dict):
        translations = torch.as_tensor(data["translation"], dtype=torch.float)
        rotations = data.get("rotation")
        if rotations is not None:
            rotations = torch.as_tensor(rotations, dtype=torch.float)
        return translations, rotations

    translations = torch.as_tensor(data, dtype=torch.float)
    if translations.ndim != 2 or translations.shape[1] != 3:
        raise ValueError(
            f"Expected [N, 3] translations in {trajectory_file}, got "
            f"{tuple(translations.shape)}"
        )
    return translations, None


def inspect_scenes(scenes_file: str, motion_file: Optional[str]) -> None:
    """Print what a scenes file actually contains, and flag silent traps.

    Args:
        scenes_file: Path to the scenes .pt.
        motion_file: Optional packaged MotionLib .pt to cross-check durations.
    """
    scenes = SceneLib._load_scenes_from_file(scenes_file, device="cpu")
    print(f"{scenes_file}: {len(scenes)} scene(s)")

    motion_lengths = None
    if motion_file is not None:
        data = torch.load(motion_file, map_location="cpu", weights_only=False)
        motion_lengths = (
            data["motion_num_frames"].float() * data["motion_dt"].float()
        ).tolist()
        print(f"{motion_file}: {len(motion_lengths)} motion(s)")

    for scene_idx, scene in enumerate(scenes):
        print(f"\nscene {scene_idx}: humanoid_motion_id={scene.humanoid_motion_id}")
        if scene.humanoid_motion_id == -1:
            print(
                "  WARNING: unpaired scene. Motions are sampled independently, so "
                "the box trajectory need not belong to the motion being tracked."
            )
        for obj_idx, obj in enumerate(scene.objects):
            frames = obj.translation.shape[0]
            duration = frames / obj.fps
            static = not obj.has_motion()
            print(
                f"  object {obj_idx} ({type(obj).__name__}): frames={frames} "
                f"fps={obj.fps} duration={duration:.2f}s "
                f"has_motion={obj.has_motion()} "
                f"fix_base_link={obj.options.fix_base_link}"
            )
            if static:
                print(
                    "    WARNING: single frame -- the reference pose is constant. "
                    "Object tracking rewards will pay this box for NOT moving."
                )
            if obj.options.fix_base_link:
                print(
                    "    WARNING: fix_base_link=True spawns a kinematic object. "
                    "It cannot be picked up or pushed."
                )
            if motion_lengths is not None and scene.humanoid_motion_id >= 0:
                motion_len = motion_lengths[scene.humanoid_motion_id]
                print(f"    paired motion duration={motion_len:.2f}s")
                if not static and abs(duration - motion_len) > 2.0 / obj.fps:
                    print(
                        "    WARNING: object and motion durations disagree. Object "
                        "poses are sampled at the motion time and clamped past the "
                        "end, so the tail of the reference will be frozen."
                    )


def main():
    args = parse_args()

    if args.inspect:
        inspect_scenes(args.inspect, args.motion_file)
        return

    translation = tuple(args.translation)
    rotation = IDENTITY_QUAT
    fps = None

    if args.motion_file:
        from protomotions.robot_configs.factory import robot_config

        body_pos, _, dt = load_packaged_motion(args.motion_file, max(args.motion_id, 0))
        body_names = robot_config(args.robot_name).kinematic_info.body_names
        carry_bodies = args.carry_body or DEFAULT_CARRY_BODIES
        missing = [b for b in carry_bodies if b not in body_names]
        if missing:
            raise ValueError(
                f"Carry bodies not found on robot '{args.robot_name}': {missing}.\n"
                f"Available: {body_names}"
            )
        carry_indices = [body_names.index(b) for b in carry_bodies]

        translation = build_carry_trajectory(
            body_pos=body_pos,
            dt=dt,
            carry_indices=carry_indices,
            rest_translation=tuple(args.translation),
            grasp_start=args.grasp_start,
            grasp_end=args.grasp_end,
        )
        rotation = torch.tensor(IDENTITY_QUAT).unsqueeze(0).repeat(len(translation), 1)
        fps = 1.0 / dt

    elif args.trajectory_file:
        translation, rotations = load_authored_trajectory(args.trajectory_file)
        if rotations is None:
            rotations = (
                torch.tensor(IDENTITY_QUAT).unsqueeze(0).repeat(len(translation), 1)
            )
        rotation = rotations
        fps = args.fps

    box = BoxSceneObject(
        width=args.width,
        depth=args.depth,
        height=args.height,
        translation=translation,
        rotation=rotation,
        fps=fps,
        options=ObjectOptions(
            fix_base_link=args.fixed,
            mass=args.mass,
            static_friction=args.friction,
            dynamic_friction=args.friction,
        ),
    )
    scenes = [Scene(objects=[box], humanoid_motion_id=args.motion_id)]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    SceneLib.save_scenes_to_file(scenes, args.output)

    num_frames = box.translation.shape[0]
    if num_frames > 1:
        print(
            f"Saved 1 scene to {args.output} "
            f"({num_frames} box frames @ {box.fps:.1f} fps, "
            f"{num_frames / box.fps:.2f}s, paired with motion {args.motion_id})"
        )
    else:
        print(f"Saved 1 scene to {args.output} (static box, no reference trajectory)")
        print(
            "NOTE: object tracking rewards score this box against a constant "
            "reference -- they will reward it for not moving. Pass --motion-file or "
            "--trajectory-file to give it a trajectory."
        )


if __name__ == "__main__":
    main()
