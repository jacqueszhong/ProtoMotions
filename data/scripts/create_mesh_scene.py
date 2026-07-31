# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create a scenes file containing a single mesh asset, for use with ``--scenes-file``.

The asset is referenced by path (``.usd``/``.usda``/``.urdf``).  A sibling
``.obj``/``.stl``/``.ply`` with the same basename must exist next to it: the bounding
box and pointcloud are computed from that mesh, not from the USD itself.  This mirrors
the layout in ``examples/data/`` (e.g. ``elephant.usda`` + ``elephant.stl``).

Example:
    python data/scripts/create_mesh_scene.py \
        --asset examples/data/elephant.usda \
        --output data/scenes/elephant.pt

    python protomotions/train_agent.py --robot-name g1 --simulator isaaclab \
        --experiment-path examples/experiments/mimic/mlp.py \
        --experiment-name mesh_test --motion-file <motions.pt> \
        --scenes-file data/scenes/elephant.pt
"""

import argparse
import os
from pathlib import Path

from protomotions.components.scene_lib import (
    MeshSceneObject,
    ObjectOptions,
    Scene,
    SceneLib,
)

# Extensions searched for the collision/bbox mesh that must accompany the asset.
_MESH_EXTENSIONS = (".obj", ".stl", ".ply")


def find_sibling_mesh(asset_path: str):
    """Return the .obj/.stl/.ply next to *asset_path*, or None if absent."""
    stem = os.path.splitext(asset_path)[0]
    for ext in _MESH_EXTENSIONS:
        if os.path.exists(stem + ext):
            return stem + ext
    return None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a scenes file with one mesh object"
    )
    parser.add_argument(
        "--asset", type=str, required=True, help="Path to the .usd/.usda/.urdf asset"
    )
    parser.add_argument("--output", type=str, required=True, help="Output .pt path")
    parser.add_argument(
        "--asset-root",
        type=str,
        default=None,
        help=(
            "Root that the asset path is stored relative to. Defaults to the scene "
            "file's grandparent, which is what SceneLib assumes when loading. Pass the "
            "same value to SceneLibConfig(asset_root=...) if you override it."
        ),
    )
    parser.add_argument(
        "--translation",
        type=float,
        nargs=3,
        default=(1.0, 0.0, 0.0),
        metavar=("X", "Y", "Z"),
        help="Object center position (m).",
    )
    parser.add_argument(
        "--scale",
        type=float,
        nargs=3,
        default=(1.0, 1.0, 1.0),
        metavar=("X", "Y", "Z"),
        help="Per-axis scale applied to the mesh.",
    )
    parser.add_argument(
        "--fixed",
        action="store_true",
        help="Pin the object in place. Omit to let it fall under gravity.",
    )
    parser.add_argument(
        "--motion-id",
        type=int,
        default=-1,
        help="Motion index this scene is paired with (-1 for any).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Check if usda asset exists
    asset_path = os.path.abspath(args.asset)
    if not os.path.exists(asset_path):
        raise FileNotFoundError(f"Asset not found: {asset_path}")

    # Check if mesh asset exists
    sibling = find_sibling_mesh(asset_path)
    if sibling is None:
        stem = os.path.splitext(asset_path)[0]
        raise FileNotFoundError(
            f"No collision mesh found next to {asset_path}.\n"
            f"MeshSceneObject computes its bounding box from a sibling mesh, so one of "
            f"{', '.join(stem + e for e in _MESH_EXTENSIONS)} must exist."
        )
    print(f"Using {os.path.basename(sibling)} for bounding box")

    # Build scene via SceneLib API
    obj = MeshSceneObject(
        object_path=asset_path,
        scale=tuple(args.scale),
        translation=tuple(args.translation),
        rotation=(0.0, 0.0, 0.0, 1.0),
        options=ObjectOptions(fix_base_link=args.fixed, density=1000.0),
    )
    scenes = [Scene(objects=[obj], humanoid_motion_id=args.motion_id)]

    output = os.path.abspath(args.output)
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    # SceneLib._load_scenes_from_file defaults asset_root to the scene file's
    # grandparent, while save_scenes_to_file defaults to its parent. Matching the
    # loader here keeps the stored relative path resolvable.
    asset_root = args.asset_root or os.path.dirname(os.path.dirname(output))
    SceneLib.save_scenes_to_file(scenes, output, asset_root=os.path.abspath(asset_root))

    print(f"Saved 1 scene to {output}")
    print(f"Asset path stored relative to {asset_root}")


if __name__ == "__main__":
    main()
