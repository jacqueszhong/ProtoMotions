# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
# Body Trajectory Extraction from a Packaged MotionLib .pt

Extracts world-frame trajectories for named bodies out of a packaged MotionLib
`.pt` and writes them to a CSV with a time column. Defaults to the two G1 hands.

Forward kinematics is already baked into the packaged file (`gts`/`grs` hold
per-body world positions/rotations), so this is a pure lookup -- no FK is run.
The raw retargeted CSVs (e.g. Kimodo `output_g1.csv`) are joint space only and
cannot be used for this without running FK yourself.

## Usage

```bash
python data/scripts/extract_body_trajectories_to_csv.py /path/to/motions.pt \\
    --robot-name g1 \\
    --output /path/to/hand_trajectories.csv

# Other bodies, repeat the flag:
python data/scripts/extract_body_trajectories_to_csv.py motions.pt \\
    --body left_wrist_yaw_link --body right_wrist_yaw_link
```

## Parameters

- `motion_file`: Path to the packaged MotionLib `.pt`.
- `--robot-name`: Robot whose body names to resolve (default: g1).
- `--body`: Body name to extract; repeatable. Defaults to the G1 rubber hands.
- `--output`: Output CSV path. Defaults to `<motion_file_dir>/body_trajectories.csv`.
- `--motion-index`: Which motion in a multi-motion file (default: 0).

## Output

One row per frame: `frame`, `time_s`, then per body `<body>_pos_{x,y,z}`,
`<body>_quat_{x,y,z,w}`, `<body>_vel_{x,y,z}`, `<body>_speed`.

Positions are world frame, in meters. Quaternions are **XYZW** -- packaging
converts from the WXYZ used by the input CSVs (see `pose_lib`, which documents
`RobotState` as XYZW). Reorder if your consumer expects WXYZ.
"""

import csv
from pathlib import Path
from typing import List, Optional

import torch
import typer

from protomotions.robot_configs.factory import robot_config

DEFAULT_BODIES = ["left_rubber_hand", "right_rubber_hand"]

app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def main(
    motion_file: Path = typer.Argument(
        ..., help="Path to the packaged MotionLib .pt file."
    ),
    robot_name: str = typer.Option(
        "g1", "--robot-name", help="Robot whose body names to resolve."
    ),
    body: Optional[List[str]] = typer.Option(
        None,
        "--body",
        help="Body name to extract; repeatable. Defaults to the G1 rubber hands.",
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", help="Output CSV path. Defaults next to the motion file."
    ),
    motion_index: int = typer.Option(
        0, "--motion-index", help="Which motion to extract from a multi-motion file."
    ),
):
    bodies = list(body) if body else list(DEFAULT_BODIES)
    if output is None:
        output = motion_file.parent / "body_trajectories.csv"

    data = torch.load(str(motion_file), map_location="cpu", weights_only=False)
    body_names = robot_config(robot_name).kinematic_info.body_names

    missing = [b for b in bodies if b not in body_names]
    if missing:
        raise typer.BadParameter(
            f"Bodies not found on robot '{robot_name}': {missing}.\n"
            f"Available: {body_names}"
        )
    indices = [body_names.index(b) for b in bodies]

    num_motions = data["motion_num_frames"].shape[0]
    if not 0 <= motion_index < num_motions:
        raise typer.BadParameter(
            f"--motion-index {motion_index} out of range; file has {num_motions} motion(s)."
        )

    # Packaged motions are concatenated; slice this one out via length_starts.
    start = int(data["length_starts"][motion_index])
    num_frames = int(data["motion_num_frames"][motion_index])
    dt = float(data["motion_dt"][motion_index])
    frames = slice(start, start + num_frames)

    positions = data["gts"][frames]
    rotations = data["grs"][frames]
    velocities = data["gvs"][frames]

    header = ["frame", "time_s"]
    for name in bodies:
        header += [f"{name}_pos_{a}" for a in "xyz"]
        header += [f"{name}_quat_{a}" for a in ("x", "y", "z", "w")]
        header += [f"{name}_vel_{a}" for a in "xyz"]
        header += [f"{name}_speed"]

    with open(output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(num_frames):
            row = [i, round(i * dt, 6)]
            for idx in indices:
                row += [round(float(v), 6) for v in positions[i, idx]]
                row += [round(float(v), 6) for v in rotations[i, idx]]
                row += [round(float(v), 6) for v in velocities[i, idx]]
                row += [round(float(velocities[i, idx].norm()), 6)]
            writer.writerow(row)

    print(f"Extracted {bodies} from {motion_file}")
    print(
        f"Wrote {num_frames} frames ({num_frames * dt:.2f}s @ {1 / dt:.1f} fps) to {output}"
    )


if __name__ == "__main__":
    app()
