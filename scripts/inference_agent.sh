#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Wrapper around protomotions/inference_agent.py that fully configures the
# boxes thrown by the "J" key (ProjectileConfig, see
# protomotions/simulator/base_simulator/config.py).
#
# Every field of ProjectileConfig is exposed below and forwarded as
# `--overrides simulator.projectile.<field>=<value>`, so the speed, size, mass,
# spawn geometry, aim noise and lifetime of the boxes are set explicitly on
# each run instead of inheriting whatever was frozen into
# resolved_configs_inference.pt.
#
# Usage:
#   scripts/inference_agent.sh [options] [-- extra inference_agent.py args]
#
#   scripts/inference_agent.sh --speed 8                  # every box at 8 m/s
#   scripts/inference_agent.sh --speed 5,12 --size 0.2    # 5-12 m/s, 40cm cubes
#   scripts/inference_agent.sh --num-boxes 10 --hide-delay 5
#   scripts/inference_agent.sh --checkpoint results/g1_box/last.ckpt --simulator isaacgym
#
# Range options accept "MIN,MAX" or a single value (which pins min == max).
# Any variable below can also be set from the environment, e.g.
#   SPEED_MIN=2 SPEED_MAX=3 scripts/inference_agent.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"

# ---------------------------------------------------------------------------
# Run configuration
# ---------------------------------------------------------------------------
CHECKPOINT="${CHECKPOINT:-results/soma_box_1/last.ckpt}"
MOTION_FILE="${MOTION_FILE:-}"          # empty -> use the checkpoint's motion file
SCENES_FILE="${SCENES_FILE:-}"          # empty -> use the checkpoint's scene file
SIMULATOR="${SIMULATOR:-isaaclab}"
NUM_ENVS="${NUM_ENVS:-1}"
HEADLESS="${HEADLESS:-0}"               # 1 -> pass --headless (no J key without a viewer)

# ---------------------------------------------------------------------------
# Box ("projectile") configuration -- the full ProjectileConfig surface
# ---------------------------------------------------------------------------
# Size of the box pool. Each J press throws the next box in the pool at every
# robot, cycling round-robin; a pool of 1 means each throw reuses the same box.
NUM_BOXES="${NUM_BOXES:-5}"

# Launch speed in m/s, sampled uniformly in [SPEED_MIN, SPEED_MAX] per throw.
# The robot's own XY velocity is added on top so the box leads a moving target.
# The upstream/ASE default is 30-40 m/s, which is a very hard hit; 5-15 is
# closer to a shove for a G1/SOMA-scale robot.
SPEED_MIN="${SPEED_MIN:-8.0}"
SPEED_MAX="${SPEED_MAX:-12.0}"

# Cube half-extent in meters, interpolated linearly across the pool
# (box i gets SIZE_MIN + (SIZE_MAX - SIZE_MIN) * i / (NUM_BOXES - 1)), so one
# pool gives a spread of box sizes over successive presses. With NUM_BOXES=1
# only SIZE_MIN is used.
SIZE_MIN="${SIZE_MIN:-0.05}"
SIZE_MAX="${SIZE_MAX:-0.15}"

# Density in kg/m^3 -> mass = density * (2 * half_size)^3. Together with the
# size range this is what sets the momentum of the hit.
DENSITY="${DENSITY:-500.0}"

# Spawn distance from the robot root, in meters (random azimuth around it).
DIST_MIN="${DIST_MIN:-4.0}"
DIST_MAX="${DIST_MAX:-5.0}"

# Spawn height relative to the robot root, in meters (negative = legs,
# positive = head/upper body).
HEIGHT_MIN="${HEIGHT_MIN:--0.65}"
HEIGHT_MAX="${HEIGHT_MAX:-1.1}"

# Std-dev of the Gaussian noise added to the aim direction. 0.0 aims exactly at
# the root; larger values scatter the hits (and produce misses).
DIR_NOISE="${DIR_NOISE:-0.1}"

# Seconds a box stays alive before it is teleported out of sight.
HIDE_DELAY="${HIDE_DELAY:-2.0}"

# Where boxes are parked while hidden: box i goes to HIDE_Z - HIDE_SPACING * i,
# so HIDE_SPACING must exceed the box size to keep parked boxes from colliding.
HIDE_Z="${HIDE_Z:--2.0}"
HIDE_SPACING="${HIDE_SPACING:-4.0}"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
EXTRA_ARGS=()
EXTRA_OVERRIDES=()

usage() {
    sed -n '15,25p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

# parse_range "MIN,MAX" | "VALUE" -> RANGE_MIN / RANGE_MAX
parse_range() {
    local value="$1"
    if [[ "$value" == *,* ]]; then
        RANGE_MIN="${value%%,*}"
        RANGE_MAX="${value##*,}"
    else
        RANGE_MIN="$value"
        RANGE_MAX="$value"
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)      usage 0 ;;
        --checkpoint)   CHECKPOINT="$2"; shift 2 ;;
        --motion-file)  MOTION_FILE="$2"; shift 2 ;;
        --scenes-file)  SCENES_FILE="$2"; shift 2 ;;
        --simulator)    SIMULATOR="$2"; shift 2 ;;
        --num-envs)     NUM_ENVS="$2"; shift 2 ;;
        --headless)     HEADLESS=1; shift ;;
        --num-boxes)    NUM_BOXES="$2"; shift 2 ;;
        --speed)        parse_range "$2"; SPEED_MIN="$RANGE_MIN"; SPEED_MAX="$RANGE_MAX"; shift 2 ;;
        --speed-min)    SPEED_MIN="$2"; shift 2 ;;
        --speed-max)    SPEED_MAX="$2"; shift 2 ;;
        --size)         parse_range "$2"; SIZE_MIN="$RANGE_MIN"; SIZE_MAX="$RANGE_MAX"; shift 2 ;;
        --density)      DENSITY="$2"; shift 2 ;;
        --distance)     parse_range "$2"; DIST_MIN="$RANGE_MIN"; DIST_MAX="$RANGE_MAX"; shift 2 ;;
        --height)       parse_range "$2"; HEIGHT_MIN="$RANGE_MIN"; HEIGHT_MAX="$RANGE_MAX"; shift 2 ;;
        --dir-noise)    DIR_NOISE="$2"; shift 2 ;;
        --hide-delay)   HIDE_DELAY="$2"; shift 2 ;;
        --hide-z)       HIDE_Z="$2"; shift 2 ;;
        --hide-spacing) HIDE_SPACING="$2"; shift 2 ;;
        -o|--override)  EXTRA_OVERRIDES+=("$2"); shift 2 ;;
        --)             shift; EXTRA_ARGS+=("$@"); break ;;
        *)              EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Error: checkpoint not found: $CHECKPOINT" >&2
    echo "Pass --checkpoint <path/to/last.ckpt> (or set CHECKPOINT=...)." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Build the command
# ---------------------------------------------------------------------------
# Values are parsed with ast.literal_eval on the Python side, so the *_range
# fields need "(min,max)" -- no spaces inside the parens.
OVERRIDES=(
    "simulator.projectile.num_projectiles=${NUM_BOXES}"
    "simulator.projectile.speed_range=(${SPEED_MIN},${SPEED_MAX})"
    "simulator.projectile.cube_half_size_range=(${SIZE_MIN},${SIZE_MAX})"
    "simulator.projectile.density=${DENSITY}"
    "simulator.projectile.spawn_distance_range=(${DIST_MIN},${DIST_MAX})"
    "simulator.projectile.spawn_height_range=(${HEIGHT_MIN},${HEIGHT_MAX})"
    "simulator.projectile.direction_noise_std=${DIR_NOISE}"
    "simulator.projectile.hide_delay=${HIDE_DELAY}"
    "simulator.projectile.hide_z=${HIDE_Z}"
    "simulator.projectile.hide_spacing=${HIDE_SPACING}"
)
OVERRIDES+=("${EXTRA_OVERRIDES[@]+"${EXTRA_OVERRIDES[@]}"}")

CMD=(
    "$PYTHON_BIN" protomotions/inference_agent.py
    --checkpoint "$CHECKPOINT"
    --simulator "$SIMULATOR"
    --num-envs "$NUM_ENVS"
)
[[ -n "$MOTION_FILE" ]] && CMD+=(--motion-file "$MOTION_FILE")
[[ -n "$SCENES_FILE" ]] && CMD+=(--scenes-file "$SCENES_FILE")
[[ "$HEADLESS" == "1" ]] && CMD+=(--headless)
CMD+=("${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}")
# Keep --overrides last: it is nargs="*" and would otherwise swallow trailing
# positional-looking arguments.
CMD+=(--overrides "${OVERRIDES[@]}")

echo "Boxes (J key): pool ${NUM_BOXES}, speed ${SPEED_MIN}-${SPEED_MAX} m/s, half-size ${SIZE_MIN}-${SIZE_MAX} m, density ${DENSITY} kg/m^3"
echo "               spawn ${DIST_MIN}-${DIST_MAX} m away at ${HEIGHT_MIN}-${HEIGHT_MAX} m, aim noise ${DIR_NOISE}, alive ${HIDE_DELAY} s"
printf '%q ' "${CMD[@]}"; echo
exec "${CMD[@]}"
