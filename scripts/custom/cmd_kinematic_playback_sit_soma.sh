#!/bin/bash
# Kinematic playback of the SOMA seated clip together with its chair scene.
# Run this BEFORE training: it replays the reference motion and the chair with
# no policy in the loop, so it shows whether the seat height is right. The
# thighs must rest on the seat with no visible gap and no penetration. If they
# float or clip, regenerate the chair with --seat-clearance instead of training.
SOMA_SIT=../training_data/configs/soma_sit_0

python examples/env_kinematic_playback.py \
    --experiment-path=examples/experiments/mimic/soma_sit_chair.py \
    --robot-name=soma23 \
    --simulator=isaaclab \
    --num-envs=1 \
    --motion-file=$SOMA_SIT/soma_sit_motion_seatcontacts.pt \
    --scenes-file=$SOMA_SIT/soma_sit_chair.pt
