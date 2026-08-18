#!/bin/bash
# Kinematic playback of the SOMA box-pickup clip together with its box scene.
# Run this BEFORE training: it replays the reference motion and the box
# trajectory with no policy in the loop, so it shows whether the two are
# actually aligned. If the box does not end up in the hands here, no amount of
# training will fix it -- regenerate the scene instead.
SOMA_BOX=../training_data/configs/soma_box

python examples/env_kinematic_playback.py \
    --experiment-path=examples/experiments/mimic/soma_pick_box.py \
    --robot-name=soma23 \
    --simulator=isaaclab \
    --num-envs=1 \
    --motion-file=$SOMA_BOX/soma_box_motion.pt \
    --scenes-file=$SOMA_BOX/soma_box_traj.pt
