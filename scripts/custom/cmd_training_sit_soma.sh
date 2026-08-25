#!/bin/bash
# SOMA seated balance on a chair -- warm start from the soma-bones tracker.
# Note --robot-name soma23 (not "soma"), and the *_seatcontacts motion file:
# soma_sit_chair.py puts LeftLeg/RightLeg in contact_bodies, so it needs the
# relabelled reference contacts or contact_match_rew penalises sitting.
# Regenerate both with data/scripts/create_chair_scene.py.
SOMA_SIT=../training_data/configs/soma_sit_0

python protomotions/train_agent.py \
    --robot-name soma23 \
    --simulator isaaclab \
    --experiment-path examples/experiments/mimic/soma_sit_chair.py \
    --experiment-name soma_sit_1 \
    --checkpoint data/pretrained_models/motion_tracker/soma-bones/last.ckpt \
    --motion-file $SOMA_SIT/soma_sit_motion_seatcontacts.pt \
    --scenes-file $SOMA_SIT/soma_sit_chair.pt \
    --num-envs 4096 \
    --batch-size 16384 \
    --ngpu 1
