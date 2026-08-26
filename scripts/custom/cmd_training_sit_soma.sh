#!/bin/bash
# SOMA seated balance on a chair -- warm start from the soma-bones tracker.
# Note --robot-name soma23 (not "soma"), and the relabelled motion file:
# soma_sit_chair.py puts LeftLeg/RightLeg *and Spine2/Chest* in contact_bodies,
# so it needs reference contacts relabelled for both the seat and the backrest,
# or contact_match_rew penalises the very pose we are training.
#
# THE SCENES FILE AND THE MOTION FILE ARE ONE ARTIFACT. The back labels are only
# true of the backrest that was fitted alongside them, so regenerate both
# together with data/scripts/create_chair_scene.py -- see
# $SOMA_SIT/create_chair_scene.sh -- and change --experiment-name whenever you
# do. Resuming reads the saved experiment state and ignores the config file, so
# a resumed run silently keeps the old chair and the old labels.
#
# soma_sit_1 was trained against the pre-backrest chair (backrest 14.5cm further
# back and 9 degrees more upright, back contacts unlabelled). Those artifacts are
# stale for this config; hence soma_sit_2 and the *_back* files below.
SOMA_SIT=../training_data/configs/soma_sit_0

python protomotions/train_agent.py \
    --robot-name soma23 \
    --simulator isaaclab \
    --experiment-path examples/experiments/mimic/soma_sit_chair.py \
    --experiment-name soma_sit_2 \
    --checkpoint data/pretrained_models/motion_tracker/soma-bones/last.ckpt \
    --motion-file $SOMA_SIT/soma_sit_motion_backcontacts.pt \
    --scenes-file $SOMA_SIT/soma_sit_chair_back.pt \
    --num-envs 4096 \
    --batch-size 16384 \
    --ngpu 1
