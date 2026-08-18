#!/bin/bash
# SOMA box pickup -- warm start from the soma-bones tracker.
# SOMA counterpart of cmd_training_box.sh. Note --robot-name soma23 (not
# "soma") and the soma-bones checkpoint: the g1-bones-deploy checkpoint is not
# loadable here (different obs widths, different model class).
SOMA_BOX=../training_data/configs/soma_box

python protomotions/train_agent.py \
    --robot-name soma23 \
    --simulator isaaclab \
    --experiment-path examples/experiments/mimic/soma_pick_box.py \
    --experiment-name soma_box_1 \
    --checkpoint data/pretrained_models/motion_tracker/soma-bones/last.ckpt \
    --motion-file $SOMA_BOX/soma_box_motion.pt \
    --scenes-file $SOMA_BOX/soma_box_traj.pt \
    --num-envs 4096 \
    --batch-size 16384 \
    --ngpu 1
