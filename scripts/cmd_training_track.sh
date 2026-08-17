python protomotions/train_agent.py \
    --robot-name g1 \
    --simulator isaaclab \
    --experiment-path data/pretrained_models/motion_tracker/g1-bones-deploy/experiment_config.py \
    --experiment-name g1_walk \
    --checkpoint data/pretrained_models/motion_tracker/g1-bones-deploy/last.ckpt \
    --motion-file ../data/g1-walk-box/g1_walk_box.pt \
    --num-envs 4096 \
    --batch-size 16384 \
    --ngpu 1
