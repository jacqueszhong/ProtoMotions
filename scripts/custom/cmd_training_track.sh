python protomotions/train_agent.py \
    --robot-name g1 \
    --simulator isaaclab \
    --experiment-path data/pretrained_models/motion_tracker/g1-bones-deploy/experiment_config.py \
    --experiment-name g1_walk \
    --checkpoint data/pretrained_models/motion_tracker/g1-bones-deploy/last.ckpt \
    --motion-file ../training_data/configs/g1_walkbox/g1_walk_box.pt \
    
    --num-envs 4096 \
    --batch-size 16384 \
    --ngpu 1
