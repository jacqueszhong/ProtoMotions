#!/bin/bash

target_dir="data/out-kimodo-g1"

# Check if directory argument is provided
if [ $# -eq 1 ]; then

    source_dir="$1"
    echo "Processing Kimodo data from $source_dir" 

    # Verify source directory exists
    if [ ! -d "$source_dir" ]; then
        echo "Error: Directory '$source_dir' does not exist"
        exit 1
    fi

    # Copy .csv
    rm -rf "$target_dir" && mkdir "$target_dir"
    mkdir -p "$target_dir"
    cp -a "$source_dir"/. "$target_dir"/
    echo "Copied data files into $target_dir"

    # Convert to .motion
    python data/scripts/convert_g1_csv_to_proto.py \
    --input-dir "$target_dir"/ \
    --output-dir "$target_dir"/proto \
    --input-fps 30 \
    --output-fps 30 \
    --pos-units m \
    --rot-format quat_wxyz \
    --joint-units rad \
    --no-has-header \
    --no-has-frame-column \
    --force-remake

    # Package
    python protomotions/components/motion_lib.py \
        --motion-path "$target_dir"/proto/ \
        --output-file "$target_dir"/kimodo_g1_motions.pt
fi

# Kinematic visualizer
# python examples/motion_libs_visualizer.py     --motion_files "$target_dir"/kimodo_g1_motions.pt     --robot g1     --simulator isaaclab


# Retargeting with pre-trained motion imitation model 
python protomotions/inference_agent.py \
    --checkpoint data/pretrained_models/motion_tracker/g1-bones-deploy/last.ckpt \
    --motion-file "$target_dir"/kimodo_g1_motions.pt \
    --simulator isaaclab \
