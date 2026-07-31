#!/bin/bash

target_dir="data/out-kimodo-soma"

# Check if directory argument is provided
if [ $# -eq 1 ]; then

    source_dir="$1"
    echo "Processing Kimodo data from $source_dir" 

    # Verify source directory exists
    if [ ! -d "$source_dir" ]; then
        echo "Error: Directory '$source_dir' does not exist"
        exit 1
    fi

    # Copy .npz
    rm -rf "$target_dir" && mkdir "$target_dir"
    mkdir -p "$target_dir"
    cp -a "$source_dir"/. "$target_dir"/

    # Convert to .motion
    python data/scripts/convert_soma23_npz_to_proto.py \
        --input-dir "$target_dir" \
        --output-dir "$target_dir"/proto \
        --input-fps 30 \
        --output-fps 30

    # Package
    python protomotions/components/motion_lib.py \
        --motion-path "$target_dir"/proto/ \
        --output-file "$target_dir"/kimodo_motion.pt
fi

# Run visualizer
python protomotions/inference_agent.py \
    --checkpoint data/pretrained_models/motion_tracker/soma-bones/last.ckpt \
    --motion-file "$target_dir"/kimodo_motion.pt \
    --simulator isaaclab \
    # --headless

