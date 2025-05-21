#!/bin/bash

current_script_path=$(realpath "$0")
current_script_dir=$(dirname "$current_script_path")
submodules_dir="$current_script_dir/submodules"

python "$submodules_dir/remote-viewer-utils/colmap_vis.py" \
        --image_dir output/matterport3d/oLBMNvg9in8/colmap_test/images \
        --colmap_path output/matterport3d/oLBMNvg9in8/colmap_test/sparse/0