#!/bin/bash

# RGBD reconstruction
# output: point cloud, mesh
python tools/rgbd_reconstruction.py --data_dir example_data/demo

# Visualization
current_script_path=$(realpath "$0")
current_script_dir=$(dirname "$current_script_path")
submodules_dir="$current_script_dir/submodules"

python "$submodules_dir/remote-viewer-utils/colmap_vis.py" \
        --image_dir example_data/demo/images \
        --colmap_path example_data/demo/sparse/0 \
        --showpcd true \
        --port 8084 \
        --is_habitat_coordinate true \

# # 3DGS reconstruction
# # output: 3DGS
# cd ../gaussian_splatting
# conda activate gaussian_splatting
# python train.py -s /home/ga/code/habitat/example_data/demo -m example_data/demo/3DGS