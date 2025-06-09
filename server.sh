#!/bin/bash

# python src/viewer.py \
#     --scene data/scene_datasets/oLBMNvg9in8/oLBMNvg9in8.glb \
#     --dataset data/scene_datasets/oLBMNvg9in8/mp3d.scene_dataset_config.json \
#     --action_path data/scene_datasets/oLBMNvg9in8/5_26.txt \
#     --output_path output/matterport3d/oLBMNvg9in8/5_26_2 \
#     --feq 3
# zip -q -r output/matterport3d/oLBMNvg9in8/second.zip output/matterport3d/oLBMNvg9in8/second

# data_dir="data/datasets/Scannet/scans"
# scene="scene0084_00"
# echo "===================${scene} Begin============================="
# glb_path="$data_dir/$scene/${scene}_vh_clean.glb"
# config_path="$data_dir/../scene_dataset_config.json"
# obs_saved_path="output/scanet/new/$scene"
# echo "Using scene: $glb_path"
# echo "Using config: $config_path"
# echo "Output will be saved to: $obs_saved_path"
# python src/viewer_server.py \
#     --scene $glb_path \
#     --dataset $config_path \
#     --output_path $obs_saved_path \
#     --sensor_height 0.7
# echo "===================== ${scene} Done==========================="

# data_dir="data/datasets/Scannet/scans"
# for scene in $(ls $data_dir); do
#     echo "===================${scene} Begin============================="
#     glb_path="$data_dir/$scene/${scene}_vh_clean.glb"
#     config_path="$data_dir/../scene_dataset_config.json"
#     obs_saved_path="output/scanet/new/$scene"
#     echo "Using scene: $glb_path"
#     echo "Using config: $config_path"
#     echo "Output will be saved to: $obs_saved_path"
#     python src/viewer_server.py \
#         --scene $glb_path \
#         --dataset $config_path \
#         --output_path $obs_saved_path \
#         --sensor_height 0.7
#     echo "===================== ${scene} Done==========================="
#     # zip -q -r output/scanet/new.zip output/scanet/new
# done



