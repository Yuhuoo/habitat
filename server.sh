#!/bin/bash

python viewer_server.py \
    --scene data/scene_datasets/oLBMNvg9in8/oLBMNvg9in8.glb \
    --dataset data/scene_datasets/oLBMNvg9in8/mp3d.scene_dataset_config.json \
    --action_path data/scene_datasets/oLBMNvg9in8/action.txt \
    --output_path output/matterport3d/oLBMNvg9in8/colmap_test \
    --feq 2
# zip -q -r output/matterport3d/oLBMNvg9in8/second.zip output/matterport3d/oLBMNvg9in8/second

# python viewer_server.py \
#     --scene data/scene_datasets/mp3d_example/17DRP5sb8fy/17DRP5sb8fy.glb \
#     --dataset data/scene_datasets/mp3d_example/mp3d.scene_dataset_config.json \
#     --action_path action.txt \
#     --output_path output/17DRP5sb8fy
# ffmpeg -framerate 24 -i output/17DRP5sb8fy/rgb/observation_rgb_%d.png -c:v libx264 -pix_fmt yuv420p output/17DRP5sb8fy/output.mp4

# python viewer_server.py \
#     --scene data/datasets/Scannet/scans/scene0000_00/scene0000_00_vh_clean.glb \
#     --dataset data/datasets/Scannet/scans/scene0000_00/scene_dataset_config.json \
#     --action_path data/datasets/Scannet/scans/scene0000_00/action.txt \
#     --output_path output/scanet/00000 \
#     --feq 5
# zip -q -r output/matterport3d/oLBMNvg9in8.zip output/matterport3d/oLBMNvg9in8



