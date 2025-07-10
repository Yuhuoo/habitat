#!/bin/bash

###### 1. 指定场景和动作文件采集
# python src/data_collect/viewer_server.py \
#     --scene data/datasets/Scannet/scans/scene0000_00/scene0000_00_vh_clean.glb \
#     --dataset data/datasets/Scannet/scene_dataset_config.json \
#     --action_path data/datasets/Scannet/scans/scene0000_00/action.txt \
#     --output_path output/demo/scene0000_00-7-10 \
#     --sensor_height 0.7 \
#     --feq 20

####### 2. 指定场景自动化采集
data_dir="data/datasets/Scannet/scans"
scene="scene0000_00"
echo "===================${scene} Begin============================="
glb_path="$data_dir/$scene/${scene}_vh_clean.glb"
config_path="$data_dir/../scene_dataset_config.json"
obs_saved_path="example_data/demo"
echo "Using scene: $glb_path"
echo "Using config: $config_path"
echo "Output will be saved to: $obs_saved_path"
python src/data_collect/viewer_server.py \
    --scene $glb_path \
    --dataset $config_path \
    --output_path $obs_saved_path \
    --sensor_height 0.7 \
    --LOOK 36
echo "===================== ${scene} Done==========================="



# ####### 指定场景和初始位置
# python src/data_collect/viewer_server.py \
#     --scene data/datasets/Scannet/scans/scene0046_00/scene0046_00_vh_clean.glb \
#     --dataset data/datasets/Scannet/scene_dataset_config.json \
#     --action_path data/datasets/Scannet/scans/scene0046_00/action.txt \
#     --output_path output/scanet/test/scene0046_00 \
#     --start_position "2.7847664,  1.5457658, -2.013033" \
#     --start_rotation "0.769103646278381, -0.111514739692211, 0.622807800769806, 0.0903028398752213" \
#     --sensor_height 0.0 \
#     --feq 4

# python src/data_collect/viewer_server.py \
#     --scene data/datasets/Scannet/scans/scene0050_00/scene0050_00_vh_clean.glb \
#     --dataset data/datasets/Scannet/scene_dataset_config.json \
#     --action_path data/datasets/Scannet/scans/scene0050_00/action.txt \
#     --output_path output/scanet/test/scene0050_00 \
#     --start_position "5.3083987,  1.8769357,  -1.778661" \
#     --start_rotation "-0.86167573928833, 0.105547100305557, -0.49615478515625, -0.0143296699970961" \
#     --sensor_height 0.0 \
#     --feq 4
# zip -q -r output/matterport3d/oLBMNvg9in8/second.zip output/matterport3d/oLBMNvg9in8/second

# python src/data_collect/viewer_server.py \
#     --scene data/datasets/Scannet/scans/scene0084_00/scene0084_00_vh_clean.glb \
#     --dataset data/datasets/Scannet/scene_dataset_config.json \
#     --action_path data/datasets/Scannet/scans/scene0084_00/action.txt \
#     --output_path output/scanet/test/scene0084_00 \
#     --start_position "0.13359478,  0.7007719,  -1.9164732" \
#     --start_rotation "0.382067292928696, -0.0216953512281179, -0.922393977642059, -0.0523773208260536" \
#     --sensor_height 0.0 \
#     --feq 4
# zip -q -r output/matterport3d/oLBMNvg9in8/second.zip output/matterport3d/oLBMNvg9in8/second

# data_dir="data/datasets/Scannet/scans"
# scene="scene0000_00"
# echo "===================${scene} Begin============================="
# glb_path="$data_dir/$scene/${scene}_vh_clean.glb"
# config_path="$data_dir/../scene_dataset_config.json"
# obs_saved_path="example_data/scanet/$scene"
# echo "Using scene: $glb_path"
# echo "Using config: $config_path"
# echo "Output will be saved to: $obs_saved_path"
# python src/data_collect/viewer_server.py \
#     --scene $glb_path \
#     --dataset $config_path \
#     --output_path $obs_saved_path \
#     --sensor_height 0.7 \
#     --width 256 \
#     --height 256 \
#     --LOOK 36
# echo "===================== ${scene} Done==========================="

######### batch Scanet data collection
# data_dir="data/datasets/Scannet/scans"
# for scene in $(ls $data_dir); do
#     echo "===================${scene} Begin============================="
#     glb_path="$data_dir/$scene/${scene}_vh_clean.glb"
#     config_path="$data_dir/../scene_dataset_config.json"
#     obs_saved_path="output/scanet/new/$scene"
#     echo "Using scene: $glb_path"
#     echo "Using config: $config_path"
#     echo "Output will be saved to: $obs_saved_path"
#     python src/data_collect/viewer_server.py \
#         --scene $glb_path \
#         --dataset $config_path \
#         --output_path $obs_saved_path \
#         --sensor_height 0.7
#         --LOOK 2.8
#     echo "===================== ${scene} Done==========================="
#     # zip -q -r output/scanet/new.zip output/scanet/new
# done



