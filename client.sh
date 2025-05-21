
# input=data/scanet/high_res/scene0001_00/scene0001_00_vh_clean.ply
# output=data/scanet/high_res/scene0000_00/scene0000_00_vh_clean.glb
# assimp export $input $output
# python viewer_client.py \
#     --scene $output \
#     --data data/scanet/high_res/scene_dataset_config.json \
#     --width 960 --height 640 \
#     --sensor_height 0.3

python viewer_client.py \
    --scene data/matterport3D/oLBMNvg9in8/oLBMNvg9in8.glb \
    --dataset data/matterport3D/mp3d.scene_dataset_config.json \
    --width 960 --height 640 \
    --saved_path test_output/action.txt