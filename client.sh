
input=data/scanet/high_res/scene0001_00/scene0001_00_vh_clean.ply
output=data/scanet/high_res/scene0000_00/scene0000_00_vh_clean.glb
# assimp export $input $output
python viewer_client.py \
    --scene $output \
    --data data/scanet/high_res/scene_dataset_config.json \
    --width 960 --height 640 \
    --saved_path test_output/scanet_action_5.27 \
    --start_position "[2.2184856, 1.6997461, -1.5994138]" \
    --start_rotation "[0.983254909515381, 0, -0.182235538959503, 0]" \
    --sensor_height 1.5 

# python viewer_client.py \
#     --scene data/matterport3D/oLBMNvg9in8/oLBMNvg9in8.glb \
#     --dataset data/matterport3D/mp3d.scene_dataset_config.json \
#     --width 960 --height 640 \
#     --saved_path test_output/scanet_action_5.27