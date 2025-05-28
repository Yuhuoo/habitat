
scene=data/scanet/high_res/scans/scene0006_00/scene0006_00_vh_clean.glb
saved_path=data/scanet/high_res/scans/scene0006_00
python viewer_client.py \
    --scene $scene \
    --data data/scanet/high_res/scene_dataset_config.json \
    --width 960 --height 640 \
    --saved_path $saved_path \
    --start_position "[2.2184856, 1.6997461, -1.5994138]" \
    --start_rotation "[0.983254909515381, 0, -0.182235538959503, 0]" \
    --sensor_height 1.5 

# python viewer_client.py \
#     --scene data/matterport3D/oLBMNvg9in8/oLBMNvg9in8.glb \
#     --dataset data/matterport3D/mp3d.scene_dataset_config.json \
#     --width 960 --height 640 \
#     --saved_path test_output/scanet_action_5.27