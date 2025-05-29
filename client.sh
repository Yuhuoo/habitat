data_dir="data/scanet/high_res/scans"
scene="scene0001_01"
glb_path="$data_dir/$scene/${scene}_vh_clean.glb"
saved_path="$data_dir/$scene"
python viewer_client.py \
    --scene $glb_path \
    --data data/scanet/high_res/scene_dataset_config.json \
    --width 960 --height 640 \
    --saved_path $saved_path \
    --sensor_height 0.7

# python viewer_client.py \
#     --scene data/matterport3D/oLBMNvg9in8/oLBMNvg9in8.glb \
#     --dataset data/matterport3D/mp3d.scene_dataset_config.json \
#     --width 960 --height 640 \
#     --saved_path test_output/scanet_action_5.27