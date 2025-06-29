data_dir="data/datasets/Scanet/high_res/scans"
scene="scene0000_00"
glb_path="$data_dir/$scene/${scene}_vh_clean.glb"
echo "GLB Path: $glb_path"
saved_path="test_output/scanet_scene0000_00_action_6.15-1"
python src/viewer_client.py \
    --scene $glb_path \
    --data data/scanet/high_res/scene_dataset_config.json \
    --width 512 --height 384 \
    --saved_path $saved_path \
    --sensor_height 0.7

# python src/viewer_client.py \
#     --scene data/datasets/Scanet/high_res/scans/scene0046_00/scene0046_00_vh_clean.glb \
#     --dataset data/datasets/Scanet/high_res/scene_dataset_config.json \
#     --width 640 --height 512 \
#     --start_position "2.7847664,  1.5457658, -2.013033" \
#     --start_rotation "0.769103646278381, -0.111514739692211, 0.622807800769806, 0.0903028398752213" \
#     --saved_path test_output/scene0046_00 \
#     --sensor_height 0.0 \
#     --debug

# python src/viewer_client.py \
#     --scene data/datasets/Scanet/high_res/scans/scene0050_00/scene0050_00_vh_clean.glb \
#     --dataset data/datasets/Scanet/high_res/scene_dataset_config.json \
#     --width 960 --height 640 \
#     --start_position "5.3083987,  1.8769357,  -1.778661" \
#     --start_rotation "-0.86167573928833, 0.105547100305557, -0.49615478515625, -0.0143296699970961" \
#     --saved_path data/datasets/Scanet/high_res/scans/scene0050_00 \
#     --sensor_height 0.0

# python src/viewer_client.py \
#     --scene data/datasets/Scanet/high_res/scans/scene0084_00/scene0084_00_vh_clean.glb \
#     --dataset data/datasets/Scanet/high_res/scene_dataset_config.json \
#     --width 960 --height 640 \
#     --start_position "0.13359478,  0.7007719,  -1.9164732" \
#     --start_rotation "0.382067292928696, -0.0216953512281179, -0.922393977642059, -0.0523773208260536" \
#     --saved_path data/datasets/Scanet/high_res/scans/scene0084_00 \
#     --sensor_height 0.0