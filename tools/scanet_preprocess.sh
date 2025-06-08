#!/bin/bash

# 定义输入和输出目录
input_dir="../data/datasets/Scannet/scans"
for scene_dir in "$input_dir"/*/; do
    scene_dir="${scene_dir%/}"
    if [ -d "$scene_dir" ]; then
    
        scene_name=$(basename "$scene_dir")
        input_file="$scene_dir/${scene_name}_vh_clean.ply"
        
        if [ -f "$input_file" ]; then

            output_file="$scene_dir/${scene_name}_vh_clean.glb"

            assimp export "$input_file" "$output_file"
            if [ $? -eq 0 ]; then
                echo "成功转换: $input_file -> $output_file"
            else
                echo "转换失败: $input_file"
            fi
        else
            echo "输入文件不存在: $input_file"
        fi
    else
        echo "目录不存在: $scene_dir"
    fi
done
