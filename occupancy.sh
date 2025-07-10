#!/bin/bash

# 显式初始化 Conda
source "$HOME/software/miniconda3/etc/profile.d/conda.sh"  # 根据你的安装路径调整
conda activate mayavi || {
    echo "ERROR: Failed to activate 'mayavi' conda environment"
    exit 1
}

# 添加错误检查
xvfb-run -s "-screen 0 1600x900x24" python src/occupancy/habitatocc.py --data_path example_data/demo || {
    echo "ERROR: Command failed with exit code $?"
    exit 1
}