# Habitat Data Collection Tool

**Description**: A efficient data collection tool (Colmap-style) based on Habitat-sim. The system optimizes rendering performance by decoupling path planning (local client) from rendering (server).

## Key Features
- Local path visualization on MacBook Air M1
- Server-based rendering on Ubuntu
- Remote Camera pose visualization

## TODO
- [ ] Integrate 3D Gaussian Splatting (3DGS) training pipeline
- [ ] Implement remote visualization for 3DGS results

## Getting Started

### Prerequisites
- Habitat-sim installation (see [official documentation](https://github.com/facebookresearch/habitat-sim))

### Getting this project
```bash
git clone --recursive git@github.com:Yuhuoo/habitat.git
cd habitat
```

## Usage
- 1、Visualize and Record the Movement Path on the Local Machine
```bash
bash client.sh
```

- 2、Copy action.txt to the Server
```bash
scp output/action.txt xxx@ip:xxx
```

- 3、Render on the Server Based on action.txt and Save the data
```bash
bash server.sh
```

## ScanNet Preprocessing
For ScanNet datasets, preprocess point clouds first:
```bash
bash tools/scanet_preprocess.sh
```
## Camera Pose Visualization
```bash
bash tools/visualize.sh
```

## Load and merge point cloud for a complete scene
```bash
python tools/load_and_merge.py
```