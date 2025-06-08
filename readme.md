# Habitat Data Collection Tool

**Description**: A tool for efficient data collection (Colmop-style) using Habitat-sim, designed to optimize rendering performance by separating path planning (local) from rendering (server).

## Key Features
- Local path visualization on MacBook Air M1
- Server-based rendering on Ubuntu
- Remote Camera pose visualization

## TODO
- 3DGS training and remote visualization

## Installation
Reference from https://github.com/facebookresearch/habitat-sim

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

## Visualize the camera pose
```bash
bash visualize.sh
```

## Load and merge point cloud for a complete scene
```bash
python tools/load_and_merge.py
```