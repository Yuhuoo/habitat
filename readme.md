# A Data Collection Tool Based on Habitat-sim
**Target:** To optimize the rendering efficiency during data collection, a method of using a local machine (MAC air m1) to visualize the movement path of the mobile robot, and then uploading the path (action.txt) to the server (Ubuntu) for rendering is adopted.

## Get Started
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