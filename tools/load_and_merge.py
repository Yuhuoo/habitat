import os
import numpy as np
import open3d as o3d
from tqdm import tqdm
from scipy.spatial.transform import Rotation
from PIL import Image
import cv2

# 设置路径
data_dir = "test_output/scanet_action_6.8"
pose_file = os.path.join(data_dir, "pose.txt")
depth_dir = os.path.join(data_dir, "depth")
color_dir = os.path.join(data_dir, "color")
output_pcd_path = os.path.join(data_dir, "colored_pointcloud.ply")

# 加载相机内参
intrinsics = np.loadtxt(os.path.join(data_dir, "intrinsics.txt"))
fx, fy = intrinsics[0, 0], intrinsics[1, 1]
cx, cy = intrinsics[0, 2], intrinsics[1, 2]

# 加载位姿文件
poses = np.loadtxt(pose_file)  # 形状为 (N, 7)
num_frames = len([f for f in os.listdir(color_dir) if f.endswith('.png')])

# 初始化点云
colored_pcd = o3d.geometry.PointCloud()

# 逐帧处理（每10帧处理一次）
target_frame_idx = range(0, num_frames, 10)
for frame_idx in tqdm(target_frame_idx, desc="Processing frames"):
    try:
        # 加载深度图
        depth = np.load(os.path.join(depth_dir, f"{frame_idx}.npy"))
        
        # 加载并处理RGBA图像（去除Alpha通道）
        color_img = Image.open(os.path.join(color_dir, f"{frame_idx}.png"))
        rgba = np.array(color_img)
        rgb = rgba[..., :3]  # 提取RGB通道（忽略Alpha）
        
        # 检查数据对齐
        assert depth.shape == rgb.shape[:2], \
               f"Depth和RGB尺寸不匹配 {depth.shape} vs {rgb.shape[:2]}"

        # 反投影点云
        height, width = depth.shape
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        z = depth
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
        colors = rgb.reshape(-1, 3) / 255.0  # 归一化
        
        # [x, y, z] -> [x, -y, -z]
        # https://github.com/facebookresearch/habitat-sim/issues/2494#issuecomment-2466709481
        transform_matrix = np.array([
            [1,  0,  0],
            [0, -1,  0],
            [0,  0, -1]
        ])
        points = points @ transform_matrix.T

        # 过滤无效点（根据场景调整阈值）
        valid_mask = (z.flatten() > 0.1) & (z.flatten() < 5.0)
        points = points[valid_mask]
        colors = colors[valid_mask]
        
        # 不可能为0
        assert len(points) != 0
        
        # 获取当前帧位姿
        pose = poses[frame_idx]
        position = pose[:3]
        quaternion = pose[3:]
        
        # 构建变换矩阵
        rotation = Rotation.from_quat(quaternion).as_matrix()
        pose_matrix = np.eye(4)
        pose_matrix[:3, :3] = rotation
        pose_matrix[:3, 3] = position

        # 坐标变换
        points_hom = np.hstack([points, np.ones((len(points), 1))])
        points_global = (pose_matrix @ points_hom.T).T[:, :3]
        
        # 添加到点云
        colored_pcd.points = o3d.utility.Vector3dVector(
            np.vstack([np.asarray(colored_pcd.points), points_global]))
        colored_pcd.colors = o3d.utility.Vector3dVector(
            np.vstack([np.asarray(colored_pcd.colors), colors]))
    
    except Exception as e:
        print(f"处理帧 {frame_idx} 时出错: {str(e)}")
        continue

# 点云优化
colored_pcd = colored_pcd.voxel_down_sample(voxel_size=0.02)  # 体素下采样
colored_pcd, _ = colored_pcd.remove_statistical_outlier(
    nb_neighbors=20, std_ratio=1.5)  # 统计滤波去噪

# 保存结果
o3d.io.write_point_cloud(output_pcd_path, colored_pcd)
print(f"带颜色的点云已保存到: {output_pcd_path}")

# 高级可视化配置
vis = o3d.visualization.Visualizer()
vis.create_window(width=1280, height=720, window_name='3D Reconstruction')
vis.add_geometry(colored_pcd)

# 设置渲染参数
render_opt = vis.get_render_option()
render_opt.background_color = np.array([0.05, 0.05, 0.05])  # 深灰背景
render_opt.point_size = 1.5  # 点大小
render_opt.light_on = True  # 启用光照

# 添加坐标系（世界坐标系，尺度1.0）
coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
vis.add_geometry(coord_frame)

vis.run()
vis.destroy_window()