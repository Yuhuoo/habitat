import os
import numpy as np
import open3d as o3d
from tqdm import tqdm
from scipy.spatial.transform import Rotation
from PIL import Image
from colmap_loader import read_extrinsics_text, read_intrinsics_text

# 坐标系转换
def transform_coordinates(points):
    """
    将点云从 Y-up (X前, Z右) 转换为 Z-up (X前, Y左)
    转换规则：
        new_x = original_x (X保持不变)
        new_y = -original_z (原来的右方向变为左方向)
        new_z = original_y (原来的上方向变为上方向)
    """
    transformed_points = np.zeros_like(points)
    transformed_points[:, 0] = points[:, 0]  # X -> X (forward)
    transformed_points[:, 1] = -points[:, 2]  # Z -> -Y (right to left)
    transformed_points[:, 2] = points[:, 1]   # Y -> Z (up to up)
    return transformed_points

def main(data_dir):
    pose_file = os.path.join(data_dir, "sparse/0/images.txt")
    intrinsics_file = os.path.join(data_dir, "sparse/0/cameras.txt")
    depth_dir = os.path.join(data_dir, "depth")
    color_dir = os.path.join(data_dir, "images")
    output_pcd_path = os.path.join(data_dir, "sparse/0/pointcloud.ply")

    # 加载相机内参
    intrinsics = read_intrinsics_text(intrinsics_file)[1]
    fx, fy, cx, cy = intrinsics.params

    # 加载位姿文件
    poses_dict = read_extrinsics_text(pose_file)  # 形状为 (N, 7)
    num_frames = len(poses_dict)

    # 初始化点云
    colored_pcd = o3d.geometry.PointCloud()

    # 逐帧处理
    target_frame_idx = range(1, num_frames+1)
    for frame_id, pose in tqdm(poses_dict.items(), desc="Processing frames"):
        idx_chr = os.path.splitext(pose.name)[0].split("_")[-1]
        frame_idx = int(idx_chr)
        try:
            # 加载深度图
            depth = np.load(os.path.join(depth_dir, f"observation_depth_{frame_idx}.npy"))
            
            # 加载并处理RGBA图像（去除Alpha通道）
            color_img = Image.open(os.path.join(color_dir, f"observation_rgb_{frame_idx}.png"))
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
            valid_mask = (z.flatten() > 0.1) & (z.flatten() < 100.0)
            points = points[valid_mask]
            colors = colors[valid_mask]
            
            # 不可能为0
            assert len(points) != 0
            
            # 获取当前帧位姿
            position = pose.tvec
            quaternion = pose.qvec
            
            # 构建变换矩阵
            rotation = Rotation.from_quat(quaternion).as_matrix()
            pose_matrix = np.eye(4)
            pose_matrix[:3, :3] = rotation
            pose_matrix[:3, 3] = position

            # 转换到世界坐标系
            points_hom = np.hstack([points, np.ones((len(points), 1))])
            points_global = (pose_matrix @ points_hom.T).T[:, :3]
            
            # 转换到scanet坐标系
            transformed_points_global = transform_coordinates(points_global)
            
            # 添加到点云
            colored_pcd.points = o3d.utility.Vector3dVector(
                np.vstack([np.asarray(colored_pcd.points), transformed_points_global]))
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
    print(f"点云已保存至: {output_pcd_path}")

        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    
    # optional arguments
    parser.add_argument(
        "--data_dir",
        default="output/scanet/test/scene0050_00",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    
    print(f"处理数据：{args.data_dir}")
    main(args.data_dir)