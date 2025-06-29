import os
import dataclasses
import numpy as np
import magnum as mn
import pickle as pkl 
from habitat.utils.humanoid_utils import MotionConverterSMPLX, smplx_body_joint_names

@dataclasses.dataclass
class Motion:
    mtype: str
    trans: np.ndarray
    orient: np.ndarray
    betas: np.ndarray
    pose_body: np.ndarray
    pose_hand: np.ndarray
    joints: np.ndarray

    @classmethod
    def from_path(cls, path: os.PathLike):
        with open(path, "rb") as fp:
            motion_data = pkl.load(fp)
            return Motion.from_tuple(motion_data)

    @classmethod
    def from_tuple(cls, motion_data: tuple):
        return cls(
            mtype=np.array2string(motion_data[0]),
            trans=motion_data[1],
            orient=motion_data[2],
            betas=motion_data[3],
            pose_body=motion_data[4],
            pose_hand=motion_data[5],
            joints=motion_data[8],
        )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    
    # optional arguments
    parser.add_argument(
        "--motion_path",
        type=str,
        default="example_data/humanise/004019_7b03303f-c9d8-44b6-88a9-3626df531d7f/motion.pkl",
    )
    parser.add_argument(
        "--PATH_TO_URDF",
        type=str,
        default="data/humanoids/humanoid_data/female_2/female_2.urdf",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="example_data/humanise/004019_7b03303f-c9d8-44b6-88a9-3626df531d7f/motion_for_habitat.pkl",
    )
    args = parser.parse_args()
    
    # 加载HUMANISE格式的运动数据
    content_motion = Motion.from_path(args.motion_path)
    
    # 加载URDF
    convert_helper = MotionConverterSMPLX(urdf_path=args.PATH_TO_URDF)
    
    # 第一种做法：work!
    # 构建pose_info字典 - 关键部分
    # 组合身体姿态和手部姿态
    full_pose = np.concatenate([
        content_motion.pose_body, 
        content_motion.pose_hand
    ], axis=1)
    
    # 添加缺失的面部关节（用零填充）
    num_frames = full_pose.shape[0]
    if full_pose.shape[1] < 162:  # 55关节 - 1根关节 = 54关节 × 3 = 162
        # 计算需要添加的零维度
        num_missing_dims = 162 - full_pose.shape[1]
        print(f"检测到缺失维度: {num_missing_dims} (可能是面部关节)")
        
        # 添加零填充
        face_pose = np.zeros((num_frames, num_missing_dims))
        full_pose = np.concatenate([full_pose, face_pose], axis=1)
    
    num_poses = content_motion.trans.shape[0]
    pose_info = {
        "trans": content_motion.trans,         # 全局平移
        "root_orient": content_motion.orient,  # 根关节朝向 (轴角表示)
        "pose": full_pose, # 身体+手部姿态 (轴角表示)
    }
    
    # ## 第二种做法：不work!
    # selected_joints_index = []
    # for model_link_id in range(len(convert_helper.joint_info)):
    #     joint_name = convert_helper.joint_info[model_link_id][1].decode("UTF-8")
    #     joint_index = convert_helper.index_joint_map[joint_name]
    #     selected_joints_index.append(joint_index - 1)
    # selected_joints = content_motion.joints[:, selected_joints_index, :] # (n, 54, 3)
    # num_poses = selected_joints.shape[0]
    # pose_info = {
    #     "trans": content_motion.trans,         # 全局平移
    #     "root_orient": content_motion.orient,  # 根关节朝向 (轴角表示)
    #     "pose": selected_joints.reshape(num_poses, -1), # 对齐后姿态 (轴角表示)
    # }
    

    # 逐帧处理转换
    transform_array = []
    joints_array = []
    for index in range(num_poses):
        # 转换单帧姿态为Habitat格式
        root_trans, root_rot, pose_quat = convert_helper.convert_pose_to_rotation(
            pose_info["trans"][index],    # 当前帧的平移
            pose_info["root_orient"][index],  # 当前帧的根关节朝向
            pose_info["pose"][index]      # 当前帧的身体姿态
        )
        
        # 构建4x4变换矩阵 (位置 + 旋转)
        transform_as_mat = np.array(mn.Matrix4.from_(root_rot, root_trans))
        transform_array.append(transform_as_mat[None, :])
        
        # 存储关节四元数
        joints_array.append(np.array(pose_quat)[None, :])

    # 合并所有帧的数据
    transform_array = np.concatenate(transform_array)
    joints_array = np.concatenate(joints_array)

    # 构建Habitat兼容的运动数据结构
    walk_motion = {
        "joints_array": joints_array,      # 关节旋转 (四元数)
        "transform_array": transform_array,  # 根变换矩阵 (4x4)
        "displacement": None,              # 位移信息 (可选)
        "fps": 30,                         # 帧率 (HUMANISE通常为30fps)
    }
    
    # 最终的Habitat运动数据
    habitat_motion = {
        "pose_motion": walk_motion,
    }
    
    # 保存为PKL文件
    with open(args.output_path, "wb+") as ff:
        pkl.dump(habitat_motion, ff)
    print(f"Successfully converted motion to Habitat format. Saved to: {args.output_path}")