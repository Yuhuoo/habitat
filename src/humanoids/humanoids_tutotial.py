
#*******************************************************************************************
#
# Keynotes: Humanoids with HUMANISE Driver
# Reference: submodules/habitat-lab/examples/tutorials/humanoids_tutorial.ipynb
# Author: Gaoao
# Time: 2025/6/29
# 
#*******************************************************************************************


import habitat_sim
import magnum as mn
import warnings
warnings.filterwarnings('ignore')
from habitat_sim.utils.settings import make_cfg
from matplotlib import pyplot as plt
from habitat_sim.utils import viz_utils as vut
from omegaconf import DictConfig
import numpy as np
from habitat.articulated_agents.robots import FetchRobot
from habitat.config.default import get_agent_config
from habitat.config.default_structured_configs import ThirdRGBSensorConfig, HeadRGBSensorConfig, HeadPanopticSensorConfig
from habitat.config.default_structured_configs import SimulatorConfig, HabitatSimV0Config, AgentConfig
from habitat.config.default import get_agent_config
import habitat
from habitat_sim.physics import JointMotorSettings, MotionType
from omegaconf import OmegaConf
from habitat.articulated_agent_controllers import (
    HumanoidRearrangeController,
    HumanoidSeqPoseController,
)
from habitat.config.default_structured_configs import HumanoidJointActionConfig, HumanoidPickActionConfig

from habitat.config.default_structured_configs import TaskConfig, EnvironmentConfig, DatasetConfig, HabitatConfig
from habitat.config.default_structured_configs import ArmActionConfig, BaseVelocityActionConfig, OracleNavActionConfig
from habitat.core.env import Env

def make_sim_cfg(agent_dict):
    # Start the scene config
    sim_cfg = SimulatorConfig(type="RearrangeSim-v0")
    
    # This is for better graphics
    sim_cfg.habitat_sim_v0.enable_hbao = True
    sim_cfg.habitat_sim_v0.enable_physics = True

    # Set up an example scene
    sim_cfg.scene = "data/hab3_bench_assets/hab3-hssd/scenes/103997919_171031233.scene_instance.json"
    sim_cfg.scene_dataset = "data/hab3_bench_assets/hab3-hssd/hab3-hssd.scene_dataset_config.json"
    sim_cfg.additional_object_paths = ['data/objects/ycb/configs/']

    # Set the scene agents
    cfg = OmegaConf.create(sim_cfg)
    cfg.agents = agent_dict
    cfg.agents_order = list(cfg.agents.keys())
    return cfg

def make_hab_cfg(agent_dict, action_dict):
    sim_cfg = make_sim_cfg(agent_dict)
    task_cfg = TaskConfig(type="RearrangeEmptyTask-v0")
    task_cfg.actions = action_dict
    env_cfg = EnvironmentConfig()
    dataset_cfg = DatasetConfig(type="RearrangeDataset-v0", data_path="data/hab3_bench_assets/episode_datasets/small_large.json.gz")
    
    hab_cfg = HabitatConfig()
    hab_cfg.environment = env_cfg
    hab_cfg.task = task_cfg
    hab_cfg.dataset = dataset_cfg
    hab_cfg.simulator = sim_cfg
    hab_cfg.simulator.seed = hab_cfg.seed

    return hab_cfg

def init_rearrange_env(agent_dict, action_dict):
    hab_cfg = make_hab_cfg(agent_dict, action_dict)
    res_cfg = OmegaConf.create(hab_cfg)
    return Env(res_cfg)


############### Initializing humanoids ######################
# Define the agent configuration
main_agent_config = AgentConfig()
urdf_path = "data/hab3_bench_assets/humanoids/female_0/female_0.urdf"
main_agent_config.articulated_agent_urdf = urdf_path
main_agent_config.articulated_agent_type = "KinematicHumanoid"
main_agent_config.motion_data_path = "data/hab3_bench_assets/humanoids/female_0/female_0_motion_data_smplx.pkl"

# Define sensors that will be attached to this agent, here a third_rgb sensor and a head_rgb.
# We will later talk about why giving the sensors these names
main_agent_config.sim_sensors = {
    "third_rgb": ThirdRGBSensorConfig(),
    "head_rgb": HeadRGBSensorConfig(),
}

# We create a dictionary with names of agents and their corresponding agent configuration
agent_dict = {"main_agent": main_agent_config}

# Define the actions
action_dict = {"humanoid_joint_action": HumanoidJointActionConfig()}
env = init_rearrange_env(agent_dict, action_dict)
env.reset()

###########################################################################################
#
#  Task Choice
#
###########################################################################################

task1 = True
task2 = False
task3 = True
task4 = False
task5 = False
task6 = False

###########################################################################################
#
#  Task 1: Set the agent joints and rotations manually.
#
# ###########################################################################################
if task1:

    observations = []
    num_iter = 100
    pos_delta = mn.Vector3(0.00, 0.00, 0.02)
    print("Position Delta:", pos_delta)
    rot_delta = np.pi / (16 * num_iter)
    print("Rotation Delta:", rot_delta)
    
    # set agent
    sim = env.sim; sim.reset()
    art_agent = sim.articulated_agent
    art_agent.base_pos = mn.Vector3(-3.5, 0.1, -1.6)
    art_agent.base_rot = -45.0

    # move and look
    for _ in range(num_iter):
        art_agent.base_pos = art_agent.base_pos + pos_delta
        art_agent.base_rot = art_agent.base_rot + rot_delta
        sim.step({})
        observations.append(sim.get_sensor_observations())

    vut.make_video(
        observations,
        "third_rgb",
        "color",
        "robot_tutorial_video_task1",
        open_vid=False,
    )
    
  
###########################################################################################
#
#  Task 2: Using the HumanoidJointAction to sample new rotations and joints
#
###########################################################################################
if task2:
    ############## new rotations and joints #######################
    # TODO: maybe we can make joint_action a subclass of dict, and have a custom function for it
    import random
    def random_rotation():
        random_dir = mn.Vector3(np.random.rand(3)).normalized()
        random_angle = random.random() * np.pi
        random_rat = mn.Quaternion.rotation(mn.Rad(random_angle), random_dir)
        return random_rat

    def custom_sample_humanoid():
        base_transform = mn.Matrix4() 
        random_rot = random_rotation()
        offset_transform = mn.Matrix4.from_(random_rot.to_matrix(), mn.Vector3())
        joints = []
        num_joints = 54
        for _ in range(num_joints):
            Q = random_rotation()
            joints = joints + list(Q.vector) + [float(Q.scalar)]
        offset_trans = list(np.asarray(offset_transform.transposed()).flatten())
        base_trans = list(np.asarray(base_transform.transposed()).flatten())
        random_vec = joints + offset_trans + base_trans
        return {
            "human_joints_trans": random_vec
        }
        
    # We can now call the defined actions
    observations = []
    num_iter = 40
    env.reset()

    for _ in range(num_iter):
        params = custom_sample_humanoid()
        action_dict = {
            "action": "humanoid_joint_action",
            "action_args": params
        }
        observations.append(env.step(action_dict))
        
    vut.make_video(
        observations,
        "third_rgb",
        "color",
        "robot_tutorial_video_task2",
        open_vid=False,
    )

###########################################################################################
# 
# task 3: HumanoidSeqPoseController
#
# While you can manually set the agent joints and rotations, or train a policy to set them, 
# in many cases you may just be interested in generating realistic motions, 
# either coming from a separate model or from motion capture data. 
# For this, we introduce the HumanoidControllers, which are helper classes to calculate humanoid poses. 
# We currently provide two of them, which we cover here.
#
###########################################################################################
if task3:
    from habitat.utils.humanoid_utils import MotionConverterSMPLX
    PATH_TO_URDF = "data/humanoids/humanoid_data/female_2/female_2.urdf"
    PATH_TO_MOTION_NPZ = "data/humanoids/humanoid_data/walk_motion/CMU_10_04_stageii.npz"
    convert_helper = MotionConverterSMPLX(urdf_path=PATH_TO_URDF)

    # import pdb; pdb.set_trace()
    convert_helper.convert_motion_file(
        motion_path = PATH_TO_MOTION_NPZ,
        output_path = PATH_TO_MOTION_NPZ.replace(".npz", ""),
    )

    # env.reset()
    motion_path = "data/humanoids/humanoid_data/walk_motion/CMU_10_04_stageii.pkl"
    
    
    PATH_TO_HUMANISE_PKL = "data/HUMANISE/motion_for_habitat.pkl"

    
    # We define here humanoid controller
    humanoid_controller = HumanoidSeqPoseController(PATH_TO_HUMANISE_PKL)

    # Because we want the humanoid controller to generate a motion relative to the current agent, we need to set
    # the reference pose
    humanoid_controller.reset(env.sim.articulated_agent.base_transformation)
    humanoid_controller.apply_base_transformation(env.sim.articulated_agent.base_transformation)

    observations = []
    for _ in range(humanoid_controller.humanoid_motion.num_poses):
        # These computes the current pose and calculates the next pose
        humanoid_controller.calculate_pose()
        humanoid_controller.next_pose()
        
        # The get_pose function gives as a humanoid pose in the same format as HumanoidJointAction
        new_pose = humanoid_controller.get_pose()
        action_dict = {
            "action": "humanoid_joint_action",
            "action_args": {"human_joints_trans": new_pose}
        }
        observations.append(env.step(action_dict))
        
    vut.make_video(
        observations,
        "third_rgb",
        "color",
        "robot_tutorial_video_task3",
        open_vid=False,
    )

## =======================================  task 4 ======================================= ##
if task4:
    ########## humanoids navigation and interaction ################
    # As before, we first define the controller, here we use a special motion file we provide for each agent.
    motion_path = "data/hab3_bench_assets/humanoids/female_0/female_0_motion_data_smplx.pkl"

    # We define here humanoid controller
    humanoid_controller = HumanoidRearrangeController(motion_path)

    ########## 1. given direction
    # We reset the controller
    env.reset()
    humanoid_controller.reset(env.sim.articulated_agent.base_transformation)
    observations = []
    print(env.sim.articulated_agent.base_pos)
    for _ in range(100):
        # This computes a pose that moves the agent to relative_position
        relative_position = env.sim.articulated_agent.base_pos + mn.Vector3(0,-1,1)
        humanoid_controller.calculate_walk_pose(relative_position)
        
        # The get_pose function gives as a humanoid pose in the same format as HumanoidJointAction
        new_pose = humanoid_controller.get_pose()
        action_dict = {
            "action": "humanoid_joint_action",
            "action_args": {"human_joints_trans": new_pose}
        }
        observations.append(env.step(action_dict))
        
    vut.make_video(
        observations,
        "third_rgb",
        "color",
        "robot_tutorial_video_task4",
        open_vid=False,
    )
    # import pdb; pdb.set_trace()

## =======================================  task 5 ======================================= ##
if task5:
    ######### 2. difference positions with the hand
    
    # As before, we first define the controller, here we use a special motion file we provide for each agent.
    motion_path = "data/hab3_bench_assets/humanoids/female_0/female_0_motion_data_smplx.pkl" 
    # We define here humanoid controller
    humanoid_controller = HumanoidRearrangeController(motion_path)

    # We reset the controller
    env.reset()
    humanoid_controller.reset(env.sim.articulated_agent.base_transformation)
    observations = []
    print(env.sim.articulated_agent.base_pos)

    # Get the hand pose
    offset =  env.sim.articulated_agent.base_transformation.transform_vector(mn.Vector3(0, 0.3, 0))
    hand_pose = env.sim.articulated_agent.ee_transform(0).translation + offset
    for _ in range(100):
        # This computes a pose that moves the agent to relative_position
        hand_pose = hand_pose + mn.Vector3((np.random.rand(3) - 0.5) * 0.2)
        humanoid_controller.calculate_reach_pose(hand_pose, index_hand=0)
        
        # The get_pose function gives as a humanoid pose in the same format as HumanoidJointAction
        new_pose = humanoid_controller.get_pose()
        action_dict = {
            "action": "humanoid_joint_action",
            "action_args": {"human_joints_trans": new_pose}
        }
        observations.append(env.step(action_dict))
        
    vut.make_video(
        observations,
        "third_rgb",
        "color",
        "robot_tutorial_video_task5",
        open_vid=False,
    )

## =======================================  task 6 ======================================= ##
if task6:
    #################### Executing Humanoid actions ############
    # Define the actions
    action_dict = {
        "humanoid_joint_action": HumanoidJointActionConfig(),
        "humanoid_navigate_action": OracleNavActionConfig(type="OracleNavCoordinateAction", 
                                                        motion_control="human_joints",
                                                        spawn_max_dist_to_obj=1.0),
        "humanoid_pick_obj_id_action": HumanoidPickActionConfig(type="HumanoidPickObjIdAction")
        
    }
    env = init_rearrange_env(agent_dict, action_dict)

    ####### excute the task
    env.reset()
    rom = env.sim.get_rigid_object_manager()
    # env.sim.articulated_agent.base_pos = init_pos
    # As before, we get a navigation point next to an object id

    obj_id = env.sim.scene_obj_ids[0]
    first_object = rom.get_object_by_id(obj_id)

    object_trans = first_object.translation
    print(first_object.handle, "is in", object_trans)
    # TODO: unoccluded object did not work
    # print(sample)
    observations = []
    delta = 2.0

    object_agent_vec = env.sim.articulated_agent.base_pos - object_trans
    object_agent_vec.y = 0
    dist_agent_object = object_agent_vec.length()
    # Walk towards the object

    agent_displ = np.inf
    agent_rot = np.inf
    prev_rot = env.sim.articulated_agent.base_rot
    prev_pos = env.sim.articulated_agent.base_pos
    while agent_displ > 1e-9 or agent_rot > 1e-9:
        prev_rot = env.sim.articulated_agent.base_rot
        prev_pos = env.sim.articulated_agent.base_pos
        action_dict = {
            "action": ("humanoid_navigate_action"), 
            "action_args": {
                "oracle_nav_lookat_action": object_trans,
                "mode": 1
            }
        }
        observations.append(env.step(action_dict))
        
        cur_rot = env.sim.articulated_agent.base_rot
        cur_pos = env.sim.articulated_agent.base_pos
        agent_displ = (cur_pos - prev_pos).length()
        agent_rot = np.abs(cur_rot - prev_rot)
        
    # Wait
    for _ in range(20):
        action_dict = {"action": (), "action_args": {}}
        observations.append(env.step(action_dict))

    # Pick object
    observations.append(env.step(action_dict))
    for _ in range(100):
        
        action_dict = {"action": ("humanoid_pick_obj_id_action"), "action_args": {"humanoid_pick_obj_id": obj_id}}
        observations.append(env.step(action_dict)) 
        
    vut.make_video(
        observations,
        "third_rgb",
        "color",
        "robot_tutorial_video_6",
        open_vid=False,
    )

    import pdb; pdb.set_trace()