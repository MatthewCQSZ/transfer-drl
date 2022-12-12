import imageio
import cv2
import numpy as np
import argparse
import robosuite
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.wrappers.gym_wrapper import GymWrapper 

from environments.lift_base import BaseLift
from environments.lift_and_place import LiftAndPlace
from environments.lift_and_place_barrier import LiftAndPlaceBarrier
from robosuite.environments.base import register_env

# collections.Iterable deprecated in python 3.10
#import collections
#collections.Iterable = collections.abc.Iterable

register_env(LiftAndPlaceBarrier)
register_env(BaseLift)
register_env(LiftAndPlace)

from stable_baselines3 import SAC, PPO, TD3
from stable_baselines3.common.base_class import BaseAlgorithm

from algorithm.sac_policy_reuse import SACPolicyReuse
from algorithm.td3_policy_reuse import TD3PolicyReuse


DEFAULT_CONTROLLER = robosuite.load_controller_config(default_controller="OSC_POSE")

# Function to create a dictionary of obs_keys, obd pairs
def make_keys(env:ManipulationEnv):
    keys = []
    keys += ["object-state"]
    # Iterate over all robots to add to state
    for idx in range(len(env.robots)):
        keys += ["robot{}_proprio-state".format(idx)]   
    return keys

# Helper function to flatten obs dictionary to np array
def flatten_obs(obs_dict, keys):
    ob_lst = []
    for key in obs_dict:
        if "_image" not in key:
            ob_lst.append(np.array(obs_dict[key]).flatten())
    return np.concatenate(ob_lst)

# Overwrite GymWrapper step and reset functions for video generation
def __step(self, action):
    ob_dict, reward, done, info = self.env.step(action)
    return ob_dict, reward, done, info

def __reset(self):
    ob_dict = self.env.reset()
    return ob_dict


HEIGHT = 512
WIDTH = 512

def generate_video(
    model_path:str, 
    video_path:str, 
    algorithm:str, 
    env_name:str, 
    robot:str, 
    camera:str = "frontview",
    num_iter: int = 10,
    timesteps: int = 500,
    reward_threshhold: float = 0.0,
    ):
    
    '''
    Generate video of manipulation task from model trained.
    
    :params model_path: directory to where the model is saved.
    :params video_path: path to the mp4 file which is the video generated.
    :params algorithm: name of the algorithm used,
        "PPO", "SAC", "TD3", "SOC", "SACPolicyReuse", "TD3PolicyReuse".
    :params env_name: name of the environment,
        "Lift", "BaseLift", "LiftAndPlace", "LiftAndPlaceBarrier".
    :params robot: name of the robot,
        "Panda", "Sawyer", "LBR IIWA 7", "Jaco", "Kinova Gen3", "UR5e".
    :params camera: name of the camera, e.g. "frontview", "agentview".
    :params num_iter: number of iterations to run for video generation.
    :params timesteps: number of timesteps per iteration.
    :paramsr eward_threshhold: cutoff for video saving. The video will not save iterations in which
        rewards at all timestamps are less than or equal to the threshold.
    '''
    
    env = robosuite.make(
        env_name,
        robot,
        controller_configs=DEFAULT_CONTROLLER,
        has_renderer=False,
        ignore_done=True,
        use_camera_obs=True,
        use_object_obs=True,
        camera_names=camera,
        camera_heights=HEIGHT,
        camera_widths=WIDTH,
        hard_reset=True,
    )
    
    writer = imageio.get_writer(video_path, fps=20)
    
    obs_keys = make_keys(env)
    
    # Overwrite GymWrapper step and reset functions for video generation
    GymWrapper.step = __step
    GymWrapper.reset = __reset
    env = GymWrapper(env)
    
    if algorithm == "SAC":
        model = SAC.load(f"{model_path}best_model.zip")
    elif algorithm == "SACPolicyReuse":
        model = SACPolicyReuse.load(f"{model_path}best_model.zip")
    elif algorithm == "TD3":
        model = TD3.load(f"{model_path}best_model.zip")
    elif algorithm == "TD3PolicyReuse":
        model = TD3PolicyReuse.load(f"{model_path}best_model.zip")
        
    for _ in range(num_iter):
        obs_dict = env.reset()
        frames = []
        save_this_iter = True
        for i in range(timesteps):

            action, _states = model.predict(flatten_obs(obs_dict, obs_keys), deterministic=True)
            obs_dict, reward, done, info = env.step(action)


            frame = obs_dict[camera + "_image"]
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            frames.append(frame)
            
            if reward > reward_threshhold:
                save_this_iter = True     

            if done:
                break
            
        if save_this_iter:
            for frame in frames:
                writer.append_data(frame)

    writer.close()
    