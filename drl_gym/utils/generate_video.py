import imageio
import cv2
import numpy as np
import argparse
import robosuite
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.wrappers.gym_wrapper import GymWrapper 

from robosuite.environments.manipulation.lift_and_place_barrier import LiftAndPlaceBarrier
from robosuite.environments.manipulation.lift_base import BaseLift
from robosuite.environments.manipulation.lift_with_terminals import LiftWithTerminals
from robosuite.environments.manipulation.lift_and_place_with_terminals import LiftAndPlaceWithTerminals
from robosuite.environments.manipulation.lift_and_place import LiftAndPlace
from robosuite.environments.base import register_env

# collections.Iterable deprecated in python 3.10
import collections
collections.Iterable = collections.abc.Iterable

register_env(LiftAndPlaceBarrier)
register_env(BaseLift)
register_env(LiftWithTerminals)
register_env(LiftAndPlaceWithTerminals)
register_env(LiftAndPlace)

from stable_baselines3 import SAC, PPO, TD3
from stable_baselines3.common.base_class import BaseAlgorithm

from sac_policy_reuse import SACPolicyReuse
from td3_policy_reuse import TD3PolicyReuse


DEFAULT_CONTROLLER = robosuite.load_controller_config(default_controller="OSC_POSE")

def make_keys(env):
    keys = []
    keys += ["object-state"]
    # Iterate over all robots to add to state
    for idx in range(len(env.robots)):
        keys += ["robot{}_proprio-state".format(idx)]   
    return keys

def flatten_obs(obs_dict, keys):
    ob_lst = []
    for key in obs_dict:
        if "_image" not in key:
            ob_lst.append(np.array(obs_dict[key]).flatten())
    return np.concatenate(ob_lst)


def step(self, action):
    ob_dict, reward, done, info = self.env.step(action)
    return ob_dict, reward, done, info
GymWrapper.step = step

def reset(self):
    ob_dict = self.env.reset()
    return ob_dict
GymWrapper.reset = reset

def main(args):
    output_path = f"{args.logdir}{args.env_name}_{args.robot}_SEED{args.seed}/{args.algorithm}/"
    
    env = robosuite.make(
        args.env_name,
        args.robot,
        controller_configs=DEFAULT_CONTROLLER,
        has_renderer=False,
        ignore_done=True,
        use_camera_obs=True,
        use_object_obs=True,
        camera_names=args.camera,
        camera_heights=args.height,
        camera_widths=args.width,
        hard_reset=True,
    )
    
    writer = imageio.get_writer(args.video_path, fps=20)
    
    obs_keys = make_keys(env)
    
    env = GymWrapper(env)
    if args.algorithm == "SAC":
        model = SAC.load(f"{output_path}best_model.zip")
    elif args.algorithm == "SACPolicyReuse":
        model = SACPolicyReuse.load(f"{args.logdir}{args.env_name}_{args.robot}_SEED{args.seed}/best_model.zip")
    elif args.algorithm == "TD3":
        model = TD3.load(f"{args.logdir}{args.env_name}_{args.robot}_SEED{args.seed}/TD3PolicyReuse/best_model.zip")
    elif args.algorithm == "TD3PolicyReuse":
        model = TD3PolicyReuse.load(f"{output_path}best_model.zip")
        
    for _ in range(args.num_iter):
        obs_dict = env.reset()
        frames = []
        have_pos_reward = True
        for i in range(args.timesteps):

            action, _states = model.predict(flatten_obs(obs_dict, obs_keys), deterministic=True)
            obs_dict, reward, done, info = env.step(action)


            frame = obs_dict[args.camera + "_image"]
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            frames.append(frame)
            
            if reward > 0:
                have_pos_reward = True     

            if done:
                break
            
        if have_pos_reward:
            for frame in frames:
                writer.append_data(frame)

    writer.close()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, default='default_log/')
    parser.add_argument("--robot", type=str, default="Panda")
    parser.add_argument("--env_name", type=str, default="BaseLift")
    parser.add_argument("--algorithm", type=str, default="SAC")
    parser.add_argument("--seed", type=int, default=69)
    parser.add_argument("--camera", type=str, default="frontview")
    parser.add_argument("--video_path", type=str, default="video.mp4")
    parser.add_argument("--timesteps", type=int, default=500)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--skip_frame", type=int, default=1)
    parser.add_argument("--num_iter", type=int, default=10)
    args = parser.parse_args()
    main(args)