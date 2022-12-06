import imageio
import numpy as np
import argparse
import robosuite
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.wrappers.gym_wrapper import GymWrapper 

from robosuite.environments.manipulation.lift_with_terminals import LiftWithTerminals
from robosuite.environments.manipulation.lift_and_place_with_terminals import LiftAndPlaceWithTerminals
from robosuite.environments.base import register_env

register_env(LiftWithTerminals)
register_env(LiftAndPlaceWithTerminals)

from stable_baselines3 import SAC, PPO, TD3
from stable_baselines3.common.base_class import BaseAlgorithm


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
    )
    
    writer = imageio.get_writer(args.video_path, fps=20)
    
    obs_keys = make_keys(env)
    
    env = GymWrapper(env)
    model = SAC.load("best_model.zip")
    obs_dict = env.reset()

    
    for i in range(args.timesteps):

        action, _states = model.predict(flatten_obs(obs_dict, obs_keys), deterministic=True)
        obs_dict, reward, done, info = env.step(action)


        frame = obs_dict[args.camera + "_image"]
        writer.append_data(frame)
        print(f"Saving frame #{i}, receiving reward {reward}")

        if done:
            break

    writer.close()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="LiftWithTerminals")
    parser.add_argument("--robot", type=str, default="Panda")
    parser.add_argument("--camera", type=str, default="frontview")
    parser.add_argument("--video_path", type=str, default="video_lift_terminal_sac.mp4")
    parser.add_argument("--timesteps", type=int, default=500)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--skip_frame", type=int, default=1)
    args = parser.parse_args()
    main(args)