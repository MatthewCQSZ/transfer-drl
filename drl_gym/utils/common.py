import numpy as np
import gym

from stable_baselines3 import SAC
from stable_baselines3 import PPO
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.base_class import BaseAlgorithm

from algorithm.sac_policy_reuse import SACPolicyReuse
from algorithm.td3_policy_reuse import TD3PolicyReuse

# collections.Iterable deprecated in python 3.10
#import collections
#collections.Iterable = collections.abc.Iterable

import robosuite
from robosuite.wrappers.gym_wrapper import GymWrapper 
from robosuite import load_controller_config
from robosuite.environments.base import register_env

from environments.lift_base import BaseLift
from environments.lift_and_place import LiftAndPlace
from environments.lift_and_place_barrier import LiftAndPlaceBarrier


register_env(LiftAndPlaceBarrier)
register_env(BaseLift)
register_env(LiftAndPlace)
    

# Default configuration for controller and environment, same for all environments
DEFAULT_CONTROLLER = load_controller_config(default_controller="OSC_POSE")
DEFAULT_CONFIG = {
        "controller_configs": DEFAULT_CONTROLLER,
        "horizon": 500,
        "control_freq": 20,
        "reward_shaping": True,
        "reward_scale": 1.0,
        "has_offscreen_renderer": False,
        "use_camera_obs": False,
        "use_object_obs": True,
        "ignore_done": False,
        "hard_reset": False,
    }
GLOBAL_STEP = 0

# Override _flatten_obs() of GymWrapper for Robosuite to unify obs space as float32
def __flatten_obs_with_type_casting(self, obs_dict, verbose=False, use_image=False):
    ob_lst = []
    for key in obs_dict:
        if verbose:
            print("adding key: {}".format(key))
                
        ob_lst.append(np.array(obs_dict[key]).flatten())
    return np.concatenate(ob_lst).astype(np.float32)
GymWrapper._flatten_obs = __flatten_obs_with_type_casting

# Add call method to GymWrapper
def ____call__(self): 
    return GymWrapper(self.env)
GymWrapper.__call__ = ____call__

def make_env(env_name:str, robot: str, seed:int) -> Monitor:
    '''
    Make DRL-Gym environment.
    
    :params env_name: name for the environment, "Lift", "BaseLift", "LiftAndPlace", "LiftAndPlaceBarrier".
    :params robot: name for the robot "Panda", "Sawyer", "LBR IIWA 7", "Jaco", "Kinova Gen3", "UR5e".
    :params seed: random seed.
    '''
    env = robosuite.make(env_name, robots=robot, **DEFAULT_CONFIG)
    env = Monitor(GymWrapper(env))
    env.seed(seed)
    return env
    
def make_algorithm(
    env: gym.Env, 
    algorithm:str, 
    buffer_size:int, 
    mu:float=0.0, 
    max_reuse_steps:int=0, 
    reuse_path:str=""
    ) -> BaseAlgorithm:
    
    '''
    Initialize algorithm model
    
    :params env: Gym environment.
    :params algorithm: name for the algorithm, "PPO", "SAC", "TD3", "SOC", "SACPolicyReuse", "TD3PolicyReuse".
    :params buffer_size: size for the replay buffer.
    :params mu: parameter mu in Policy Reuse
    :params max_reuse_steps: maximum number of steps for per episode in Policy Reuse.
    :params reuse_path: directory where the teacher policy for Policy Reuse is loaded from
    
    :return: stable baselines 3 BaseAlgorithm.
    '''
    
    if "PolicyReuse" in algorithm:
        assert mu != 0
        assert max_reuse_steps != 0 
        assert reuse_path != ""
        
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.2 * np.ones(n_actions))
    if algorithm == "SAC":
        model = SAC("MlpPolicy", 
                    env, 
                    learning_rate=3e-4,
                    buffer_size=int(buffer_size),
                    #batch_size=1024,
                    action_noise=action_noise, 
                    tau=0.005, 
                    gamma=0.99,
                    target_update_interval=2, 
                    verbose=1)
    elif algorithm == "PPO":
        model = PPO("MlpPolicy", env, buffer_size=int(buffer_size), verbose=1)
    elif algorithm == "TD3":
        model = TD3("MlpPolicy", 
                    env,
                    learning_rate=3e-4,
                    buffer_size=int(buffer_size),
                    action_noise=action_noise, 
                    tau=0.005,  
                    policy_delay=2,
                    verbose=1)
    elif algorithm == "SACPolicyReuse":
        model = SAC.load(f"{reuse_path}best_model")
        model = SACPolicyReuse("MlpPolicy", 
                               env, 
                               verbose=1, 
                               old_policy=model.policy,
                               reuse_mu=mu,
                               max_reuse_steps=max_reuse_steps)
    elif algorithm == "TD3PolicyReuse":
        model = TD3.load(f"{reuse_path}best_model")
        model = TD3PolicyReuse("MlpPolicy", 
                               env, 
                               verbose=1, 
                               old_policy=model.policy,
                               reuse_mu=mu,
                               max_reuse_steps=max_reuse_steps)
    else:
        raise ValueError("Not Supported Algorithm")
        
    return model