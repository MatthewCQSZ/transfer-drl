import argparse
import numpy as np
import time

from stable_baselines3 import SAC
from stable_baselines3 import PPO
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

import robosuite
from robosuite.wrappers.gym_wrapper import GymWrapper 
from robosuite import load_controller_config
from robosuite.environments.manipulation.lift_with_terminals import LiftWithTerminals
from robosuite.environments.manipulation.lift_and_place_with_terminals import LiftAndPlaceWithTerminals
from robosuite.environments.base import register_env

register_env(LiftWithTerminals)
register_env(LiftAndPlaceWithTerminals)

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Log rewards
        self.logger.record("rewards", self.locals["rewards"][0])
        return True

# Default configuration for controller and environment, same for all environments
DEFAULT_CONTROLLER = load_controller_config(default_controller="OSC_POSE")
DEFAULT_CONFIG = {
        "controller_configs": DEFAULT_CONTROLLER,
        "horizon": 500,
        "control_freq": 20,
        "reward_shaping": False,
        "reward_scale": 1.0,
        "has_offscreen_renderer": False,
        "use_camera_obs": False,
        "use_object_obs": True,
        "ignore_done": False,
        "hard_reset": False,
    }
GLOBAL_STEP = 0

# Override _flatten_obs() of GymWrapper for Robosuite to unify obs space as float32
def _flatten_obs_with_type_casting(self, obs_dict, verbose=False, use_image=False):
    ob_lst = []
    for key in obs_dict:
        if verbose:
            print("adding key: {}".format(key))
                
        # Don't train on all camera pixels
        #if "_image" not in key or use_image:
        ob_lst.append(np.array(obs_dict[key]).flatten())
    return np.concatenate(ob_lst).astype(np.float32)
GymWrapper._flatten_obs = _flatten_obs_with_type_casting

# Add call method to GymWrapper
def __call__(self): 
    return GymWrapper(self.env)
GymWrapper.__call__ = __call__

def make_env(args):
    env = robosuite.make(args.env_name, robots=args.robot, **DEFAULT_CONFIG)

    #obs_dict = env.reset()
    #for k in obs_dict:
    #    print(k)
    #    print(obs_dict[k].shape)
    #    print(obs_dict[k])  
    #exit()  
    env = Monitor(GymWrapper(env))

    env.seed(args.seed)
    return env
    
def train(env, algorithm, output_path, num_steps, buffer_size, seed=None):
    # TODO: figure out why logging is not working
    logger = configure(output_path, ["stdout", "csv", "tensorboard"])
    eval_callback = EvalCallback(env, best_model_save_path=output_path,
                             log_path=output_path, eval_freq=500,
                             deterministic=True, render=False)
    checkpoint_callback = CheckpointCallback(save_freq=100000, 
                                             save_path=output_path,
                                             name_prefix="rl_model",
                                             save_replay_buffer=False,
                                             save_vecnormalize=False)
    
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.2 * np.ones(n_actions))
    #TODO: Increase GPU usage
    if algorithm == "SAC":
        model = SAC("MlpPolicy", 
                    env, 
                    learning_rate=3e-4,
                    buffer_size=int(buffer_size),
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
    else:
        raise ValueError("Not Supported Algorithm")
        
    model.set_random_seed(seed=seed)
    model.set_logger(logger)
    
    model.learn(total_timesteps=num_steps, callback=[checkpoint_callback, eval_callback])
    model.save(f"{output_path}parameters")
    
def main(args):
    seed =  np.random.seed(args.seed)
    time_str = time.strftime("%b_%d_%H_%M", time.localtime())
    output_path = f"{args.logdir}{args.env_name}_{args.robot}_SEED{args.seed}/{args.algorithm}/"
    if args.train:
        # Turn off renderer to speed up training
        #DEFAULT_CONFIG["has_offscreen_renderer"] = False
        env = make_env(args)

        train(env, args.algorithm, output_path, args.num_steps, args.buffer_size, seed)
    else:
        model = SAC.load(f"{output_path}parameters")
        print(model.get_parameters()['policy'])
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, default='default_log/')
    parser.add_argument("--robot", type=str, default="Panda")
    parser.add_argument("--env_name", type=str, default="LiftWithTerminals")
    parser.add_argument("--algorithm", type=str, default="SAC")
    parser.add_argument("--buffer_size", type=int, default=1e7)
    parser.add_argument("--num_steps", type=int, default=2e6)
    parser.add_argument("--seed", type=int, default=69)
    parser.add_argument('--train', dest='train', action='store_true', default=False)
    args = parser.parse_args()
    main(args)