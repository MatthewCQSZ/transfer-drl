import argparse
import numpy as np
import time

from stable_baselines3 import SAC
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env

from robosuite.environments.manipulation.lift import Lift as BlockLift
from robosuite.environments.manipulation.stack import Stack as BlockStacking
from robosuite.wrappers.gym_wrapper import GymWrapper 
from robosuite import load_controller_config

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
        "reward_shaping": True,
        "reward_scale": 1.0,
        "use_camera_obs": False,
        "ignore_done": True,
        "hard_reset": False,
    }

# Override _flatten_obs() of GymWrapper for Robosuite to unify obs space as float32
def _flatten_obs_with_type_casting(self, obs_dict, verbose=False):
    ob_lst = []
    for key in self.keys:
        if key in obs_dict:
            if verbose:
                print("adding key: {}".format(key))
            ob_lst.append(np.array(obs_dict[key]).flatten())
    return np.concatenate(ob_lst).astype(np.float32)
GymWrapper._flatten_obs = _flatten_obs_with_type_casting

def make_env(name, robot):
    if name == "BlockLift":
        env = GymWrapper(BlockLift(robots=robot, **DEFAULT_CONFIG))
    elif name == "BlockStacking":
        env = GymWrapper(BlockStacking(robots=robot, **DEFAULT_CONFIG))
    else:
        env = None
    return env
    
def train(env, algorithm, output_path, num_steps, seed=None):
    # TODO: figure out why logging is not working
    logger = configure(output_path, ["stdout", "csv", "tensorboard"])
    
    if algorithm == "SAC":
        model = SAC("MlpPolicy", env, verbose=1)
    elif algorithm == "TD3":
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1)
        
    model.set_random_seed(seed=seed)
    model.set_logger(logger)
    
    model.learn(total_timesteps=num_steps)  #, callback=TensorboardCallback())
                                            # custom callback failed to solve logger issue
    model.save(f"{output_path}parameters")
    
def main(args):
    seed =  np.random.seed(args.seed)
    time_str = time.strftime("%b_%d_%H_%M", time.localtime())
    output_path = f"{args.logdir}{args.env_name}_{args.robot}_SEED{args.seed}/{args.algorithm}/"
    if args.train:
        # Turn off renderer to speed up training
        DEFAULT_CONFIG["has_offscreen_renderer"] = False
        env = make_env(args.env_name, args.robot)
        
        check_env(env)
        train(env, args.algorithm, output_path, args.num_steps, seed)
    else:
        model = SAC.load(f"{output_path}parameters")
        print(model.get_parameters()['policy'])
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, default='log/')
    parser.add_argument("--robot", type=str, default="Panda")
    parser.add_argument("--env_name", type=str, default="BlockLift")
    parser.add_argument("--algorithm", type=str, default="SAC")
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=69)
    parser.add_argument('--train', dest='train', action='store_true', default=False)
    args = parser.parse_args()
    main(args)