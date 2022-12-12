import numpy as np
import argparse

from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from utils.common import make_env, make_algorithm
from utils.video_generation import generate_video
from algorithm.soc.soc import train_SOC

class TransferDRLGym:
    '''
    Base Class for DRLGym
    A integrated Deep Reinforcement Learning and Transfer Learning library based on
    Stable Baselines 3  https://stable-baselines3.readthedocs.io/en/master/
    Robosuite           https://robosuite.ai/
    
    :params logdir: the directory for logging.
    :params reuse_logdir: the directory for teacher policy in Policy Reuse.
    :params robot: the name for robosuite robots, like "Panda", "Sawyer", "LBR IIWA 7",
        "Jaco", "Kinova Gen3", "UR5e".
    :params env_name: the name for the environment, currently supporting "Lift", "BaseLift",
        "LiftAndPlace", "LiftAndPlaceBarrier".
    :params algorithm: the name of the algorithm for training, currently supporting "PPO", "SAC", 
        "TD3", "SOC", "SACPolicyReuse", "TD3PolicyReuse"; other SB3 algorithms are supported but not thoroughly tested.
    :params buffersize: size for the replay buffer.
    :params num_steps: total number of steps for training.
    :params mu: parameter mu in Policy Reuse.
    :params max_reuse_steps: maximum number of steps for per episode in Policy Reuse.
    :params seed: random seed.
    '''
    def __init__(
        self,
        logdir: str = "default_log",
        reuse_logdir: str = "default_log",
        robot: str = "Panda",
        env_name: str = "BaseLift",
        algorithm: str = "SAC",
        buffer_size: int = 1_000_000, #1e6
        num_steps: int = 2_000_000,  #2e6
        mu: float = 0.95,
        max_reuse_steps: int = 500,
        seed: int = 69,
        ):
        
        self.logdir = logdir
        self.reuse_logdir = reuse_logdir
        self.robot = robot
        self.env_name = env_name
        self.algorithm = algorithm
        self.buffer_size = buffer_size
        self.num_steps = num_steps
        self.mu = mu
        self.max_reuse_steps = max_reuse_steps
        self.seed = np.random.seed(seed)
        
    def make(self):
        '''
        Function to initialize environment and model for experiments.
        Must call before training.
        '''
        self.env = make_env(self.env_name, self.robot, self.seed)
        self.model_path = f"{self.logdir}/{self.env_name}_{self.robot}_SEED{self.seed}/{self.algorithm}/"
        self.reuse_path = f"{self.reuse_logdir}/{self.env_name}_{self.robot}_SEED{self.seed}/{self.algorithm}/"

    
    def train(self):
        '''
        Train on environments ("Lift", "BaseLift", "LiftAndPlace", "LiftAndPlaceBarrier") with 
        algorithms ("PPO", "SAC", "TD3", "SOC", "SACPolicyReuse", "TD3PolicyReuse").
        Model checkpoints and weights are saved in "<logdir>/<env_name>_<robot>_SEED<seed>/<algorithm>/"
        '''
        if "SOC" in self.env:
            train_SOC(self.model_path)
        else:
            self.model = make_algorithm(self.env, 
                                        self.algorithm, 
                                        self.buffer_size, 
                                        self.mu, 
                                        self.max_reuse_steps, 
                                        self.reuse_path)
            
            logger = configure(self.model_path, ["stdout", "csv", "tensorboard"])
            eval_callback = EvalCallback(self.model_path, best_model_save_path=self.model_path,
                                log_path=self.model_path, eval_freq=500,
                                deterministic=True, render=False)
            checkpoint_callback = CheckpointCallback(save_freq=100000, 
                                                save_path=self.model_path,
                                                name_prefix="rl_model",
                                                save_replay_buffer=False,
                                                save_vecnormalize=False)
            
            self.model.set_random_seed(seed=self.seed)
            self.model.set_logger(logger)
        
            self.model.learn(total_timesteps=self.num_steps, callback=[checkpoint_callback, eval_callback])
            self.model.save(f"{self.model_path}parameters")
        
    def generate_video(
        self, 
        video_path:str="video", 
        camera:str = "frontview",
        num_iter: int = 10,
        timesteps: int = 500,
        reward_threshhold: float = 0.0,
        ):
        '''
        Generate video of manipulation task from model trained.
    
        :params video_path: path to the file which is the video generated, no need to specify file format as 
            mp4 is used automatically.
        :params camera: name of the camera, e.g. "frontview", "agentview".
        :params num_iter: number of iterations to run for video generation.
        :params timesteps: number of timesteps per iteration.
        :paramsr eward_threshhold: cutoff for video saving. The video will not save iterations in which
            rewards at all timestamps are less than or equal to the threshold.
    '''
        
        video_path += ".mp4"
        generate_video(
            self.model_path, 
            video_path, 
            self.algorithm, 
            self.env_name, 
            self.robot, 
            camera,
            num_iter,
            timesteps,
            reward_threshhold,
        )


def main(args):
    transfer_learning = TransferDRLGym(**vars(args))
    if args.train:
        transfer_learning.make()
        transfer_learning.train()
    
    if args.video:
        transfer_learning.generate_video(args.video_path, 
                                         args.camera,
                                         args.video_num_iter,
                                         args.video_timesteps,
                                         args.reward_threshhold)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog = 'TransferFRLGym',
                                     description = 'A library for Deep Reinforcement Learning and Transfer Learning.')
    parser.add_argument('--logdir', type=str, default='default_log/',
                        help="the directory for logging.")
    parser.add_argument('--reuse_logdir', type=str, default='default_log/',
                        help="the directory for teacher policy in Policy Reuse.")
    parser.add_argument("--robot", type=str, default="Panda",
                        help="the name for robosuite robots, like \"Panda\", \"Sawyer\".")
    parser.add_argument("--env_name", type=str, default="BaseLift",
                        help="the name for the environment, like \"BaseLift\", \"LiftAndPlace\".")
    parser.add_argument("--algorithm", type=str, default="SAC",
                        help="the name of the algorithm for training, like \"SAC\", \"SACPolicyReuse\".")
    parser.add_argument("--buffer_size", type=int, default=1e6,
                        help="size for the replay buffer.")
    parser.add_argument("--num_steps", type=int, default=1e6,
                        help=" total number of steps for training.")
    parser.add_argument("--mu", type=float, default=0.95,
                        help="parameter mu in Policy Reuse.")
    parser.add_argument("--max_reuse_steps", type=int, default=500,
                        help="maximum number of steps for per episode in Policy Reuse.")
    parser.add_argument("--seed", type=int, default=69,
                        help="random seed.")
    parser.add_argument('--train', dest='train', action='store_true', default=False,
                        help="set for training.")
    parser.add_argument('--video', dest='video', action='store_true', default=False,
                        help="set for video generation.")
    parser.add_argument("--camera", type=str, default="frontview",
                        help="name of the camera, e.g. \"frontview\", \"agentview\".")
    parser.add_argument("--video_path", type=str, default="video",
                        help="path to the video.")
    parser.add_argument("--video_timesteps", type=int, default=500,
                        help="number of timesteps per iteration.")
    parser.add_argument("--video_num_iter", type=int, default=10,
                        help="number of iterations to run for video generation.")
    parser.add_argument("--reward_threshhold", type=float, default=0.00,
                        help="threshhold for video saving.")
    args = parser.parse_args()
    main(args)