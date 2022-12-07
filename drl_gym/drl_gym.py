import numpy as np
import time
import argparse

from utils.training import make_env, train

class DRLGym:
    def __init__(self) -> None:
        pass
    
    def train():
        pass


def main(args):
    seed =  np.random.seed(args.seed)
    time_str = time.strftime("%b_%d_%H_%M", time.localtime())
    output_path = f"{args.logdir}{args.env_name}_{args.robot}_SEED{args.seed}/{args.algorithm}/"
    if args.train:
        env = make_env(args)

        train(env, output_path, args, seed)
    else:
        env = make_env(args)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, default='default_log/')
    parser.add_argument('--reuse_logdir', type=str, default='default_log/')
    parser.add_argument("--robot", type=str, default="Panda")
    parser.add_argument("--env_name", type=str, default="BaseLift")
    parser.add_argument("--algorithm", type=str, default="SAC")
    parser.add_argument("--buffer_size", type=int, default=1e6)
    parser.add_argument("--num_steps", type=int, default=1e6)
    parser.add_argument("--mu", type=float, default=0.95)
    parser.add_argument("--max_reuse_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=69)
    parser.add_argument('--train', dest='train', action='store_true', default=False)
    parser.add_argument('--inherit', dest='inherit', action='store_true', default=False)
    args = parser.parse_args()
    main(args)