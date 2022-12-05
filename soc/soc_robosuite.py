# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import argparse
import os
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import pybullet_envs  # noqa
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from buffers import OptionReplayBuffer
from torch.utils.tensorboard import SummaryWriter

from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *
from robosuite.environments.manipulation.lift_with_terminals import LiftWithTerminals
from robosuite.environments.manipulation.lift_and_place_with_terminals import LiftAndPlaceWithTerminals
from robosuite.environments.base import register_env
from robosuite.wrappers.gym_wrapper import GymWrapper 


register_env(LiftWithTerminals)
register_env(LiftAndPlaceWithTerminals)

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

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="LiftWithTerminals",
        help="the id of the environment")
    parser.add_argument("--robot", type = str, default="Panda")
    parser.add_argument("--total-timesteps", type=int, default=1250000,
        help="total timesteps of the experiments")
    parser.add_argument("--buffer-size", type=int, default=int(1e6),
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.005,
        help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--batch-size", type=int, default=256,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--exploration-noise", type=float, default=0.1,
        help="the scale of exploration noise")
    parser.add_argument("--learning-starts", type=int, default=20000,
        help="timestep to start learning")
    parser.add_argument("--policy-lr", type=float, default=3e-4,
        help="the learning rate of the policy network optimizer")
    parser.add_argument("--q-lr", type=float, default=1e-3,
        help="the learning rate of the Q network network optimizer")
    parser.add_argument("--qHigh-lr", type=float, default=3e-4,
        help="the learning rate of the Q network network optimizer")
    parser.add_argument("--term-lr", type=float, default=3e-4,
        help="the learning rate of the Q network network optimizer")
    parser.add_argument("--policy-frequency", type=int, default=2,
        help="the frequency of training policy (delayed)")
    parser.add_argument("--target-network-frequency", type=int, default=1, # Denis Yarats' implementation delays this by 2.
        help="the frequency of updates for the target nerworks")
    parser.add_argument("--noise-clip", type=float, default=0.5,
        help="noise clip parameter of the Target Policy Smoothing Regularization")
    parser.add_argument("--alpha", type=float, default=0.2,
            help="Entropy regularization coefficient.")
    parser.add_argument("--autotune", type=lambda x:bool(strtobool(x)), default=True, nargs="?", const=True,
        help="automatic tuning of the entropy coefficient")
    parser.add_argument("--num-options", type=int, default=2, nargs="?", const=True,
        help="number of options")
    parser.add_argument("--eps-steps", type=int, default=200000, nargs="?", const=True,
        help="over how many steps to decay random option epsilon")
    parser.add_argument("--terminal-eps", type=float, default=0.2, nargs="?", const=True,
        help="minimum random option epsilon")
    parser.add_argument("--termination-reg", type=float, default=0.01, nargs="?", const=True,
        help="penalty for termination an option, promotes continuity")

    args = parser.parse_args()
    # fmt: on
    return args


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk

#Logic for high-level option Q-Network
class Option2QNetwork(nn.Module):

    def __init__(self, env, num_options, device = "cuda"):
        super().__init__()
        self.fc1_1 = nn.Linear(np.array(env.observation_space.shape).prod(), 256)
        self.fc1_2 = nn.Linear(256, 256)
        self.fc1_3 = nn.Linear(256, num_options)
        self.fc2_1 = nn.Linear(np.array(env.observation_space.shape).prod(), 256)
        self.fc2_2 = nn.Linear(256, 256)
        self.fc2_3 = nn.Linear(256, num_options)
        self.num_options = num_options
    
    def forward(self, x):
        x1 = F.relu(self.fc1_1(x))
        x1 = F.relu(self.fc2_2(x1))
        x1 = self.fc1_3(x1)
        x2 = F.relu(self.fc2_1(x))
        x2 = F.relu(self.fc2_2(x2))
        x2 = self.fc2_3(x2)
        y = torch.min(x1, x2)
        return y, x1, x2
    
    def greedy_option(self, x):
        Q, _, _= self(x) #q values for options
        option = torch.argmax(Q, dim=-1).detach().cpu().numpy()
        return option
    
    def get_option(self, x, options, option_termination, eps):
        rands = np.random.random((x.shape[0]))
        new_options = np.random.choice(self.num_options, (x.shape[0])) # start with randomly chosen options
        greedy_options = self.greedy_option(x)
        indices = (rands >= eps).nonzero()[0]
        new_options[indices] = greedy_options[indices] # for ones above eps, choose the greedy Q action
        terminated_indices = option_termination.nonzero()[0]
        options[terminated_indices] = new_options[terminated_indices]
        return options


#Logic for option termination net
class OptionTermNet(nn.Module):
    def __init__(self, env, num_options):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, num_options)
        self.num_options = num_options

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x

    def get_terminations(self, x):
        vals = self(x)
        dist = torch.distributions.Bernoulli(vals)
        sample = dist.sample()
        num_terms = sample.sum()
        return vals, sample.detach().cpu().numpy() 

# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env, num_options, device = "cuda"):
        super().__init__()
        self.num_options = num_options
        self.fc1 = torch.nn.ModuleList([nn.Linear(np.array(env.observation_space.shape).prod() + np.prod(env.action_space.shape), 256) for i in range(num_options)])
        self.fc2 = torch.nn.ModuleList([nn.Linear(256, 256) for i in range(num_options)])
        self.fc3 = torch.nn.ModuleList([nn.Linear(256, 1) for i in range(num_options)])

    def forward(self, x, a, o):
        #o options should be a numpy array
        x = torch.cat([x, a], 1)
        y = torch.zeros((x.shape[0], 1), device = x.device)

        for i in range(self.num_options):
            indices = ((o==i).nonzero())[0]
            v = x[indices]
            v = F.relu(self.fc1[i](v))
            v = F.relu(self.fc2[i](v))
            v = self.fc3[i](v)
            v = v.squeeze(dim=-1)
            #if i == 0:
            #    print("zeros shape", v.shape, y[indices, 0].shape, indices.shape)
            #else:
            #    print("ones shape", v.shape, y[indices, 0].shape, indices.shape)
            y[indices, 0] = v
        
        return y


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env, num_options, device = "cuda"):
        super().__init__()
        self.env = env
        self.num_options = num_options
        self.action_size = np.prod(env.action_space.shape)
        self.fc1 = torch.nn.ModuleList([nn.Linear(np.array(env.observation_space.shape).prod(), 256) for i in range(num_options)])
        self.fc2 = torch.nn.ModuleList([nn.Linear(256, 256) for i in range(num_options)])
        self.fc_mean = torch.nn.ModuleList([nn.Linear(256, self.action_size) for i in range(num_options)])
        self.fc_logstd = torch.nn.ModuleList([nn.Linear(256, self.action_size) for i in range(num_options)])

        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )


    def forward(self, x, o):
        mean_y = torch.zeros(x.shape[0], np.prod(self.action_size), device = x.device)
        log_std_y = torch.zeros(x.shape[0], np.prod(self.action_size), device = x.device)
        for i in range(self.num_options):
            indices = ((o==i).nonzero())[0]
            v = x[indices]
            v = F.relu(self.fc1[i](v))
            v = F.relu(self.fc2[i](v))
            mean_v = self.fc_mean[i](v)
            log_std_v = self.fc_logstd[i](v)
            mean_y[indices] = mean_v
            log_std_y[indices] = log_std_v

        log_std_y = torch.tanh(log_std_y)
        log_std_y = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std_y + 1)  # From SpinUp / Denis Yarats

        return mean_y, log_std_y

    def get_action(self, x, o):
        mean, log_std = self(x, o)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean
    

if __name__ == "__main__":
    curr_ep_return = 0
    curr_ep_length = 0
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    running_mean_ep_length = 10
    running_mean_ep_count = 0
    running_mean_episodic_reward = np.zeros((running_mean_ep_length))
    best_mean_reward = -99999999999.0
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    model_output_folder = f"saved_best_models/{run_name}"
    os.mkdir(model_output_folder)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup

    #envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    envs = suite.make(args.env_id, robots = args.robot, **DEFAULT_CONFIG)
    envs = GymWrapper(envs)
    #assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    actor = Actor(envs, args.num_options, device = device).to(device)
    qf1 = SoftQNetwork(envs, args.num_options, device = device).to(device)
    qf2 = SoftQNetwork(envs, args.num_options, device = device).to(device)
    qf1_target = SoftQNetwork(envs, args.num_options, device = device).to(device)
    qf2_target = SoftQNetwork(envs, args.num_options, device = device).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    termNet = OptionTermNet(envs, args.num_options).to(device)
    termNet_target = OptionTermNet(envs, args.num_options).to(device)
    termNet_target.load_state_dict(termNet.state_dict())
    term_optimizer = optim.Adam(termNet.parameters(), lr = args.term_lr)
    qHigh = Option2QNetwork(envs, args.num_options).to(device)
    qHigh_target = Option2QNetwork(envs, args.num_options).to(device)
    qHigh_target.load_state_dict(qHigh.state_dict())
    qHigh_optimizer = optim.Adam(qHigh.parameters(), lr = args.qHigh_lr)

    option_prop_sum = np.zeros((args.num_options), dtype=np.float32)
    eps = 1.0

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    envs.observation_space.dtype = np.float32
    rb = OptionReplayBuffer(
        args.buffer_size,
        envs.observation_space,
        envs.action_space,
        device,
        handle_timeout_termination=True,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    options = np.zeros((1,), dtype=np.int32)
    obs = np.expand_dims(envs.reset(), axis=0)

    greedy_options = qHigh.greedy_option(torch.Tensor(obs).to(device))
    option_termination = np.zeros((1,))

    option_length = np.zeros((args.num_options))
    option_end_count = np.zeros((args.num_options))
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        eps = max(args.terminal_eps + (1.0 - args.terminal_eps) * (args.eps_steps - global_step) / args.eps_steps, args.terminal_eps)
        if global_step < args.learning_starts:
            actions = np.array([envs.action_space.sample() for _ in range(1)])
            #randomly choose options at early start
            options = np.random.randint(0, args.num_options, size = 1, dtype=np.int32) 
        else:
            options = qHigh.get_option(torch.Tensor(obs).to(device), options, option_termination, eps)
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device), options)
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, dones, infos = envs.step(actions[0])

        # Simulate Vectorized Env
        next_obs = np.expand_dims(next_obs.astype(np.float32), axis=0)
        rewards = np.array([rewards], dtype=np.float32)
        dones = np.array([dones])
        infos = [infos]
        options_termination_probs, option_termination = termNet.get_terminations(torch.tensor(next_obs).to(device))
        mean_term_prob = options_termination_probs.detach().mean().cpu().numpy()

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        curr_ep_return += rewards[0]
        curr_ep_length += 1

        if dones[0]:
            print(f"global_step={global_step}, episodic_return={curr_ep_return}")
            writer.add_scalar("charts/episodic_return", curr_ep_return, global_step)
            writer.add_scalar("charts/episodic_length", curr_ep_length, global_step)
            running_mean_ep_count += 1
            running_mean_ep_count = running_mean_ep_count % running_mean_ep_length
            running_mean_episodic_reward[running_mean_ep_count] = curr_ep_return
            curr_ep_return = 0
            curr_ep_length = 0
        


        batch_idx = np.arange((options.shape[0]))
        for i in range(args.num_options):
            option_prop_sum[i] += np.sum(np.equal(options, i))
            option_length[i] += np.sum(np.equal(options, i))
            option_end_count[i] += np.sum(np.logical_and(option_termination[batch_idx, options], np.equal(options, i)))

        # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
        real_next_obs = next_obs.copy()
        for idx, d in enumerate(dones):
            if d:
                #manually set terminal observation to be all zeros
                real_next_obs[idx] = next_obs * 0#infos[idx]["terminal_observation"]

                #only works for single degree env
                obs = np.expand_dims(envs.reset().astype(np.float32), axis=0)
        rb.add(obs, real_next_obs, actions, rewards, dones, infos, options)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs


        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                termination_probs, termination_sample = termNet.get_terminations(data.observations)
                terminations = termination_sample[np.arange(data.options.shape[0]), data.options]
                next_state_options = qHigh.get_option(data.next_observations, data.options, terminations, eps)
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations, next_state_options)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions, next_state_options)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions, next_state_options)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

                #compute target for high-level Q learning:
                batch_idx = np.arange(data.options.shape[0])
                next_high_q_prime, _, _ = qHigh_target(data.next_observations)
                next_term_prob_prime, _ = termNet_target.get_terminations(data.next_observations)
                next_term_prob_prime = next_term_prob_prime[batch_idx, data.options]
                no_term_value_target = (1.0 - next_term_prob_prime) * next_high_q_prime[batch_idx, data.options]
                term_value_target = next_term_prob_prime * next_high_q_prime.max(dim=-1)[0]
                high_q_gt = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (term_value_target + no_term_value_target)

            qf1_a_values = qf1(data.observations, data.actions, data.options).view(-1)
            qf2_a_values = qf2(data.observations, data.actions, data.options).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            #add high Level Q Learning here:
            qHigh_val, q1High_val, q2High_val = qHigh(data.observations)
            qHighLoss1 = (q1High_val[batch_idx, data.options] - high_q_gt.detach()).pow(2).mul(0.5).mean()
            qHighLoss2 = (q2High_val[batch_idx, data.options] - high_q_gt.detach()).pow(2).mul(0.5).mean()
            qHighLoss = qHighLoss1 + qHighLoss2

            qHigh_optimizer.zero_grad()
            qHighLoss.backward()
            qHigh_optimizer.step()

            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(data.observations, data.options)
                    qf1_pi = qf1(data.observations, pi, data.options)
                    qf2_pi = qf2(data.observations, pi, data.options)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi).view(-1)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    #get termination loss
                    option_term_prob, _, = termNet.get_terminations(data.observations)
                    option_term_prob = option_term_prob[batch_idx, data.options]
                    qht, _, _ = qHigh_target(data.next_observations)

                    term_loss = ((1 - data.dones.flatten()) * option_term_prob * (qht[batch_idx, data.options] - qht.max(dim=-1)[0] + args.termination_reg)).mean()
                    term_optimizer.zero_grad()
                    term_loss.backward()
                    term_optimizer.step()


                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations, data.options)
                        alpha_loss = (-log_alpha * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qHigh.parameters(), qHigh_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                writer.add_scalar("losses/HighQ_loss", qHighLoss.item(), global_step)
                writer.add_scalar("losses/term_loss", term_loss.item(), global_step)

                prop_sum = np.sum(option_prop_sum)
                for i in range(args.num_options):
                    writer.add_scalar("options/op_" + str(i) + "_proportion", option_prop_sum[i] / prop_sum, global_step)
                    writer.add_scalar("options/op_" + str(i) + "_mean_length", option_length[i] / option_end_count[i], global_step)
                option_length *= 0
                option_end_count *= 0
                option_prop_sum *= 0
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)
            if global_step % 2000 == 0:
                avg_reward = np.mean(running_mean_episodic_reward)
                if avg_reward > best_mean_reward:
                    best_mean_reward = avg_reward
                    torch.save(actor.state_dict(), f"{model_output_folder}/actor")
                    torch.save(qf1.state_dict(), f"{model_output_folder}/qf1")
                    torch.save(qf2.state_dict(), f"{model_output_folder}/qf2")
                    torch.save(termNet.state_dict(), f"{model_output_folder}/termNet")
                    torch.save(qHigh.state_dict(), f"{model_output_folder}/qHigh")


    envs.close()
    writer.close()
