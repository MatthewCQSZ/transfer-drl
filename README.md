# TransferDRLGym

A library for Deep Reinforcement Learning and Transfer Learning.

The library uses [Robosuite](https://robosuite.ai/) for environment implementation and [Stable Baseline 3](https://stable-baselines3.readthedocs.io/en/master/) for algorithm implementation.

Please follow [installation instructions](install_instructions/) to set up Mujoco and install required package.

## Base Library

To run experiments, simply
 
```
python transfer_drl_gym.py
```

You can also set up experiments in script.

```
from transfer_drl_gym import TransferDRLGym

transfer_learning = TransferDRLGym(env_name="BaseLift", algorithm="SAC", num_steps=2000)

# Make env
transfer_learning.make()

# Training
transfer_learning.train()

# Save video
transfer_learning.generate_video()
```

Plot for loss, return, entropy... can auto-generated during training. They can be accessed by using tensorboard.

```
tensorboard --logdir <logging_directory>
```

### Using Our Algorithms

You can use our Probabilistic Policy Reuse Algorithm, SAC Policy Reuse and TD3 Policy Reuse, directly by simply import and use them as Stable Baseline 3 algorithms. Like this: 

```
from algorithm.sac_policy_reuse import SACPolicyReuse

reuse_path = "../report_log/default_log_Nov25_Lift_vanilla_sac/BaseLift_Panda_SEED69/SAC/"
model = SAC.load(f"{reuse_path}best_model")
model = SACPolicyReuse("MlpPolicy", 
                        env, 
                        verbose=1, 
                        old_policy=model.policy,
                       )
model.learn()
```

Our Probabilistic Policy Reuse Algorithm, `SACPolicyReuse` and `TD3PolicyReuse`, extend our `OffPolicyAlgorithmPolicyReuse` class. In theory, any other SB3 off policy algorithm can extend our `OffPolicyAlgorithmPolicyReuse`, though it is not tested.

### Transfer Learning Metric Plotter

To use the transfer learning metric plotter (transfer_metric_plotter.py), you will need to fill out the following terminal
arguments:
1. --no_transfer_logdir with the directory path that contains your progress.csv file 
for your RL agent trained without transfer learning (from scratch). 
2. --transfer_logdir with the directory path of your progress.csv file for your RL agent 
trained with transfer learning.
3. --smooth with an integer value that is larger than 1 if you want to see a smoothed version of your rewards plot over your raw 
rewards plot.
4. --sample_count with an integer value of how many timesteps you would like displayed in the plot. This will cut the data of
your reward plot to this value.
5. --threshold with an integer value of the performance value where the agent is performing its trained task correctly.

NOTE: Make sure there are these two labeled columns in your progress.csv file. 
1. eval/mean_rewards
2. time/total_timesteps

Below is an example terminal command that ran the transfer metric plotter with data from our block lifting task with 
and without using reward shaping. For every run of the transfer metric plotter, the transfer metrics will be printed to 
the terminal and saved in a csv file named transfer_metrics.csv in the [/transfer_metrics_log](transfer_metrics_log) 
directory.

```
python transfer_metric_plotter.py --no_transfer_logdir ./report_log/default_log_Nov17_lift_noshaping/Lift_Panda_SEED69/SAC/ --transfer_logdir ./report_log/default_log_Nov25_Lift_vanilla_sac/BaseLift_Panda_SEED69/SAC/ --smooth 4 --sample_count 1140000 --threshold 350
```

### Extra

For help, run
```
python transfer_drl_gym.py -h
python transfer_metric_plotter.py -h
```

Unit tests can be found in the [/tests](tests) directory. Please run unit tests from within tests directory.
If you receive a ModuleNotFoundError while running these tests 
make sure that your PYTHONPATH is correct. Example: `PYTHONPATH=.:/(PathToTransferDRLRepo)/transfer-drl-git/`
