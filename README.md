# TransferDRLGym

A library for Deep Reinforcement Learning and Transfer Learning.

The library uses [Robosuite](https://robosuite.ai/) for environment implementation and [Stable Baseline 3](https://stable-baselines3.readthedocs.io/en/master/) for algorithm implementation.

Please follow [installation instruction](install%instruction/) to set up Mujoco and Robosuite and install required package.

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

Get Transfer Learning Evaluation Metric by running 

```
python transfer_metric_plotter.py --no_transfer_logdir ./report_log/default_log_Nov17_lift_noshaping/Lift_Panda_SEED69/SAC/ --transfer_logdir ./report_log/default_log_Nov25_Lift_vanilla_sac/BaseLift_Panda_SEED69/SAC/
```

For help, run
```
python transfer_drl_gym.py -h
python transfer_metric_plotter.py -h
```

More unit tests at [here](tests/).
