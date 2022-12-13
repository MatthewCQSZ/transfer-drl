import gym
import unittest
from transfer_drl_gym import *
import os

class TransferAlgorithmTester(unittest.TestCase):
    """
    Unit test code for transfer algorithm python file. 
    """   
    
    def test_env_make(self):
        """
        Test the environment initilization.
        Set up the TransferDRLGym class.
        Check if the environment created is Gym environment.
        """
        self.transfer_learning = TransferDRLGym(logdir="./tests/test_log", num_steps = 2000)
        self.transfer_learning.make()
        self.assertTrue(isinstance(self.transfer_learning.env, gym.Env))
        
    def test_training_rl(self):
        """
        Test training using our env and RL algorithm, may take a while.
        Check if there's model saved at the end of training.
        """
        self.transfer_learning = TransferDRLGym(logdir="./tests/test_log", num_steps = 2000)
        self.transfer_learning.make()
        print(f"\nTesting training on {self.transfer_learning.env_name} with {self.transfer_learning.algorithm}, this may take a few minutes...")
        self.transfer_learning.train()
        self.assertTrue(os.path.isfile(f"{self.transfer_learning.model_path}best_model.zip"))
        
        
    def test_training_policy_reuse(self):
        """
        Test training using our env and our Policy Reuse algorithm, may take a while.
        Check if there's model saved at the end of training.
        """
        self.transfer_learning = TransferDRLGym(logdir="./tests/test_log", algorithm = "SACPolicyReuse", num_steps = 2000)
        self.transfer_learning.make()
        # Use one of our pre-trained policy for reuse
        self.transfer_learning.reuse_path = "../report_log/default_log_Nov25_Lift_vanilla_sac/BaseLift_Panda_SEED69/SAC/"
        print(f"\nTesting training on {self.transfer_learning.env_name} with {self.transfer_learning.algorithm}, this may take a few minutes...")
        self.transfer_learning.train()
        self.assertTrue(os.path.isfile(f"{self.transfer_learning.model_path}best_model.zip"))
        


if __name__ == '__main__':
    unittest.main()
