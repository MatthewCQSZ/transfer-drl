import gym
import unittest
from transfer_drl_gym import *
import os

class TransferAlgorithmTester(unittest.TestCase):
    """
    Unit test code for transfer algorithm python file. 
    """    

    def test_training_soc(self):
        """
        Test training using our env and SOC algorithm, may take a while.
        Select 3 when wandb prompted.
        """
        self.transfer_learning = TransferDRLGym(logdir="./test_log", algorithm = "SOC", num_steps = 2000)
        self.transfer_learning.make()
        print(f"\nTesting training on {self.transfer_learning.env_name} with {self.transfer_learning.algorithm}, this may take a few minutes...")
        self.transfer_learning.train()
        self.assertTrue(os.path.exists("./wandb"))
        
        


if __name__ == '__main__':
    unittest.main()
