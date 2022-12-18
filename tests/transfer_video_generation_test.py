import gym
import unittest
from transfer_drl_gym import *
import os

class TransferAlgorithmTester(unittest.TestCase):
    """
    Unit test code for transfer algorithm python file. 
    """    

    
    def test_generating_video(self):
        """
        Test generating video from our saved model, may take a while.
        Check if there's video saved at the end of training.
        """
        self.transfer_learning = TransferDRLGym()
        # Use one of our pre-trained policy for video generation
        self.transfer_learning.model_path = "../report_log/default_log_Nov25_Lift_vanilla_sac/BaseLift_Panda_SEED69/SAC/"
        print(f"\nTesting generating video from checkpoint saved at {self.transfer_learning.model_path}, this may take a few minutes...")
        self.transfer_learning.generate_video(video_path="test_video", timesteps=100, reward_threshhold=-1)
        self.assertTrue(os.path.isfile(f"./test_video.mp4"))

        
        


if __name__ == '__main__':
    unittest.main()
