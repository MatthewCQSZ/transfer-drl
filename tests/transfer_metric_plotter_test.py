"""
Unit test code for transfer metric plotter python file. There are only two unit tests which test the
get_datasets and get_all_datasets functions. Running tests on the other functions would simply be a full function test
and no longer a unit test.
"""

import unittest
from ..transfer_metric_plotter import *



class TransferMetricPlotterTester(unittest.TestCase):
    def test_get_datasets(self):
        """
        This will test get_datasets function to verify that the return of the function is a list and another check
        to make sure that the list is not empty since we know the logdir is valid.
        """
        logdir = '../report_log/default_log_Nov17_lift_noshaping/Lift_Panda_SEED69/SAC/'
        sample_count = 1140000
        result = get_datasets(logdir, None, sample_count)
        self.assertTrue(type(result) == list and len(result) != 0)  # add assertion here

    def test_get_all_datasets(self):
        """
        This will test get_all_datasets function to verify that the return of the function is a list and another check
        to make sure that the list is not empty since we know the logdirs are valid.
        """
        logdirs = ['../report_log/default_log_Nov17_lift_noshaping/Lift_Panda_SEED69/SAC/',
                   '../report_log/default_log_Nov25_Lift_vanilla_sac/BaseLift_Panda_SEED69/SAC/']
        sample_count = 1140000
        result = get_all_datasets(logdirs, None, None, None, sample_count)
        self.assertTrue(type(result) == list and len(result) != 0)  # add assertion here


if __name__ == '__main__':
    unittest.main()
