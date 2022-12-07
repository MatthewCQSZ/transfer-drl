import unittest
from utils.transfer_metric_plotter import *



class TransferMetricPlotterTester(unittest.TestCase):
    def test_get_datasets(self):
        """
        This will test get_datasets function to verify that the return of the function is a list and another check
        to make sure that the list is not empty since we know the logdir is valid.
        """
        logdir = '/home/umdworkspace/git-workspace/transfer-drl-git/report_log/default_log_Nov17_lift_noshaping/Lift_Panda_SEED69/SAC/'
        sample_count = 1140000
        result = get_datasets(logdir, None, sample_count)
        self.assertTrue(type(result) == list and len(result) != 0)  # add assertion here


if __name__ == '__main__':
    unittest.main()
