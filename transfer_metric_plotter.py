# Need to have Pandas as a requirement
# Plotting tool for transfer methods

import pandas as pd
import matplotlib.pyplot as plot
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        prog='TransferMetricPlotter',
        description='Plotting tool for transfer learning')

    parser.add_argument('--no_transfer_file', type=str,
                        required=True, help='path to the no transfer method progress csv file')
    parser.add_argument('--transfer_file', type=str,
                        required=True, help='path to the transfer method progress csv file')

    # First, parse args
    args = parser.parse_args()

    # Pull source and target data from csv files
    try:
        no_transfer_data = pd.read_csv(args.no_transfer_file)
    except FileNotFoundError:
        print("Error opening source filepath csv at: {}. "
              "Please check filepath and try again.".format(args.no_transfer_file))

    try:
        transfer_data = pd.read_csv(args.transfer_file)
    except FileNotFoundError:
        print("Error opening target csv at: {}. "
              "Please check filepath and try again.".format(args.transfer_file))



    no_transfer_data['No Transfer Mean Reward'] = no_transfer_data['eval/mean_reward']
    print(no_transfer_data.head())
    # target_data.insert(0, 'sourceOrTarget', 'target')
    transfer_data['Transfer Mean Reward'] = transfer_data['eval/mean_reward']
    print(transfer_data.head())

    no_and_transfer_data = pd.concat([no_transfer_data, transfer_data], axis=0)
    print(no_and_transfer_data.head())

    print('Shape of the no transfer data table: ', no_transfer_data.shape)
    print('Shape of the transfer data table: ', transfer_data.shape)
    print('Shape of the no transfer and transfer data table: ', no_and_transfer_data.shape)

    no_and_transfer_data.plot(y=['No Transfer Mean Reward', 'Transfer Mean Reward'], kind="line", xlabel='Training Time',
                                ylabel='Performance', title='Transfer Learning Metrics')
    plot.show()

