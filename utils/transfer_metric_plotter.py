"""
Base code taken from https://github.com/openai/spinningup/blob/master/spinup/utils/plot.py,
Modified base code to work with pulling transfer learning results and
all transfer learning metrics related work added by Charles Meehan
"""

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import os.path as osp
import numpy as np
from numpy import trapz
from pathlib import Path

DIV_LINE_WIDTH = 50

# Global vars for tracking and labeling data at load time.
exp_idx = 0
units = dict()


def plot_transfer_metrics(data, xaxis='time/total_timesteps', value="Performance", condition="Condition1", smooth=1,
                          sample_count=1210000, threshold=350, **kwargs):
    """
    Plotting function for transfer metrics.
    :return: Transfer metrics plotted on top of reward lines. Also creates a csv file with transfer metrics saved to
    the file.
    """

    # Dictionary and then dataframe to hold transfer metric results
    transfer_metrics_dict = {'Condition': ['No Transfer Method', 'Transfer Method', 'One Metric Only'],
                             'Jumpstart Performance': ['NA', 'NA', 0],
                             'Asymptotic Performance': ['NA', 'NA', 0],
                             'Time to Threshold': [0, 0, 'NA'],
                             'Total Rewards': [0, 0, 'NA']}
    transfer_metrics_dataframe = pd.DataFrame.from_dict(transfer_metrics_dict)
    data2 = data
    if isinstance(data2, list):
        data2 = pd.concat(data2, ignore_index=True)

    # Jumpstart Performance Metric Line plot
    line = sns.lineplot(data=data2.loc[data2['time/total_timesteps'] == 500], x=xaxis, y=value,
                        color='red', palette='bright', legend=False, style='time/total_timesteps',
                        markers=True, dashes=False, estimator=None, linewidth='2.5')
    line.annotate('Jumpstart', xy=(0, 10), color='red')

    jumpstart_data = data2.loc[data2['time/total_timesteps'] == 500]
    transfer_jumpstart_perf = list(jumpstart_data.loc[jumpstart_data['Condition1'] == 'Transfer_Method'][value])[0] - \
                              list(jumpstart_data.loc[jumpstart_data['Condition1'] == 'No_Transfer_Method'][value])[0]
    print('Jumpstart Performance: ', transfer_jumpstart_perf)
    transfer_metrics_dataframe.at[2, 'Jumpstart Performance'] = transfer_jumpstart_perf

    # Asymptotic Performance Metric Line plot
    last_timestep = data2.tail(1)['time/total_timesteps']
    last_value = data2.tail(1)[value]
    line2 = sns.lineplot(data=data2.loc[data2['time/total_timesteps'] == last_timestep.iloc[0]], x=xaxis, y=value,
                         color='red', palette='bright', legend=False, style='time/total_timesteps',
                         markers=True, dashes=False, estimator=None, linewidth='2.5')
    line2.annotate(' Asymptotic Performance', xy=(last_timestep.iloc[0], last_value.iloc[0]), color='red')
    asymptotic_data = data2.loc[data2['time/total_timesteps'] == last_timestep.iloc[0]]
    transfer_asymptotic_perf = list(asymptotic_data.loc[asymptotic_data['Condition1'] == 'Transfer_Method'][value])[0] - \
                              list(asymptotic_data.loc[asymptotic_data['Condition1'] == 'No_Transfer_Method'][value])[0]
    print('Asymptotic Performance: ', transfer_asymptotic_perf)
    transfer_metrics_dataframe.at[2, 'Asymptotic Performance'] = transfer_asymptotic_perf

    # Threshold Performance horizontal line plot
    kwargs.pop("estimator")
    kwargs.update({"label": "Threshold Performance"})
    kwargs.update({"color": "black"})
    kwargs.update({"ls": "--"})
    line2.axhline(threshold, **kwargs)

    # Time to threshold calculation
    time_to_thresh_no_transfer_data = data2.loc[data2['Condition1'] == 'No_Transfer_Method']
    time_to_thresh_no_transfer_data = time_to_thresh_no_transfer_data.loc[time_to_thresh_no_transfer_data[value] > threshold]
    if time_to_thresh_no_transfer_data.empty:
        print("No Transfer Method Never Reached Threshold")
        transfer_metrics_dataframe.at[0, 'Time to Threshold'] = "No Transfer Method Never Reached Threshold"
    else:
        time_to_thresh_no_transfer = list(time_to_thresh_no_transfer_data['time/total_timesteps'])[0]
        print("Time to Threshold for No Transfer Method (total timesteps): ", time_to_thresh_no_transfer)
        transfer_metrics_dataframe.at[0, 'Time to Threshold'] = time_to_thresh_no_transfer

    time_to_thresh_transfer_data = data2.loc[data2['Condition1'] == 'Transfer_Method']
    time_to_thresh_transfer_data = time_to_thresh_transfer_data.loc[time_to_thresh_transfer_data[value] > threshold]
    if time_to_thresh_transfer_data.empty:
        print("Transfer Method Never Reached Threshold")
        transfer_metrics_dataframe.at[1, 'Time to Threshold'] = "Transfer Method Never Reached Threshold"
    else:
        time_to_thresh_transfer = list(time_to_thresh_transfer_data['time/total_timesteps'])[0]
        print("Time to Threshold for Transfer Method (total timesteps): ", time_to_thresh_transfer)
        transfer_metrics_dataframe.at[1, 'Time to Threshold'] = time_to_thresh_transfer

    # Calculate the area under the curve for each no transfer and transfer methods
    no_transfer_data = data2.loc[data2['Condition1'] == 'No_Transfer_Method']
    no_transfer_y = no_transfer_data[value]
    no_transfer_y = list(no_transfer_y.dropna())
    no_transfer_area = trapz(no_transfer_y, dx=5)
    print("No Transfer Method Total Rewards: ", no_transfer_area)
    transfer_metrics_dataframe.at[0, 'Total Rewards'] = no_transfer_area

    transfer_data = data2.loc[data2['Condition1'] == 'Transfer_Method']
    transfer_y = transfer_data[value]
    transfer_y = list(transfer_y.dropna())
    transfer_area = trapz(transfer_y, dx=5)
    print("Transfer Method Total Rewards: ", transfer_area)
    transfer_metrics_dataframe.at[1, 'Total Rewards'] = transfer_area

    # Output tansfer metrics to a csv file within top repository directory
    # Written so that the file will overwrite file of the same name within transfer_metrics_log directory
    transfer_metrics_dataframe_filepath = Path('./transfer_metrics_log/transfer_metrics.csv')
    transfer_metrics_dataframe_filepath.parent.mkdir(parents=True, exist_ok=True)
    transfer_metrics_dataframe.to_csv(transfer_metrics_dataframe_filepath)



def plot_data(data, xaxis='time/total_timesteps', value="Performance", condition="Condition1", smooth=1, **kwargs):
    """
    Plotting function for the reward lines
    :return: two plots if smooth > 1; original data plot and smoothed data plot on top of the original plot
    """
    data2 = data

    if isinstance(data2, list):
        data2 = pd.concat(data2, ignore_index=True)
    sns.set(style="whitegrid", font_scale=1.5)
    sns.lineplot(data=data2, x=xaxis, y=value, hue=condition, palette='pastel', legend=False, ci='sd', **kwargs)

    # This portion in original file
    xscale = np.max(np.asarray(data2[xaxis])) > 5e3
    if xscale:
        # Just some formatting niceness: x-axis scale in scientific notation if max x is large
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    plt.tight_layout(pad=0.5)

    # If smoothing greater than 1, plot will include smoothed line in darker color on top of regular line plot.
    if smooth > 1:
        """
        smooth data with moving window average.
        that is,
            smoothed_y[t] = average(y[t-k], y[t-k+1], ..., y[t+k-1], y[t+k])
        where the "smooth" param is width of that window (2k+1)
        """
        y = np.ones(smooth)
        for datum in data:
            x = np.asarray(datum[value])
            z = np.ones(len(x))
            smoothed_x = np.convolve(x,y,'same') / np.convolve(z,y,'same')
            datum[value] = smoothed_x

    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True)
    sns.set(style="whitegrid", font_scale=1.5)
    sns.lineplot(data=data, x=xaxis, y=value, hue=condition, palette='dark', ci='sd', **kwargs)

    # From original file.
    plt.legend(loc='best').set_draggable(True)
    xscale = np.max(np.asarray(data[xaxis])) > 5e3
    if xscale:
        # Just some formatting niceness: x-axis scale in scientific notation if max x is large
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    plt.tight_layout(pad=0.5)

def get_datasets(logdir, condition=None, sample_count=1210000):
    """
    Recursively look through transfer and no transfer directories provided as a terminal argument by the user.

    Assumes that any file "progress.csv" is a valid hit.
    """
    global exp_idx
    global units
    datasets = []
    for root, _, files in os.walk(logdir):
        if 'progress.csv' in files:
            if exp_idx < 1:
                exp_name = 'No_Transfer_Method'
                condition1 = exp_name
            else:
                exp_name = 'Transfer_Method'
                condition1 = exp_name

            exp_idx += 1
            if condition1 not in units:
                units[condition1] = 0
            unit = units[condition1]
            units[condition1] += 1

            try:
                exp_data = pd.read_csv(os.path.join(root,'progress.csv'))
            except:
                print('Could not read from %s'%os.path.join(root,'progress.csv'))
                continue
            exp_data_shortened = exp_data.loc[exp_data['time/total_timesteps'] < sample_count]
            performance = 'eval/mean_reward'
            exp_data_shortened.insert(len(exp_data_shortened.columns), 'Unit', unit)
            exp_data_shortened.insert(len(exp_data_shortened.columns), 'Condition1', condition1)
            exp_data_shortened.insert(len(exp_data_shortened.columns), 'Performance', exp_data_shortened[performance])
            datasets.append(exp_data_shortened)
    return datasets


def get_all_datasets(all_logdirs, legend=None, select=None, exclude=None, sample_count=1210000):
    """
    For every entry in all_logdirs,
        1) check if the entry is a real directory and if it is,
           pull data from it;

        2) if not, check to see if the entry is a prefix for a
           real directory, and pull data from that.
    This method from original file.
    """
    logdirs = []
    for logdir in all_logdirs:
        print(logdir)
        if osp.isdir(logdir) and logdir[-1]==os.sep:
            logdirs += [logdir]
        else:
            basedir = osp.dirname(logdir)
            fulldir = lambda x : osp.join(basedir, x)
            prefix = logdir.split(os.sep)[-1]
            listdir= os.listdir(basedir)
            logdirs += sorted([fulldir(x) for x in listdir if prefix in x])

    """
    Enforce selection rules, which check logdirs for certain substrings.
    Makes it easier to look at graphs from particular ablations, if you
    launch many jobs at once with similar names.
    """
    if select is not None:
        logdirs = [log for log in logdirs if all(x in log for x in select)]
    if exclude is not None:
        logdirs = [log for log in logdirs if all(not(x in log) for x in exclude)]

    # Verify logdirs
    print('Plotting from...\n' + '='*DIV_LINE_WIDTH + '\n')
    for logdir in logdirs:
        print(logdir)
    print('\n' + '='*DIV_LINE_WIDTH)

    # Make sure the legend is compatible with the logdirs
    assert not(legend) or (len(legend) == len(logdirs)), \
        "Must give a legend title for each set of experiments."

    # Load data from logdirs
    data = []
    if legend:
        for log, leg in zip(logdirs, legend):
            data += get_datasets(log, leg, sample_count)
    else:
        for log in logdirs:
            data += get_datasets(log, None, sample_count)
    return data


def make_plots(all_logdirs, legend=None, xaxis=None, values=None, count=False,
               font_scale=1.5, smooth=1, select=None, exclude=None, estimator='mean', sample_count=1210000,
               threshold=350):
    """
    Main method called to start the process of pulling data from given csv progress files and then plotting
    the average rewards data and transfer metrics data.
    """
    data = get_all_datasets(all_logdirs, legend, select, exclude, sample_count)
    values = values if isinstance(values, list) else [values]
    condition = 'Condition1'
    estimator = getattr(np, estimator)      # choose what to show on main curve: mean? max? min?
    for value in values:
        plt.figure()
        plot_transfer_metrics(data, xaxis=xaxis, value=value, condition=condition, smooth=smooth, estimator=estimator,
                              sample_count=sample_count, threshold=threshold)
        plot_data(data, xaxis=xaxis, value=value, condition=condition, smooth=smooth, estimator=estimator)

    plt.show()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_transfer_logdir', nargs='*')
    parser.add_argument('--transfer_logdir', nargs='*')
    parser.add_argument('--legend', '-l', nargs='*')
    parser.add_argument('--xaxis', '-x', default='time/total_timesteps')
    parser.add_argument('--sample_count', type=int, default=1210000)
    parser.add_argument('--threshold', type=int, default=350)
    parser.add_argument('--value', '-y', default='Performance', nargs='*')
    parser.add_argument('--count', action='store_true')
    parser.add_argument('--smooth', '-s', type=int, default=1)
    parser.add_argument('--select', nargs='*')
    parser.add_argument('--exclude', nargs='*')
    parser.add_argument('--est', default='mean')
    args = parser.parse_args()
    """

    Args: 
        no transfer logdir (string): File path of the log directory for
            the learning method without transfer.
            
        transfer logdir (string): File path of the log directory for 
            the learning method using transfer learning.

        legend (strings): Optional way to specify legend for the plot. The 
            plotter legend will automatically use the ``exp_name`` from the
            config.json file, unless you tell it otherwise through this flag.
            This only works if you provide a name for each directory that
            will get plotted. (Note: this may not be the same as the number
            of logdir args you provide! Recall that the plotter looks for
            autocompletes of the logdir args: there may be more than one 
            match for a given logdir prefix, and you will need to provide a 
            legend string for each one of those matches---unless you have 
            removed some of them as candidates via selection or exclusion 
            rules (below).)

        xaxis (string): Pick what column from data is used for the x-axis.
             Defaults to ``time/total_timesteps``.
             
        sample_count (int): number of total_timesteps to pull from the given
             data.
             
        threshold (int): threshold performance number for a given task.
             threshold 150 for Lift and place; 350 for Lift task

        value (strings): Pick what columns from data to graph on the y-axis. 
            Submitting multiple values will produce multiple graphs. Defaults
            to ``Performance``, which is not an actual output of any algorithm.
            Instead, ``Performance`` refers to either ``AverageEpRet``, the 
            correct performance measure for the on-policy algorithms, or
            ``AverageTestEpRet``, the correct performance measure for the 
            off-policy algorithms. The plotter will automatically figure out 
            which of ``AverageEpRet`` or ``AverageTestEpRet`` to report for 
            each separate logdir.

        count: Optional flag. By default, the plotter shows y-values which
            are averaged across all results that share an ``exp_name``, 
            which is typically a set of identical experiments that only vary
            in random seed. But if you'd like to see all of those curves 
            separately, use the ``--count`` flag.

        smooth (int): Smooth data by averaging it over a fixed window. This 
            parameter says how wide the averaging window will be.

        select (strings): Optional selection rule: the plotter will only show
            curves from logdirs that contain all of these substrings.

        exclude (strings): Optional exclusion rule: plotter will only show 
            curves from logdirs that do not contain these substrings.

    """

    logdir = [args.no_transfer_logdir[0], args.transfer_logdir[0]]

    make_plots(logdir, args.legend, args.xaxis, args.value, args.count,
               smooth=args.smooth, select=args.select, exclude=args.exclude,
               estimator=args.est, sample_count=args.sample_count, threshold=args.threshold)


if __name__ == "__main__":
    main()


