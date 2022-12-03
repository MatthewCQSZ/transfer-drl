# Code taken from https://github.com/openai/spinningup/blob/master/spinup/utils/plot.py
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import os.path as osp
import numpy as np

DIV_LINE_WIDTH = 50

# Global vars for tracking and labeling data at load time.
exp_idx = 0
units = dict()


def plot_transfer_metrics(data, xaxis='time/total_timesteps', value="Performance", condition="Condition1", smooth=1,
                          sample_count=1210000, **kwargs):
    global threshold_idx
    data2 = data
    if isinstance(data2, list):
        data2 = pd.concat(data2, ignore_index=True)
    sample_count = 1140000
    #print(data2.loc[data2['time/total_timesteps'] == sample_count])
    line = sns.lineplot(data=data2.loc[data2['time/total_timesteps'] == 0], x=xaxis, y=value,
                        color='black', palette='bright', legend=False, style='time/total_timesteps', estimator=None, linewidth='2.5')
    line.annotate(' Jumpstart', xy=(0, 10))
    line2 = sns.lineplot(data=data2.loc[data2['time/total_timesteps'] == sample_count], x=xaxis, y=value,
                  color='black', palette='bright', legend=False, style='time/total_timesteps', estimator=None, linewidth='2.5')
    line2.annotate(' Asymptotic Performance', xy=(sample_count, 150))

    # threshold_one_condition = data2.loc[data2[condition] == 'No_Transfer_Method']
    # threshold_one_condition[value] = 300
    # threshold = threshold_one_condition.copy()
    # line3 = sns.lineplot(data=threshold, x=xaxis, y=value, hue=condition,
    #                      color='black', palette='bright', legend=False, style='time/total_timesteps', estimator=None, linewidth='2.5')
    # line3.annotate(' Threshold Performance', xy=(0, 305))


def plot_data(data, xaxis='time/total_timesteps', value="Performance", condition="Condition1", smooth=1, **kwargs):
    data2 = data

    if isinstance(data2, list):
        data2 = pd.concat(data2, ignore_index=True)
    sns.set(style="whitegrid", font_scale=1.5)
    sns.lineplot(data=data2, x=xaxis, y=value, hue=condition, palette='pastel', legend=False, ci='sd', **kwargs)
    #sns.tsplot(data=data, time=xaxis, value=value, unit="Unit", condition=condition, ci='sd', **kwargs)
    """
    If you upgrade to any version of Seaborn greater than 0.8.1, switch from 
    tsplot to lineplot replacing L29 with:

        sns.lineplot(data=data, x=xaxis, y=value, hue=condition, ci='sd', **kwargs)

    Changes the colorscheme and the default legend style, though.
    """
    #plt.legend(loc='best').set_draggable(True)
    #plt.legend(loc='upper center', ncol=3, handlelength=1,
    #           borderaxespad=0., prop={'size': 13})

    """
    For the version of the legend used in the Spinning Up benchmarking page, 
    swap L38 with:

    plt.legend(loc='upper center', ncol=6, handlelength=1,
               mode="expand", borderaxespad=0., prop={'size': 13})
    """

    xscale = np.max(np.asarray(data2[xaxis])) > 5e3
    if xscale:
        # Just some formatting niceness: x-axis scale in scientific notation if max x is large
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    plt.tight_layout(pad=0.5)

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
    #sns.tsplot(data=data, time=xaxis, value=value, unit="Unit", condition=condition, ci='sd', **kwargs)
    """
    If you upgrade to any version of Seaborn greater than 0.8.1, switch from 
    tsplot to lineplot replacing L29 with:

        sns.lineplot(data=data, x=xaxis, y=value, hue=condition, ci='sd', **kwargs)

    Changes the colorscheme and the default legend style, though.
    """
    plt.legend(loc='best').set_draggable(True)
    #plt.legend(loc='upper center', ncol=3, handlelength=1,
    #           borderaxespad=0., prop={'size': 13})

    """
    For the version of the legend used in the Spinning Up benchmarking page, 
    swap L38 with:

    plt.legend(loc='upper center', ncol=6, handlelength=1,
               mode="expand", borderaxespad=0., prop={'size': 13})
    """

    xscale = np.max(np.asarray(data[xaxis])) > 5e3
    if xscale:
        # Just some formatting niceness: x-axis scale in scientific notation if max x is large
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    plt.tight_layout(pad=0.5)

def get_datasets(logdir, condition=None, sample_count=1210000):
    """
    Recursively look through logdir for output files produced by
    spinup.logx.Logger.

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
            #print(exp_data_shortened.shape)
            performance = 'eval/mean_reward'
            exp_data_shortened.insert(len(exp_data_shortened.columns), 'Unit', unit)
            exp_data_shortened.insert(len(exp_data_shortened.columns), 'Condition1', condition1)
            #exp_data.insert(len(exp_data.columns), 'Condition2', condition2)
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
               font_scale=1.5, smooth=1, select=None, exclude=None, estimator='mean', sample_count=1210000):
    data = get_all_datasets(all_logdirs, legend, select, exclude, sample_count)
    values = values if isinstance(values, list) else [values]
    condition = 'Condition1'
    estimator = getattr(np, estimator)      # choose what to show on main curve: mean? max? min?
    for value in values:
        plt.figure()
        plot_transfer_metrics(data, xaxis=xaxis, value=value, condition=condition, smooth=smooth, estimator=estimator,
                              sample_count=sample_count)
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
    #print("no transfer log: ", args.no_transfer_logdir)
    logdir = [args.no_transfer_logdir[0], args.transfer_logdir[0]]

    make_plots(logdir, args.legend, args.xaxis, args.value, args.count,
               smooth=args.smooth, select=args.select, exclude=args.exclude,
               estimator=args.est, sample_count=args.sample_count)

if __name__ == "__main__":
    main()


# import pandas as pd
# import matplotlib.pyplot as plot
# import argparse
#
#
# if __name__ == '__main__':
#
#     parser = argparse.ArgumentParser(
#         prog='TransferMetricPlotter',
#         description='Plotting tool for transfer learning')
#
#     parser.add_argument('--no_transfer_file', type=str,
#                         required=True, help='path to the no transfer method progress csv file')
#     parser.add_argument('--transfer_file', type=str,
#                         required=True, help='path to the transfer method progress csv file')
#
#     # First, parse args
#     args = parser.parse_args()
#
#     # Pull source and target data from csv files
#     try:
#         no_transfer_data = pd.read_csv(args.no_transfer_file)
#     except FileNotFoundError:
#         print("Error opening source filepath csv at: {}. "
#               "Please check filepath and try again.".format(args.no_transfer_file))
#
#     try:
#         transfer_data = pd.read_csv(args.transfer_file)
#     except FileNotFoundError:
#         print("Error opening target csv at: {}. "
#               "Please check filepath and try again.".format(args.transfer_file))
#
#
#
#     no_transfer_data['No Transfer Mean Reward'] = no_transfer_data['eval/mean_reward']
#     print(no_transfer_data.head())
#     # target_data.insert(0, 'sourceOrTarget', 'target')
#     transfer_data['Transfer Mean Reward'] = transfer_data['eval/mean_reward']
#     print(transfer_data.head())
#
#     no_and_transfer_data = pd.concat([no_transfer_data, transfer_data], axis=0)
#     print(no_and_transfer_data.head())
#
#     print('Shape of the no transfer data table: ', no_transfer_data.shape)
#     print('Shape of the transfer data table: ', transfer_data.shape)
#     print('Shape of the no transfer and transfer data table: ', no_and_transfer_data.shape)
#
#     no_and_transfer_data.plot(y=['No Transfer Mean Reward', 'Transfer Mean Reward'], kind="line", xlabel='Training Time',
#                                 ylabel='Performance', title='Transfer Learning Metrics')
#     plot.show()

