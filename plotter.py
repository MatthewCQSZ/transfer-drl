import glob
from csv import DictReader
import argparse
from torch.utils.tensorboard import SummaryWriter
import multiprocessing
import random

def log_to_writer(logdir, key, files_dictionary):
    writer  = SummaryWriter(log_dir=logdir+key)
    for index, row in enumerate(files_dictionary):
        for kk, vv in row.items():
            writer.add_scalar(kk, float(vv), index)
    writer.close()

def log_from_csv(runs_path, logdir):
    csv_files = glob.glob(f"{runs_path}**/**/*.csv")
    csv_files_dictionaries = {f.split('/')[-2]: list(DictReader(open(f))) for f in csv_files}
    
    # Use multiprocessing to speed up logging
    keys = random.choices(list(csv_files_dictionaries.keys()), k=10)    # randomly select 10 to plot
    p_args_list = [[logdir, key, csv_files_dictionaries[key]] for key in keys]
    with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
            p.starmap(log_to_writer, p_args_list)

def main(args):
    log_from_csv(args.runs_path, args.logdir)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, default='log/')
    parser.add_argument('--runs_path', type=str, default='robosuite-benchmark/runs/')
    args = parser.parse_args()
    main(args)