from __future__ import print_function, division, absolute_import

import argparse
import json
import os

import pandas as pd

parser = argparse.ArgumentParser(description='Create a report file for a given dataset')
parser.add_argument('-d', '--data_log_folder', type=str, default="", required=True, help='Path to a dataset log folder')
args = parser.parse_args()

assert os.path.isdir(args.data_log_folder), "--data_log_folder must be a path to a valid folder"

dataset_logfolder = args.data_log_folder
experiments = [item for item in os.listdir(dataset_logfolder) if os.path.isdir('{}/{}'.format(dataset_logfolder, item))]
experiments.sort()
print("Found {} experiments".format(len(experiments)))

knn_mse = []
# Add here keys from exp_config.json that should be saved in the csv report file
exp_configs = {'architecture_name': [], 'model_type': [], 'state_dim': []}

for experiment in experiments:

    with open('{}/{}/exp_config.json'.format(dataset_logfolder, experiment)) as f:
        exp_config = json.load(f)
    for key in exp_configs.keys():
        exp_configs[key].append(exp_config[key])

    try:
        with open('{}/{}/knn_mse.json'.format(dataset_logfolder, experiment)) as f:
            knn_mse.append(json.load(f)['knn_mse'])
    except IOError:
        knn_mse.append(-1)
        print("knn_mse.json not found for {}".format(experiment))

exp_configs.update({'experiments': experiments, 'knn_mse': knn_mse})
result_df = pd.DataFrame(exp_configs)
result_df.to_csv('{}/results.csv'.format(dataset_logfolder), sep=",", index=False)
print("Saved results to {}/results.csv".format(dataset_logfolder))
print("Last 10 experiments:")
print(result_df.tail(10))
