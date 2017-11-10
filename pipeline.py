# coding: utf-8
"""
Pipeline script to preprocess data, learn state representation and evaluate the representation.
It can also perform grid search or reproduce an experiment
"""
from __future__ import print_function, division

import argparse
import datetime
import subprocess
import json
import os
import sys
from collections import OrderedDict
from pprint import pprint

from utils import printRed, printGreen, printBlue, parseDataFolder, \
                  printYellow, priorsToString, createFolder

# Fix for matplotlib non-zero return
# Apparently due to seg fault
# (https://stackoverflow.com/questions/24139389/unable-to-find-out-what-return-code-of-11-means)
MATPLOTLIB_WARNING_CODE = -11


def getLogFolderName(exp_config):
    """
    Create experiment name using experiment config and current time.
    It also try to create the experiment folder.
    It returns both full path to the log folder and experiment_name
    :param exp_config: (dict)
    :return: (str, str)
    """
    now = datetime.datetime.now()
    date = "Y{}_M{:02d}_D{:02d}_H{:02d}M{:02d}S{:02d}".format(now.year, now.month, now.day, now.hour, now.minute,
                                                              now.second)
    model_str = "_{}".format(exp_config['architecture_name'])  # exp_config['data_folder'],
    srl_str = "{}_ST_DIM{}_SEED{}".format(priorsToString(exp_config['priors']), exp_config['state_dim'],
                                          exp_config['seed'])

    if exp_config['use_continuous']:
        raise NotImplementedError("Continous actions not supported yet")
        continuous_str = "_cont_MCD{}_S{}".format(MAX_COS_DIST_AMONG_ACTIONS_THRESHOLD, CONTINUOUS_ACTION_SIGMA)
        continuous_str = continuous_str.replace(".", "_")  # replace decimal points by '_' for folder naming
    else:
        continuous_str = ""

    experiment_name = "model{}{}{}{}_{}".format(date, model_str, continuous_str, srl_str, exp_config['model_approach'])

    printBlue("\nExperiment: {}\n".format(experiment_name))
    log_folder = "logs/{}/{}".format(exp_config['data_folder'], experiment_name)
    createFolder(log_folder, "Experiment folder already exist")

    return log_folder, experiment_name


def preprocessingCall(exp_config, force=False):
    """
    Preprocess the data, if the data are already preprocessed
    (i.e. the file preprocessed_data.npz exists) it will skip it
    unless you set the `force` flag to True
    :param exp_config: (dict)
    :param force: (bool)
    """
    preprocessed_file_exist = os.path.isfile('data/{}/preprocessed_data.npz'.format(exp_config['data_folder']))
    if not force and preprocessed_file_exist:
        printYellow('Dataset already preprocessed, skipping...')
        return

    printGreen("\nPreprocessing dataset...")
    args = ['--data_folder', exp_config['data_folder'], '--no-warnings']
    ok = subprocess.call(['python', '-m', 'preprocessing.preprocess'] + args)
    if ok != 0:
        printRed("An error occured, error code: {}".format(ok))
        pprint(exp_config)
        raise RuntimeError("Error during preprocessing (config file above)")
    print("End of preprocessing.\n")


def stateRepresentationLearningCall(exp_config):
    """
    :param exp_config: (dict)
    """
    printGreen("\nLearning a state representation...")

    npz_file = "data/{}/preprocessed_data.npz".format(exp_config['data_folder'])
    args = ['--path', npz_file, '--no-plots']
    for arg in ['learning_rate', 'l1_reg', 'batch_size',
                'state_dim', 'epochs', 'seed', 'model_type',
                'log_folder', 'data_folder']:
        args.extend(['--{}'.format(arg), str(exp_config[arg])])

    ok = subprocess.call(['python', 'train.py'] + args)
    if ok != 0:
        printRed("An error occured, error code: {}".format(ok))
        if ok != MATPLOTLIB_WARNING_CODE:
            pprint(exp_config)
            raise RuntimeError("Error during state representation learning (config file above)")

    print("End of state representation learning.\n")


def knnCall(exp_config):
    """
    Evaluate the representation using knn
    and compute knn-mse on a set of images.
    :param exp_config: (dict)
    """
    folder_path = '{}/NearestNeighbors/'.format(exp_config['log_folder'])
    createFolder(folder_path, "NearestNeighbors folder already exist")

    printGreen("\nEvaluating the state representation with KNN")
    args = ['--seed', str(exp_config['knn_seed']), '--n_samples', str(exp_config['knn_samples'])]
    for arg in ['log_folder', 'n_neighbors']:
        args.extend(['--{}'.format(arg), str(exp_config[arg])])

    ok = subprocess.call(['python', '-m', 'plotting.knn_images'] + args)
    if ok != 0:
        printRed("An error occured, error code: {}".format(ok))
        pprint(exp_config)
        raise RuntimeError("Error during knn plotting (config file above)")
    print("End of evaluation.\n")


def saveConfig(exp_config, print_config=False):
    """
    Save the experiment config to a json file
    :param exp_config: (dict)
    :param print_config: (bool)
    """
    if print_config:
        pprint(exp_config)
    # Sort by keys
    exp_config = OrderedDict(sorted(exp_config.items()))

    with open("{}/exp_config.json".format(exp_config['log_folder']), "wb") as f:
        json.dump(exp_config, f)
    print("Saved config to log folder: {}".format(exp_config['log_folder']))


parser = argparse.ArgumentParser(description='Pipeline script for state representation learning')
parser.add_argument('-c', '--exp_config', type=str, default="", help='Path to an experiment config file')
parser.add_argument('--data_folder', type=str, default="", help='Path to a dataset folder')
parser.add_argument('--base_config', type=str, default="configs/default.json",
                    help='Path to overall config file, it contains variables independent from datasets (default: '
                         '/configs/default.json)')
args = parser.parse_args()

if args.exp_config != "":
    with open(args.exp_config, 'rb') as f:
        exp_config = json.load(f)

    print("\n Pipeline using json config file: {} \n".format(args.exp_config))

    experiment_name = exp_config['experiment_name']
    data_folder = exp_config['data_folder']
    printGreen("\nDataset folder: {}".format(data_folder))
    # Update and save config
    log_folder, experiment_name = getLogFolderName(exp_config)
    exp_config['log_folder'] = log_folder
    exp_config['experiment_name'] = experiment_name
    # Save config in log folder
    saveConfig(exp_config)
    # Preprocess data if needed
    preprocessingCall(exp_config)
    # Learn a state representation and plot it
    stateRepresentationLearningCall(exp_config)
    # Evaluate the representation with kNN
    knnCall(exp_config)

elif args.data_folder != "":
    if not os.path.isfile(args.base_config):
        printRed("You must specify a valid --base_config json file")
        sys.exit(-1)

    args.data_folder = parseDataFolder(args.data_folder)
    dataset_path = "data/{}".format(args.data_folder)

    assert os.path.isdir(dataset_path), "Path to dataset folder is not valid: {}".format(dataset_path)

    printGreen("\n Grid search on a dataset folder: {} \n".format(args.data_folder))

    with open(args.base_config, 'rb') as f:
        exp_config = json.load(f)
    exp_config['data_folder'] = args.data_folder

    createFolder("logs/{}".format(args.data_folder), "Dataset log folder already exist")
    createFolder("logs/{}/baselines".format(args.data_folder), "Baseline folder already exist")

    # Preprocessing
    preprocessingCall(exp_config)

    # Grid search
    for seed in [1]:
        exp_config['seed'] = seed
        for state_dim in [2, 3, 4]:
            # Update config
            exp_config['state_dim'] = state_dim
            log_folder, experiment_name = getLogFolderName(exp_config)
            exp_config['log_folder'] = log_folder
            exp_config['experiment_name'] = experiment_name
            # Save config in log folder
            saveConfig(exp_config, print_config=True)

            # Learn a state representation and plot it
            stateRepresentationLearningCall(exp_config)
            # Evaluate the representation with kNN
            knnCall(exp_config)

else:
    printYellow("Please specify one of --exp_config or --data_folder")
