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
# Apparently due to segmentation fault
# (https://stackoverflow.com/questions/24139389/unable-to-find-out-what-return-code-of-11-means)
MATPLOTLIB_WARNING_CODE = -11
NO_PAIRS_ERROR = 10  # return code when no dissimilar/reference pairs where found
NAN_ERROR = 11  # return code when loss is NaN, consider increasing the NOISE_STD


def getLogFolderName(exp_config):
    """
    Create experiment name using experiment config and current time.
    It also try to create the experiment folder.
    It returns both full path to the log folder and experiment_name
    :param exp_config: (dict)
    :return: (str, str)
    """
    date = datetime.datetime.now().strftime("%y-%m-%d_%Hh%M_%S")
    model_str = "_{}".format(exp_config['model_type'])
    srl_str = "{}_ST_DIM{}_SEED{}".format(priorsToString(exp_config['priors']), exp_config['state_dim'],
                                          exp_config['seed'])

    if exp_config['use_continuous']:
        raise NotImplementedError("Continous actions not supported yet")
        # continuous_str = "_cont_MCD{}_S{}".format(MAX_COS_DIST_AMONG_ACTIONS_THRESHOLD, CONTINUOUS_ACTION_SIGMA)
        # continuous_str = continuous_str.replace(".", "_")  # replace decimal points by '_' for folder naming
    else:
        continuous_str = ""

    experiment_name = "{}{}{}{}_{}".format(date, model_str, continuous_str, srl_str, exp_config['model_approach'])

    printBlue("\nExperiment: {}\n".format(experiment_name))
    log_folder = "logs/{}/{}".format(exp_config['data_folder'], experiment_name)
    createFolder(log_folder, "Experiment folder already exist")

    return log_folder, experiment_name


def printConfigOnError(return_code, exp_config, step_name):
    """
    :param return_code: (int)
    :param exp_config: (dict)
    :param step_name: (str)
    """
    if return_code != 0:
        printRed("An error occured, error code: {}".format(return_code))
        pprint(exp_config)
        raise RuntimeError("Error during {} (config file above)".format(step_name))
    print("End of " + step_name)


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
    printConfigOnError(ok, exp_config, "preprocessing")


def stateRepresentationLearningCall(exp_config):
    """
    :param exp_config: (dict)
    :return: (bool) True if no error occured
    """
    printGreen("\nLearning a state representation...")

    args = ['--no-plots']
    if "Reference" in exp_config["priors"]:
        args.extend(['--ref_prior'])

    if "SameEnv" in exp_config["priors"]:
        args.extend(['--same_env_prior'])

    # TODO: Remove as soon as possible (only here for backward compatibility)
    if 'training_set_size' not in exp_config.keys():
        exp_config['training_set_size'] = -1

    for arg in ['learning_rate', 'l1_reg', 'batch_size',
                'state_dim', 'epochs', 'seed', 'model_type',
                'log_folder', 'data_folder', 'training_set_size']:
        args.extend(['--{}'.format(arg), str(exp_config[arg])])

    ok = subprocess.call(['python', 'train.py'] + args)
    if ok == 0:
        print("End of state representation learning.\n")
        return True
    else:
        printRed("An error occured, error code: {}".format(ok))
        pprint(exp_config)
        if ok == NO_PAIRS_ERROR:
            printRed("No Pairs found, consider increasing the batch_size or using a different seed")
            return False
        elif ok == NAN_ERROR:
            printRed("NaN Loss, consider increasing NOISE_STD in the gaussian noise layer")
            return False
        elif ok != MATPLOTLIB_WARNING_CODE:
            raise RuntimeError("Error during state representation learning (config file above)")
        else:
            return False


def baselineCall(exp_config, baseline="supervised"):
    """
    :param exp_config: (dict)
    :param baseline: (str) one of "supervised" or "autoencoder"
    """
    printGreen("\n Baseline {}...".format(baseline))

    args = ['--no-plots']
    config_args = ['epochs', 'seed', 'model_type',
                   'data_folder', 'training_set_size']

    if baseline in ["autoencoder", "vae"]:
        config_args += ['state_dim']
    elif baseline == "supervised" and exp_config['relative_pos']:
        args += ['--relative_pos']

    for arg in config_args:
        args.extend(['--{}'.format(arg), str(exp_config[arg])])

    ok = subprocess.call(['python', '-m', 'baselines.{}'.format(baseline)] + args)
    printConfigOnError(ok, exp_config, "baselineCall")


def dimReductionCall(exp_config, baseline="pca"):
    """
    :param exp_config: (dict)
    :param baseline: (str) one of "pca" or "tsne"
    """
    printGreen("\n Baseline {}...".format(baseline))

    args = ['--no-plots', '--method', baseline]
    config_args = ['data_folder', 'training_set_size', 'state_dim']

    for arg in config_args:
        args.extend(['--{}'.format(arg), str(exp_config[arg])])

    ok = subprocess.call(['python', '-m', 'baselines.pca_tsne'] + args)
    printConfigOnError(ok, exp_config, "dimReductionCall")


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

    for arg in ['log_folder', 'n_neighbors', 'n_to_plot']:
        args.extend(['--{}'.format(arg), str(exp_config[arg])])

    ok = subprocess.call(['python', '-m', 'plotting.knn_images'] + args)
    printConfigOnError(ok, exp_config, "knnCall")


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

    with open("{}/exp_config.json".format(exp_config['log_folder']), "w") as f:
        json.dump(exp_config, f)
    print("Saved config to log folder: {}".format(exp_config['log_folder']))


def useRelativePosition(data_folder):
    """
    :param data_folder: (str)
    :return: (bool)
    """
    with open('data/{}/dataset_config.json'.format(data_folder), 'r') as f:
        relative_pos = json.load(f).get('relative_pos', False)
    return relative_pos


def getBaseExpConfig(args):
    """
    :param args: (parsed args object)
    :return: (str)
    """
    if not os.path.isfile(args.base_config):
        printRed("You must specify a valid --base_config json file")
        sys.exit(1)

    args.data_folder = parseDataFolder(args.data_folder)
    dataset_path = "data/{}".format(args.data_folder)
    assert os.path.isdir(dataset_path), "Path to dataset folder is not valid: {}".format(dataset_path)

    with open(args.base_config, 'r') as f:
        exp_config = json.load(f)
    exp_config['data_folder'] = args.data_folder
    exp_config['relative_pos'] = useRelativePosition(args.data_folder)
    return exp_config


def evaluateBaseline(base_config):
    """
    Retrieve baseline exp_config by reading last
    folder created in baselines directory and
    evaluate the learned representation
    :param base_config: (dict)
    """
    log_folder = "logs/{}/baselines/".format(base_config['data_folder'])
    # Get Latest edited folder
    path = max([log_folder + d for d in os.listdir(log_folder)], key=os.path.getmtime)
    with open("{}/exp_config.json".format(path), "rb") as f:
        exp_config = json.load(f)

    # Update exp config params (knn evaluation)
    for param in ['knn_samples', 'knn_seed', 'n_neighbors', 'n_to_plot', 'relative_pos']:
        exp_config[param] = base_config[param]

    # Save knn params
    with open("{}/exp_config.json".format(path), "wb") as f:
        json.dump(exp_config, f)

    knnCall(exp_config)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Pipeline script for state representation learning')
    parser.add_argument('-c', '--exp_config', type=str, default="", help='Path to an experiment config file')
    parser.add_argument('--data_folder', type=str, default="", help='Path to a dataset folder')
    parser.add_argument('--baselines', action='store_true', default=False, help='Run grid search for baselines')
    parser.add_argument('--base_config', type=str, default="configs/default.json",
                        help='Path to overall config file, it contains variables independent from datasets (default: '
                             '/configs/default.json)')
    args = parser.parse_args()

    # Grid Search on Baselines
    if args.baselines and args.data_folder != "":
        exp_config = getBaseExpConfig(args)
        # WARNING: batch_size and learning_rate in the base config
        # are NOT currently taken into account for baselines
        base_config = exp_config.copy()
        createFolder("logs/{}/baselines".format(exp_config['data_folder']), "Baseline folder already exist")
        # Preprocessing if needed
        preprocessingCall(exp_config)

        # Grid search for baselines
        for seed in [1]:
            exp_config['seed'] = seed
            # Supervised Learning
            for model_type in ['resnet', 'custom_cnn', 'triplet_cnn']:
                exp_config['model_type'] = model_type
                baselineCall(exp_config, 'supervised')
                evaluateBaseline(base_config)

            # Autoencoder and VAE
            exp_config['model_type'] = "cnn"
            for baseline in ['autoencoder', 'vae']:
                for state_dim in [3, 4, 5, 6]:
                    # Update config
                    exp_config['state_dim'] = state_dim
                    baselineCall(exp_config, baseline)
                    evaluateBaseline(base_config)

        # PCA
        for state_dim in [3, 4, 5, 6]:
            # Update config
            exp_config['state_dim'] = state_dim
            dimReductionCall(exp_config, 'pca')
            evaluateBaseline(base_config)

        # t-SNE
        for state_dim in [2, 3]:
            # Update config
            exp_config['state_dim'] = state_dim
            dimReductionCall(exp_config, 'tsne')
            evaluateBaseline(base_config)

    # Reproduce a previous experiment using "exp_config.json"
    elif args.exp_config != "":
        with open(args.exp_config, 'r') as f:
            exp_config = json.load(f)

        print("\n Pipeline using json config file: {} \n".format(args.exp_config))

        experiment_name = exp_config['experiment_name']
        data_folder = exp_config['data_folder']
        printGreen("\nDataset folder: {}".format(data_folder))
        # Update and save config
        log_folder, experiment_name = getLogFolderName(exp_config)
        exp_config['log_folder'] = log_folder
        exp_config['experiment_name'] = experiment_name
        exp_config['relative_pos'] = useRelativePosition(data_folder)
        # Save config in log folder
        saveConfig(exp_config)
        # Preprocess data if needed
        preprocessingCall(exp_config)
        # Learn a state representation and plot it
        ok = stateRepresentationLearningCall(exp_config)
        if ok:
            # Evaluate the representation with kNN
            knnCall(exp_config)

    # Grid on State Representation Learning with Priors
    elif args.data_folder != "":
        exp_config = getBaseExpConfig(args)

        printGreen("\n Grid search on several state_dim on dataset folder: {} \n".format(exp_config['data_folder']))

        createFolder("logs/{}".format(exp_config['data_folder']), "Dataset log folder already exist")
        createFolder("logs/{}/baselines".format(exp_config['data_folder']), "Baseline folder already exist")

        # Preprocessing
        preprocessingCall(exp_config)

        # Grid search
        for seed in [0]:
            exp_config['seed'] = seed
            for state_dim in [3, 4, 6, 10]:
                # Update config
                exp_config['state_dim'] = state_dim
                log_folder, experiment_name = getLogFolderName(exp_config)
                exp_config['log_folder'] = log_folder
                exp_config['experiment_name'] = experiment_name
                # Save config in log folder
                saveConfig(exp_config, print_config=True)

                # Learn a state representation and plot it
                ok = stateRepresentationLearningCall(exp_config)
                if not ok:
                    printYellow("Skipping evaluation...")
                    continue
                # Evaluate the representation with kNN
                knnCall(exp_config)

    else:
        printYellow("Please specify one of --exp_config or --data_folder")
