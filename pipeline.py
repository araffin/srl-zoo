# coding: utf-8
"""
Pipeline script to preprocess data, learn state representation and evaluate the representation.
It can also perform grid search or reproduce an experiment
"""
from __future__ import print_function, division

import argparse
import datetime
import json
import os
import subprocess
import sys
from collections import OrderedDict
from pprint import pprint

from utils import printRed, printGreen, printBlue, parseDataFolder, \
    printYellow, createFolder

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
    model_str = "_{}_".format(exp_config['model-type'])

    srl_str = "ST_DIM{}".format(exp_config['state-dim'])


    losses = exp_config["losses"]
    if losses is not str():
        losses = "_".join(losses)
    experiment_name = "{}{}{}_{}".format(date, model_str, srl_str, losses)

    printBlue("\nExperiment: {}\n".format(experiment_name))
    log_folder = "logs/{}/{}".format(exp_config['data-folder'], experiment_name)
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


def preprocessingCheck(exp_config):
    """
    Check that the data are already preprocessed
    (i.e. the file preprocessed_data.npz exists)
    :param exp_config: (dict)
    """
    preprocessed_file_exist = os.path.isfile('data/{}/preprocessed_data.npz'.format(exp_config['data-folder']))
    assert preprocessed_file_exist, "Dataset must be preprocessed"


def stateRepresentationLearningCall(exp_config):
    """
    :param exp_config: (dict)
    :return: (bool) True if no error occured
    """
    printGreen("\nLearning a state representation...")

    args = ['--no-display-plots']

    if exp_config.get('multi-view', False):
        args.extend(['--multi-view'])

    for arg in ['learning-rate', 'l1-reg', 'batch-size',
                'state-dim', 'epochs', 'seed', 'model-type',
                'log-folder', 'data-folder', 'training-set-size']:
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
    :param baseline: (str) one of "supervised" , "autoencoder" or "vae"
    """
    printGreen("\n Baseline {}...".format(baseline))
    ok = False
    args = ['--no-display-plots']

    config_args = ['epochs', 'seed', 'model-type',
                   'data-folder', 'training-set-size', 'batch-size']

    if 'log-folder' in exp_config.keys():
        config_args += ['log-folder']

    if baseline in ["supervised", "autoencoder", "vae"]:

        if baseline == "supervised":
            if exp_config['relative-pos']:
                args += ['--relative-pos']
        else:
            config_args += ['state-dim']
            # because ae & vae use the script train.py with loss argument
            args += ['--losses', baseline]
        exp_config['losses'] = [baseline]

        for arg in config_args:
            args.extend(['--{}'.format(arg), str(exp_config[arg])])

        if baseline == "supervised":
            ok = subprocess.call(['python', '-m', 'srl_baselines.{}'.format(baseline)] + args)
        else:
            ok = subprocess.call(['python', 'train.py'.format(baseline)] + args)

    printConfigOnError(ok, exp_config, "baselineCall")


def pcaCall(exp_config):
    """
    :param exp_config: (dict)
    """
    printGreen("\n Baseline PCA...")

    args = ['--no-display-plots']
    config_args = ['data-folder', 'training-set-size', 'state-dim']

    for arg in config_args:
        args.extend(['--{}'.format(arg), str(exp_config[arg])])

    ok = subprocess.call(['python', '-m', 'srl_baselines.pca'] + args)
    printConfigOnError(ok, exp_config, "pcaCall")


def createGroundTruthFolder(exp_config):
    """
    Create folder and save exp_config in order to compute knn-mse
    :param exp_config: (dict)
    :return: (dict)
    """
    log_folder = "logs/{}/baselines/ground_truth/".format(exp_config['data-folder'])
    createFolder(log_folder, "")
    exp_config['log-folder'] = log_folder
    exp_config['ground-truth'] = True
    saveConfig(exp_config)
    return exp_config


def correlationCall(exp_config, plot=False):
    """
    Evaluate the representation using correlation measurement
    :param exp_config: (dict)
    :param plot: (bool)
    """
    log_folder = exp_config["log-folder"] + "/states_rewards.npz"
    data_folder = 'data/' + exp_config['data-folder']
    use_plot = [] if plot else ["--print-corr"]
    ok = subprocess.call(["python", "-m", "plotting.representation_plot", "-i", log_folder,
                          "--data-folder", data_folder, "--correlation"] + use_plot)
    printConfigOnError(ok, exp_config, "correlationCall")


def knnCall(exp_config):
    """
    Evaluate the representation using knn
    and compute knn-mse on a set of images.
    :param exp_config: (dict)
    """
    folder_path = '{}/NearestNeighbors/'.format(exp_config['log-folder'])
    createFolder(folder_path, "NearestNeighbors folder already exist")

    printGreen("\nEvaluating the state representation with KNN")

    args = ['--seed', str(exp_config['knn-seed']), '--n-samples', str(exp_config['knn-samples'])]

    if exp_config.get('ground-truth', False):
        args.extend(['--ground-truth'])

    if exp_config.get('multi-view', False):
        args.extend(['--multi-view'])

    if exp_config.get('relative-pos', False):
        args.extend(['--relative-pos'])

    for arg in ['log-folder', 'n-neighbors', 'n-to-plot']:
        args.extend(['--{}'.format(arg), str(exp_config[arg])])

    ok = subprocess.call(['python', '-m', 'evaluation.knn_images'] + args)
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

    with open("{}/exp_config.json".format(exp_config['log-folder']), "w") as f:
        json.dump(exp_config, f)
    print("Saved config to log folder: {}".format(exp_config['log-folder']))


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
        printRed("You must specify a valid --base-config json file")
        sys.exit(1)

    args.data_folder = parseDataFolder(args.data_folder)
    dataset_path = "data/{}".format(args.data_folder)
    assert os.path.isdir(dataset_path), "Path to dataset folder is not valid: {}".format(dataset_path)
    with open(args.base_config, 'r') as f:
        exp_config = json.load(f)
    exp_config['data-folder'] = args.data_folder
    exp_config['relative-pos'] = useRelativePosition(args.data_folder)
    return exp_config


def evaluateBaseline(base_config):
    """
    Retrieve baseline exp_config by reading last
    folder created in baselines directory and
    evaluate the learned representation
    :param base_config: (dict)
    """
    log_folder = "logs/{}/baselines/".format(base_config['data-folder'])
    # Get Latest edited folder
    path = max([log_folder + d for d in os.listdir(log_folder)], key=os.path.getmtime)
    with open("{}/exp_config.json".format(path), "r") as f:
        exp_config = json.load(f)

    # Update exp config params (knn evaluation)
    for param in ['knn-samples', 'knn-seed', 'n-neighbors', 'n-to-plot', 'relative-pos']:
        exp_config[param] = base_config[param]

    # Save knn params
    with open("{}/exp_config.json".format(path), "w") as f:
        json.dump(exp_config, f)

    knnCall(exp_config)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Pipeline script for state representation learning')
    parser.add_argument('-c', '--exp-config', type=str, default="", help='Path to an experiment config file')
    parser.add_argument('--data-folder', type=str, default="", help='Path to a dataset folder')
    parser.add_argument('--baselines', action='store_true', default=False, help='Run grid search for baselines')
    parser.add_argument('--base-config', type=str, default="configs/default.json",
                        help='Path to overall config file, it contains variables independent from datasets (default: '
                             '/configs/default.json)')

    args = parser.parse_args()

    # Grid Search on Baselines
    if args.baselines and args.data_folder != "":
        exp_config = getBaseExpConfig(args)
        # WARNING: learning_rate in the base config
        # is NOT currently taken into account for baselines
        base_config = exp_config.copy()
        createFolder("logs/{}/baselines".format(exp_config['data-folder']), "Baseline folder already exist")
        # Check that the dataset is already preprocessed
        preprocessingCheck(exp_config)

        # Grid search for baselines
        for seed in [1]:
            exp_config['seed'] = seed

            # Supervised Learning
            for model_type in ['resnet', 'custom_cnn']:
                exp_config['model-type'] = model_type

                baselineCall(exp_config, 'supervised')
                evaluateBaseline(base_config)

            # Autoencoder and VAE
            exp_config['model-type'] = "custom_cnn"
            for baseline in ['autoencoder', 'vae']:
                for state_dim in [6, 12, 32]:
                    # Update config
                    exp_config['state-dim'] = state_dim
                    baselineCall(exp_config, baseline)
                    evaluateBaseline(base_config)

        # PCA
        for state_dim in [12, 32]:
            # Update config
            exp_config['state-dim'] = state_dim
            pcaCall(exp_config)
            evaluateBaseline(base_config)

        # KNN-MSE for ground_truth
        exp_config = base_config.copy()
        exp_config = createGroundTruthFolder(exp_config)
        knnCall(exp_config)

    # Reproduce a previous experiment using "exp_config.json"
    elif args.exp_config != "":
        with open(args.exp_config, 'r') as f:
            exp_config = json.load(f)

        print("\n Pipeline using json config file: {} \n".format(args.exp_config))
        exp_config = {k.replace('_', '-'): v for k, v in exp_config.items()}

        baseline = None
        for name in ['vae', 'autoencoder', 'supervised']:
            if name in exp_config['log-folder']:
                baseline = name
                break

        data_folder = exp_config['data-folder']
        printGreen("\nDataset folder: {}".format(data_folder))
        # Update and save config
        log_folder, experiment_name = getLogFolderName(exp_config)
        exp_config['log-folder'] = log_folder
        exp_config['experiment-name'] = experiment_name
        exp_config['relative-pos'] = useRelativePosition(data_folder)
        # Save config in log folder
        saveConfig(exp_config)
        # Check that the dataset is already preprocessed
        preprocessingCheck(exp_config)

        if baseline is None:
            # Learn a state representation and plot it
            ok = stateRepresentationLearningCall(exp_config)
            if ok:
                # Evaluate the representation with kNN
                knnCall(exp_config)
        else:
            baselineCall(exp_config, baseline)
            evaluateBaseline(exp_config)

    # Grid on State Representation Learning with Priors
    # If using multi_view=true with custom_cnn : make sure you set N_CHANNELS to 6 in preprocess.py
    # If using multi_view=true with triplet_cnn: set N_CHANNELS to 9. Also disable priors with no_priors=true
    elif args.data_folder != "":
        exp_config = getBaseExpConfig(args)

        printGreen("\n Grid search on several state_dim on dataset folder: {} \n".format(exp_config['data-folder']))

        createFolder("logs/{}".format(exp_config['data-folder']), "Dataset log folder already exist")
        createFolder("logs/{}/baselines".format(exp_config['data-folder']), "Baseline folder already exist")

        # Check that the dataset is already preprocessed
        preprocessingCheck(exp_config)

        # Grid search
        for seed in [0]:
            exp_config['seed'] = seed
            for state_dim in [3, 6]:
                # Update config
                exp_config['state-dim'] = state_dim
                log_folder, experiment_name = getLogFolderName(exp_config)
                exp_config['log-folder'] = log_folder
                exp_config['experiment-name'] = experiment_name
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
        printYellow("Please specify one of --exp-config or --data-folder")
