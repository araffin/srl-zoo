from __future__ import print_function, division

import argparse
import datetime
import subprocess
import json
import os
from collections import OrderedDict
from pprint import pprint

from utils import printRed, printGreen, printBlue, printYellow, priorsToString


def getLogFolderName(exp_config):
    """
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

    try:
        os.makedirs(log_folder)
    except OSError:
        print("Experiment folder already exist")

    return log_folder, experiment_name


def preprocessingCall(exp_config, force=False):
    """
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
        printRed("An error occured")
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
        printRed("An error occured")
        pprint(exp_config)
        raise RuntimeError("Error during state representation learning (config file above)")

    print("End of state representation learning.\n")


def knnCall(exp_config):
    try:
        os.makedirs('{}/NearestNeighbors/'.format(exp_config['log_folder']))
    except OSError:
        print("NearestNeighbors folder already exist")

    printGreen("\nEvaluating the state representation with KNN")
    args = ['--seed', str(exp_config['knn_seed']), '--n_samples', str(exp_config['knn_samples'])]
    for arg in ['log_folder', 'data_folder', 'n_neighbors']:
        args.extend(['--{}'.format(arg), str(exp_config[arg])])

    ok = subprocess.call(['python', '-m', 'plotting.knn_images'] + args)
    if ok != 0:
        printRed("An error occured")
        pprint(exp_config)
        raise RuntimeError("Error during knn plotting (config file above)")
    print("End of evluation.\n")

def saveConfig(exp_config):
    """
    :param exp_config: (dict)
    """
    # Sort by keys
    exp_config = OrderedDict(sorted(exp_config.items()))

    with open("{}/exp_config.json".format(exp_config['log_folder']), "wb") as f:
        json.dump(exp_config, f)
    print("Saved config to log folder: {}".format(exp_config['log_folder']))


parser = argparse.ArgumentParser(description='Pipeline script for state representation learning')
parser.add_argument('-c', '--exp_config', type=str, default="", help='Path to an experiment config file')
parser.add_argument('--data_folder', type=str, default="", help='Path to a dataset folder')
parser.add_argument('--model_approach', type=str, default="priors", help='Model approach (default: priors)')
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
    if "data/" in args.data_folder:
        args.data_folder = args.data_folder.split('data/')[1].strip("/")
    dataset_path = "data/{}".format(args.data_folder)

    assert os.path.isdir(dataset_path), "Path to dataset folder is not valid: {}".format(dataset_path)

    printGreen("\n Grid search on a dataset folder: {} \n".format(args.data_folder))

    # TODO: replace default argument by a json config file (see configs/ folder)
    exp_config = {
        'data_folder': args.data_folder,
        'use_continuous': False,
        'model_approach': args.model_approach,
    }

    try:
        os.makedirs("logs/{}".format(args.data_folder))
    except OSError:
        printYellow("Dataset log folder already exist")

    # Preprocessing
    preprocessingCall(exp_config)
    exp_config['priors'] = ['Proportionality', 'Temporal', 'Causality', 'Repetability']
    exp_config['model_type'] = "cnn"

    exp_config['learning_rate'] = 0.001
    exp_config['l1_reg'] = 0.0
    exp_config['batch_size'] = 256
    exp_config['epochs'] = 1
    exp_config['architecture_name'] = "resnet"
    exp_config['knn_seed'] = 1
    exp_config['n_neighbors'] = 5
    exp_config['knn_samples'] = 5  # 5 test images for KNN

    # Grid search
    for seed in [1]:
        exp_config['seed'] = seed
        for state_dim in [3]:
            # Update config
            exp_config['state_dim'] = state_dim
            log_folder, experiment_name = getLogFolderName(exp_config)
            exp_config['log_folder'] = log_folder
            exp_config['experiment_name'] = experiment_name
            # Run the pipeline
            saveConfig(exp_config)
            stateRepresentationLearningCall(exp_config)
            knnCall(exp_config)

else:
    print("Please specify one of --exp_config or --data_folder")
