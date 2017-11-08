from __future__ import print_function, division

import argparse
import datetime
import subprocess
import json
import os
from pprint import pprint

# from const import LOG_FOLDER, MODEL_APPROACH, USE_CONTINUOUS, MAX_COS_DIST_AMONG_ACTIONS_THRESHOLD
# from const import CONTINUOUS_ACTION_SIGMA, STATES_DIMENSION, PRIORS_CONFIGS_TO_APPLY, PROP, TEMP, CAUS, REP, \
#     BRING_CLOSER_REF_POINT

def priorsToString(listOfPriors):
    string = '_'
    for index, priors in enumerate(listOfPriors):
        string = string + priors[0:3]
    return string


def getLogFolder(exp_config):
    """
    :param exp_config: (dict)
    :return: (str, str)
    """
    now = datetime.datetime.now()
    date = "Y{}_M{:02d}_D{:02d}_H{:02d}M{:02d}S{:02d}".format(now.year, now.month, now.day, now.hour, now.minute, now.second)
    model_str = "_{}".format(exp_config['architecture_name'])
    srl_str = "{}_ST_DIM{}".format(priorsToString(exp_config['priors']), exp_config['state_dim'])

    if exp_config['use_continuous']:
        continuous_str = "_cont_MCD{}_S{}".format(MAX_COS_DIST_AMONG_ACTIONS_THRESHOLD, CONTINUOUS_ACTION_SIGMA)
    else:
        continuous_str = ""

    experiment_name = "model{}{}{}{}_{}".format(date, model_str, continuous_str, srl_str, exp_config['model_approach'])

    print(experiment_name)
    log_folder = "logs/{}/{}".format(exp_config['data_folder'], experiment_name)

    try:
        os.makedirs(log_folder)
    except OSError:
        print("Experiment folder already exist")

    return log_folder, experiment_name

def preprocessingCall(exp_config):
    args = ['--data_folder', exp_config['data_folder'], '--no-warnings']
    ok = subprocess.call(['python', '-m', 'preprocessing.preprocess'] + args)
    if ok != 0:
        pprint(exp_config)
        raise RuntimeError("Error during preprocessing (config file above)")

# def createLogFolders():
#     pass

def saveConfig(exp_config):
    """
    :param exp_config: (dict)
    """
    with open("{}/exp_config.json".format(exp_config['log_folder']), "wb") as f:
        json.dump(exp_config, f)
    print("Saved config to log folder: {}".format(exp_config['log_folder']))

parser = argparse.ArgumentParser(description='Pipeline script for state representation learning')
parser.add_argument("-c", '--config', type=str, default="", help='Path to an experiment config file')
parser.add_argument('--data_folder', type=str, default="", help='Path to a dataset folder')
parser.add_argument('--model_approach', type=str, default="priors", help='Model approach (default: priors)')
parser.add_argument('--architecture_name', type=str, default="resnet", help='Architecture name (default: resnet)')
args = parser.parse_args()

if args.config != "":
    with open(args.config, 'rb') as f:
        exp_config = json.load(f)

    print("\n Pipeline using json config file: {} \n".format(args.config))

    experiment_name = exp_config['experiment_name']
    data_folder = exp_config['data_folder']
    # Update and save config
    log_folder, experiment_name = getLogFolder(exp_config)
    exp_config['log_folder'] = log_folder
    exp_config['experiment_name'] = experiment_name
    saveConfig(exp_config)

elif args.data_folder != "":
    if "data/" in args.data_folder:
        args.data_folder = args.data_folder.split('data/')[1].strip("/")
    dataset_path = "data/{}".format(args.data_folder)

    assert os.path.isdir(dataset_path), "Path to dataset folder is not valid: {}".format(dataset_path)

    print("\n Grid search on a dataset folder: {} \n".format(args.data_folder))

    exp_config = {'data_folder': args.data_folder,
                   'use_continuous': False,
                   'model_approach': args.model_approach,
                   'architecture_name': args.architecture_name
    }

    try:
        os.makedirs("logs/{}".format(args.data_folder))
    except OSError:
        print("Dataset log folder already exist")

    for state_dim in [2]:
        exp_config['priors'] = ['causality', 'prop', 'temp', 'rep']
        exp_config['state_dim'] = state_dim
        log_folder, experiment_name = getLogFolder(exp_config)
        exp_config['log_folder'] = log_folder
        exp_config['experiment_name'] = experiment_name
        saveConfig(exp_config)
        # Test with a preprocessing call
        preprocessingCall(exp_config)

else:
    print("Please specify one of --config or --data_folder")
