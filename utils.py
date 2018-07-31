from __future__ import print_function, division

import os
import subprocess
import json

import torch as th
import numpy as np
from termcolor import colored


def loadData(data_folder):
    """
    :param data_folder: (str) path to the data_folder to be loaded
    :return: (Numpy dictionary-like objects and numpy arrays)
    """
    training_data = np.load('data/{}/preprocessed_data.npz'.format(data_folder))
    episode_starts = training_data['episode_starts']

    ground_truth = np.load('data/{}/ground_truth.npz'.format(data_folder))
    # Backward compatibility with previous names
    true_states = ground_truth['ground_truth_states' if 'ground_truth_states' in ground_truth.keys() else 'arm_states']
    target_positions = \
        ground_truth['target_positions' if 'target_positions' in ground_truth.keys() else 'button_positions']

    with open('data/{}/dataset_config.json'.format(data_folder), 'r') as f:
        relative_pos = json.load(f).get('relative_pos', False)

    target_pos_ = []
    # True state is the relative position to the target
    if relative_pos:
        target_idx = -1
        for i in range(len(episode_starts)):
            if episode_starts[i] == 1:
                target_idx += 1
            true_states[i] -= target_positions[target_idx]
            target_pos_.append(target_positions[target_idx])
    target_pos_ = np.array(target_pos_)

    return training_data, ground_truth, true_states, target_pos_


def getInputBuiltin():
    """
    Python 2/3 compatibility
    Returns the python 'input' builtin
    :return: (input)
    """
    try:
        return raw_input
    except NameError:
        return input


def importMaplotlib():
    """
    Fix for plotting when x11 is not available
    """
    p = subprocess.Popen(["xset", "-q"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p.communicate()
    x11_available = p.returncode == 0
    if not x11_available:
        import matplotlib
        matplotlib.use('Agg')


def detachToNumpy(tensor):
    """
    Gets a pytorch tensor and returns a numpy array
    :param tensor: (th.Tensor)
    :return: (numpy float)
    """
    return tensor.to(th.device('cpu')).detach().numpy()


def parseDataFolder(path):
    """
    Remove `data/` from dataset folder path
    if needed
    :param path: (str)
    :return: (str) name of the dataset folder
    """
    if path.startswith('data/'):
        path = path[5:]
    return path


def createFolder(path_to_folder, exist_msg):
    """
    Try to create a folder (and parents if needed)
    print a message in case the folder already exist
    :param path_to_folder: (str)
    :param exist_msg:
    """
    try:
        os.makedirs(path_to_folder)
    except OSError:
        print(exist_msg)


def printGreen(string):
    """
    Print a string in green in the terminal
    :param string: (str)
    """
    print(colored(string, 'green'))


def printYellow(string):
    """
    :param string: (str)
    """
    print(colored(string, 'yellow'))


def printRed(string):
    """
    :param string: (str)
    """
    print(colored(string, 'red'))


def printBlue(string):
    """
    :param string: (str)
    """
    print(colored(string, 'blue'))
