from __future__ import print_function, division

import os

import torch
from termcolor import colored

# Python 2/3 compatibility
try:
    input = raw_input
except NameError:
    # Assign the built-in function to a variable
    input = input


def detachToNumpy(tensor):
    """
    gets a pytorch tensor and returns a numpy array
    :param tensor: (pytorch tensor)
    :return: (numpy float)
    """
    return tensor.to(torch.device('cpu')).detach().numpy()


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
