from __future__ import print_function, division

import os
import subprocess
import torch
from termcolor import colored


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