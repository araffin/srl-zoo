from __future__ import print_function, division

from termcolor import colored


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


def priorsToString(priors_list):
    """
    Convert a list of priors to a string
    (for the experiment name)
    :param priors_list: [str]
    :return: (str)
    """
    string = '_'
    for index, priors in enumerate(priors_list):
        # Keep only first three letters
        string = string + priors[0:3]
    return string
