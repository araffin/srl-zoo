from __future__ import print_function, division

from termcolor import colored


def printGreen(string):
    print(colored(string, 'green'))


def printYellow(string):
    print(colored(string, 'yellow'))


def printRed(string):
    print(colored(string, 'red'))


def printBlue(string):
    print(colored(string, 'blue'))


def priorsToString(priors_list):
    """
    :param priors_list: [str]
    :return: (str)
    """
    string = '_'
    for index, priors in enumerate(priors_list):
        string = string + priors[0:3]
    return string
