from __future__ import print_function, division

import argparse

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Init seaborn
sns.set()


def plotLosses(loss_history, path=None):
    """
    :param loss_history: (dict)
    :param path: (str)
    """
    keys = list(loss_history.keys())
    keys.sort()

    x = np.arange(1, len(loss_history[keys[0]]) + 1)

    plt.figure("Losses")
    for key in keys:
        plt.plot(x, loss_history[key], label=key, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.title("Evolution of the losses")
    if path is not None:
        plt.savefig(path + "/losses.png")
    else:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot losses')
    parser.add_argument('-i', '--input-file', type=str, default="", required=True,
                        help='Path to a npz file containing losses history')
    parser.add_argument('--log-folder', type=str, default="", help='Path to a log folder')
    args = parser.parse_args()

    path = args.log_folder if args.log_folder != "" else None

    plotLosses(np.load(args.input_file), path)
