from __future__ import print_function, division

import argparse
from textwrap import fill

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

# Python 2/3 compatibility
try:
    input = raw_input
except NameError:
    pass

# Init seaborn
sns.set()
INTERACTIVE_PLOT = True
TITLE_MAX_LENGTH = 60


def updateDisplayMode():
    """
    Enable or disable interactive plot
    see: http://matplotlib.org/faq/usage_faq.html#what-is-interactive-mode
    """
    if INTERACTIVE_PLOT:
        plt.ion()
    else:
        plt.ioff()

def pauseOrClose(fig):
    """
    :param fig: (matplotlib figure object)
    """
    if INTERACTIVE_PLOT:
        plt.draw()
        plt.pause(0.0001)  # Small pause to update the plot
    else:
        plt.close(fig)

def plot_representation(states, rewards, name="Learned State Representation",
                        add_colorbar=True, path=None, fit_pca=True):
    """
    :param states: (numpy array)
    :param reward: (numpy 1D array)
    :param name: (str)
    :param add_colorbar: (bool)
    :param path: (str)
    :param fit_pca: (bool)
    """
    state_dim = states.shape[1]
    if state_dim != 1 and (fit_pca or state_dim > 3):
        name += " (PCA)"
        n_components = min(state_dim, 3)
        print("Fitting PCA with {} components".format(n_components))
        states = PCA(n_components=n_components).fit_transform(states)

    if state_dim == 1:
        # Extend states as 2D:
        states_matrix = np.zeros((states.shape[0], 2))
        states_matrix[:, 0] = states[:, 0]
        plot_2d_representation(states_matrix, rewards, name, add_colorbar, path)
    elif state_dim == 2:
        plot_2d_representation(states, rewards, name, add_colorbar, path)
    else:
        plot_3d_representation(states, rewards, name, add_colorbar, path)



def plot_2d_representation(states, rewards, name="Learned State Representation", add_colorbar=True, path=None):
    updateDisplayMode()
    fig = plt.figure(name)
    plt.clf()
    plt.scatter(states[:, 0], states[:, 1], s=7, c=np.clip(rewards, -1, 1), cmap='coolwarm', linewidths=0.1)
    plt.xlabel('State dimension 1')
    plt.ylabel('State dimension 2')
    plt.title(fill(name, TITLE_MAX_LENGTH))
    fig.tight_layout()
    if add_colorbar:
        plt.colorbar(label='Reward')
    if path is not None:
        plt.savefig(path)
    pauseOrClose(fig)


def plot_3d_representation(states, rewards, name="Learned State Representation",
                           add_colorbar=True, path=None):
    updateDisplayMode()
    fig = plt.figure(name)
    plt.clf()
    ax = fig.add_subplot(111, projection='3d')
    im = ax.scatter(states[:, 0], states[:, 1], states[:, 2],
                    s=7, c=np.clip(rewards, -1, 1), cmap='coolwarm', linewidths=0.1)
    ax.set_xlabel('State dimension 1')
    ax.set_ylabel('State dimension 2')
    ax.set_zlabel('State dimension 3')
    ax.set_title(fill(name, TITLE_MAX_LENGTH))
    fig.tight_layout()
    if add_colorbar:
        fig.colorbar(im, label='Reward')
    if path is not None:
        plt.savefig(path)
    pauseOrClose(fig)


def plot_observations(observations, name='Observation Samples'):
    updateDisplayMode()
    fig = plt.figure(name)
    m, n = 8, 10
    for i in range(m * n):
        plt.subplot(m, n, i + 1)
        plt.imshow(observations[i].reshape(16, 16, 3), interpolation='nearest')
        plt.gca().invert_yaxis()
        plt.xticks([])
        plt.yticks([])
    pauseOrClose(fig)


def plot_image(image, name='Observation Sample'):
    """
    Display an image
    :param image: (numpy tensor) (with values in [0, 1])
    """
    # Reorder channels
    if image.shape[0] == 3 and len(x.shape) == 3:
        # (n_channels, height, width) -> (width, height, n_channels)
        image = np.transpose(image, (2, 1, 0))
    updateDisplayMode()
    fig = plt.figure(name)
    plt.imshow(image, interpolation='nearest')
    plt.gca().invert_yaxis()
    plt.xticks([])
    plt.yticks([])
    pauseOrClose(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plotting script for representation')
    parser.add_argument('-i', '--input_file', type=str, default="",
                        help='Path to a npz file containing states and rewards')
    parser.add_argument('--data_folder', type=str, default="",
                        help='Path to a dataset folder, it will plot ground truth states')
    args = parser.parse_args()

    if args.input_file != "":
        print("Loading {}...".format(args.input_file))
        states_rewards = np.load(args.input_file)
        plot_representation(states_rewards['states'], states_rewards['rewards'])
        input('\nPress any key to exit.')

    elif args.data_folder != "":
        # Remove `data/` from the path if needed
        if "data/" in args.data_folder:
            args.data_folder = args.data_folder.split('data/')[1].strip("/")
        print("Plotting ground truth...")
        states = np.load('data/{}/ground_truth.npz'.format(args.data_folder))['arm_states']
        rewards = np.load('data/{}/preprocessed_data.npz'.format(args.data_folder))['rewards']
        name = "Ground Truth States - {}".format(args.data_folder)
        plot_representation(states, rewards, name)
        input('\nPress any key to exit.')

    else:
        print("You must specify one of --input_file or --data_folder")
