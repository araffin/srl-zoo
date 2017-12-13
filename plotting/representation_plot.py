from __future__ import print_function, division

import json
import argparse
from textwrap import fill

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa
from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
# Faster implementation of t-SNE:
from MulticoreTSNE import MulticoreTSNE as TSNE
# Python 2/3 compatibility
try:
    input = raw_input
except NameError:
    pass

# Init seaborn
sns.set()
INTERACTIVE_PLOT = True
TITLE_MAX_LENGTH = 50


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

def plot_tsne(states, rewards, name="T-SNE of Learned States", add_colorbar=True, path=None,
                n_components=3, perplexity=100.0, learning_rate=200.0, n_iter=1000, cmap="coolwarm"):
    """
    :param states: (numpy array)
    :param rewards: (numpy 1D array)
    :param name: (str)
    :param add_colorbar: (bool)
    :param path: (str)
    :param n_components: (int)
    :param perplexity: (float)
    :param learning_rate: (float)
    :param n_iter: (int)
    :param cmap: (str)
    """
    assert n_components in [2, 3], "You cannot applied t-SNE with n_components={}".format(n_components)
    t_sne = TSNE(n_components=n_components, perplexity=perplexity,
                learning_rate=learning_rate, n_iter=n_iter, verbose=1, n_jobs=4)
    s_tranformed = t_sne.fit_transform(states)
    plot_representation(s_tranformed, rewards, name, add_colorbar, path, cmap=cmap, fit_pca=False)


def plot_representation(states, rewards, name="Learned State Representation",
                        add_colorbar=True, path=None, fit_pca=True, cmap='coolwarm'):
    """
    :param states: (numpy array)
    :param rewards: (numpy 1D array)
    :param name: (str)
    :param add_colorbar: (bool)
    :param path: (str)
    :param fit_pca: (bool)
    :param cmap: (str)
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
        plot_2d_representation(states_matrix, rewards, name, add_colorbar, path, cmap)
    elif state_dim == 2:
        plot_2d_representation(states, rewards, name, add_colorbar, path, cmap)
    else:
        plot_3d_representation(states, rewards, name, add_colorbar, path, cmap)


def plot_2d_representation(states, rewards, name="Learned State Representation",
                           add_colorbar=True, path=None, cmap='coolwarm'):
    updateDisplayMode()
    fig = plt.figure(name)
    plt.clf()
    plt.scatter(states[:, 0], states[:, 1], s=7, c=rewards, cmap=cmap, linewidths=0.1)
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
                           add_colorbar=True, path=None, cmap='coolwarm'):
    updateDisplayMode()
    fig = plt.figure(name)
    plt.clf()
    ax = fig.add_subplot(111, projection='3d')
    im = ax.scatter(states[:, 0], states[:, 1], states[:, 2],
                    s=7, c=rewards, cmap=cmap, linewidths=0.1)
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
    :param name: (str)
    """
    # Reorder channels
    if image.shape[0] == 3 and len(image.shape) == 3:
        # (n_channels, height, width) -> (width, height, n_channels)
        image = np.transpose(image, (2, 1, 0))
    updateDisplayMode()
    fig = plt.figure(name)
    plt.imshow(image, interpolation='nearest')
    plt.gca().invert_yaxis()
    plt.xticks([])
    plt.yticks([])
    pauseOrClose(fig)


def colorPerEpisode(episode_starts):
    """
    :param episode_starts: (numpy 1D array)
    :return: (numpy 1D array)
    """
    colors = np.zeros(len(episode_starts))
    color_idx = -1
    print(np.sum(episode_starts))
    for i in range(len(episode_starts)):
        # New episode
        if episode_starts[i] == 1:
            color_idx += 1
        colors[i] = color_idx
    return colors


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plotting script for representation')
    parser.add_argument('-i', '--input_file', type=str, default="",
                        help='Path to a npz file containing states and rewards')
    parser.add_argument('--data_folder', type=str, default="",
                        help='Path to a dataset folder, it will plot ground truth states')
    parser.add_argument('--t-sne', action='store_true', default=False, help='Use t-SNE instead of PCA')
    parser.add_argument('--per-episode', action='store_true', default=False, help='Color states per episodes instead of reward')
    args = parser.parse_args()

    cmap = "tab20" if args.per_episode else "coolwarm"
    assert not (args.per_episode and args.data_folder == ""), "You must specify a datafolder when using per-episode color"
    # Remove `data/` from the path if needed
    if "data/" in args.data_folder:
        args.data_folder = args.data_folder.split('data/')[1].strip("/")

    if args.input_file != "":
        print("Loading {}...".format(args.input_file))
        states_rewards = np.load(args.input_file)
        rewards = states_rewards['rewards']
        if args.per_episode:
            episode_starts = np.load('data/{}/preprocessed_data.npz'.format(args.data_folder))['episode_starts']
            rewards = colorPerEpisode(episode_starts)[:len(rewards)]

        if args.t_sne:
            print("Using t-SNE...")
            plot_tsne(states_rewards['states'], rewards, cmap=cmap)
        else:
            plot_representation(states_rewards['states'], rewards, cmap=cmap)
        input('\nPress any key to exit.')

    elif args.data_folder != "":

        print("Plotting ground truth...")
        training_data = np.load('data/{}/preprocessed_data.npz'.format(args.data_folder))
        ground_truth = np.load('data/{}/ground_truth.npz'.format(args.data_folder))
        true_states = ground_truth['arm_states']
        name = "Ground Truth States - {}".format(args.data_folder)
        episode_starts, rewards = training_data['episode_starts'], training_data['rewards']
        button_positions = ground_truth['button_positions']
        with open('data/{}/dataset_config.json'.format(args.data_folder), 'rb') as f:
            relative_pos = json.load(f).get('relative_pos', False)

        # True state is the relative position to the button
        if relative_pos:
            button_idx = -1
            for i in range(len(episode_starts)):
                if episode_starts[i] == 1:
                    button_idx += 1
                true_states[i] -= button_positions[button_idx]

        if args.per_episode:
            rewards = colorPerEpisode(episode_starts)

        plot_representation(true_states, rewards, name, cmap=cmap)
        input('\nPress any key to exit.')

    else:
        print("You must specify one of --input_file or --data_folder")
