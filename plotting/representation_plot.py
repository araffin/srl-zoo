from __future__ import print_function, division

import json
import argparse
from textwrap import fill

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
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
    assert n_components in [2, 3], "You cannot apply t-SNE with n_components={}".format(n_components)
    t_sne = TSNE(n_components=n_components, perplexity=perplexity,
                 learning_rate=learning_rate, n_iter=n_iter, verbose=1, n_jobs=4)
    s_transformed = t_sne.fit_transform(states)
    plot_representation(s_transformed, rewards, name, add_colorbar, path, cmap=cmap, fit_pca=False)


def plot_representation(states, rewards, name="Learned State Representation",
                        add_colorbar=True, path=None, fit_pca=True, cmap='coolwarm'):
    """
    Plot learned state representation using rewards for coloring
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


def plot_against(states, rewards, title="Representation", fit_pca=False, cmap='coolwarm'):
    """
    State dimensions are plotted one against the other (it creates a matrix of 2d representation)
    using rewards for coloring
    :param states: (numpy tensor)
    :param reward: (numpy array)
    :param title: (str)
    :param fit_pca: (bool)
    :param cmap: (str)
    """
    n = states.shape[1]
    fig, ax_mat = plt.subplots(n, n, figsize=(10, 10), sharex=False, sharey=False)
    fig.subplots_adjust(hspace=0.0, wspace=0.0)

    if fit_pca:
        title += " (PCA)"
        states = PCA(n_components=n).fit_transform(states)

    for i in range(n):
        for j in range(n):
            x, y = states[:, i], states[:, j]
            ax = ax_mat[i, j]
            ax.scatter(x, y, c=rewards, cmap=cmap, s=5)
            ax.set_xlim([np.min(x), np.max(x)])
            ax.set_ylim([np.min(y), np.max(y)])

            # Hide ticks
            if i != 0 and i != n - 1:
                ax.xaxis.set_visible(False)
            if j != 0 and j != n - 1:
                ax.yaxis.set_visible(False)

            # Set up ticks only on one side for the "edge" subplots...
            if j == 0:
                ax.yaxis.set_ticks_position('left')
            if j == n - 1:
                ax.yaxis.set_ticks_position('right')
            if i == 0:
                ax.set_title("Dim {}".format(j), y=1.2)
                ax.xaxis.set_ticks_position('top')
            if i == n - 1:
                ax.xaxis.set_ticks_position('bottom')


    plt.suptitle(title, fontsize=16)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plotting script for representation')
    parser.add_argument('-i', '--input_file', type=str, default="",
                        help='Path to a npz file containing states and rewards')
    parser.add_argument('--data_folder', type=str, default="",
                        help='Path to a dataset folder, it will plot ground truth states')
    parser.add_argument('--t-sne', action='store_true', default=False, help='Use t-SNE instead of PCA')
    parser.add_argument('--color-episode', action='store_true', default=False,
                        help='Color states per episodes instead of reward')
    parser.add_argument('--plot-against', action='store_true', default=False,
                        help='Plot against each dimension')
    args = parser.parse_args()

    cmap = "tab20" if args.color_episode else "coolwarm"
    assert not (args.color_episode and args.data_folder == ""),\
           "You must specify a datafolder when using per-episode color"
    # Remove `data/` from the path if needed
    if "data/" in args.data_folder:
        args.data_folder = args.data_folder.split('data/')[1].strip("/")

    if args.input_file != "":
        print("Loading {}...".format(args.input_file))
        states_rewards = np.load(args.input_file)
        rewards = states_rewards['rewards']
        if args.color_episode:
            episode_starts = np.load('data/{}/preprocessed_data.npz'.format(args.data_folder))['episode_starts']
            rewards = colorPerEpisode(episode_starts)[:len(rewards)]

        if args.t_sne:
            print("Using t-SNE...")
            plot_tsne(states_rewards['states'], rewards, cmap=cmap)
        elif args.plot_against:
            print("Plotting against")
            plot_against(states_rewards['states'], rewards, cmap=cmap)
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
        with open('data/{}/dataset_config.json'.format(args.data_folder), 'r') as f:
            relative_pos = json.load(f).get('relative_pos', False)

        # True state is the relative position to the button
        if relative_pos:
            button_idx = -1
            for i in range(len(episode_starts)):
                if episode_starts[i] == 1:
                    button_idx += 1
                true_states[i] -= button_positions[button_idx]

        if args.color_episode:
            rewards = colorPerEpisode(episode_starts)

        if args.plot_against:
            plot_against(true_states, rewards, cmap=cmap)
        else:
            plot_representation(true_states, rewards, name, cmap=cmap)
        input('\nPress any key to exit.')

    else:
        print("You must specify one of --input_file or --data_folder")
