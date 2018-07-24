from __future__ import print_function, division

import json
import argparse
from textwrap import fill

import matplotlib.pyplot as plt
from matplotlib import cm, colors
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


def plotRepresentation(states, rewards, name="Learned State Representation",
                       add_colorbar=True, path=None, fit_pca=False, cmap='coolwarm', true_states=None):
    """
    Plot learned state representation using rewards for coloring
    :param states: (numpy array)
    :param rewards: (numpy 1D array)
    :param name: (str)
    :param add_colorbar: (bool)
    :param path: (str)
    :param fit_pca: (bool)
    :param cmap: (str)
    :param true_states: project a 1D predicted states onto the ground_truth
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
        plot2dRepresentation(states_matrix, rewards, name, add_colorbar, path, cmap, true_states=true_states)
    elif state_dim == 2:
        plot2dRepresentation(states, rewards, name, add_colorbar, path, cmap)
    else:
        plot3dRepresentation(states, rewards, name, add_colorbar, path, cmap)


def plot2dRepresentation(states, rewards, name="Learned State Representation",
                         add_colorbar=True, path=None, cmap='coolwarm', true_states=None):
    updateDisplayMode()
    fig = plt.figure(name)
    plt.clf()
    if true_states is not None:
        plt.scatter(true_states[:len(states), 0], true_states[:len(states), 1], s=7, c=states[:, 0], cmap=cmap,
                    linewidths=0.1)
    else:
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


def plot3dRepresentation(states, rewards, name="Learned State Representation",
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


def plotImage(image, name='Observation Sample'):
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
    # plt.gca().invert_yaxis()
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


def prettyPlotAgainst(states, rewards, title="Representation", fit_pca=False, cmap='coolwarm'):
    """
    State dimensions are plotted one against the other (it creates a matrix of 2d representation)
    using rewards for coloring, the diagonal is a distribution plot, and the scatter plots have a density outline.
    :param states: (numpy tensor)
    :param rewards: (numpy array)
    :param title: (str)
    :param fit_pca: (bool)
    :param cmap: (str)
    """
    with sns.axes_style('white'):
        n = states.shape[1]
        fig, ax_mat = plt.subplots(n, n, figsize=(10, 10), sharex=False, sharey=False)
        fig.subplots_adjust(hspace=0.2, wspace=0.2)

        if fit_pca:
            title += " (PCA)"
            states = PCA(n_components=n).fit_transform(states)

        c_idx = cm.get_cmap(cmap)
        norm = colors.Normalize(vmin=np.min(rewards), vmax=np.max(rewards))

        for i in range(n):
            for j in range(n):
                x, y = states[:, i], states[:, j]
                ax = ax_mat[i, j]
                if i != j:
                    ax.scatter(x, y, c=rewards, cmap=cmap, s=5)
                    sns.kdeplot(x, y, cmap="Greys", ax=ax, shade=True, shade_lowest=False, alpha=0.2)
                    ax.set_xlim([np.min(x), np.max(x)])
                    ax.set_ylim([np.min(y), np.max(y)])
                else:
                    if len(np.unique(rewards)) < 10:
                        for r in np.unique(rewards):
                            sns.distplot(x[rewards == r], color=c_idx(norm(r)), ax=ax)
                    else:
                        sns.distplot(x, ax=ax)

                if i == 0:
                    ax.set_title("Dim {}".format(j), y=1.2)
                if i != j:
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
                        ax.xaxis.set_ticks_position('top')
                    if i == n - 1:
                        ax.xaxis.set_ticks_position('bottom')

        plt.suptitle(title, fontsize=16)
        plt.show()


def plotAgainst(states, rewards, title="Representation", fit_pca=False, cmap='coolwarm'):
    """
    State dimensions are plotted one against the other (it creates a matrix of 2d representation)
    using rewards for coloring
    :param states: (numpy tensor)
    :param rewards: (numpy array)
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


def plotCorrelation(states_rewards, ground_truth):
    """
    Correlation matrix: Target pos/GT vs. States predicted
    :param states_rewards:
    :param ground_truth:
    :return:
    """
    correlation_scalar = 0
    for index, ground_truth_name in enumerate([" Agent's position ", "Target Position"]):
        if ground_truth_name == " Agent's position ":
            key = 'ground_truth_states' if 'ground_truth_states' in ground_truth.keys() else 'arm_states'
            X = ground_truth[key][:len(rewards)]
        else:
            X = button_pos_[:len(rewards)]

        # adding epsilon in case of little variance in samples of X & Ys
        eps = 1e-12
        corr = np.corrcoef(x=X + eps, y=states_rewards['states'] + eps, rowvar=False)
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        labels = [r'$\tilde{s}_' + str(i_) + '$' for i_ in range(X.shape[1])]
        labels += [r'$s_' + str(i_) + '$' for i_ in range(states_rewards['states'].shape[1])]
        cax = ax.matshow(corr, cmap=cmap, vmin=-1, vmax=1)
        ax.set_xticklabels([''] + labels)
        ax.set_yticklabels([''] + labels)
        ax.grid(False)
        plt.title(r'Correlation Matrix: S = Predicted states | $\tilde{S}$ = ' + ground_truth_name)
        fig.colorbar(cax, label='correlation coefficient')

        ground_truth_dim = X.shape[1]
        corr_copy = corr
        for iter_gt in range(ground_truth_dim):
            corr_copy[iter_gt, iter_gt] = 0.0
            correlation_scalar += max(abs(corr_copy[iter_gt])) ** 2
    correlation_scalar = np.sqrt(correlation_scalar)
    print("Correlation value of the model with GT: ",correlation_scalar)
    pauseOrClose(fig)


def loadData(data_folder):
    """
    :param data_folder: path to the data_folder to be loaded (Str)
    :return: training_data, ground_truth, true_states, target_positions (Numpy dictionary-like objects) X 3
    """
    training_data = np.load('data/{}/preprocessed_data.npz'.format(data_folder))
    ground_truth = np.load('data/{}/ground_truth.npz'.format(data_folder))
    # Backward compatibility with previous names
    true_states = ground_truth['ground_truth_states' if 'ground_truth_states' in ground_truth.keys() else 'arm_states']
    target_positions = \
        ground_truth['target_positions' if 'target_positions' in ground_truth.keys() else 'button_positions']
    return training_data, ground_truth, true_states, target_positions


def loadOffsets(training_data, data_folder):
    """
    :param training_data:
    :param data_folder:
    :return: episode_starts, name
    """
    return training_data['episode_starts'], "Ground Truth States - {}".format(data_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plotting script for representation')
    parser.add_argument('-i', '--input-file', type=str, default="",
                        help='Path to a npz file containing states and rewards')
    parser.add_argument('--data-folder', type=str, default="",
                        help='Path to a dataset folder, it will plot ground truth states')
    parser.add_argument('--color-episode', action='store_true', default=False,
                        help='Color states per episodes instead of reward')
    parser.add_argument('--plot-against', action='store_true', default=False,
                        help='Plot against each dimension')
    parser.add_argument('--pretty-plot-against', action='store_true', default=False,
                        help='Plot against each dimension (diagonals are distributions + cleaner look)')
    parser.add_argument('--correlation', action='store_true', default=False,
                        help='Plot correlation coeff against each dimension')
    parser.add_argument('--projection', action='store_true', default=False,
                        help='Plot 1D projection of predicted state on ground truth')

    args = parser.parse_args()

    cmap = "tab20" if args.color_episode else "coolwarm"
    assert not (args.color_episode and args.data_folder == ""), \
        "You must specify a datafolder when using per-episode color"
    assert not (args.correlation and args.data_folder == ""), \
        "You must specify a datafolder when using the correlation plot"

    # Remove `data/` from the path if needed
    if args.data_folder.startswith('data/'):
        args.data_folder = args.data_folder[5:]

    if args.input_file != "":
        print("Loading {}...".format(args.input_file))
        states_rewards = np.load(args.input_file)
        rewards = states_rewards['rewards']

        if args.color_episode:
            episode_starts = np.load('data/{}/preprocessed_data.npz'.format(args.data_folder))['episode_starts']
            rewards = colorPerEpisode(episode_starts)[:len(rewards)]

        if args.plot_against:
            print("Plotting against")
            plotAgainst(states_rewards['states'], rewards, cmap=cmap)
        elif args.pretty_plot_against:
            print("Pretty plotting against")
            prettyPlotAgainst(states_rewards['states'], rewards, cmap=cmap)

        elif args.projection:
            training_data, ground_truth, true_states, _ = loadData(args.data_folder)
            plotRepresentation(states_rewards['states'], rewards, cmap=cmap, true_states=true_states)

        elif args.correlation:
            training_data, ground_truth, true_states, target_positions = loadData(args.data_folder)
            episode_starts = training_data['episode_starts']
            with open('data/{}/dataset_config.json'.format(args.data_folder), 'r') as f:
                relative_pos = json.load(f).get('relative_pos', False)

            button_pos_ = []
            # True state is the relative position to the button
            if relative_pos:
                button_idx = -1
                for i in range(len(episode_starts)):
                    if episode_starts[i] == 1:
                        button_idx += 1
                    true_states[i] -= target_positions[button_idx]
                    button_pos_.append(target_positions[button_idx])
            button_pos_ = np.array(button_pos_[:len(rewards)])

            if args.color_episode:
                rewards = colorPerEpisode(episode_starts)

            plotCorrelation(states_rewards, ground_truth)
        else:
            plotRepresentation(states_rewards['states'], rewards, cmap=cmap)
        input('\nPress any key to exit.')

    elif args.data_folder != "":

        print("Plotting ground truth...")
        training_data, ground_truth, true_states, target_positions = loadData(args.data_folder)
        episode_starts = training_data['episode_starts']
        rewards = training_data['rewards']
        name = "Ground Truth States - {}".format(args.data_folder)

        with open('data/{}/dataset_config.json'.format(args.data_folder), 'r') as f:
            relative_pos = json.load(f).get('relative_pos', False)

        # True state is the relative position to the button
        if relative_pos:
            button_idx = -1
            for i in range(len(episode_starts)):
                if episode_starts[i] == 1:
                    button_idx += 1
                true_states[i] -= target_positions[button_idx]

        if args.color_episode:
            rewards = colorPerEpisode(episode_starts)

        if args.plot_against:
            plotAgainst(true_states, rewards, cmap=cmap)
        elif args.pretty_plot_against:
            prettyPlotAgainst(true_states, rewards, cmap=cmap)
        else:
            plotRepresentation(true_states, rewards, name, fit_pca=False, cmap=cmap)
        input('\nPress any key to exit.')

    else:
        print("You must specify one of --input-file or --data-folder")
