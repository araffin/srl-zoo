from __future__ import print_function, division

import argparse

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA

# Python 2/3 compatibility
try:
    input = raw_input
except NameError:
    pass

# Init seaborn
sns.set()


def plot_3d_representation(states, rewards, name="Learned State Representation",
                           add_colorbar=True, path=None):
    plt.ion()
    fig = plt.figure(name)
    plt.clf()
    ax = fig.add_subplot(111, projection='3d')
    im = ax.scatter(states[:, 0], states[:, 1], states[:, 2],
                    s=7, c=np.clip(rewards, -1, 1), cmap='coolwarm', linewidths=0.1)
    ax.set_xlabel('State dimension 1')
    ax.set_ylabel('State dimension 2')
    ax.set_zlabel('State dimension 3')
    ax.set_title(name)
    if add_colorbar:
        fig.colorbar(im, label='Reward')
    plt.draw()
    plt.pause(0.0001)
    if path is not None:
        plt.savefig(path)


def plot_representation(states, rewards, name="Learned State Representation",
                        add_colorbar=True, path=None):
    state_dim = states.shape[1]
    if state_dim == 2:
        plot_2d_representation(states, rewards, name, add_colorbar, path)
    elif state_dim == 3:
        plot_3d_representation(states, rewards, name, add_colorbar, path)
    else:
        if state_dim > 3:
            # PCA with 3 components by default
            print("Fitting PCA with 3 components")
            pca = PCA(n_components=3)
            pca.fit(states)
            plot_3d_representation(pca.transform(states), rewards, name, add_colorbar, path)
        else:
            # TODO: 1d plot
            print("[WARNING] state dim = {} is not supported for plotting".format(state_dim))


def plot_2d_representation(states, rewards, name="Learned State Representation", add_colorbar=True, path=None):
    plt.ion()
    plt.figure(name)
    plt.clf()
    plt.scatter(states[:, 0], states[:, 1], s=7, c=np.clip(rewards, -1, 1), cmap='coolwarm', linewidths=0.1)
    plt.xlabel('State dimension 1')
    plt.ylabel('State dimension 2')
    plt.title(name)
    if add_colorbar:
        plt.colorbar(label='Reward')
    plt.pause(0.0001)
    if path is not None:
        plt.savefig(path)


def plot_observations(observations, name='Observation Samples'):
    plt.ion()
    plt.figure(name)
    m, n = 8, 10
    for i in range(m * n):
        plt.subplot(m, n, i + 1)
        plt.imshow(observations[i].reshape(16, 16, 3), interpolation='nearest')
        plt.gca().invert_yaxis()
        plt.xticks([])
        plt.yticks([])
    plt.pause(0.0001)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plotting script for representation')
    parser.add_argument('-i', '--input_file', type=str, default="", help='Path to a npz file containing states and rewards')
    parser.add_argument('--data_folder', type=str, default="", help='Path to a dataset folder, it will plot ground truth states')
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
