from __future__ import print_function, division

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA

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
            plot_3d_representation(pca.transform(states), name, add_colorbar, path)
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
