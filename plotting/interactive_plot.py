from __future__ import print_function, division

import argparse

import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


# Python 2/3 compatibility
try:
    input = raw_input
except NameError:
    pass

# Init seaborn
sns.set()

def createInteractivePlot(fig, states, images_path):
    fig2 = plt.figure("Image")
    image_plot = plt.imshow(cv2.imread('data/' + images_path[0]) / 255.)
    callback = ImageFinder(states, image_plot, images_path)
    fig.canvas.mpl_connect('button_press_event', callback)

def plot_2d_representation(states, rewards, images_path, name="Learned State Representation",
                            add_colorbar=True):
    plt.ion()
    fig = plt.figure(name)
    plt.clf()
    plt.scatter(states[:, 0], states[:, 1], s=7, c=np.clip(rewards, -1, 1), cmap='coolwarm', linewidths=0.1)
    plt.xlabel('State dimension 1')
    plt.ylabel('State dimension 2')
    plt.title(name)
    if add_colorbar:
        plt.colorbar(label='Reward')

    createInteractivePlot(fig, states, images_path)
    plt.show()



def plot_3d_representation(states, rewards, images_path, name="Learned State Representation",
                           add_colorbar=True):
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

    createInteractivePlot(fig, states, images_path)
    plt.show()


def plot_representation(states, rewards, images_path, name="Learned State Representation",
                        add_colorbar=True, fit_pca=True):
    """
    :param states: (numpy array)
    :param reward: (numpy 1D array)
    :param name: (str)
    :param add_colorbar: (bool)
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
        plot_2d_representation(states_matrix, rewards, images_path, name, add_colorbar)
    elif state_dim == 2:
        plot_2d_representation(states, rewards, images_path, name, add_colorbar)
    else:
        plot_3d_representation(states, rewards, images_path, name, add_colorbar)

class ImageFinder(object):
    """
    Callback for matplotlib to display an annotation when points are
    clicked on.  The point which is closest to the click and within
    xtol and ytol is identified.
    """
    def __init__(self, states, image_plot, images_path):

        self.image_plot = image_plot
        self.images_path = images_path
        self.knn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(states)
        self.state_dim = states.shape[1]

    def __call__(self, event):
        if event.inaxes:
            click_x = event.xdata
            click_y = event.ydata
            click_state = np.array([click_x, click_y])
            if self.state_dim == 3:
                # TODO: get current state of the plot
                # and apply the right transformation
                print("WARNING: 3D click not supported yet")
                # click_z = event.zdata
                click_z = 0
                click_state = np.array([click_x, click_y, click_z])

            click_state = click_state.reshape(1, -1)
            _, neighbors_indices = self.knn.kneighbors(click_state)
            path = self.images_path[neighbors_indices.flatten()[0]]
            # Load the image that corresponds to the clicked point in the space
            self.image_plot.set_data(cv2.imread('data/' + path) / 255.)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plotting script for representation')
    parser.add_argument('-i', '--input_file', type=str, default="",
                        help='Path to a npz file containing states and rewards')
    parser.add_argument('--data_folder', type=str, default="", required=True,
                        help='Path to a dataset folder, it will plot ground truth states')
    args = parser.parse_args()

    # Remove `data/` from the path if needed
    if "data/" in args.data_folder:
        args.data_folder = args.data_folder.split('data/')[1].strip("/")

    if args.input_file != "":
        print("Loading {}...".format(args.input_file))
        states_rewards = np.load(args.input_file)
        images_path = np.load('data/{}/ground_truth.npz'.format(args.data_folder))['images_path']
        plot_representation(states_rewards['states'], states_rewards['rewards'], images_path)

        input('\nPress any key to exit.')

    else:

        print("Plotting ground truth...")
        states = np.load('data/{}/ground_truth.npz'.format(args.data_folder))['arm_states']
        images_path = np.load('data/{}/ground_truth.npz'.format(args.data_folder))['images_path']
        rewards = np.load('data/{}/preprocessed_data.npz'.format(args.data_folder))['rewards']
        name = "Ground Truth States - {}".format(args.data_folder)

        print("[WARNING] Using 2D plot instead of 3D")
        states = np.concatenate((states[:, 0].reshape(-1, 1), states[:, 1].reshape(-1, 1)), axis=1)
        plot_representation(states, rewards, images_path, name)
        input('\nPress any key to exit.')
