from __future__ import print_function, division

import argparse
import re
from textwrap import fill

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
TITLE_MAX_LENGTH = 60


def loadImage(path):
    """
    Load an image and convert it to matplotlib format
    :param path: (str)
    """
    bgr_img = cv2.imread('data/' + path)
    return cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB) / 255.


def createInteractivePlot(fig, ax, states, rewards, images_path):
    fig2 = plt.figure("Image")
    image_plot = plt.imshow(loadImage(images_path[0]))
    # Disable seaborn grid
    image_plot.axes.grid(False)
    callback = ImageFinder(states, rewards, image_plot, ax, images_path)
    fig.canvas.mpl_connect('button_release_event', callback)


def plot_2d_representation(states, rewards, images_path, name="Learned State Representation",
                           add_colorbar=True):
    plt.ion()
    fig = plt.figure(name)
    plt.clf()
    ax = fig.add_subplot(111)
    im = ax.scatter(states[:, 0], states[:, 1], s=7, c=np.clip(rewards, -1, 1), cmap='coolwarm', linewidths=0.1)
    ax.set_xlabel('State dimension 1')
    ax.set_ylabel('State dimension 2')
    ax.set_title(fill(name, TITLE_MAX_LENGTH))
    fig.tight_layout()
    if add_colorbar:
        fig.colorbar(im, label='Reward')

    createInteractivePlot(fig, ax, states, rewards, images_path)
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
    ax.set_title(fill(name, TITLE_MAX_LENGTH))
    fig.tight_layout()
    if add_colorbar:
        fig.colorbar(im, label='Reward')

    createInteractivePlot(fig, ax, states, rewards, images_path)
    plt.show()


def plot_representation(states, rewards, images_path, name="Learned State Representation",
                        add_colorbar=True, fit_pca=True):
    """
    :param states: (numpy array)
    :param rewards: (numpy 1D array)
    :param images_path: (str)
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

    def __init__(self, states, rewards, image_plot, ax, images_path):

        self.image_plot = image_plot
        self.images_path = images_path
        self.knn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(states)
        self.state_dim = states.shape[1]
        self.ax = ax
        self.states = states
        self.rewards = rewards

    def __call__(self, event):
        if event.inaxes:
            click_x = event.xdata
            click_y = event.ydata
            click_state = np.array([click_x, click_y])
            if self.state_dim == 3:
                # Return a string: 'x=0.222, y=0.452, z=0.826'
                coord_string = self.ax.format_coord(click_x, click_y)
                if 'azimuth' in coord_string:
                    # Left click -> right click should be used for 3D
                    return
                # Extract coordinates
                float_num = r"[0-9\.-]+"
                regex = r"x=(?P<x>" + float_num + ")\s*,\s*y=(?P<y>" + float_num + ")\s*,\s*z=(?P<z>" + float_num + ")"
                result = re.match(regex, coord_string)
                click_state = np.array([result.group("x"), result.group("y"), result.group("z")])

            click_state = click_state.reshape(1, -1)
            _, neighbors_indices = self.knn.kneighbors(click_state)

            state_idx = neighbors_indices.flatten()[0]
            title = "{}\n {}\n reward={}".format(fill(self.images_path[state_idx], TITLE_MAX_LENGTH),
                                                 self.states[state_idx], self.rewards[state_idx])
            self.image_plot.axes.set_title(title)
            path = self.images_path[state_idx]
            # Load the image that corresponds to the clicked point in the space
            self.image_plot.set_data(loadImage(path))


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

        plot_representation(states, rewards, images_path, name)
        input('\nPress any key to exit.')
