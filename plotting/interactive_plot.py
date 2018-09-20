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

from utils import parseDataFolder, getInputBuiltin, loadData

# Init seaborn
sns.set()
TITLE_MAX_LENGTH = 60


def loadImage(path, view=0):
    """
    Load an image and convert it to matplotlib format
    :param path: (str)
    :param view: (int) : 0 for normal, {1, 2} for multi_view
    """
    if view > 0:
        path += "_" + str(view)
    path += ".jpg"
    bgr_img = cv2.imread('data/' + path)
    return cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB) / 255.


def createInteractivePlot(fig, ax, states, rewards, images_path, view=0):
    name_img = "Image"
    if view > 0:
        name_img += "_" + str(view)
    plt.figure(name_img)
    image_plot = plt.imshow(loadImage(images_path[0], view))

    # Disable seaborn grid
    image_plot.axes.grid(False)
    callback = ImageFinder(states, rewards, image_plot, ax, images_path, view)
    fig.canvas.mpl_connect('button_release_event', callback)


def plot2dRepresentation(states, rewards, images_path, name="Learned State Representation",
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


def plot3dRepresentation(states, rewards, images_path, name="Learned State Representation",
                         add_colorbar=True, multi_view=False):
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

    if multi_view:
        createInteractivePlot(fig, ax, states, rewards, images_path, view=1)
        createInteractivePlot(plt.figure(name), ax, states, rewards, images_path, view=2)
    else:
        createInteractivePlot(fig, ax, states, rewards, images_path)

    plt.show()


def plotRepresentation(states, rewards, images_path, name="Learned State Representation",
                       add_colorbar=True, fit_pca=True, multi_view=False):
    """
    :param states: (np.ndarray)
    :param rewards: (numpy 1D array)
    :param images_path: (str)
    :param name: (str)
    :param add_colorbar: (bool)
    :param fit_pca: (bool)
    :param multi_view: (bool)
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
        plot2dRepresentation(states_matrix, rewards, images_path, name, add_colorbar)
    elif state_dim == 2:
        plot2dRepresentation(states, rewards, images_path, name, add_colorbar)
    else:
        plot3dRepresentation(states, rewards, images_path, name, add_colorbar, multi_view)


class ImageFinder(object):
    """
    Callback for matplotlib to display an annotation when points are
    clicked on.  The point which is closest to the click.
    """

    def __init__(self, states, rewards, image_plot, ax, images_path, view=0):

        self.image_plot = image_plot
        self.images_path = images_path
        self.knn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(states)
        self.state_dim = states.shape[1]
        self.ax = ax
        self.states = states
        self.rewards = rewards
        self.view = view

        # Highlight the selected state
        self.kwargs = dict(s=130, color='green', alpha=0.7)
        coords = self.getCoords(0)
        if states.shape[1] > 2:
            self.dot = ax.scatter([coords[0]], [coords[1]], [coords[2]], **self.kwargs)
        else:
            self.dot = ax.scatter([coords[0]], [coords[1]], **self.kwargs)

    def getCoords(self, state_idx):
        return self.states[state_idx].tolist()

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
            self.image_plot.set_data(loadImage(path, self.view))
            coords = self.getCoords(state_idx)
            self.dot.set_offsets(coords[:2])
            if self.states.shape[1] > 2:
                # Recreate the highlighted dot each time because set_offsets does not work in 3d
                self.dot.remove()
                self.dot = self.ax.scatter([coords[0]], [coords[1]], [coords[2]], **self.kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Interactive plot of representation (left click for 2D, right click for 3D)')
    parser.add_argument('-i', '--input-file', type=str, default="",
                        help='Path to a npz file containing states and rewards')
    parser.add_argument('--data-folder', type=str, default="", required=True,
                        help='Path to a dataset folder, it will plot ground truth states')
    parser.add_argument('--multi-view', action='store_true', default=False,
                        help='Enable use of multiple camera')
    args = parser.parse_args()

    args.data_folder = parseDataFolder(args.data_folder)

    if args.input_file != "":
        print("Loading {}...".format(args.input_file))
        states_rewards = np.load(args.input_file)
        images_path = np.load('data/{}/ground_truth.npz'.format(args.data_folder))['images_path']
        plotRepresentation(states_rewards['states'], states_rewards['rewards'], images_path,
                           multi_view=args.multi_view)

        getInputBuiltin()('\nPress any key to exit.')

    else:

        print("Plotting ground truth...")
        training_data, ground_truth, true_states, _ = loadData(args.data_folder)
        images_path = ground_truth['images_path']
        rewards = training_data['rewards']
        name = "Ground Truth States - {}".format(args.data_folder)

        plotRepresentation(true_states, rewards, images_path, name, fit_pca=False, multi_view=args.multi_view)
        getInputBuiltin()('\nPress any key to exit.')
