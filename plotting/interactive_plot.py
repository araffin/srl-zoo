from __future__ import print_function, division

import argparse

import cv2
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

def createImagePlot(images_path):
    fig = plt.figure("Image")
    return plt.imshow(cv2.imread('data/' + images_path[0]) / 255.)


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

    image_plot = createImagePlot(images_path)

    callback = ImageFinder(states[:, 0], states[:, 1], image_plot, images_path, x_tol=0.1, y_tol=0.1)
    fig.canvas.mpl_connect('button_press_event', callback)

    plt.show()



def plot_3d_representation(states, rewards, name="Learned State Representation",
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

    plt.show()


def plot_representation(states, rewards, name="Learned State Representation",
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
        plot_2d_representation(states_matrix, rewards, name, add_colorbar)
    elif state_dim == 2:
        plot_2d_representation(states, rewards, name, add_colorbar)
    else:
        plot_3d_representation(states, rewards, name, add_colorbar)

class ImageFinder(object):
    """
    Callback for matplotlib to display an annotation when points are
    clicked on.  The point which is closest to the click and within
    xtol and ytol is identified.
    """
    def __init__(self, xdata, ydata, image_plot, images_path, x_tol=1, y_tol=1):

        self.data = list(zip(xdata, ydata, images_path))
        self.x_tol, self.y_tol = x_tol, y_tol
        self.image_plot = image_plot

    @staticmethod
    def distance(x1, x2, y1, y2):
        """
        return the distance between two points
        """
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def __call__(self, event):
        if event.inaxes:
            clickX = event.xdata
            clickY = event.ydata
            annotes = []
            # TODO use nearest neighbors
            for i, (x, y, path) in enumerate(self.data):
                if ((clickX - self.x_tol < x < clickX + self.x_tol) and
                (clickY - self.y_tol < y < clickY + self.y_tol)):
                    annotes.append((self.distance(x, clickX, y, clickY), x, y, path))
            if len(annotes) > 0:
                annotes.sort()
                distance, x, y, path = annotes[0]
                if self.image_plot is not None:
                    self.image_plot.set_data(cv2.imread('data/' + path) / 255.)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plotting script for representation')
    parser.add_argument('-i', '--input_file', type=str, default="",
                        help='Path to a npz file containing states and rewards')
    parser.add_argument('--data_folder', type=str, default="",
                        help='Path to a dataset folder, it will plot ground truth states')
    args = parser.parse_args()

    # Remove `data/` from the path if needed
    if "data/" in args.data_folder:
        args.data_folder = args.data_folder.split('data/')[1].strip("/")

    if args.input_file != "" and args.data_folder != "":
        print("Loading {}...".format(args.input_file))
        states_rewards = np.load(args.input_file)
        images_path = np.load('data/{}/ground_truth.npz'.format(args.data_folder))['images_path']
        plot_representation(states_rewards['states'], states_rewards['rewards'], images_path)

        input('\nPress any key to exit.')

    elif args.data_folder != "":

        print("Plotting ground truth...")
        states = np.load('data/{}/ground_truth.npz'.format(args.data_folder))['arm_states']
        images_path = np.load('data/{}/ground_truth.npz'.format(args.data_folder))['images_path']
        rewards = np.load('data/{}/preprocessed_data.npz'.format(args.data_folder))['rewards']
        name = "Ground Truth States - {}".format(args.data_folder)

        plot_2d_representation(states, rewards, images_path, name)
        input('\nPress any key to exit.')
    else:
        print("You must specify one of --input_file or --data_folder")
