from __future__ import print_function, division

import argparse

# Python 2/3 compatibility
try:
    import cPickle as pkl
except ImportError:
    import pickle as pkl

import numpy as np
from tqdm import tqdm
from sklearn.decomposition import IncrementalPCA

from models.learner import BaseLearner
from pipeline import saveConfig
import plotting.representation_plot as plot_script
from plotting.representation_plot import plotRepresentation
from preprocessing.data_loader import DataLoader
from utils import parseDataFolder, createFolder, getInputBuiltin


def getModelName(args):
    """
    :param args: (parsed args object)
    :return: (str)
    """
    return "{}_ST_DIM{}".format(args.method, args.state_dim)


def saveExpConfig(args, log_folder):
    """
    :param args: (parsed args object)
    :param log_folder: (str)
    """
    exp_config = {
        "batch-size": args.batch_size,
        "data-folder": args.data_folder,
        "training-set-size": args.training_set_size,
        "log-folder": log_folder,
        "state-dim": args.state_dim,
    }

    saveConfig(exp_config, print_config=True)


def toNumpyMatrix(obs_var):
    """
    :param obs_var: (th.Tensor)
    :return: (np.ndarray)
    """
    obs_tensor = obs_var.data.numpy()
    n_features = np.prod(obs_tensor.shape[1:])
    return obs_tensor.reshape(-1, n_features)


parser = argparse.ArgumentParser(description='Dimension Reduction using PCA')
parser.add_argument('-bs', '--batch-size', type=int, default=16, help='batch_size for IncrementalPCA (default: 16)')
parser.add_argument('--no-display-plots', action='store_true', default=False, help='disables live plots of the representation learned')
parser.add_argument('--data-folder', type=str, default="", help='Dataset folder', required=True)
parser.add_argument('--training-set-size', type=int, default=-1, help='Limit size of the training set (default: -1)')
parser.add_argument('--state-dim', type=int, default=3, help='State dimension')

input = getInputBuiltin()
args = parser.parse_args()
DISPLAY_PLOTS = not args.no_display_plots
plot_script.INTERACTIVE_PLOT = DISPLAY_PLOTS
args.data_folder = parseDataFolder(args.data_folder)
args.method = "pca"
log_folder = "logs/{}/baselines/{}".format(args.data_folder, getModelName(args))

createFolder(log_folder, "{} folder already exist".format(args.method))
folder_path = '{}/NearestNeighbors/'.format(log_folder)
createFolder(folder_path, "NearestNeighbors folder already exist")

saveExpConfig(args, log_folder)
print('Log folder: {}'.format(log_folder))

print('Loading data ... ')
rewards = np.load("data/{}/preprocessed_data.npz".format(args.data_folder))['rewards']
images_path = np.load("data/{}/ground_truth.npz".format(args.data_folder))['images_path']

if args.training_set_size > 0:
    limit = args.training_set_size
    images_path = images_path[:limit]
    rewards = rewards[:limit]

x_indices = np.arange(len(images_path)).astype(np.int64)

# Reduce dimension to 50 before applying t-SNE
# WARNING: it seems that if n_components > batch_size, it breaks
n_components = args.state_dim if args.method == "pca" else 50

# Avoid "Mean of empty slice." in sklearn:
batch_size = max(n_components + 1, args.batch_size)
print("batch_size = {}".format(batch_size))

# Create data loader
minibatchlist = DataLoader.createTestMinibatchList(len(images_path), batch_size)
# Training = False -> outputs only the current observation, not a tuple
data_loader = DataLoader(minibatchlist, images_path, n_workers=4, is_training=False)

print("Fitting PCA with n_components={}".format(n_components))
ipca = IncrementalPCA(n_components=n_components)

pbar = tqdm(total=len(data_loader))
for obs_var in data_loader:
    ipca.partial_fit(toNumpyMatrix(obs_var))
    pbar.update(1)
pbar.close()
# Save PCA transformation
with open(log_folder + "/pca.pkl", "wb") as f:
    pkl.dump(ipca, f)

print("Transforming observations to states")
predictions = []
for obs_var in data_loader:
    predictions.append(ipca.transform(toNumpyMatrix(obs_var)))
predictions = np.concatenate(predictions, axis=0)

BaseLearner.saveStates(predictions, images_path, rewards, log_folder)

path = "{}/learned_states.png".format(log_folder)
name = "Learned State Representation - {} \n {}".format(args.data_folder, args.method)
plotRepresentation(predictions, rewards, name, add_colorbar=True, path=path, fit_pca=False)

if DISPLAY_PLOTS:
    input('\nPress any key to exit.')
