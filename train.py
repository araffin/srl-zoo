"""
This is a PyTorch implementation based on the method
for state representation learning described in the paper "Learning State
Representations with Robotic Priors" (Jonschkowski & Brock, 2015).

This program is based on the original implementation by Rico Jonschkowski (rico.jonschkowski@tu-berlin.de):
https://github.com/tu-rbo/learning-state-representations-with-robotic-priors

"""
from __future__ import print_function, division, absolute_import

import argparse

import numpy as np
import torch as th

import models.learner as learner
import plotting.representation_plot as plot_script
import preprocessing
from models.learner import SRL4robotics
from pipeline import getLogFolderName, saveConfig
from plotting.losses_plot import plotLosses
from plotting.representation_plot import plotRepresentation
from utils import parseDataFolder, createFolder, getInputBuiltin, loadData


def buildConfig(args):
    """
    :param args: (parsed args object)
    :return: (dict)
    """
    # Fix to use this function in srl_baselines/
    split_index = args.split_index if hasattr(args, "split_index") else -1
    beta = args.beta if hasattr(args, "beta") else -1
    exp_config = {
        "batch-size": args.batch_size,
        "beta": beta,
        "data-folder": args.data_folder,
        "epochs": args.epochs,
        "learning-rate": args.learning_rate,
        "training-set-size": args.training_set_size,
        "log-folder": "",
        "model-type": args.model_type,
        "seed": args.seed,
        "state-dim": args.state_dim,
        "knn-samples": 200,
        "knn-seed": 1,
        "l1-reg": 0,
        "losses": args.losses,
        "n-neighbors": 5,
        "n-to-plot": 5,
        "split-index": split_index
    }
    return exp_config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='State Representation Learning with PyTorch')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--state-dim', type=int, default=2, help='state dimension (default: 2)')
    parser.add_argument('-bs', '--batch-size', type=int, default=256, help='batch_size (default: 256)')
    parser.add_argument('--val-size', type=float, default=0.2, help='Validation set size in percentage (default: 0.2)')
    parser.add_argument('--training-set-size', type=int, default=-1,
                        help='Limit size (number of samples) of the training set (default: -1)')
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.005, help='learning rate (default: 0.005)')
    parser.add_argument('--l1-reg', type=float, default=0.0, help='L1 regularization coeff (default: 0.0)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--no-display-plots', action='store_true', default=False,
                        help='disables live plots of the representation learned')
    parser.add_argument('--model-type', type=str, default="custom_cnn",
                        choices=['custom_cnn', 'resnet', 'mlp', 'linear'],
                        help='Model architecture (default: "custom_cnn")')
    parser.add_argument('--data-folder', type=str, default="", help='Dataset folder', required=True)
    parser.add_argument('--log-folder', type=str, default="",
                        help='Folder where the experiment model and plots will be saved. ' +
                             'By default, automatically computing KNN-MSE and saving logs at location ' +
                             'logs/DatasetName/YY-MM-DD_HHhMM_SS_ModelType_ST_DIMN_LOSEES')
    parser.add_argument('--multi-view', action='store_true', default=False,
                        help='Enable use of multiple camera')
    parser.add_argument('--balanced-sampling', action='store_true', default=False,
                        help='Force balanced sampling for episode independent prior instead of uniform')
    parser.add_argument('--losses', type=str, nargs='+', default=["priors"], help='losses(s)',
                        choices=["forward", "inverse", "reward", "priors", "episode-prior", "reward-prior", "triplet",
                                 "autoencoder", "vae"], )
    parser.add_argument('--beta', type=float, default=1.0,
                        help='(For beta-VAE only) Factor on the KL divergence, higher value means more disentangling.')
    parser.add_argument('--split-index', type=int, default=-1,
                        help='Split representation models (default: -1, no split)')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and th.cuda.is_available()
    args.data_folder = parseDataFolder(args.data_folder)
    learner.DISPLAY_PLOTS = not args.no_display_plots
    learner.N_EPOCHS = args.epochs
    learner.BATCH_SIZE = args.batch_size
    learner.VALIDATION_SIZE = args.val_size
    learner.BALANCED_SAMPLING = args.balanced_sampling
    plot_script.INTERACTIVE_PLOT = learner.DISPLAY_PLOTS

    # Dealing with losses to use
    losses = list(set(args.losses))

    if args.multi_view is True:
        # Setting variables involved data-loading from multiple cameras,
        # involved also in adapting the input layers of NN to that data
        # PS: those are stacked images - 3 if triplet loss, 2 otherwise
        if "triplet" in losses:
            preprocessing.preprocess.N_CHANNELS = 9
        else:
            preprocessing.preprocess.N_CHANNELS = 6

    assert not ("autoencoder" in losses and "vae" in losses), "Model cannot be both an Autoencoder and a VAE (come on!)"
    assert not (("autoencoder" in losses or "vae" in losses)
                and args.model_type == "resnet"), "Model cannot be an Autoencoder or VAE using ResNet Architecture !"
    assert not ("vae" in losses and args.model_type == "linear"), "Model cannot be VAE using Linear Architecture !"
    assert not (args.multi_view and args.model_type == "resnet"), \
        "Default ResNet input layer is not suitable for stacked images!"

    print('Loading data ... ')
    training_data, ground_truth, _, _ = loadData(args.data_folder)
    rewards, episode_starts = training_data['rewards'], training_data['episode_starts']
    actions = training_data['actions']
    n_actions = int(np.max(actions) + 1)

    # Try to convert old python 2 format
    try:
        images_path = np.array([path.decode("utf-8") for path in ground_truth['images_path']])
    except AttributeError:
        images_path = ground_truth['images_path']

    # Building the experiment config file
    exp_config = buildConfig(args)
    if args.log_folder == "":
        # Automatically create dated log folder for configs
        createFolder("logs/{}".format(exp_config['data-folder']), "Dataset log folder already exist")
        # Check that the dataset is already preprocessed
        log_folder, experiment_name = getLogFolderName(exp_config)
        args.log_folder = log_folder
    else:
        experiment_name = "{}_{}".format(args.model_type, losses)
    exp_config['log-folder'] = args.log_folder
    exp_config['experiment-name'] = experiment_name
    exp_config['n_actions'] = n_actions
    exp_config['multi-view'] = args.multi_view
    print('Log folder: {}'.format(args.log_folder))

    print('Learning a state representation ... ')
    srl = SRL4robotics(args.state_dim, model_type=args.model_type, seed=args.seed,
                       log_folder=args.log_folder, learning_rate=args.learning_rate,
                       l1_reg=args.l1_reg, cuda=args.cuda, multi_view=args.multi_view,
                       losses=losses, n_actions=n_actions, beta=args.beta, split_index=args.split_index)

    if args.training_set_size > 0:
        limit = args.training_set_size
        actions = actions[:limit]
        images_path = images_path[:limit]
        rewards = rewards[:limit]
        episode_starts = episode_starts[:limit]

    # Save configs in log folder
    saveConfig(exp_config, print_config=True)

    loss_history, learned_states, pairs_name_weights = srl.learn(images_path, actions, rewards, episode_starts)

    # Update config with weights for each losses
    exp_config['losses_weights'] = pairs_name_weights
    saveConfig(exp_config, print_config=True)

    # Save plot
    plotLosses(loss_history, args.log_folder)
    srl.saveStates(learned_states, images_path, rewards, args.log_folder)
    # Save losses losses history
    np.savez('{}/loss_history.npz'.format(args.log_folder), **loss_history)

    name = "Learned State Representation\n {}".format(args.log_folder.split('/')[-1])
    path = "{}/learned_states.png".format(args.log_folder)

    # PLOT REPRESENTATION
    plotRepresentation(learned_states, rewards, name, add_colorbar=True, path=path)

    # Do not close plot at the end of training
    if learner.DISPLAY_PLOTS:
        getInputBuiltin()('\nPress any key to exit.')
