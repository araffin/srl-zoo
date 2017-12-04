# coding: utf-8
"""
This is a PyTorch implementation of the method for state representation learning described in the paper "Learning State
Representations with Robotic Priors" (Jonschkowski & Brock, 2015).

This program is based on the original implementation by Rico Jonschkowski (rico.jonschkowski@tu-berlin.de):
https://github.com/tu-rbo/learning-state-representations-with-robotic-priors

"""
from __future__ import print_function, division, absolute_import

import argparse
import time
import sys

import numpy as np
import torch as th
import torch.nn as nn
from tqdm import tqdm

import plotting.representation_plot as plot_script
from models.base_learner import BaseLearner
from models.models import SRLConvolutionalNetwork, SRLDenseNetwork
from plotting.representation_plot import plot_representation, plt
from preprocessing.data_loader import BaxterImageLoader
from preprocessing.preprocess import INPUT_DIM
from utils import parseDataFolder
from pipeline import NO_PAIRS_ERROR

# Python 2/3 compatibility
try:
    input = raw_input
except NameError:
    pass

try:
    from functools import reduce
except ImportError:
    pass

DISPLAY_PLOTS = True
EPOCH_FLAG = 1  # Plot every 1 epoch
BATCH_SIZE = 256  #
NOISE_STD = 1e-6  # To avoid NaN (states must be different)


class RoboticPriorsLoss(nn.Module):
    """
    :param model: (PyTorch model)
    :param l1_reg: (float) l1 regularization coeff
    """

    def __init__(self, model, l1_reg=0.0):
        super(RoboticPriorsLoss, self).__init__()
        # Retrieve only trainable and regularizable parameters (we should exclude biases)
        self.reg_params = [param for name, param in model.named_parameters() if
                           ".bias" not in name and param.requires_grad]
        n_params = sum([reduce(lambda x, y: x * y, param.size()) for param in self.reg_params])
        self.l1_coeff = (l1_reg / n_params)

    def forward(self, states, next_states,
                dissimilar_pairs, same_actions_pairs, ref_point_pairs=None):
        """
        :param states: (th Variable)
        :param next_states: (th Variable)
        :param dissimilar_pairs: (th tensor)
        :param same_actions_pairs: (th tensor)
        :param ref_point_pairs: (th tensor)
        :return: (th Variable)
        """
        state_diff = next_states - states
        state_diff_norm = state_diff.norm(2, dim=1)
        similarity = lambda x, y: th.exp(-(x - y).norm(2, dim=1) ** 2)
        temp_coherence_loss = (state_diff_norm ** 2).mean()
        causality_loss = similarity(states[dissimilar_pairs[:, 0]],
                                    states[dissimilar_pairs[:, 1]]).mean()
        proportionality_loss = ((state_diff_norm[same_actions_pairs[:, 0]] -
                                 state_diff_norm[same_actions_pairs[:, 1]]) ** 2).mean()

        repeatability_loss = (
            similarity(states[same_actions_pairs[:, 0]], states[same_actions_pairs[:, 1]]) *
            (state_diff[same_actions_pairs[:, 0]] - state_diff[same_actions_pairs[:, 1]]).norm(2, dim=1) ** 2).mean()

        if len(ref_point_pairs) > 0:
            # Apply reference point prior
            # It assumes all sequences in the dataset share
            # at least one same 3D pos input image of Baxter arm
            states_same_ref_pos_t1 = states[ref_point_pairs[:, 0]]
            states_same_ref_pos_t2 = states[ref_point_pairs[:, 1]]

            same_pos_states_diff = states_same_ref_pos_t1 - states_same_ref_pos_t2
            same_pos_states_diff_norm = same_pos_states_diff.norm(2, dim=1)
            fixed_ref_point_loss = (same_pos_states_diff_norm ** 2).mean()
        else:
            fixed_ref_point_loss = 0

        l1_loss = sum([th.sum(th.abs(param)) for param in self.reg_params])

        loss = 1 * temp_coherence_loss + 1 * causality_loss + 5 * proportionality_loss \
               + 5 * repeatability_loss + 1 * fixed_ref_point_loss + self.l1_coeff * l1_loss
        return loss


class SRL4robotics(BaseLearner):
    """
    :param state_dim: (int)
    :param model_type: (str) one of "resnet" or "mlp"
    :param seed: (int)
    :param learning_rate: (float)
    :param l1_reg: (float)
    :param cuda: (bool)
    """

    def __init__(self, state_dim, model_type="resnet", log_folder="logs/default",
                 seed=1, learning_rate=0.001, l1_reg=0.0, cuda=False):

        super(SRL4robotics, self).__init__(state_dim, BATCH_SIZE, seed, cuda)

        if model_type == "resnet":
            self.model = SRLConvolutionalNetwork(self.state_dim, self.batch_size, cuda, noise_std=NOISE_STD)
        elif model_type == "mlp":
            self.model = SRLDenseNetwork(INPUT_DIM, self.state_dim, self.batch_size, cuda, noise_std=NOISE_STD)
        else:
            raise ValueError("Unknown model: {}".format(model_type))
        print("Using {} model".format(model_type))

        if cuda:
            self.model.cuda()
        learnable_params = [param for param in self.model.parameters() if param.requires_grad]
        self.optimizer = th.optim.Adam(learnable_params, lr=learning_rate)
        self.l1_reg = l1_reg
        self.log_folder = log_folder

    def learn(self, images_path, actions, rewards,
              episode_starts, is_ref_point_list=None):
        """
        Learn a state representation
        :param images_path: (numpy 1D array)
        :param actions: (numpy matrix)
        :param rewards: (numpy 1D array)
        :param episode_starts: (numpy 1D array) boolean array
                                the ith index is True if one episode starts at this frame
        :param is_ref_point_list: (numpy 1D array) Boolean array where True values represent states
                                that corresponds to the reference position
                                (when using the reference prior)
        :return: (numpy tensor) the learned states for the given observations
        """

        # PREPARE DATA -------------------------------------------------------------------------------------------------
        # here, we organize the data into minibatches
        # and find pairs for the respective loss terms
        if is_ref_point_list is None:
            is_ref_point_list = []

        num_samples = images_path.shape[0] - 1  # number of samples

        # indices for all time steps where the episode continues
        indices = np.array([i for i in range(num_samples) if not episode_starts[i + 1]], dtype='int64')
        np.random.shuffle(indices)

        # split indices into minibatches. minibatchlist is a list of lists; each
        # list is the id of the observation preserved thorough the training
        minibatchlist = [np.array(sorted(indices[start_idx:start_idx + self.batch_size]))
                         for start_idx in range(0, num_samples - self.batch_size + 1, self.batch_size)]

        if len(minibatchlist[-1]) < self.batch_size:
            print("Removing last minibatch of size {} < batch_size".format(len(minibatchlist[-1])))
            del minibatchlist[-1]

        def find_same_actions(index, minibatch):
            """
            Get observations indices where the same action was performed
            as in a reference observation
            :param index: (int)
            :param minibatch: (numpy array)
            :return: (numpy array)
            """
            return np.where(actions[minibatch] == actions[minibatch[index]])[0]

        # same_actions: list of arrays, each containing one pair of observation ids
        same_actions = [
            np.array([[i, j] for i in range(self.batch_size) for j in find_same_actions(i, minibatch) if j > i],
                     dtype='int64') for minibatch in minibatchlist]

        def find_dissimilar(index, minibatch):
            """
            check which samples should be dissimilar
            because they lead to different rewards aften the same actions
            :param index: (int)
            :param minibatch: (numpy array)
            :return: (numpy array)
            """
            return np.where((actions[minibatch] == actions[minibatch[index]]) *
                            (rewards[minibatch + 1] != rewards[minibatch[index] + 1]))[0]

        dissimilar = [np.array([[i, j] for i in range(self.batch_size) for j in find_dissimilar(i, minibatch) if j > i],
                               dtype='int64') for minibatch in minibatchlist]

        for item in same_actions + dissimilar:
            if len(item) == 0:
                msg = "No similar or dissimilar pairs found for at least one minibatch (currently is {})\n".format(
                    BATCH_SIZE)
                msg += "=> Consider increasing the batch_size or changing the seed"
                print(msg)
                sys.exit(NO_PAIRS_ERROR)

        ref_point_pairs = []
        if len(is_ref_point_list) > 0:
            def find_ref_point(index, minibatch):
                """
                Find observations corresponding to the reference
                :param index: (int)
                :param minibatch: (numpy array)
                :return: (numpy array)
                """
                return np.where(is_ref_point_list[minibatch] * is_ref_point_list[minibatch[index]])[0]

            ref_point_pairs = [np.array([[i, j] for i in range(self.batch_size)
                                         for j in find_ref_point(i, minibatch) if j > i],
                                        dtype='int64') for minibatch in minibatchlist]

            for item in ref_point_pairs:
                if len(item) == 0:
                    msg = "No same ref point position observation of the arm was found \
                            for at least one minibatch (current batch size is {})\n".format(BATCH_SIZE)
                    msg += "=> Consider increasing the batch_size or changing the seed\n same_ref_point_positions: {}".format(
                        ref_point_pairs)
                    print(msg)
                    sys.exit(NO_PAIRS_ERROR)

        baxter_data_loader = BaxterImageLoader(minibatchlist, images_path,
                                               same_actions, dissimilar, ref_point_pairs,
                                               cache_capacity=5000)

        # TRAINING -----------------------------------------------------------------------------------------------------
        criterion = RoboticPriorsLoss(self.model, self.l1_reg)
        best_error = np.inf
        best_model_path = "{}/srl_model.pth".format(self.log_folder)
        self.model.train()
        start_time = time.time()

        for epoch in range(N_EPOCHS):
            # In each epoch, we do a full pass over the training data:
            epoch_loss, epoch_batches = 0, 0
            pbar = tqdm(total=len(minibatchlist))
            baxter_data_loader.resetAndShuffle()
            for obs, next_obs, same, diss, is_ref_point_list in baxter_data_loader:
                if self.cuda:
                    obs, next_obs = obs.cuda(), next_obs.cuda()
                    same, diss = same.cuda(), diss.cuda()  #
                    if len(is_ref_point_list) > 0:
                        is_ref_point_list = is_ref_point_list.cuda()

                # Predict states given observations
                states, next_states = self.model(obs), self.model(next_obs)

                self.optimizer.zero_grad()
                loss = criterion(states, next_states, diss, same, is_ref_point_list)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.data[0]
                epoch_batches += 1
                pbar.update(1)
            pbar.close()

            train_loss = epoch_loss / epoch_batches

            # Save best model
            # TODO: use a validation set
            if train_loss < best_error:
                best_error = train_loss
                th.save(self.model.state_dict(), best_model_path)

            # Then we print the results for this epoch:
            if (epoch + 1) % EPOCH_FLAG == 0:
                print("Epoch {:3}/{}, loss:{:.4f}".format(epoch + 1, N_EPOCHS, epoch_loss / epoch_batches))
                print("{:.2f}s/epoch".format((time.time() - start_time) / (epoch + 1)))
                if DISPLAY_PLOTS:
                    # Optionally plot the current state space
                    plot_representation(self.predStatesWithDataLoader(baxter_data_loader, restore_train=True), rewards,
                                        add_colorbar=epoch == 0,
                                        name="Learned State Representation (Training Data)")
        if DISPLAY_PLOTS:
            plt.close("Learned State Representation (Training Data)")

        # Load best model before predicting states
        self.model.load_state_dict(th.load(best_model_path))

        print("Predicting states for all the observations...")
        # return predicted states for training observations
        return self.predStatesWithDataLoader(baxter_data_loader, restore_train=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch SRL with robotic priors')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--state_dim', type=int, default=2, help='state dimension (default: 2)')
    parser.add_argument('-bs', '--batch_size', type=int, default=256, help='batch_size (default: 256)')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.005, help='learning rate (default: 0.005)')
    parser.add_argument('--l1_reg', type=float, default=0.0, help='L1 regularization coeff (default: 0.0)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--no-plots', action='store_true', default=False, help='disables plots')
    parser.add_argument('--model_type', type=str, default="resnet", help='Model architecture (default: "resnet")')
    parser.add_argument('--data_folder', type=str, default="", help='Dataset folder', required=True)
    parser.add_argument('--log_folder', type=str, default='logs/default_folder',
                        help='Folder within logs/ where the experiment model and plots will be saved')
    parser.add_argument('--no_ref_prior', action='store_false', default=False,
                        help='Disable Fixed Reference Point Prior')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and th.cuda.is_available()
    args.data_folder = parseDataFolder(args.data_folder)
    DISPLAY_PLOTS = not args.no_plots
    N_EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    APPLY_5TH_PRIOR = not args.no_ref_prior
    plot_script.INTERACTIVE_PLOT = DISPLAY_PLOTS

    print('Log folder: {}'.format(args.log_folder))

    print('Loading data ... ')
    training_data = np.load("data/{}/preprocessed_data.npz".format(args.data_folder))
    actions = training_data['actions']
    rewards, episode_starts = training_data['rewards'], training_data['episode_starts']

    ground_truth = np.load("data/{}/ground_truth.npz".format(args.data_folder))

    print('Learning a state representation ... ')
    srl = SRL4robotics(args.state_dim, model_type=args.model_type, seed=args.seed,
                       log_folder=args.log_folder, learning_rate=args.learning_rate,
                       l1_reg=args.l1_reg, cuda=args.cuda)

    is_ref_point_list = None
    if APPLY_5TH_PRIOR:
        print('Applying 5th fixed ref_point prior...')
        is_ref_point_list = training_data['is_ref_point_list']

    learned_states = srl.learn(ground_truth['images_path'], actions,
                               rewards, episode_starts, is_ref_point_list)

    srl.saveStates(learned_states, ground_truth['images_path'], rewards, args.log_folder)

    name = "Learned State Representation\n {}".format(args.log_folder.split('/')[-1])
    path = "{}/learned_states.png".format(args.log_folder)
    plot_representation(learned_states, rewards, name, add_colorbar=True, path=path)

    # Do not close plot at the end of training
    if DISPLAY_PLOTS:
        input('\nPress any key to exit.')
