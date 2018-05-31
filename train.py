# coding: utf-8
"""
This is a PyTorch implementation based on the method
for state representation learning described in the paper "Learning State
Representations with Robotic Priors" (Jonschkowski & Brock, 2015).

This program is based on the original implementation by Rico Jonschkowski (rico.jonschkowski@tu-berlin.de):
https://github.com/tu-rbo/learning-state-representations-with-robotic-priors

"""
from __future__ import print_function, division, absolute_import

import argparse
import time
import sys
from collections import defaultdict

import numpy as np
import torch as th
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import plotting.representation_plot as plot_script
from models.base_learner import BaseLearner
from models import SRLConvolutionalNetwork, SRLDenseNetwork, SRLCustomCNN, TripletNet, Discriminator
from models.priors import ReverseLayerF
from plotting.representation_plot import plotRepresentation, plt
from plotting.losses_plot import plotLosses
from preprocessing.data_loader import CustomDataLoader
from preprocessing.preprocess import INPUT_DIM
from utils import parseDataFolder, printRed, printYellow
from pipeline import NO_PAIRS_ERROR, NAN_ERROR

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
VALIDATION_SIZE = 0.2  # 20% of training data for validation

# Experimental: episode independent prior
BALANCED_SAMPLING = False  # Whether to do Uniform (default) or balanced sampling


def overSampling(batch_size, m_list, pairs, function_on_pairs):
    """
    Look for minibatches missing pairs of observations with the similar/dissimilar rewards (see params)
    Sample for each of those minibatches an observation from another batch that satisfies the
    similarity/dissimilarity with the 1rst observation.
    return the new pairs & the modified minibatch list
    :param batch_size: (int)
    :param m_list: (list) mini-batch list
    :param pairs: similar / dissimilar pairs
    :param function_on_pairs: (function) findDissimilar applied to pairs
    :return: (list, list) pairs, mini-batch list modified
    """
    # For a each minibatch_id
    if function_on_pairs.__name__ == "findDissimilar":
        pair_name = 'dissimilar pairs'
    else:
        pair_name = 'Unknown pairs'
    counter = 0
    for minibatch_id, d in enumerate(pairs):
        do = True
        if len(d) == 0:
            counter += 1
        # Do if it contains no similar pairs of samples
        while do and len(d) == 0:
            # for every minibatch & obs of a mini-batch list
            for m_id, minibatch in enumerate(m_list):
                for i in range(batch_size):
                    # Look for similar samples j in other minibatches m_id
                    for j in function_on_pairs(i, m_list[minibatch_id], minibatch):
                        # Copy samples - done once
                        if (j != i) & (minibatch_id != m_id) and do:
                            m_list[minibatch_id][j] = minibatch[j]
                            pairs[minibatch_id] = np.array([[i, j]])
                            do = False
    print('Dealt with {} minibatches - {}'.format(counter, pair_name))
    return pairs, m_list


class RoboticPriorsLoss(nn.Module):
    """
    :param model: (PyTorch model)
    :param l1_reg: (float) l1 regularization coeff
    :param loss_history: (dict) will be modified
    """

    def __init__(self, model, l1_reg=0.0, loss_history=None):
        super(RoboticPriorsLoss, self).__init__()
        # Retrieve only trainable and regularizable parameters (we should exclude biases)
        self.reg_params = [param for name, param in model.named_parameters() if
                           ".bias" not in name and param.requires_grad]
        n_params = sum([reduce(lambda x, y: x * y, param.size()) for param in self.reg_params])
        self.l1_coeff = (l1_reg / n_params)
        self.loss_history = loss_history

    def forward(self, states, next_states, dissimilar_pairs, same_actions_pairs):
        """
        :param states: (th Variable)
        :param next_states: (th Variable)
        :param dissimilar_pairs: (th tensor)
        :param same_actions_pairs: (th tensor)
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
                (state_diff[same_actions_pairs[:, 0]] - state_diff[same_actions_pairs[:, 1]]).norm(2,
                                                                                                   dim=1) ** 2).mean()

        l1_loss = sum([th.sum(th.abs(param)) for param in self.reg_params])

        total_loss = 1 * temp_coherence_loss + 1 * causality_loss + 1 * proportionality_loss \
                     + 1 * repeatability_loss + self.l1_coeff * l1_loss

        if self.loss_history is not None:
            weights = [1, 1, 1, 1, self.l1_coeff]
            names = ['temp_coherence_loss', 'causality_loss', 'proportionality_loss',
                     'repeatability_loss', 'l1_loss']
            losses = [temp_coherence_loss, causality_loss, proportionality_loss,
                      repeatability_loss, l1_loss]
            for name, w, loss in zip(names, weights, losses):
                if w > 0:
                    if len(self.loss_history[name]) > 0:
                        self.loss_history[name][-1] += w * loss.data[0]
                    else:
                        self.loss_history[name].append(w * loss.data[0])

        return total_loss


class RoboticPriorsTripletLoss(nn.Module):
    """
    :param model: (PyTorch model)
    :param l1_reg: (float) l1 regularization coeff
    :param loss_history: (dict) will be modified
    """

    def __init__(self, model, l1_reg=0.0, loss_history=None):
        super(RoboticPriorsTripletLoss, self).__init__()
        # Retrieve only trainable and regularizable parameters (we should exclude biases)
        self.reg_params = [param for name, param in model.named_parameters() if
                           ".bias" not in name and param.requires_grad]
        n_params = sum([reduce(lambda x, y: x * y, param.size()) for param in self.reg_params])
        self.l1_coeff = (l1_reg / n_params)
        self.loss_history = loss_history

    @staticmethod
    def priorsOnStates(s, next_s, same_actions_pairs, dissimilar_pairs):
        """
        :param s: (th Variable) states
        :param next_s: (th Variable) next states
        :param dissimilar_pairs: (th tensor)
        :param same_actions_pairs: (th tensor)
        """

        state_diff = next_s - s
        state_diff_norm = state_diff.norm(2, dim=1)
        similarity = lambda x, y: th.exp(-(x - y).norm(2, dim=1) ** 2)
        temp_coherence_loss = (state_diff_norm ** 2).mean()
        causality_loss = similarity(s[dissimilar_pairs[:, 0]],
                                    s[dissimilar_pairs[:, 1]]).mean()
        proportionality_loss = ((state_diff_norm[same_actions_pairs[:, 0]] -
                                 state_diff_norm[same_actions_pairs[:, 1]]) ** 2).mean()

        repeatability_loss = (
                similarity(s[same_actions_pairs[:, 0]], s[same_actions_pairs[:, 1]]) *
                (state_diff[same_actions_pairs[:, 0]] - state_diff[same_actions_pairs[:, 1]]).norm(2,
                                                                                                   dim=1) ** 2).mean()

        return temp_coherence_loss, causality_loss, proportionality_loss, repeatability_loss

    # Override in the case of use of Time-Contrastive Triplet Loss
    def forward(self, states, p_states, n_states, next_states, next_p_st,
                dissimilar_pairs, same_actions_pairs,
                alpha=0.2, no_priors=False):
        """
        :param alpha: (float) margin that is enforced between positive & neg observation (TCN Triplet Loss)
        :param states: (th Variable) states for the anchor obs
        :param p_states: (th Variable) states for the positive obs
        :param n_states: (th Variable) states for the negative obs
        :param next_states: (th Variable)
        :param next_p_st: (th Variable) next states for the positive obs
        :param dissimilar_pairs: (th Tensor)
        :param same_actions_pairs: (th Tensor)
        :param alpha: (float) gap value in the triplet loss
        :param no_priors: (bool) no use of priors in the loss/ Only triplets
        :return: (th Variable)
        """
        l1_loss = sum([th.sum(th.abs(param)) for param in self.reg_params])
        total_loss = self.l1_coeff * l1_loss

        # Applying the priors on the 1st view
        first_view_losses = self.priorsOnStates(states, next_states, same_actions_pairs, dissimilar_pairs)
        temp_coherence_loss, causality_loss, proportionality_loss, repeatability_loss = first_view_losses

        # Applying the priors on the 2nd view
        second_view_losses = self.priorsOnStates(p_states, next_p_st, same_actions_pairs, dissimilar_pairs)
        temp_coherence_loss_2, causality_loss_2, proportionality_loss_2, repeatability_loss_2 = second_view_losses

        temp_coherence_loss += temp_coherence_loss_2
        causality_loss += causality_loss_2
        proportionality_loss += proportionality_loss_2
        repeatability_loss += repeatability_loss_2

        if not no_priors:
            total_loss += 1 * temp_coherence_loss + 1 * causality_loss + 1 * proportionality_loss \
                          + 1 * repeatability_loss
            if self.loss_history is not None:
                weights = [1, 1, 1, 1, self.l1_coeff]
                names = ['temp_coherence_loss', 'causality_loss', 'proportionality_loss',
                         'repeatability_loss', 'l1_loss']
                losses = [temp_coherence_loss, causality_loss, proportionality_loss,
                          repeatability_loss, l1_loss]
                for name, w, loss in zip(names, weights, losses):
                    if w > 0:
                        if len(self.loss_history[name]) > 0:
                            self.loss_history[name][-1] += w * loss.data[0]
                        else:
                            self.loss_history[name].append(w * loss.data[0])

        # Time-Contrastive Triplet Loss
        distance_positive = (states - p_states).pow(2).sum(1)
        distance_negative = (states - n_states).pow(2).sum(1)
        tcn_trplet_loss = F.relu(distance_positive - distance_negative + alpha)
        tcn_trplet_loss = tcn_trplet_loss.mean()
        total_loss += 1 * tcn_trplet_loss

        return total_loss


class SRL4robotics(BaseLearner):
    """
    :param state_dim: (int)
    :param model_type: (str) one of "resnet", "mlp" or "custom_cnn"
    :param seed: (int)
    :param learning_rate: (float)
    :param l1_reg: (float)
    :param cuda: (bool)
    :param multi_view (bool)
    :param episode_prior (bool)
    """

    def __init__(self, state_dim, model_type="resnet", log_folder="logs/default",
                 seed=1, learning_rate=0.001, l1_reg=0.0, cuda=False,
                 multi_view=False, no_priors=False, episode_prior=False):

        super(SRL4robotics, self).__init__(state_dim, BATCH_SIZE, seed, cuda)

        self.multi_view = multi_view
        self.episode_prior = episode_prior

        if model_type == "resnet":
            self.model = SRLConvolutionalNetwork(self.state_dim, cuda, noise_std=NOISE_STD)
        elif model_type == "custom_cnn":
            self.model = SRLCustomCNN(self.state_dim, cuda, noise_std=NOISE_STD)
        elif model_type == "triplet_cnn":
            self.model = TripletNet(self.state_dim)
        elif model_type == "mlp":
            self.model = SRLDenseNetwork(INPUT_DIM, self.state_dim, self.batch_size, cuda, noise_std=NOISE_STD)
        else:
            raise ValueError("Unknown model: {}".format(model_type))
        print("Using {} model".format(model_type))

        if self.episode_prior:
            self.discriminator = Discriminator(2 * self.state_dim)

        if cuda:
            self.model.cuda()
            if self.episode_prior:
                self.discriminator.cuda()

        learnable_params = [param for param in self.model.parameters() if param.requires_grad]

        if self.episode_prior:
            learnable_params += [p for p in self.discriminator.parameters()]

        self.optimizer = th.optim.Adam(learnable_params, lr=learning_rate)
        self.l1_reg = l1_reg
        self.log_folder = log_folder
        self.model_type = model_type
        self.no_priors = no_priors

    def learn(self, images_path, actions, rewards, episode_starts):
        """
        Learn a state representation
        :param images_path: (numpy 1D array)
        :param actions: (numpy matrix)
        :param rewards: (numpy 1D array)
        :param episode_starts: (numpy 1D array) boolean array
                                the ith index is True if one episode starts at this frame
        :return: (numpy tensor) the learned states for the given observations
        """

        # PREPARE DATA -------------------------------------------------------------------------------------------------
        # here, we organize the data into minibatches
        # and find pairs for the respective loss terms

        num_samples = images_path.shape[0] - 1  # number of samples

        # indices for all time steps where the episode continues
        indices = np.array([i for i in range(num_samples) if not episode_starts[i + 1]], dtype='int64')
        np.random.shuffle(indices)

        # split indices into minibatches. minibatchlist is a list of lists; each
        # list is the id of the observation preserved through the training
        minibatchlist = [np.array(sorted(indices[start_idx:start_idx + self.batch_size]))
                         for start_idx in range(0, len(indices) - self.batch_size + 1, self.batch_size)]

        if len(minibatchlist[-1]) < self.batch_size:
            printYellow("Removing last minibatch of size {} < batch_size".format(len(minibatchlist[-1])))
            del minibatchlist[-1]

        # Number of minibatches used for validation:
        n_val_batches = np.round(VALIDATION_SIZE * len(minibatchlist)).astype(np.int64)
        val_indices = np.random.permutation(len(minibatchlist))[:n_val_batches]
        # Print some info
        print("{} minibatches for training, {} samples".format(len(minibatchlist) - n_val_batches,
                                                               (len(minibatchlist) - n_val_batches) * BATCH_SIZE))
        print("{} minibatches for validation, {} samples".format(n_val_batches, n_val_batches * BATCH_SIZE))
        assert n_val_batches > 0, "Not enough sample to create a validation set"

        def findDissimilar(index, minibatch1, minibatch2):
            """
            check which samples should be dissimilar
            because they lead to different rewards after the same actions
            :param index: (int)
            :param minibatch1: (numpy array)
            :param minibatch2: (numpy array)
            :return: (dict, numpy array)
            """
            return np.where((actions[minibatch2] == actions[minibatch1[index]]) *
                            (rewards[minibatch2 + 1] != rewards[minibatch1[index] + 1]))[0]

        dissimilar = [
            np.array([[i, j] for i in range(self.batch_size) for j in findDissimilar(i, minibatch, minibatch) if j > i],
                     dtype='int64') for minibatch in minibatchlist]

        # sampling relevant pairs to have at least a pair of dissimilar obs in every minibatches
        dissimilar, minibatchlist = overSampling(self.batch_size, minibatchlist, dissimilar, findDissimilar)

        def findSameActions(index, minibatch):
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
            np.array([[i, j] for i in range(self.batch_size) for j in findSameActions(i, minibatch) if j > i],
                     dtype='int64') for minibatch in minibatchlist]

        # Stats about pairs
        action_set = set(actions)
        n_actions = np.max(actions) + 1
        print("{} unique actions / {} actions".format(len(action_set), n_actions))
        n_pairs_per_action = np.zeros(n_actions, dtype=np.int64)
        n_obs_per_action = np.zeros(n_actions, dtype=np.int64)

        for i in range(n_actions):
            n_obs_per_action[i] = np.sum(actions == i)

        print("Number of observations per action")
        print(n_obs_per_action)

        for pair, minibatch in zip(same_actions, minibatchlist):
            for i in range(n_actions):
                n_pairs_per_action[i] += np.sum(actions[minibatch[pair[:, 0]]] == i)

        print("Number of pairs per action:")
        print(n_pairs_per_action)
        print("Pairs of {} unique actions".format(np.sum(n_pairs_per_action > 0)))

        for item in same_actions + dissimilar:
            if len(item) == 0:
                msg = "No same actions or dissimilar pairs found for at least one minibatch (currently is {})\n".format(
                    BATCH_SIZE)
                msg += "=> Consider increasing the batch_size or changing the seed"
                printRed(msg)
                sys.exit(NO_PAIRS_ERROR)

        # For episode prior
        idx_to_episode = {idx: episode_idx for idx, episode_idx in enumerate(np.cumsum(episode_starts))}
        minibatch_episodes = [[idx_to_episode[i] for i in minibatch] for minibatch in minibatchlist]

        data_loader = CustomDataLoader(minibatchlist, images_path,
                                       same_actions, dissimilar, cache_capacity=100,
                                       multi_view=self.multi_view,
                                       triplets=(self.model_type == "triplet_cnn"))
        # TRAINING -----------------------------------------------------------------------------------------------------
        loss_history = defaultdict(list)
        if self.model_type == "triplet_cnn":
            criterion = RoboticPriorsTripletLoss(self.model, self.l1_reg, loss_history)
        else:
            criterion = RoboticPriorsLoss(self.model, self.l1_reg, loss_history)
        criterion_episode = nn.BCELoss(size_average=False)
        best_error = np.inf
        best_model_path = "{}/srl_model.pth".format(self.log_folder)
        self.model.train()
        start_time = time.time()
        for epoch in range(N_EPOCHS):
            # In each epoch, we do a full pass over the training data:
            epoch_loss, epoch_batches = 0, 0
            val_loss = 0
            pbar = tqdm(total=len(minibatchlist))
            data_loader.resetAndShuffle()

            for minibatch_num, _input in enumerate(data_loader):
                # Unpack input
                minibatch_idx, obs, next_obs, same_actions, diss_pairs = _input
                if self.cuda:
                    obs, next_obs = obs.cuda(), next_obs.cuda()
                    same_actions, diss_pairs = same_actions.cuda(), diss_pairs.cuda()

                self.optimizer.zero_grad()

                # Predict states given observations as in Time Contrastive Network (Triplet Loss) [Sermanet et al.]
                if self.model_type == "triplet_cnn":
                    states, positive_states, negative_states = self.model(obs[:, :3:, :, :], obs[:, 3:6, :, :],
                                                                          obs[:, 6:, :, :])

                    next_states, next_positive_states, next_negative_states = self.model(next_obs[:, :3:, :, :],
                                                                                         next_obs[:, 3:6, :, :],
                                                                                         next_obs[:, 6:, :, :])

                    loss = criterion(states, positive_states, negative_states, next_states, next_positive_states,
                                     diss_pairs, same_actions, no_priors=self.no_priors)

                else:
                    states, next_states = self.model(obs), self.model(next_obs)
                    loss = criterion(states, next_states, diss_pairs, same_actions)

                    if self.episode_prior:
                        # The "episode prior" idea is really close
                        # to http://proceedings.mlr.press/v37/ganin15.pdf and GANs
                        # We train a discriminator that try to distinguish states for same/different episodes
                        # and then use the opposite gradient to update the states in order to fool it

                        # lambda_ is the weight we give to the episode prior loss
                        # lambda_ from 0 to 1 (as in original paper)
                        # p = (minibatch_num + epoch * len(data_loader)) / (N_EPOCHS * len(data_loader))
                        # lambda_ = 2. / (1. + np.exp(-10 * p)) - 1
                        lambda_ = 1
                        # Reverse gradient
                        reverse_states = ReverseLayerF.apply(states, lambda_)

                        # Get episodes indices for current minibatch
                        episodes = np.array(minibatch_episodes[minibatch_idx])

                        # Sample other states
                        if BALANCED_SAMPLING:
                            # Balanced sampling
                            others_idx = np.arange(len(episodes))
                            for i in range(len(episodes)):
                                if np.random.rand() > 0.5:
                                    others_idx[i] = np.random.choice(np.where(episodes != episodes[i])[0])
                                else:
                                    others_idx[i] = np.random.choice(np.where(episodes == episodes[i])[0])
                        else:
                            # Uniform (unbalanced) sampling
                            others_idx = np.random.permutation(len(states))


                        # Create input for episode discriminator
                        episode_input = th.cat((reverse_states, reverse_states[others_idx, :]), dim=1)
                        episode_output = self.discriminator(episode_input)

                        others_episodes = episodes[others_idx]
                        same_episodes = Variable(th.from_numpy((episodes == others_episodes).astype(np.float32)))

                        if self.cuda:
                            same_episodes = same_episodes.cuda()
                        # TODO: classification accuracy/loss
                        loss_episode = criterion_episode(episode_output.squeeze(1), same_episodes)
                        loss += 1 * loss_episode
                # We have to call backward in both train/val
                # to avoid memory error
                loss.backward()
                if minibatch_idx in val_indices:
                    val_loss += loss.data[0]
                    # We do not optimize on validation data
                    # so optimizer.step() is not called
                else:
                    self.optimizer.step()
                    epoch_loss += loss.data[0]
                    epoch_batches += 1
                pbar.update(1)
            pbar.close()

            train_loss = epoch_loss / epoch_batches
            val_loss /= n_val_batches
            # Even if loss_history is modified by RoboticPriorsLoss object
            # we make it explicit
            loss_history = criterion.loss_history
            loss_history['train_loss'].append(train_loss)
            loss_history['val_loss'].append(val_loss)
            for key in loss_history.keys():
                if key in ['train_loss', 'val_loss']:
                    continue
                loss_history[key][-1] /= epoch_batches
                if epoch + 1 < N_EPOCHS:
                    loss_history[key].append(0)

            # Save best model
            if val_loss < best_error:
                best_error = val_loss
                th.save(self.model.state_dict(), best_model_path)

            if np.isnan(train_loss):
                print("NaN Loss, consider increasing NOISE_STD in the gaussian noise layer")
                sys.exit(NAN_ERROR)

            # Then we print the results for this epoch:
            if (epoch + 1) % EPOCH_FLAG == 0:
                print("Epoch {:3}/{}, train_loss:{:.4f} val_loss:{:.4f}".format(epoch + 1, N_EPOCHS, train_loss,
                                                                                val_loss))
                print("{:.2f}s/epoch".format((time.time() - start_time) / (epoch + 1)))
                if DISPLAY_PLOTS:
                    # Optionally plot the current state space
                    plotRepresentation(self.predStatesWithDataLoader(data_loader, restore_train=True), rewards,
                                       add_colorbar=epoch == 0,
                                       name="Learned State Representation (Training Data)")
        if DISPLAY_PLOTS:
            plt.close("Learned State Representation (Training Data)")

        # Load best model before predicting states
        self.model.load_state_dict(th.load(best_model_path))

        print("Predicting states for all the observations...")
        # return predicted states for training observations
        return loss_history, self.predStatesWithDataLoader(data_loader, restore_train=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch SRL with robotic priors')
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
    parser.add_argument('--no-plots', action='store_true', default=False, help='disables plots')
    parser.add_argument('--model-type', type=str, default="custom_cnn",
                        choices=['custom_cnn', 'resnet', 'mlp', 'triplet_cnn'],
                        help='Model architecture (default: "custom_cnn")')
    parser.add_argument('--data-folder', type=str, default="", help='Dataset folder', required=True)
    parser.add_argument('--log-folder', type=str, default='logs/default_folder',
                        help='Folder within logs/ where the experiment model and plots will be saved')
    parser.add_argument('--multi-view', action='store_true', default=False,
                        help='Enable use of multiple camera')
    parser.add_argument('--episode-prior', action='store_true', default=False,
                        help='Enable episode independent prior')
    parser.add_argument('--balanced-sampling', action='store_true', default=False,
                        help='Force balanced sampling for episode independent prior instead of uniform')
    parser.add_argument('--no-priors', action='store_true', default=False,
                        help='Disable use of priors - in case of triplet loss')


    args = parser.parse_args()
    args.cuda = not args.no_cuda and th.cuda.is_available()
    args.data_folder = parseDataFolder(args.data_folder)
    DISPLAY_PLOTS = not args.no_plots
    N_EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    VALIDATION_SIZE = args.val_size
    BALANCED_SAMPLING = args.balanced_sampling
    plot_script.INTERACTIVE_PLOT = DISPLAY_PLOTS

    print('Log folder: {}'.format(args.log_folder))

    print('Loading data ... ')
    training_data = np.load("data/{}/preprocessed_data.npz".format(args.data_folder))
    actions = training_data['actions']
    rewards, episode_starts = training_data['rewards'], training_data['episode_starts']

    ground_truth = np.load("data/{}/ground_truth.npz".format(args.data_folder))
    # Try to convert old python 2 format
    try:
        images_path = np.array([path.decode("utf-8") for path in ground_truth['images_path']])
    except AttributeError:
        images_path = ground_truth['images_path']

    print('Learning a state representation ... ')
    srl = SRL4robotics(args.state_dim, model_type=args.model_type, seed=args.seed,
                       log_folder=args.log_folder, learning_rate=args.learning_rate,
                       l1_reg=args.l1_reg, cuda=args.cuda, multi_view=args.multi_view,
                       no_priors=args.no_priors, episode_prior=args.episode_prior)

    if args.training_set_size > 0:
        limit = args.training_set_size
        actions = actions[:limit]
        images_path = images_path[:limit]
        rewards = rewards[:limit]
        episode_starts = episode_starts[:limit]

    loss_history, learned_states = srl.learn(images_path, actions, rewards, episode_starts)
    # Save losses losses history
    np.savez('{}/loss_history.npz'.format(args.log_folder), **loss_history)
    # Save plot
    plotLosses(loss_history, args.log_folder)

    srl.saveStates(learned_states, images_path, rewards, args.log_folder)

    name = "Learned State Representation\n {}".format(args.log_folder.split('/')[-1])
    path = "{}/learned_states.png".format(args.log_folder)
    plotRepresentation(learned_states, rewards, name, add_colorbar=True, path=path)

    # Do not close plot at the end of training
    if DISPLAY_PLOTS:
        input('\nPress any key to exit.')
