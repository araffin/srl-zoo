from __future__ import print_function, division, absolute_import

import sys

import numpy as np
import torch as th
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from utils import printRed, printYellow
from pipeline import NO_PAIRS_ERROR, NAN_ERROR
from models.models import encodeOneHot
from models.priors import ReverseLayerF

try:
    from functools import reduce
except ImportError:
    pass

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
        self.names, self.weights, self.losses = [], [], []

    def addToLosses(self, name, weight, loss_value):
        self.names.append(name)
        self.weights.append(weight)
        self.losses.append(loss_value)

    def updateLossHistory(self):
        if self.loss_history is not None:
            for name, w, loss in zip(self.names, self.weights, self.losses):
                if w > 0:
                    if len(self.loss_history[name]) > 0:
                        self.loss_history[name][-1] += w * loss.data[0]
                    else:
                        self.loss_history[name].append(w * loss.data[0])

    def computeTotalLoss(self):
        return sum([self.weights[i] * self.losses[i] for i in range(len(self.losses))])

    def resetLosses(self):
        self.names, self.weights, self.losses = [], [], []

    def forward(self, states, next_states,
                dissimilar_pairs=None, same_actions_pairs=None):
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
        self.weights = [1, 1, 1, 1]
        self.names = ['temp_coherence_loss', 'causality_loss', 'proportionality_loss', 'repeatability_loss']
        self.losses = [temp_coherence_loss, causality_loss, proportionality_loss, repeatability_loss]

        if self.l1_coeff > 0:
            l1Loss(self.reg_params, self.l1_coeff, self)

        return self.computeTotalLoss()


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


def overSampling(batch_size, m_list, pairs, function_on_pairs, actions, rewards):
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
                    for j in function_on_pairs(i, m_list[minibatch_id], minibatch, actions, rewards):
                        # Copy samples - done once
                        if (j != i) & (minibatch_id != m_id) and do:
                            m_list[minibatch_id][j] = minibatch[j]
                            pairs[minibatch_id] = np.array([[i, j]])
                            do = False
    print('Dealt with {} minibatches - {}'.format(counter, pair_name))
    return pairs, m_list


def findDissimilar(index, minibatch1, minibatch2, actions, rewards):
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


def findSameActions(index, minibatch, actions):
    """
    Get observations indices where the same action was performed
    as in a reference observation
    :param index: (int)
    :param minibatch: (numpy array)
    :return: (numpy array)
    """
    return np.where(actions[minibatch] == actions[minibatch[index]])[0]


def findPriorsPairs(batch_size, minibatchlist, actions, rewards, n_actions, n_pairs_per_action):
    dissimilar_pairs = [
        np.array(
            [[i, j] for i in range(batch_size) for j in findDissimilar(i, minibatch, minibatch, actions, rewards) if
             j > i],
            dtype='int64') for minibatch in minibatchlist]

    # sampling relevant pairs to have at least a pair of dissimilar obs in every minibatches
    dissimilar_pairs, minibatchlist = overSampling(batch_size, minibatchlist, dissimilar_pairs,
                                                   findDissimilar, actions, rewards)
    # same_actions: list of arrays, each containing one pair of observation ids
    same_actions_pairs = [
        np.array([[i, j] for i in range(batch_size) for j in findSameActions(i, minibatch, actions) if j > i],
                 dtype='int64') for minibatch in minibatchlist]

    for pair, minibatch in zip(same_actions_pairs, minibatchlist):
        for i in range(n_actions):
            n_pairs_per_action[i] += np.sum(actions[minibatch[pair[:, 0]]] == i)

    # Stats about pairs
    print("Number of pairs per action:")
    print(n_pairs_per_action)
    print("Pairs of {} unique actions".format(np.sum(n_pairs_per_action > 0)))

    for item in same_actions_pairs + dissimilar_pairs:
        if len(item) == 0:
            msg = "No same actions or dissimilar pairs found for at least one minibatch (currently is {})\n".format(
                batch_size)
            msg += "=> Consider increasing the batch_size or changing the seed"
            printRed(msg)
            sys.exit(NO_PAIRS_ERROR)
    return dissimilar_pairs, same_actions_pairs


def forwardModelLoss(next_states_pred, next_states, weight, loss_object):
    forward_loss = F.mse_loss(next_states_pred, next_states, size_average=True)
    loss_object.addToLosses('forward_loss', weight, forward_loss)
    return weight * forward_loss


def inverseModelLoss(actions_pred, actions_st, weight, loss_object):
    lossFn = nn.CrossEntropyLoss()
    inverse_loss = lossFn(actions_pred, actions_st.squeeze(1))
    loss_object.addToLosses('inverse_loss', weight, inverse_loss)
    return weight * inverse_loss


def l1Loss(params, weight, loss_object):
    l1_loss = sum([th.sum(th.abs(param)) for param in params])
    loss_object.addToLosses('l1_loss', weight, l1_loss)
    return weight * l1_loss


def rewardModelLoss(rewards_pred, rewards_st, weight, loss_object):
    reward_loss = F.mse_loss(rewards_pred, rewards_st, size_average=True)
    loss_object.addToLosses('reward_loss', weight, reward_loss)
    return weight * reward_loss


# reconstructionLoss = nn.MSELoss(size_average=True)
# Redefine MSE otherwise PyTorch won't less us compute gradient w.r.t. input
def reconstructionLoss(_input, target):
    return th.sum((_input - target) ** 2) / _input.data.nelement()


def autoEncoderLoss(obs, decoded_obs, next_obs, decoded_next_obs, weight, loss_object):
    ae_loss = reconstructionLoss(obs, decoded_obs) + reconstructionLoss(next_obs, decoded_next_obs)
    loss_object.addToLosses('reconstruction_loss', weight, ae_loss)
    return weight * ae_loss


def rewardPriorLoss(states, rewards_st, actions_st, n_actions, weight, loss_object):
        
    concat_var = th.cat((states, encodeOneHot(actions_st, n_dim=n_actions).float()), 1)
    ###### Mutual information loss
    X = concat_var
    Y = rewards_st
    I = 0
    eps = 1e-10
    p_x  = float(1/np.sqrt(2*np.pi)) * th.exp(-th.pow(th.norm((X - th.mean(X, dim=0))/(th.std(X, dim=0)+eps), 2, dim=1), 2)/2) + eps
    p_y  = float(1/np.sqrt(2*np.pi)) * th.exp(-th.pow(th.norm((Y - th.mean(Y, dim=0))/(th.std(Y, dim=0)+eps), 2, dim=1), 2)/2) + eps
    for x in range(X.shape[0]):
        for y in range(Y.shape[0]):
             p_xy = float(1/np.sqrt(2*np.pi)) * th.exp(-th.pow(th.norm((th.cat([X[x],Y[y]]) - th.mean(th.cat([X,Y], dim=1), dim=0))/(th.std(th.cat([X,Y], dim=1), dim=0)+ eps), 2), 2) /2)  + eps
             I += p_xy * th.log(p_xy  / (p_x[x] * p_y[y]))
             #print('inter I:',p_xy,p_x[x],p_y[y])

    #VI = - th.sum(p_x * th.log(p_x)) - th.sum(p_y * th.log(p_y)) - 2*I
    #print('VI,I:',VI,I)
    reward_prior_loss = th.exp(-I)

    ############ Correlation loss (st+at) vs. rt
    #concat_var = th.cat((states, encodeOneHot(actions_st, n_dim=n_actions).float()), 1)
    #reward_loss = th.mean(
    #th.mm((concat_var - th.mean(concat_var, dim=0)).t(), (rewards_st - th.mean(rewards_st, dim=0))))
    #reward_prior_loss = th.exp(-reward_loss)
    ######################
    loss_object.addToLosses('reward_prior', weight, reward_prior_loss)
    return weight * reward_prior_loss


def episodePriorLoss(minibatch_idx, minibatch_episodes, states, discriminator, balanced_sampling, weight, loss_object, cuda=False):
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

    criterion_episode = nn.BCELoss(size_average=False)
    # Get episodes indices for current minibatch
    episodes = np.array(minibatch_episodes[minibatch_idx])

    # Sample other states
    if balanced_sampling:
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
    episode_output = discriminator(episode_input)

    others_episodes = episodes[others_idx]
    same_episodes = Variable(th.from_numpy((episodes == others_episodes).astype(np.float32)))

    if cuda:
        same_episodes = same_episodes.cuda()
    # TODO: classification accuracy/loss
    episode_loss = criterion_episode(episode_output.squeeze(1), same_episodes)
    loss_object.addToLosses('episode_prior', weight, episode_loss)
    return weight * episode_loss
