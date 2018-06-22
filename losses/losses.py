from __future__ import print_function, division, absolute_import

import sys

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from models.priors import ReverseLayerF
from pipeline import NO_PAIRS_ERROR
from utils import printRed

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
                        self.loss_history[name][-1] += w * loss.item()
                    else:
                        self.loss_history[name].append(w * loss.item())

    def computeTotalLoss(self):
        return sum([self.weights[i] * self.losses[i] for i in range(len(self.losses))])

    def resetLosses(self):
        self.names, self.weights, self.losses = [], [], []

    def forward(self, states, next_states, minibatch_idx,
                dissimilar_pairs, same_actions_pairs):
        """
        :param states: (th.Tensor)
        :param next_states: (th.Tensor)
        :param minibatch_idx: (int)
        :param dissimilar_pairs: ([numpy array])
        :param same_actions_pairs: ([numpy array])
        :return: (th.Tensor)
        """
        dissimilar_pairs = th.from_numpy(dissimilar_pairs[minibatch_idx]).to(states.device)
        same_actions_pairs = th.from_numpy(same_actions_pairs[minibatch_idx]).to(states.device)

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
    def priorsOnStates(s, next_s, dissimilar_pairs, same_actions_pairs):
        """
        :param s: (th.Tensor) states
        :param next_s: (th.Tensor) next states
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
                minibatch_idx, dissimilar_pairs, same_actions_pairs,
                alpha=0.2, no_priors=False):
        """
        :param alpha: (float) margin that is enforced between positive & neg observation (TCN Triplet Loss)
        :param states: (th.Tensor) states for the anchor obs
        :param p_states: (th.Tensor) states for the positive obs
        :param n_states: (th.Tensor) states for the negative obs
        :param next_states: (th.Tensor)
        :param next_p_st: (th.Tensor) next states for the positive obs
        :param minibatch_idx: (int)
        :param dissimilar_pairs: ([numpy array])
        :param same_actions_pairs: ([numpy array])
        :param alpha: (float) gap value in the triplet loss
        :param no_priors: (bool) no use of priors in the loss/ Only triplets
        :return: (th.Tensor)
        """
        dissimilar_pairs = dissimilar_pairs[minibatch_idx]
        same_actions_pairs = same_actions_pairs[minibatch_idx]

        l1_loss = sum([th.sum(th.abs(param)) for param in self.reg_params])
        total_loss = self.l1_coeff * l1_loss

        # Applying the priors on the 1st view
        first_view_losses = self.priorsOnStates(states, next_states, dissimilar_pairs, same_actions_pairs)
        temp_coherence_loss, causality_loss, proportionality_loss, repeatability_loss = first_view_losses

        # Applying the priors on the 2nd view
        second_view_losses = self.priorsOnStates(p_states, next_p_st, dissimilar_pairs, same_actions_pairs)
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
                            self.loss_history[name][-1] += w * loss.item()
                        else:
                            self.loss_history[name].append(w * loss.item())

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
    :param actions: (numpy array)
    :param rewards: (numpy array)
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
    :param actions: (numpy array)
    :param rewards: (numpy array)
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
    :param actions: (numpy array)
    :return: (numpy array)
    """
    return np.where(actions[minibatch] == actions[minibatch[index]])[0]


def findPriorsPairs(batch_size, minibatchlist, actions, rewards, n_actions, n_pairs_per_action):
    """

    :param batch_size: (int)
    :param minibatchlist: ([[int]])
    :param actions: (numpy array)
    :param rewards: (numpy array)
    :param n_actions: (int)
    :param n_pairs_per_action: ([int])
    :return: ([numpy array], [numpy array])
    """
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
    """
    :param next_states_pred: (th.Tensor)
    :param next_states: (th.Tensor)
    :param weight: coefficient to weight the loss
    :param loss_object: loss criterion needed to log the loss value
    :return:
    """
    forward_loss = F.mse_loss(next_states_pred, next_states, size_average=True)
    loss_object.addToLosses('forward_loss', weight, forward_loss)
    return weight * forward_loss


def inverseModelLoss(actions_pred, actions_st, weight, loss_object):
    """
    Inverse model's loss: Cross-entropy between predicted categoriacal actions and true actions
    :param actions_pred: (th.Tensor)
    :param actions_st: (th.Tensor)
    :param weight: coefficient to weight the loss
    :param loss_object: loss criterion needed to log the loss value
    :return:
    """
    loss_fn = nn.CrossEntropyLoss()
    inverse_loss = loss_fn(actions_pred, actions_st.squeeze(1))
    loss_object.addToLosses('inverse_loss', weight, inverse_loss)
    return weight * inverse_loss


def l1Loss(params, weight, loss_object):
    """
    L1 regularization loss
    :param params: NN's weights to regularize
    :param weight: coefficient to weight the loss (float)
    :param loss_object: loss criterion needed to log the loss value
    :return:
    """
    l1_loss = sum([th.sum(th.abs(param)) for param in params])
    loss_object.addToLosses('l1_loss', weight, l1_loss)
    return weight * l1_loss


def rewardModelLoss(rewards_pred, rewards_st, weight, loss_object):
    """
    Categorical Reward prediction Loss (Cross-entropy)
    :param rewards_pred: predicted reward - categorical (th.Tensor)
    :param rewards_st: (th.Tensor)
    :param weight: coefficient to weight the loss
    :param loss_object: loss criterion needed to log the loss value
    :return:
    """
    loss_fn = nn.CrossEntropyLoss()
    reward_loss = loss_fn(rewards_pred, target=rewards_st.squeeze(1))
    loss_object.addToLosses('reward_loss', weight, reward_loss)
    return weight * reward_loss


# reconstructionLoss = nn.MSELoss(size_average=True)
# Redefine MSE otherwise PyTorch won't less us compute gradient w.r.t. input
def reconstructionLoss(_input, target):
    """
    TODO: fill out
    :param _input:
    :param target:
    :return:
    """
    return th.sum((_input - target) ** 2) / _input.data.nelement()


def autoEncoderLoss(obs, decoded_obs, next_obs, decoded_next_obs, weight, loss_object):
    """
    TODO: fill out
    :param obs:
    :param decoded_obs:
    :param next_obs:
    :param decoded_next_obs:
    :param weight: coefficient to weight the loss (float)
    :param loss_object: loss criterion needed to log the loss value
    :return:
    """
    ae_loss = reconstructionLoss(obs, decoded_obs) + reconstructionLoss(next_obs, decoded_next_obs)
    loss_object.addToLosses('reconstruction_loss', weight, ae_loss)
    return weight * ae_loss


def vaeLoss(decoded, obs, mu, logvar, weight, loss_object, beta=1):
    """
    Reconstruction + KL divergence losses summed over all elements and batch
    :param decoded: (th.Tensor)
    :param obs: (th.Tensor)
    :param mu: (th.Tensor)
    :param logvar: (th.Tensor)
    :param weight: coefficient to weight the loss (float)
    :param loss_object: loss criterion needed to log the loss value
    :param beta: (float) used to weight the KL divergence for disentangling
    :return: (th.Tensor)
    """
    generation_loss = F.mse_loss(decoded, obs, size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    kl_divergence = -0.5 * th.sum(1 + logvar - mu.pow(2) - logvar.exp())

    vae_loss = generation_loss + beta * kl_divergence
    loss_object.addToLosses('kl_loss', weight, vae_loss)
    return weight * vae_loss


def mutualInformationLoss(states, rewards_st, weight, loss_object):
    """
    Loss criterion to assess mutual information between predicted states and rewards
    :param states: (th.Tensor)
    :param rewards_st:(th.Tensor)
    :param weight: coefficient to weight the loss (float)
    :param loss_object: loss criterion needed to log the loss value
    :return:
    """
    # concat_var = th.cat((states, encodeOneHot(actions_st, n_dim=n_actions).float()), 1)
    X = states
    Y = rewards_st
    I = 0
    eps = 1e-10
    p_x = float(1 / np.sqrt(2 * np.pi)) * \
          th.exp(-th.pow(th.norm((X - th.mean(X, dim=0)) / (th.std(X, dim=0) + eps), 2, dim=1), 2) / 2) + eps
    p_y = float(1 / np.sqrt(2 * np.pi)) * \
          th.exp(-th.pow(th.norm((Y - th.mean(Y, dim=0)) / (th.std(Y, dim=0) + eps), 2, dim=1), 2) / 2) + eps
    for x in range(X.shape[0]):
        for y in range(Y.shape[0]):
            p_xy = float(1 / np.sqrt(2 * np.pi)) * \
                   th.exp(-th.pow(th.norm((th.cat([X[x], Y[y]]) - th.mean(th.cat([X, Y], dim=1), dim=0)) /
                                          (th.std(th.cat([X, Y], dim=1), dim=0) + eps), 2), 2) / 2) + eps
            I += p_xy * th.log(p_xy / (p_x[x] * p_y[y]))

    # VI = - th.sum(p_x * th.log(p_x)) - th.sum(p_y * th.log(p_y)) - 2*I
    reward_prior_loss = th.exp(-I)
    loss_object.addToLosses('reward_prior', weight, reward_prior_loss)
    return weight * reward_prior_loss


def rewardPriorLoss(states, rewards_st, weight, loss_object):
    """
    Loss expressing Correlation between predicted states and reward
    :param states: (th.Tensor)
    :param rewards_st: rewards at timestep t (th.Tensor)
    :param weight: coefficient to weight the los s
    :param loss_object: loss criterion needed to log the loss value
    :return:
    """

    reward_loss = th.mean(
        th.mm((states - th.mean(states, dim=0)).t(), (rewards_st - th.mean(rewards_st, dim=0))))
    reward_prior_loss = th.exp(-th.abs(reward_loss))
    loss_object.addToLosses('reward_prior', weight, reward_prior_loss)
    return weight * reward_prior_loss


def episodePriorLoss(minibatch_idx, minibatch_episodes, states, discriminator, balanced_sampling, weight, loss_object):
    """
    TODO: fill out
    :param minibatch_idx:
    :param minibatch_episodes:
    :param states: (th.Tensor)
    :param discriminator:
    :param balanced_sampling:
    :param weight: coefficient to weight the loss
    :param loss_object: loss criterion needed to log the loss value
    :return:
    """
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
    same_episodes = th.from_numpy((episodes == others_episodes).astype(np.float32))
    same_episodes = same_episodes.to(states.device)

    # TODO: classification accuracy/loss
    episode_loss = criterion_episode(episode_output.squeeze(1), same_episodes)
    loss_object.addToLosses('episode_prior', weight, episode_loss)
    return weight * episode_loss
