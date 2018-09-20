from __future__ import print_function, division, absolute_import

import sys

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from models.priors import ReverseLayerF
from .utils import correlationMatrix

try:
    from functools import reduce
except ImportError:
    pass


class LossManager:
    """
    Class in charge of Computing and Saving history of Losses
    """

    def __init__(self, model, loss_history=None):
        """
        :param model: (PyTorch model)
        :param loss_history: (dict)
        """
        # Retrieve only trainable and regularizable parameters (we should exclude biases)
        self.reg_params = [param for name, param in model.named_parameters() if
                           ".bias" not in name and param.requires_grad]
        self.loss_history = loss_history
        self.names, self.weights, self.losses = [], [], []

    def addToLosses(self, name, weight, loss_value):
        """
        :param name: (str)
        :param weight: (float)
        :param loss_value: (FloatTensor)
        :return:
        """
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


def roboticPriorsLoss(states, next_states, minibatch_idx,
            dissimilar_pairs, same_actions_pairs, weight, loss_manager):
    """
    Computing the 4 Robotic priors: Temporal coherence, Causality, Proportionality, Repeatability
    :param states: (th.Tensor)
    :param next_states: (th.Tensor)
    :param minibatch_idx: (int)
    :param dissimilar_pairs: ([np.ndarray])
    :param same_actions_pairs: ([np.ndarray])
    :param weight: coefficient to weight the loss
    :param loss_manager: loss criterion needed to log the loss value (LossManager)
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
    weights = [1, 1, 1, 1]
    names = ['temp_coherence_loss', 'causality_loss', 'proportionality_loss', 'repeatability_loss']
    losses = [temp_coherence_loss, causality_loss, proportionality_loss, repeatability_loss]

    total_loss = 0
    for idx in range(len(weights)):
        loss_manager.addToLosses(names[idx], weights[idx], losses[idx])
        total_loss += losses[idx]
    return weight * total_loss


def forwardModelLoss(next_states_pred, next_states, weight, loss_manager):
    """
    :param next_states_pred: (th.Tensor)
    :param next_states: (th.Tensor)
    :param weight: coefficient to weight the loss
    :param loss_manager: loss criterion needed to log the loss value (LossManager)
    :return:
    """
    # TODO: replace with mse_loss once the bug is fixed in PyTorch (see reconstructionLoss)
    # forward_loss = F.mse_loss(next_states_pred, next_states, reduction='elementwise_mean')
    forward_loss = reconstructionLoss(next_states_pred, next_states)
    loss_manager.addToLosses('forward_loss', weight, forward_loss)
    return weight * forward_loss


def inverseModelLoss(actions_pred, actions_st, weight, loss_manager):
    """
    Inverse model's loss: Cross-entropy between predicted categoriacal actions and true actions
    :param actions_pred: (th.Tensor)
    :param actions_st: (th.Tensor)
    :param weight: coefficient to weight the loss
    :param loss_manager: loss criterion needed to log the loss value (LossManager)
    :return:
    """
    loss_fn = nn.CrossEntropyLoss()
    inverse_loss = loss_fn(actions_pred, actions_st.squeeze(1))
    loss_manager.addToLosses('inverse_loss', weight, inverse_loss)
    return weight * inverse_loss


def l1Loss(params, weight, loss_manager):
    """
    L1 regularization loss
    :param params: NN's weights to regularize
    :param weight: coefficient to weight the loss (float)
    :param loss_manager: loss criterion needed to log the loss value (LossManager)
    :return:
    """
    l1_loss = sum([th.sum(th.abs(param)) for param in params])
    loss_manager.addToLosses('l1_loss', weight, l1_loss)
    return weight * l1_loss


def l2Loss(params, weight, loss_manager):
    """
    L2 regularization loss
    :param params: NN's weights to regularize
    :param weight: coefficient to weight the loss (float)
    :param loss_manager: loss criterion needed to log the loss value (LossManager)
    :return:
    """
    l2_loss = sum([param.norm(2) for param in params]) / len(params)
    loss_manager.addToLosses('l2_loss', weight, l2_loss)
    return weight * l2_loss


def rewardModelLoss(rewards_pred, rewards_st, weight, loss_manager):
    """
    Categorical Reward prediction Loss (Cross-entropy)
    :param rewards_pred: predicted reward - categorical (th.Tensor)
    :param rewards_st: (th.Tensor)
    :param weight: coefficient to weight the loss
    :param loss_manager: loss criterion needed to log the loss value (LossManager)
    :return:
    """
    loss_fn = nn.CrossEntropyLoss()
    reward_loss = loss_fn(rewards_pred, target=rewards_st)
    loss_manager.addToLosses('reward_loss', weight, reward_loss)
    return weight * reward_loss

def reconstructionLoss(input_image, target_image):
    """
    Reconstruction Loss for Autoencoders
    :param input_image: Observation (th.Tensor)
    :param target_image: Reconstructed observation (th.Tensor)
    :return:
    """
    # TODO: replace with mse_loss when new release is out.
    # We use a custom version because of this issue: https://github.com/pytorch/pytorch/issues/10009
    return th.sum((input_image - target_image) ** 2) / input_image.data.nelement()


def autoEncoderLoss(obs, decoded_obs, next_obs, decoded_next_obs, weight, loss_manager):
    """
    :param obs: Observation (th.Tensor)
    :param decoded_obs: reconstructed Observation (th.Tensor)
    :param next_obs: next Observation (th.Tensor)
    :param decoded_next_obs: next reconstructed Observation (th.Tensor)
    :param weight: coefficient to weight the loss (float)
    :param loss_manager: loss criterion needed to log the loss value (LossManager)
    :return:
    """
    ae_loss = reconstructionLoss(obs, decoded_obs) + reconstructionLoss(next_obs, decoded_next_obs)
    loss_manager.addToLosses('reconstruction_loss', weight, ae_loss)
    return weight * ae_loss


def generationLoss(decoded, next_decoded, obs, next_obs, weight, loss_manager):
    """
    Pixel-wise generation Loss
    :param loss_manager: loss criterion needed to log the loss value (LossManager)
    :param decoded: reconstructed Observation (th.Tensor)
    :param next_decoded: next reconstructed Observation (th.Tensor)
    :param obs: Observation (th.Tensor)
    :param next_obs: next Observation (th.Tensor)
    :param weight: (float)
    :return: (th.Tensor)
    """
    generation_loss = F.mse_loss(decoded, obs, reduction='sum')
    generation_loss += F.mse_loss(next_decoded, next_obs, reduction='sum')
    loss_name = 'generation_loss'
    loss_manager.addToLosses(loss_name, weight, generation_loss)
    return weight * generation_loss


def perceptualSimilarityLoss(encoded_real, encoded_prediction, next_encoded_real, next_encoded_prediction,
                            weight, loss_manager):
    """
    Perceptual similarity Loss for VAE as in
    # "DARLA: Improving Zero-Shot Transfer in Reinforcement Learning", Higgins et al.
    # see https://arxiv.org/pdf/1707.08475.pdf

    :param loss_manager: loss criterion needed to log the loss value (LossManager)
    :param encoded_real: states encoding the real observation by the DAE (th.Tensor)
    :param encoded_prediction: states encoding the vae's predicted observation by the DAE  (th.Tensor)
    :param next_encoded_real: states encoding the next real observation by the DAE (th.Tensor)
    :param next_encoded_prediction: states encoding the vae's predicted next observation by the DAE (th.Tensor)
    :param weight: loss for the DAE's embedding l2 distance (float)
    :return: (th.Tensor)
    """

    pretrained_dae_encoding_loss = F.mse_loss(encoded_real, encoded_prediction, reduction='sum')
    pretrained_dae_encoding_loss += F.mse_loss(next_encoded_real, next_encoded_prediction, reduction='sum')
    loss_manager.addToLosses("denoising perceptual similarity", weight, pretrained_dae_encoding_loss)
    return weight * pretrained_dae_encoding_loss


def kullbackLeiblerLoss(mu, next_mu, logvar, next_logvar, loss_manager, beta=1):
    """
    KL divergence losses summed over all elements and batch
    :param mu: mean of the distribution of samples (th.Tensor)
    :param next_mu: mean of the distribution of next samples (th.Tensor)
    :param logvar: log of the variance of the distribution of samples (th.Tensor)
    :param next_logvar: log of the variance of the distribution of next samples (th.Tensor)
    :param loss_manager: loss criterion needed to log the loss value (LossManager)
    :param beta: (float) used to weight the KL divergence for disentangling
    """

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    kl_divergence = -0.5 * th.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_divergence += -0.5 * th.sum(1 + next_logvar - next_mu.pow(2) - next_logvar.exp())
    loss_manager.addToLosses('kl_loss', beta, kl_divergence)
    return beta * kl_divergence


def mutualInformationLoss(states, rewards_st, weight, loss_manager):
    """
    TODO: Equation needs to be fixed for faster computation
    Loss criterion to assess mutual information between predicted states and rewards
    see: https://en.wikipedia.org/wiki/Mutual_information
    :param states: (th.Tensor)
    :param rewards_st:(th.Tensor)
    :param weight: coefficient to weight the loss (float)
    :param loss_manager: loss criterion needed to log the loss value
    :return:
    """
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

    mutual_info_loss = th.exp(-I)
    loss_manager.addToLosses('mutual_info', weight, mutual_info_loss)
    return weight * mutual_info_loss


def rewardPriorLoss(states, rewards_st, weight, loss_manager):
    """
    Loss expressing correlation between predicted states and reward
    :param states: (th.Tensor)
    :param rewards_st: rewards at timestep t (th.Tensor)
    :param weight: coefficient to weight the loss
    :param loss_manager: loss criterion needed to log the loss value
    :return:
    """
    corr_matrix = correlationMatrix(th.cat([states, rewards_st], dim=1).t())
    # Maximise correlation between states and rewards (last rows)
    reward_prior_loss = 1 - th.mean(th.abs(corr_matrix[-rewards_st.shape[1]:, :]))

    loss_manager.addToLosses('reward_prior', weight, reward_prior_loss)
    return weight * reward_prior_loss


def episodePriorLoss(minibatch_idx, minibatch_episodes, states, discriminator, balanced_sampling, weight, loss_manager):
    """
    :param minibatch_idx:
    :param minibatch_episodes:
    :param states: (th.Tensor)
    :param discriminator: (model)
    :param balanced_sampling: (boool)
    :param weight: coefficient to weight the loss (float)
    :param loss_manager: loss criterion needed to log the loss value (LossManager)
    :return:
    """
    # The "episode prior" idea is really close
    # to http://proceedings.mlr.press/v37/ganin15.pdf and GANs
    # We train a discriminator that try to distinguish states from same/different episodes
    # and then use the opposite gradient to update the states in order to fool it

    # lambda_ is the weight we give to the episode prior loss
    # lambda_ from 0 to 1 (as in original paper)
    # p = (minibatch_num + epoch * len(data_loader)) / (N_EPOCHS * len(data_loader))
    # lambda_ = 2. / (1. + np.exp(-10 * p)) - 1
    lambda_ = 1
    # Reverse gradient
    reverse_states = ReverseLayerF.apply(states, lambda_)

    criterion_episode = nn.BCELoss(reduction='sum')
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
    loss_manager.addToLosses('episode_prior', weight, episode_loss)
    return weight * episode_loss


def tripletLoss(states, p_states, n_states, weight, loss_manager, alpha=0.2):
    """
    :param alpha: (float) margin that is enforced between positive & neg observation (TCN Triplet Loss)
    :param states: (th.Tensor) states for the anchor obs
    :param p_states: (th.Tensor) states for the positive obs
    :param n_states: (th.Tensor) states for the negative obs
    :return: (th.Tensor)
    """
    # Time-Contrastive Triplet Loss
    distance_positive = (states - p_states).pow(2).sum(1)
    distance_negative = (states - n_states).pow(2).sum(1)
    tcn_triplet_loss = F.relu(distance_positive - distance_negative + alpha)
    tcn_triplet_loss = tcn_triplet_loss.mean()
    loss_manager.addToLosses('triplet_loss', weight, tcn_triplet_loss)
    return weight * tcn_triplet_loss
