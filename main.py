"""
This is a PyTorch implementation of the method for state representation learning described in the paper "Learning State
Representations with Robotic Priors" (Jonschkowski & Brock, 2015).

This program is based on the original implementation by Rico Jonschkowski (rico.jonschkowski@tu-berlin.de):
https://github.com/tu-rbo/learning-state-representations-with-robotic-priors

TODO: generator to load images on the fly
"""
from __future__ import print_function, division

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
import torch as th
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable

# Python 2/3 compatibility
try:
    input = raw_input
except NameError:
    pass

try:
    from functools import reduce
except ImportError:
    pass

BATCHSIZE = 256
NOISE_STD = 1e-6  # To avoid NaN (states must be different)


class SRLNetwork(nn.Module):
    """
    Neural Net for State Representation Learning (SRL)
    input shape : 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224
    :param state_dim: (int)
    :param batchsize: (int)
    :param cuda: (bool)
    """

    def __init__(self, state_dim=2, batchsize=256, cuda=False):
        super(SRLNetwork, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = False
        # Replace the last fully-connected layer
        self.resnet.fc = nn.Linear(512, state_dim)
        if cuda:
            self.resnet.cuda()
        self.noise = GaussianNoise(batchsize, state_dim, NOISE_STD, cuda=cuda)

    def forward(self, x):
        x = self.resnet(x)
        x = self.noise(x)
        return x


class GaussianNoise(nn.Module):
    """
    Gaussian Noise layer
    :param batchsize: (int)
    :param input_dim: (int)
    :param std: (float) standard deviation
    :param mean: (float)
    :param cuda: (bool)
    """

    def __init__(self, batchsize, input_dim, std, mean=0, cuda=False):
        super(GaussianNoise, self).__init__()
        self.std = std
        self.mean = mean
        self.noise = Variable(th.zeros(batchsize, input_dim))
        if cuda:
            self.noise = self.noise.cuda()

    def forward(self, x):
        if self.training:
            self.noise.data.normal_(self.mean, std=self.std)
            return x + self.noise
        return x


class RoboticPriorsLoss(nn.Module):
    def __init__(self, model, l1_reg=0):
        super(RoboticPriorsLoss, self).__init__()
        # Retrieve only trainable and regularizable parameters (we should exclude biases)
        self.reg_params = [param for name, param in model.named_parameters() if
                           ".bias" not in name and param.requires_grad]
        n_params = sum([reduce(lambda x, y: x * y, param.size()) for param in self.reg_params])
        self.l1_coeff = (l1_reg / n_params)

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
            (state_diff[same_actions_pairs[:, 0]] - state_diff[same_actions_pairs[:, 1]]).norm(2, dim=1) ** 2).mean()

        l1_loss = sum([th.sum(th.abs(param)) for param in self.reg_params])

        loss = 1 * temp_coherence_loss + 1 * causality_loss + 5 * proportionality_loss + 5 * repeatability_loss + self.l1_coeff * l1_loss
        return loss


class SRL4robotics:
    """
    :param state_dim: (int)
    :param seed: (int)
    :param learning_rate: (float)
    :param l1_reg: (float)
    :param cuda: (bool)
    """

    def __init__(self, state_dim, seed=1, learning_rate=0.001, l1_reg=0.001, cuda=False):

        self.state_dim = state_dim
        self.batchsize = BATCHSIZE
        self.cuda = cuda

        np.random.seed(seed)
        th.manual_seed(seed)
        if cuda:
            th.cuda.manual_seed(seed)

        self.model = SRLNetwork(self.state_dim, self.batchsize, cuda)
        if cuda:
            self.model.cuda()
        learnable_params = [param for param in self.model.parameters() if param.requires_grad]
        self.optimizer = th.optim.Adam(learnable_params, lr=learning_rate)
        self.l1_reg = l1_reg

    def _predFn(self, observations, restore_train=True):
        # test mode
        self.model.eval()
        states = self.model(observations)
        if restore_train:
            self.model.train()
        if self.cuda:
            return states.data.cpu().numpy()
        return states.data.numpy()

    def learn(self, observations, actions, rewards, episode_starts):

        # PREPARE DATA -------------------------------------------------------------------------------------------------
        # here, we normalize the observations, organize the data into minibatches
        # and find pairs for the respective loss terms

        # We assume that observations are already preprocessed
        observations = observations.astype(np.float32)

        # For testing
        obs_var = Variable(th.from_numpy(observations), volatile=True)
        if self.cuda:
            obs_var = obs_var.cuda()

        num_samples = observations.shape[0] - 1  # number of samples

        # indices for all time steps where the episode continues
        indices = np.array([i for i in range(num_samples) if not episode_starts[i + 1]], dtype='int64')
        np.random.shuffle(indices)

        # split indices into minibatches
        minibatchlist = [np.array(sorted(indices[start_idx:start_idx + self.batchsize]))
                         for start_idx in range(0, num_samples - self.batchsize + 1, self.batchsize)]

        find_same_actions = lambda index, minibatch: \
            np.where(np.prod(actions[minibatch] == actions[minibatch[index]], axis=1))[0]
        same_actions = [
            np.array([[i, j] for i in range(self.batchsize) for j in find_same_actions(i, minibatch) if j > i],
                     dtype='int64') for minibatch in minibatchlist]

        # check with samples should be dissimilar because they lead to different rewards aften the same actions
        find_dissimilar = lambda index, minibatch: \
            np.where(np.prod(actions[minibatch] == actions[minibatch[index]], axis=1) *
                     (rewards[minibatch + 1] != rewards[minibatch[index] + 1]))[0]
        dissimilar = [np.array([[i, j] for i in range(self.batchsize) for j in find_dissimilar(i, minibatch) if j > i],
                               dtype='int64') for minibatch in minibatchlist]

        # TRAINING -----------------------------------------------------------------------------------------------------
        criterion = RoboticPriorsLoss(self.model, self.l1_reg)

        self.model.train()
        start_time = time.time()
        for epoch in range(N_EPOCHS):
            # In each epoch, we do a full pass over the training data:
            epoch_loss, epoch_batches = 0, 0
            enumerated_minibatches = list(enumerate(minibatchlist))
            np.random.shuffle(enumerated_minibatches)
            for i, batch in enumerated_minibatches:
                diss = dissimilar[i][np.random.permutation(dissimilar[i].shape[0])]  # [:10*self.batchsize]
                same = same_actions[i][np.random.permutation(same_actions[i].shape[0])]  # [:10*self.batchsize]
                diss, same = th.from_numpy(diss), th.from_numpy(same)
                obs = Variable(th.from_numpy(observations[batch]))
                next_obs = Variable(th.from_numpy(observations[batch + 1]))
                if self.cuda:
                    obs, next_obs = obs.cuda(), next_obs.cuda()
                    same, diss = same.cuda(), diss.cuda()

                states, next_states = self.model(obs), self.model(next_obs)
                self.optimizer.zero_grad()
                loss = criterion(states, next_states, diss, same)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.data[0]
                epoch_batches += 1

            # Then we print the results for this epoch:
            if (epoch + 1) % 5 == 0:
                print("Epoch {:3}/{}, loss:{:.4f}".format(epoch + 1, N_EPOCHS, epoch_loss / epoch_batches))
                print("{:.2f}s/epoch".format((time.time() - start_time) / (epoch + 1)))

                # Optionally plot the current state space
                plot_representation(self._predFn(obs_var), rewards, add_colorbar=epoch == 0,
                                    name="Learned State Representation (Training Data)")

        plt.close("Learned State Representation (Training Data)")

        # return predicted states for training observations
        return self._predFn(obs_var)

    def predStates(self, observations):
        observations = observations.astype(np.float32)
        obs_var = Variable(th.from_numpy(observations), volatile=True)
        if self.cuda:
            obs_var = obs_var.cuda()
        states = self._predFn(obs_var, restore_train=False)
        return states


def plot_representation(states, rewards, name="Learned State Representation", add_colorbar=True):
    plt.ion()
    plt.figure(name)
    plt.clf()
    plt.scatter(states[:, 0], states[:, 1], s=7, c=np.clip(rewards, -1, 1), cmap='bwr', linewidths=0.1)
    plt.xlim([-2, 2])
    plt.xlabel('State dimension 1')
    plt.ylim([-2, 2])
    plt.ylabel('State dimension 2')
    if add_colorbar:
        plt.colorbar(label='Reward')
    plt.pause(0.0001)


def plot_observations(observations, name='Observation Samples'):
    plt.ion()
    plt.figure(name)
    m, n = 8, 10
    for i in range(m * n):
        plt.subplot(m, n, i + 1)
        plt.imshow(observations[i].reshape(16, 16, 3), interpolation='nearest')
        plt.gca().invert_yaxis()
        plt.xticks([])
        plt.yticks([])
    plt.pause(0.0001)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch SRL with robotic priors')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--path', type=str, default="", help='Path to npz folder')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and th.cuda.is_available()
    N_EPOCHS = args.epochs

    print('\nExperiment: {}\n'.format(args.path))

    print('Loading data ... ')
    training_data = np.load(args.path)

    observations, actions = training_data['observations'], training_data['actions']
    rewards, episode_starts = training_data['rewards'], training_data['episode_starts']
    # (batchsize, width, height, n_channels) -> (batchsize, n_channels, height, width)
    observations = np.transpose(observations, (0, 3, 2, 1))

    print('Learning a state representation ... ')
    srl = SRL4robotics(2, args.seed, learning_rate=0.001, l1_reg=0.001, cuda=args.cuda)
    training_states = srl.learn(observations, actions, rewards, episode_starts)
    plot_representation(training_states, training_data['rewards'], name='Training Data', add_colorbar=True)

    input('\nPress any key to exit.')
