"""
This is a PyTorch implementation of the method for state representation learning described in the paper "Learning State
Representations with Robotic Priors" (Jonschkowski & Brock, 2015).

This program is based on the original implementation by Rico Jonschkowski (rico.jonschkowski@tu-berlin.de):
https://github.com/tu-rbo/learning-state-representations-with-robotic-priors

TODO: Add cuda support
"""
from __future__ import print_function, division

import matplotlib.pyplot as plt
import numpy as np
import torch as th
import torch.nn as nn
from torch.autograd import Variable

# Python 2/3 compatibility
try:
    input = raw_input
except NameError:
    pass

N_EPOCHS = 50
BATCHSIZE = 256
NOISE_STD = 1e-6  # To avoid NaN (states must be different)

class SRLNetwork(nn.Module):
    """
    Neural Net for State Representation Learning (SRL)
    :param obs_dim: (int)
    :param state_dim: (int)
    :param batchsize: (int)
    """
    def __init__(self, obs_dim, state_dim=2, batchsize=256):
        super(SRLNetwork, self).__init__()
        self.l1 = nn.Linear(obs_dim, state_dim)
        self.noise = GaussianNoise(batchsize, state_dim, NOISE_STD)

    def forward(self, x):
        x = self.l1(x)
        x = self.noise(x)
        return x

class GaussianNoise(nn.Module):
    """
    Gaussian Noise layer
    :param batchsize: (int)
    :param input_dim: (int)
    :param std: (float) standard deviation
    :param mean: (float)
    """
    def __init__(self, batchsize, input_dim, std, mean=0):
        super(GaussianNoise, self).__init__()
        self.std = std
        self.mean = mean
        # TODO: handle GPU variable
        self.noise = Variable(th.zeros(batchsize, input_dim))

    def forward(self, x):
        if self.training:
            self.noise.data.normal_(self.mean, std=self.std)
            return x + self.noise
        return x


class RoboticPriorsLoss(nn.Module):
    def __init__(self, model, l1_reg=0):
        super(RoboticPriorsLoss, self).__init__()
        # Retrieve only regularizable parameters (we should exclude biases)
        self.reg_params = [param for name, param in model.named_parameters() if ".bias" not in name]
        n_params = sum([reduce(lambda x,y: x*y, param.size()) for param in self.reg_params])
        self.l1_coeff = (l1_reg / n_params)


    def forward(self, states, next_states, dissimilar_pairs, same_actions_pairs):
        """
        WARNING: l1 regularization is missing
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
    def __init__(self, obs_dim, state_dim, seed=1, learning_rate=0.001, l1_reg=0.001):

        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.batchsize = BATCHSIZE

        np.random.seed(seed)
        th.manual_seed(seed)

        # init values
        self.mean_obs = np.zeros(self.obs_dim)
        self.std_obs = 1

        self.model = SRLNetwork(self.obs_dim, self.state_dim, self.batchsize)
        self.optimizer = th.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.l1_reg = l1_reg

    def _predFn(self, observations, restore_train=True):
        # test mode
        self.model.eval()
        states = self.model(observations)
        if restore_train:
            self.model.train()
        return states.data.numpy()

    def learn(self, observations, actions, rewards, episode_starts):

        # PREPARE DATA -------------------------------------------------------------------------------------------------
        # here, we normalize the observations, organize the data into minibatches
        # and find pairs for the respective loss terms
        observations = observations.astype(np.float32)

        self.mean_obs = np.mean(observations, axis=0, keepdims=True)
        self.std_obs = np.std(observations, ddof=1)
        observations = (observations - self.mean_obs) / self.std_obs
        # For testing
        obs_var = Variable(th.from_numpy(observations), volatile=True)

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

                # Optionally plot the current state space
                plot_representation(self._predFn(obs_var), rewards, add_colorbar=epoch==0,
                                         name="Learned State Representation (Training Data)")

        plt.close("Learned State Representation (Training Data)")

        # return predicted states for training observations
        return self._predFn(obs_var)

    def predStates(self, observations):
        observations = (observations - self.mean_obs) / self.std_obs
        observations = observations.astype(np.float32)
        obs_var = Variable(th.from_numpy(observations), volatile=True)
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
    # print('\nSIMPLE NAVIGATION TASK\n')
    #
    # print('Loading and displaying training data ... ')
    # training_data = np.load('simple_navigation_task_train.npz')
    #
    # print('Learning a state representation ... ')
    # srl = SRL4robotics(16 * 16 * 3, 2, learning_rate=0.0001, l1_reg=0.3)
    # training_states = srl.learn(**training_data)
    # plot_representation(training_states, training_data['rewards'],
    #                         name='Observation-State-Mapping Applied to Training Data -- Simple Navigation Task',
    #                         add_colorbar=True)

    ####################################################################################################################

    print('\nSLOT CAR TASK\n')

    print('Loading and displaying training data ... ')
    training_data = np.load('slot_car_task_train.npz')

    print('Learning a state representation ... ')
    srl = SRL4robotics(16 * 16 * 3, 2, learning_rate=0.001, l1_reg=0.001)
    training_states = srl.learn(**training_data)
    plot_representation(training_states, training_data['rewards'],
                        name='Observation-State-Mapping Applied to Training Data -- Slot Car Task',
                        add_colorbar=True)

    input('\nPress any key to exit.')
