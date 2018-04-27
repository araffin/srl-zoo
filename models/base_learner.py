from __future__ import print_function, division, absolute_import

import json

import numpy as np
import torch as th

MAX_BATCH_SIZE_GPU = 512  # For plotting, max batch_size before having memory issues


def observationsGenerator(observations, device, batch_size=64):
    """
    Python generator to avoid out of memory issues
    when predicting states for all the observations
    :param observations: (torch tensor)
    :param batch_size: (int)
    :param device: (pytorch device)
    """
    n_minibatches = len(observations) // batch_size + 1
    for i in range(n_minibatches):
        start_idx, end_idx = batch_size * i, batch_size * (i + 1)
        obs_var = observations[start_idx:end_idx].set_grad_enabled(False)
        obs_var = obs_var.to(device)
        yield obs_var


class BaseLearner(object):
    """
    Base class for a method that learn a state representation
    from observations
    :param state_dim: (int)
    :param batch_size: (int)
    :param seed: (int)
    :param cuda: (bool)
    """

    def __init__(self, state_dim, batch_size, seed=1, cuda=False):
        super(BaseLearner, self).__init__()
        self.state_dim = state_dim
        self.batch_size = batch_size
        self.cuda = cuda
        self.model = None
        self.seed = seed
        # Seed the random generator
        np.random.seed(seed)
        th.manual_seed(seed)
        if cuda:
            th.cuda.manual_seed(seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")

    def _predFn(self, observations, restore_train=True):
        """
        Predict states in test mode given observations
        :param observations: (PyTorch Variable)
        :param restore_train: (bool) whether to restore training mode after prediction
        :return: (numpy tensor)
        """
        # Switch to test mode
        self.model.eval()
        states = self.model.getStates(observations)
        if restore_train:
            # Restore training mode
            self.model.train()
        # Move the tensor back to the cpu
        return states.detach().numpy()

    def predStates(self, observations):
        """
        Predict states for given observations
        WARNING: you should use _batchPredStates
        if observations tensor is large to avoid memory issues
        :param observations: (numpy tensor)
        :return: (numpy tensor)
        """
        observations = observations.astype(np.float32)
        obs_var = th.from_numpy(observations).set_grad_enabled(False)
        obs_var = obs_var.to(self.device)
        states = self._predFn(obs_var, restore_train=False)
        return states

    def _batchPredStates(self, observations):
        """
        Predict states using minibatches to avoid memory issues
        :param observations: (numpy tensor)
        :return: (numpy tensor)
        """
        predictions = []
        for obs_var in observationsGenerator(th.from_numpy(observations), self.device, MAX_BATCH_SIZE_GPU):
            predictions.append(self._predFn(obs_var))
        return np.concatenate(predictions, axis=0)

    def predStatesWithDataLoader(self, data_loader, restore_train=False):
        """
        Predict states using minibatches to avoid memory issues
        :param data_loader: (Baxter Data Loader object)
        :param restore_train: (bool) restore train mode (model + dataLoader) after predicting states
        :return: (numpy tensor)
        """
        # Switch to test mode and reset the iterator
        data_loader.testMode()
        predictions = []
        for obs_var in data_loader:
            obs_var = obs_var.to(self.device)
            predictions.append(self._predFn(obs_var, restore_train))
        # Switch back to train mode
        if restore_train:
            data_loader.trainMode()
        return np.concatenate(predictions, axis=0)

    def learn(self, *args, **kwargs):
        """
        Function called to learn a state representation
        it returns the learned states for the given observations
        """
        raise NotImplementedError("Learn method not implemented")

    @staticmethod
    def saveStates(states, images_path, rewards, log_folder, name=""):
        """
        Save learned states to json and npz files
        :param states: (numpy array)
        :param images_path: ([str])
        :param rewards: (rewards)
        :param log_folder: (str)
        :param name: (str)
        """
        print("Saving image path to state representation")
        image_to_state = {path: list(map(str, state)) for path, state in zip(images_path, states)}
        with open("{}/image_to_state{}.json".format(log_folder, name), 'w') as f:
            json.dump(image_to_state, f, sort_keys=True)
        print("Saving states and rewards")
        states_rewards = {'states': states, 'rewards': rewards}
        np.savez('{}/states_rewards{}.npz'.format(log_folder, name), **states_rewards)
