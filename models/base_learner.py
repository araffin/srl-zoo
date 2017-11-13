from __future__ import print_function, division, absolute_import

import numpy as np
import torch as th
from torch.autograd import Variable

MAX_BACTHSIZE_GPU = 512  # For plotting, max batch_size before having memory issues


def observationsGenerator(observations, batch_size=64, cuda=False):
    """
    Python generator to avoid out of memory issues
    when predicting states for all the observations
    :param observations: (torch tensor)
    :param batch_size: (int)
    :param cuda: (bool)
    """
    n_minibatches = len(observations) // batch_size + 1
    for i in range(n_minibatches):
        start_idx, end_idx = batch_size * i, batch_size * (i + 1)
        obs_var = Variable(observations[start_idx:end_idx], volatile=True)
        if cuda:
            obs_var = obs_var.cuda()
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

    def _predFn(self, observations, restore_train=True):
        """
        Predict states in test mode given observations
        :param observations: (PyTorch Variable)
        :param restore_train: (bool) whether to restore training mode after prediction
        :return: (numpy tensor)
        """
        # Switch to test mode
        self.model.eval()
        states = self.model(observations)
        if restore_train:
            # Restore training mode
            self.model.train()
        if self.cuda:
            # Move the tensor back to the cpu
            return states.data.cpu().numpy()
        return states.data.numpy()

    def predStates(self, observations):
        """
        Predict states for given observations
        WARNING: you should use _batchPredStates
        if observations tensor is large to avoid memory issues
        :param observations: (numpy tensor)
        :return: (numpy tensor)
        """
        observations = observations.astype(np.float32)
        obs_var = Variable(th.from_numpy(observations), volatile=True)
        if self.cuda:
            obs_var = obs_var.cuda()
        states = self._predFn(obs_var, restore_train=False)
        return states

    def _batchPredStates(self, observations):
        """
        Predict states using minibatches to avoid memory issues
        :param observations: (numpy tensor)
        :return: (numpy tensor)
        """
        predictions = []
        for obs_var in observationsGenerator(th.from_numpy(observations), MAX_BACTHSIZE_GPU, cuda=self.cuda):
            predictions.append(self._predFn(obs_var))
        return np.concatenate(predictions, axis=0)

    def learn(self, *args, **kwargs):
        """
        Function called to learn a state representation
        it returns the learned states for the given observations
        """
        raise NotImplementedError("Learn method not implemented")
