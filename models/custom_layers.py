from __future__ import print_function, division, absolute_import

import torch as th
import torch.nn as nn
from torch.autograd import Variable


class GaussianNoise(nn.Module):
    """
    Gaussian Noise layer
    :param batch_size: (int)
    :param input_dim: (int)
    :param std: (float) standard deviation
    :param mean: (float)
    :param cuda: (bool)
    """

    def __init__(self, batch_size, input_dim, std, mean=0, cuda=False):
        super(GaussianNoise, self).__init__()
        self.std = std
        self.mean = mean
        self.noise = Variable(th.zeros(batch_size, input_dim))
        if cuda:
            self.noise = self.noise.cuda()

    def forward(self, x):
        if self.training:
            self.noise.data.normal_(self.mean, std=self.std)
            return x + self.noise
        return x
