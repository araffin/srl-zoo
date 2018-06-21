from __future__ import print_function, division, absolute_import

import torch as th
import torch.nn as nn


class GaussianNoise(nn.Module):
    """
    Gaussian Noise layer
    :param batch_size: (int)
    :param input_dim: (int)
    :param std: (float) standard deviation
    :param mean: (float)
    :param device: (pytorch device)
    """

    def __init__(self, batch_size, input_dim, device, std, mean=0):
        super(GaussianNoise, self).__init__()
        self.std = std
        self.mean = mean
        self.device = device
        self.noise = th.zeros(batch_size, input_dim, device=self.device)

    def forward(self, x):
        if self.training:
            self.noise.data.normal_(self.mean, std=self.std)
            return x + self.noise
        return x


class GaussianNoiseVariant(nn.Module):
    """
    Variant of the Gaussian Noise layer that does not require fixed batch_size
    It recreates a tensor at each call
    :param device: (pytorch device)
    :param std: (float) standard deviation
    :param mean: (float)
    """

    def __init__(self, device, std, mean=0):
        super(GaussianNoiseVariant, self).__init__()
        self.std = std
        self.mean = mean
        self.device = device

    def forward(self, x):
        if self.training:
            noise = th.zeros(x.size(), device=self.device)
            noise.data.normal_(self.mean, std=self.std)
            return x + noise
        return x
