from __future__ import print_function, division, absolute_import

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .custom_layers import GaussianNoiseVariant

try:
    from preprocessing.preprocess import getNChannels
except ImportError:
    from ..preprocessing.preprocess import getNChannels


class BaseModelSRL(nn.Module):
    """
    Base Class for a SRL network
    It implements a getState method to retrieve a state from observations
    """

    def __init__(self):
        super(BaseModelSRL, self).__init__()

    def getStates(self, observations):
        """
        :param observations: (th.Tensor)
        :return: (th.Tensor)
        """
        return self.forward(observations)

    def forward(self, x):
        raise NotImplementedError


class BaseModelAutoEncoder(BaseModelSRL):
    """
    Base Class for a SRL network (autoencoder family)
    It implements a getState method to retrieve a state from observations
    """

    def __init__(self):
        super(BaseModelAutoEncoder, self).__init__()

        # Inspired by ResNet:
        # conv3x3 followed by BatchNorm2d
        self.encoder_conv = nn.Sequential(
            # 224x224xN_CHANNELS -> 112x112x64
            nn.Conv2d(getNChannels(), 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 56x56x64

            conv3x3(in_planes=64, out_planes=64, stride=1),  # 56x56x64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 27x27x64

            conv3x3(in_planes=64, out_planes=64, stride=2),  # 14x14x64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)  # 6x6x64
        )

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2),  # 13x13x64
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2),  # 27x27x64
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2),  # 55x55x64
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2),  # 111x111x64
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, getNChannels(), kernel_size=4, stride=2),  # 224x224xN_CHANNELS
        )

    def getStates(self, observations):
        """
        :param observations: (th.Tensor)
        :return: (th.Tensor)
        """
        return self.encode(observations)

    def encode(self, x):
        """
        :param x: (th.Tensor)
        :return: (th.Tensor)
        """
        raise NotImplementedError

    def decode(self, x):
        """
        :param x: (th.Tensor)
        :return: (th.Tensor)
        """
        raise NotImplementedError

    def forward(self, x):
        """
        :param x: (th.Tensor)
        :return: (th.Tensor)
        """
        input_shape = x.size()
        encoded = self.encode(x)
        decoded = self.decode(encoded).view(input_shape)
        return encoded, decoded


class BaseModelVAE(BaseModelAutoEncoder):
    """
    Base Class for a SRL network (VAE family)
    It implements a getState method to retrieve a state from observations
    """

    def __init__(self):
        super(BaseModelVAE, self).__init__()

    def getStates(self, observations):
        """
        :param observations: (th.Tensor)
        :return: (th.Tensor)
        """
        return self.encode(observations)[0]

    def encode(self, x):
        """
        :param x: (th.Tensor)
        :return: (th.Tensor)
        """
        raise NotImplementedError

    def decode(self, x):
        """
        :param x: (th.Tensor)
        :return: (th.Tensor)
        """
        raise NotImplementedError

    def reparameterize(self, mu, logvar):
        """
        Reparameterize for the backpropagation of z instead of q.
        (See "The reparameterization trick" section of https://arxiv.org/abs/1312.6114)
        :param mu: (th.Tensor)
        :param logvar: (th.Tensor)
        """
        if self.training:
            # logvar = \log(\sigma^2) = 2 * \log(\sigma)
            # \sigma = \exp(0.5 * logvar)
            std = logvar.mul(0.5).exp_()
            # Sample \epsilon from normal distribution
            # use std to create a new tensor, so we don't have to care
            # about running on GPU or not
            eps = std.new(std.size()).normal_()
            # Then multiply with the standard deviation and add the mean
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        """
        :param x: (th.Tensor)
        :return: (th.Tensor)
        """
        input_shape = x.size()
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        decoded = self.decode(z).view(input_shape)
        return decoded, mu, logvar


class CustomCNN(BaseModelSRL):
    """
    Convolutional Neural Network
    input shape : 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224
    :param state_dim: (int)
    """

    def __init__(self, state_dim=2):
        super(CustomCNN, self).__init__()
        # Inspired by ResNet:
        # conv3x3 followed by BatchNorm2d
        self.conv_layers = nn.Sequential(
            # 224x224x3 -> 112x112x64
            nn.Conv2d(getNChannels(), 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 56x56x64

            conv3x3(in_planes=64, out_planes=64, stride=1),  # 56x56x64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 27x27x64

            conv3x3(in_planes=64, out_planes=64, stride=2),  # 14x14x64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)  # 6x6x64
        )

        self.fc = nn.Linear(6 * 6 * 64, state_dim)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def conv3x3(in_planes, out_planes, stride=1):
    """"
    From PyTorch Resnet implementation
    3x3 convolution with padding
    :param in_planes: (int)
    :param out_planes: (int)
    :param stride: (int)
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def encodeOneHot(tensor, n_dim):
    """
    One hot encoding for a given tensor
    :param tensor: (th Tensor)
    :param n_dim: (int) Number of dimensions
    :return: (th.Tensor)
    """
    encoded_tensor = th.Tensor(tensor.shape[0], n_dim).zero_().to(tensor.device)
    return encoded_tensor.scatter_(1, tensor.data, 1.)
