from __future__ import print_function, division, absolute_import

import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .custom_layers import GaussianNoise, GaussianNoiseVariant


class SRLConvolutionalNetwork(nn.Module):
    """
    Convolutional Neural Net for State Representation Learning (SRL)
    input shape : 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224
    :param state_dim: (int)
    :param batch_size: (int)
    :param cuda: (bool)
    :param noise_std: (float)  To avoid NaN (states must be different)
    """

    def __init__(self, state_dim=2, batch_size=256, cuda=False, noise_std=1e-6):
        super(SRLConvolutionalNetwork, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        # TODO: add squeezeNet support
        # self.squeezeNet = models.squeezenet1_0(pretrained=True)
        # Freeze params
        for param in self.resnet.parameters():
            param.requires_grad = False
        # Replace the last fully-connected layer
        n_units = self.resnet.fc.in_features
        print("{} units in the last layer".format(n_units))
        self.resnet.fc = nn.Linear(n_units, state_dim)
        if cuda:
            self.resnet.cuda()
        self.noise = GaussianNoise(batch_size, state_dim, noise_std, cuda=cuda)

    def forward(self, x):
        x = self.resnet(x)
        x = self.noise(x)
        return x


class SRLDenseNetwork(nn.Module):
    """
    Feedforward Neural Net for State Representation Learning (SRL)
    input shape : 3-channel RGB images of shape (3 x H x W) (to be consistent with CNN network)
    :param input_dim: (int) 3 x H x H
    :param state_dim: (int)
    :param noise_std: (float)  To avoid NaN (states must be different)
    :param batch_size: (int)
    :param cuda: (bool)
    :param n_hidden: (int)
    """

    def __init__(self, input_dim, state_dim=2, batch_size=256,
                 cuda=False, n_hidden=32, noise_std=1e-6):
        super(SRLDenseNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, n_hidden)
        self.fc2 = nn.Linear(n_hidden, state_dim)
        self.noise = GaussianNoise(batch_size, state_dim, noise_std, cuda=cuda)

    def forward(self, x):
        # Flatten input
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.noise(x)
        return x


class DenseNetwork(nn.Module):
    """
    Feedforward Neural Net for State Representation Learning (SRL)
    input shape : 3-channel RGB images of shape (3 x H x W) (to be consistent with CNN network)
    :param input_dim: (int) 3 x H x H
    :param state_dim: (int)
    :param n_hidden: (int)
    :param drop_p: (float) Dropout proba
    """

    def __init__(self, input_dim, state_dim=2, n_hidden=64, drop_p=0.5):
        super(DenseNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, n_hidden)
        self.fc2 = nn.Linear(n_hidden, state_dim)
        self.drop_p = drop_p

    def forward(self, x):
        # Flatten input
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc2(x)
        return x


class ConvolutionalNetwork(nn.Module):
    """
    Convolutional Neural Net for State Representation Learning (SRL)
    input shape : 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224
    :param state_dim: (int)
    :param cuda: (bool)
    """

    def __init__(self, state_dim=2, cuda=False):
        super(ConvolutionalNetwork, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        # Freeze params
        for param in self.resnet.parameters():
            param.requires_grad = False
        # Replace the last fully-connected layer
        n_units = self.resnet.fc.in_features
        print("{} units in the last layer".format(n_units))
        self.resnet.fc = nn.Linear(n_units, state_dim)
        if cuda:
            self.resnet.cuda()

    def forward(self, x):
        x = self.resnet(x)
        return x


class LinearAutoEncoder(nn.Module):
    def __init__(self, input_dim, state_dim=3, noise_std=0.0, cuda=False):
        super(LinearAutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            GaussianNoiseVariant(noise_std, cuda=cuda),
            nn.Linear(input_dim, state_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(state_dim, input_dim),
        )

    def forward(self, x):
        input_shape = x.size()
        # Flatten input
        x = x.view(x.size(0), -1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        # Transform to a 4D tensor
        decoded = decoded.view(input_shape)
        return encoded, decoded


class DenseAutoEncoder(nn.Module):
    def __init__(self, input_dim, state_dim=3, noise_std=0.0, cuda=False):
        super(DenseAutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            GaussianNoiseVariant(noise_std, cuda=cuda),
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 12),
            nn.Tanh(),
            nn.Linear(12, state_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(state_dim, 12),
            nn.Tanh(),
            nn.Linear(12, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, input_dim),
        )

    def forward(self, x):
        input_shape = x.size()
        # Flatten input
        x = x.view(x.size(0), -1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        # Transform to a 4D tensor
        decoded = decoded.view(input_shape)
        return encoded, decoded
