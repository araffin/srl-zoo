from __future__ import print_function, division, absolute_import

from .models import *


class DenseVAE(nn.Module):
    """
    Dense VAE network
    :param input_dim: (int)
    :param state_dim: (int)
    """

    def __init__(self, input_dim, state_dim=3):
        super(DenseVAE, self).__init__()

        self.input_dim = input_dim

        self.encoder_fc1 = nn.Linear(input_dim, 50)
        self.encoder_fc21 = nn.Linear(50, state_dim)
        self.encoder_fc22 = nn.Linear(50, state_dim)

        self.decoder = nn.Sequential(
            nn.Linear(state_dim, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, input_dim),
        )

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def getState(self, x):
        """
        :param x: (PyTorch Variable)
        :return: (PyTorch Variable)
        """
        return self.encode(x)[0]

    def encode(self, x):
        # Flatten input
        x = x.view(x.size(0), -1)
        x = self.relu(self.encoder_fc1(x))
        return self.encoder_fc21(x), self.encoder_fc22(x)

    def reparameterize(self, mu, logvar):
        """
        Reparameterize for the backpropagation of z instead of q.
        (See "The reparameterization trick" section of https://arxiv.org/abs/1312.6114)
        :param mu: (Pytorch Variable)
        :param logvar: (Pytorch Variable)
        """
        if self.training:
            # logvar = \log(\sigma^2) = 2 * \log(\sigma)
            # \sigma = \exp(0.5 * logvar)
            std = logvar.mul(0.5).exp_()
            # Sample \epsilon from normal distribution
            # use std to create a new variable, so we don't have to care
            # about running on GPU or not
            eps = Variable(std.data.new(std.size()).normal_())
            # Then multiply with the standard deviation and add the mean
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        input_shape = x.size()
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z).view(input_shape), mu, logvar


class CNNVAE(nn.Module):
    """
    Custom convolutional VAE network
    Input dim (same as ResNet): 3x224x224
    :param state_dim: (int)
    """

    def __init__(self, state_dim=3):
        super(CNNVAE, self).__init__()
        # Inspired by ResNet:
        # conv3x3 followed by BatchNorm2d
        # TODO: implement residual connection
        self.encoder_conv = nn.Sequential(
            # 224x224x3 -> 112x112x64
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
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

        self.encoder_fc1 = nn.Linear(6 * 6 * 64, state_dim)
        self.encoder_fc2 = nn.Linear(6 * 6 * 64, state_dim)

        self.decoder_fc = nn.Sequential(
            nn.Linear(state_dim, 6 * 6 * 64)
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

            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2),  # 224x224x3
        )

    def getState(self, x):
        """
        :param x: (PyTorch Variable)
        :return: (PyTorch Variable)
        """
        return self.encode(x)[0]

    def encode(self, x):
        x = self.encoder_conv(x)
        x = x.view(x.size(0), -1)
        return self.encoder_fc1(x), self.encoder_fc2(x)

    def decode(self, z):
        z = self.decoder_fc(z)
        z = z.view(z.size(0), 64, 6, 6)
        return self.decoder_conv(z)

    def reparameterize(self, mu, logvar):
        """
        Reparameterize for the backpropagation of z instead of q.
        (See "The reparameterization trick" section of https://arxiv.org/abs/1312.6114)
        :param mu: (Pytorch Variable)
        :param logvar: (Pytorch Variable)
        """
        if self.training:
            # logvar = \log(\sigma^2) = 2 * \log(\sigma)
            # \sigma = \exp(0.5 * logvar)
            std = logvar.mul(0.5).exp_()
            # Sample \epsilon from normal distribution
            eps = Variable(std.data.new(std.size()).normal_())
            # Then multiply with the standard deviation and add the mean
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        decoded = self.decode(z)
        return decoded, mu, logvar
