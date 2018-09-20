from __future__ import print_function, division, absolute_import

from .models import *


class DenseVAE(BaseModelVAE):
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

    def encode(self, x):
        # Flatten input
        x = x.view(x.size(0), -1)
        x = self.relu(self.encoder_fc1(x))
        return self.encoder_fc21(x), self.encoder_fc22(x)

    def decode(self, z):
        return self.decoder(z)


class CNNVAE(BaseModelVAE):
    """
    Custom convolutional VAE network
    Input dim (same as ResNet): 3x224x224
    :param state_dim: (int)
    """

    def __init__(self, state_dim=3):
        super(CNNVAE, self).__init__()
        self.encoder_fc1 = nn.Linear(6 * 6 * 64, state_dim)
        self.encoder_fc2 = nn.Linear(6 * 6 * 64, state_dim)

        self.decoder_fc = nn.Sequential(
            nn.Linear(state_dim, 6 * 6 * 64)
        )

    def encode(self, x):
        """
        :param x: (th.Tensor)
        :return: (th.Tensor)
        """
        x = self.encoder_conv(x)
        x = x.view(x.size(0), -1)
        return self.encoder_fc1(x), self.encoder_fc2(x)

    def decode(self, z):
        """
        :param z: (th.Tensor)
        :return: (th.Tensor)
        """
        z = self.decoder_fc(z)
        z = z.view(z.size(0), 64, 6, 6)
        return self.decoder_conv(z)
