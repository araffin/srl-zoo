from __future__ import print_function, division, absolute_import

from .models import *


class LinearAutoEncoder(nn.Module):
    """
    :param input_dim: (int)
    :param state_dim: (int)
    """

    def __init__(self, input_dim, state_dim=3):
        super(LinearAutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, state_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(state_dim, input_dim),
        )

    def getState(self, x):
        """
        :param x: (PyTorch Variable)
        :return: (PyTorch Variable)
        """
        return self.encode(x)

    def encode(self, x):
        """
        :param x: (PyTorch Variable)
        :return: (PyTorch Variable)
        """
        # Flatten input
        x = x.view(x.size(0), -1)
        return self.encoder(x)

    def decode(self, x):
        """
        :param x: (PyTorch Variable)
        :return: (PyTorch Variable)
        """
        decoded = self.decode(x)
        # Transform to a 4D tensor
        return decoded.view(x.size(0), 3, 224, 224)

    def forward(self, x):
        """
        :param x: (PyTorch Variable)
        :return: (PyTorch Variable, PyTorch Variable)
        """
        encoded = self.encode(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class DenseAutoEncoder(nn.Module):
    """
    Dense autoencoder network
    Known issue: it reconstructs the image but omits the robot arm
    :param input_dim: (int)
    :param state_dim: (int)
    """

    def __init__(self, input_dim, state_dim=3):
        super(DenseAutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, state_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(state_dim, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, input_dim),
        )

    def encode(self, x):
        """
        :param x: (PyTorch Variable)
        :return: (PyTorch Variable)
        """
        # Flatten input
        x = x.view(x.size(0), -1)
        return self.encoder(x)

    def decode(self, x):
        """
        :param x: (PyTorch Variable)
        :return: (PyTorch Variable)
        """
        decoded = self.decode(x)
        # Transform to a 4D tensor
        return decoded.view(x.size(0), 3, 224, 224)

    def forward(self, x):
        """
        :param x: (PyTorch Variable)
        :return: (PyTorch Variable, PyTorch Variable)
        """
        encoded = self.encode(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class CNNAutoEncoder(nn.Module):
    """
    Custom convolutional autoencoder network
    Input dim (same as ResNet): 3x224x224
    :param state_dim: (int)
    """

    def __init__(self, state_dim=3):
        super(CNNAutoEncoder, self).__init__()
        # Inspired by ResNet:
        # conv3x3 followed by BatchNorm2d
        # TODO: implement residual connection
        self.encoder_conv = nn.Sequential(
            # 224x224x3 -> 112x112x64
            nn.Conv2d(N_CHANNELS, 64, kernel_size=7, stride=2, padding=3, bias=False),
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

        self.encoder_fc = nn.Sequential(
            nn.Linear(6 * 6 * 64, state_dim)
        )

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
        return self.encode(x)

    def encode(self, x):
        """
        :param x: (PyTorch Variable)
        :return: (PyTorch Variable)
        """
        encoded = self.encoder_conv(x)
        encoded = encoded.view(encoded.size(0), -1)
        return self.encoder_fc(encoded)

    def decode(self, x):
        """
        :param x: (PyTorch Variable)
        :return: (PyTorch Variable)
        """
        decoded = self.decoder_fc(x)
        decoded = decoded.view(x.size(0), 64, 6, 6)
        return self.decoder_conv(decoded)

    def forward(self, x):
        """
        :param x: (PyTorch Variable)
        :return: (PyTorch Variable)
        """
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return encoded, decoded
