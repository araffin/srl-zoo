from __future__ import print_function, division, absolute_import

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable

from .custom_layers import GaussianNoiseVariant
try:
    from preprocessing.preprocess import N_CHANNELS
except ImportError:
    from ..preprocessing.preprocess import N_CHANNELS


class SRLConvolutionalNetwork(nn.Module):
    """
    Convolutional Neural Net for State Representation Learning (SRL)
    input shape : 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224
    :param state_dim: (int)
    :param cuda: (bool)
    :param noise_std: (float)  To avoid NaN (states must be different)
    """

    def __init__(self, state_dim=2, cuda=False, noise_std=1e-6):
        super(SRLConvolutionalNetwork, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        # TODO: add squeezeNet support
        # self.squeezeNet = models.squeezenet1_0(pretrained=True)
        # TODO: freeze less layers
        # Freeze params
        for param in self.resnet.parameters():
            param.requires_grad = False
        # Replace the last fully-connected layer
        n_units = self.resnet.fc.in_features
        print("{} units in the last layer".format(n_units))
        self.resnet.fc = nn.Linear(n_units, state_dim)
        if cuda:
            self.resnet.cuda()
        # This variant does not require the batch_size
        self.noise = GaussianNoiseVariant(noise_std, cuda=cuda)
        # self.noise = GaussianNoise(batch_size, state_dim, noise_std, cuda=cuda)

    def forward(self, x):
        x = self.resnet(x)
        x = self.noise(x)
        return x


class SRLDenseNetwork(nn.Module):
    """
    Dense Neural Net for State Representation Learning (SRL)
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
        self.noise = GaussianNoiseVariant(noise_std, cuda=cuda)
        # self.noise = GaussianNoise(batch_size, state_dim, noise_std, cuda=cuda)

    def forward(self, x):
        # Flatten input
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.noise(x)
        return x


class DenseNetwork(nn.Module):
    """
    Dense Neural Net for State Representation Learning (SRL)
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
    Convolutional Neural Network using pretrained ResNet
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


class EmbeddingNet(nn.Module):
    def __init__(self, state_dim=2, embedding_size=128):
        """
        Resnet18 + FC layer (Embedding to learn a metric)
        input shape : 2 X 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224
        :param state_dim: (int)
        :embedding_size: (int) size of TCN embedding
        """
        super(EmbeddingNet, self).__init__()
        # ResNet 18
        self.conv_layers = models.resnet18(pretrained=True)
        # Freeze params
        for param in self.conv_layers.parameters():
            param.requires_grad = False
        # Replace the last fully-connected layer
        n_units = self.conv_layers.fc.in_features
        print("{} units in the last layer".format(n_units))
        self.conv_layers.fc = nn.Linear(n_units, embedding_size)
        self.fc = nn.Sequential(nn.PReLU(),
                                nn.Linear(embedding_size, state_dim)
                                )        

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class TripletNet(nn.Module):
    def __init__(self, state_dim=2):
        super(TripletNet, self).__init__()
        self.embedding = EmbeddingNet(state_dim)

    def forward(self, anchor, positive, negative):
        """
        anchor : observation
        positive : observation
        negative : observation
        """
        return self.embedding(anchor), self.embedding(positive), self.embedding(negative)

    def encode(self, x):
        return self.embedding(x)


class CustomCNN(nn.Module):
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

        self.fc = nn.Linear(6 * 6 * 64, state_dim)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class SRLCustomCNN(nn.Module):
    """
    Convolutional Neural Network for State Representation Learning
    input shape : 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224
    :param state_dim: (int)
    :param cuda: (bool)
    :param noise_std: (float)  To avoid NaN (states must be different)
    """

    def __init__(self, state_dim=2, cuda=False, noise_std=1e-6):
        super(SRLCustomCNN, self).__init__()
        self.cnn = CustomCNN(state_dim)
        if cuda:
            self.cnn.cuda()
        self.noise = GaussianNoiseVariant(noise_std, cuda=cuda)

    def forward(self, x):
        x = self.cnn(x)
        return self.noise(x)


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

    def forward(self, x):
        input_shape = x.size()
        # Flatten input
        x = x.view(x.size(0), -1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        # Transform to a 4D tensor
        decoded = decoded.view(input_shape)
        return encoded, decoded


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

    def encode(self, x):
        x = self.encoder_conv(x)
        x = x.view(x.size(0), -1)
        return self.encoder_fc1(x), self.encoder_fc2(x)

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
        decoded = self.decoder_fc(z)
        decoded = decoded.view(x.size(0), 64, 6, 6)
        decoded = self.decoder_conv(decoded)
        return decoded, mu, logvar
