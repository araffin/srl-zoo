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

    def getState(self, x):
        """
        :param x: (PyTorch Variable)
        :return: (PyTorch Variable)
        """
        return self.forward(x)

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
