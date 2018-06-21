from __future__ import print_function, division, absolute_import

from .models import *


class DenseNetwork(BaseModelSRL):
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


class ConvolutionalNetwork(BaseModelSRL):
    """
    Convolutional Neural Network using ResNet
    input shape : 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224
    :param state_dim: (int)
    :param cuda: (bool)
    """

    def __init__(self, state_dim=2, cuda=False):
        super(ConvolutionalNetwork, self).__init__()
        self.resnet = models.resnet18(pretrained=False)
        # Freeze params
        # for param in self.resnet.parameters():
        #     param.requires_grad = False
        # Replace the last fully-connected layer
        n_units = self.resnet.fc.in_features
        print("{} units in the last layer".format(n_units))
        self.resnet.fc = nn.Linear(n_units, state_dim)
        self.device = th.device("cuda" if th.cuda.is_available() and cuda else "cpu")
        self.resnet = self.resnet.to(self.device)

    def forward(self, x):
        x = self.resnet(x)
        return x
