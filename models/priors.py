from __future__ import print_function, division, absolute_import

from torch.autograd import Function

from .models import *
import torch.nn.functional as F


class SRLConvolutionalNetwork(BaseModelSRL):
    """
    Convolutional Neural Net for State Representation Learning (SRL)
    input shape : 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224
    :param state_dim: (int)
    :param cuda: (bool)
    :param noise_std: (float)  To avoid NaN (states must be different)
    """

    def __init__(self, state_dim=2, cuda=False, noise_std=1e-6):
        super(SRLConvolutionalNetwork, self).__init__()
        self.resnet = models.resnet18(pretrained=False)

        # Replace the last fully-connected layer
        n_units = self.resnet.fc.in_features
        print("{} units in the last layer".format(n_units))

        self.resnet.fc = nn.Sequential(
            nn.Linear(n_units, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, state_dim),
        )
        if cuda:
            self.resnet.cuda()
        # This variant does not require the batch_size
        self.noise = GaussianNoiseVariant(noise_std, cuda=cuda)
        # self.noise = GaussianNoise(batch_size, state_dim, noise_std, cuda=cuda)

    def forward(self, x):
        x = self.resnet(x)
        if self.training:
            x = self.noise(x)
        return x


class SRLCustomCNN(BaseModelSRL):
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
        if self.training:
            x = self.noise(x)
        return x

class SRLDenseNetwork(BaseModelSRL):
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

    def __init__(self, input_dim, state_dim=2, cuda=False,
                 n_hidden=64, noise_std=1e-6):
        super(SRLDenseNetwork, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_dim, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, state_dim)
            )
        
        self.noise = GaussianNoiseVariant(noise_std, cuda=cuda)
        if cuda:
            self.fc.cuda()
            
    def forward(self, x):
        # Flatten input
        x = x.view(x.size(0), -1)
        x = self.fc(x) 
        if self.training:
            x = self.noise(x)
        return x


class SRLLinear(BaseModelSRL):
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

    def __init__(self, input_dim, state_dim=2, cuda=False, noise_std=1e-6):
        super(SRLLinear, self).__init__()

        self.fc = nn.Linear(input_dim, state_dim)
        self.noise = GaussianNoiseVariant(noise_std, cuda=cuda)
        if cuda:
            self.fc.cuda()
            
    def forward(self, x):
        # Flatten input
        x = x.view(x.size(0), -1)
        x = self.fc(x) 
        return x


# From https://github.com/fungtion/DANN
class ReverseLayerF(Function):
    """
    Fonction to backpropagate the opposite of the gradient
    scaled by a constant
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        """
        :param x: (PyTorch Tensor)
        :param lambda_: (float) scaling factor
        :return: (PyTorch Tensor)
        """
        ctx.lambda_ = lambda_
        # Equivalent to return x ?
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param grad_output: (PyTorch Tensor)
        :return: (PyTorch Tensor, None)
        """
        # Compute the opposite of the gradient
        output = grad_output.neg() * ctx.lambda_
        return output, None


class Discriminator(nn.Module):
    """
    Discriminator network to distinguish states from two different episodes
    :input_dim: (int) input_dim = 2 * state_dim
    """

    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)
