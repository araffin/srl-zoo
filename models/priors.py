from __future__ import print_function, division, absolute_import

from .models import *
import torch
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

class SRLCustomForward(BaseModelSRL):
    def __init__(self, state_dim=2, action_dim=6, cuda=False, noise_std=1e-6, type='linear'):
        """
        :param state_dim:
        :param action_dim:
        :param cuda:
        :param noise_std:
        :param type:
        """
        super(SRLCustomForward, self).__init__()
        self.cnn = CustomCNN(state_dim)

        self.forward_l1 = nn.Linear(state_dim + action_dim, 256)
        self.forward_l2 = nn.Linear(256, state_dim)
        if cuda:
            self.cnn.cuda()
            self.forward_l1.cuda()
            self.forward_l2.cuda()

        self.noise = GaussianNoiseVariant(noise_std, cuda=cuda)

    def forward(self, x):
        x = self.cnn(x)
        return self.noise(x)

    def forward_extra(self, s_t, a_t):
        """
        #TODO: add bias to for
        :param s_t: s(t)
        :param a_t: a(t)
        :return: s(t+1)
        """

        # Onehot encoding of the action
        a_one_hot = torch.Tensor(a_t.shape[0], 6).zero_() 
        if a_t.is_cuda:
            a_one_hot = a_one_hot.cuda()
        a_one_hot = torch.autograd.Variable(a_one_hot.scatter_(1, a_t.data, 1.))

        # Forward pass
        concat = torch.cat((s_t, a_one_hot),1)
        inter = nn.functional.relu( self.forward_l1(concat))
        return self.forward_l2(inter)

class SRLCustomInverse(BaseModelSRL):
    def __init__(self, state_dim=2, action_dim=6, cuda=False, noise_std=1e-6, type='linear'):
        """
        :param state_dim:
        :param action_dim:
        :param cuda:
        :param noise_std:
        :param type:
        """
        super(SRLCustomInverse, self).__init__()
        self.cnn = CustomCNN(state_dim)

        self.inverse_l1 = nn.Linear(state_dim * 2, 256)  #+ action_dim, 256)
        self.inverse_l2 = nn.Linear(256, 128)
        self.inverse_l3 = nn.Linear(128, action_dim)
        if cuda:
            self.cnn.cuda()
            self.inverse_l1.cuda()
            self.inverse_l2.cuda()
            self.inverse_l3.cuda()

        self.noise = GaussianNoiseVariant(noise_std, cuda=cuda)

    def forward(self, x):
        x = self.cnn(x)
        return self.noise(x)

    def inverse(self, s_t, s_t_plus):
        """
        #TODO: add bias to for
        :param s_t: s(t)
        :param s_t_plus: s(t+1)
        :return: probability of a_t
        """
        concat = torch.cat((s_t, s_t_plus), 1)
        inter = nn.functional.relu(self.inverse_l1(concat))
        inter2 = nn.functional.relu(self.inverse_l2(inter))
        return self.inverse_l3(inter2)

class SRLCustomForwardInverse(SRLCustomForward):
    def __init__(self, state_dim=2, action_dim=6, cuda=False, noise_std=1e-6, type='linear'):
        """
        :param state_dim:
        :param action_dim:
        :param cuda:
        :param noise_std:
        :param type:
        """
        super(SRLCustomForwardInverse, self).__init__()
        self.cnn = CustomCNN(state_dim)

        self.forward_l1 = nn.Linear(state_dim + action_dim, 256)
        self.forward_l2 = nn.Linear(256, state_dim)
        self.inverse_l1 = nn.Linear(state_dim * 2, 256)  # + action_dim, 256)
        self.inverse_l2 = nn.Linear(256, 128)
        self.inverse_l3 = nn.Linear(128, action_dim)

        if cuda:
            self.cnn.cuda()
            self.forward_l1.cuda()
            self.forward_l2.cuda()
            self.inverse_l1.cuda()
            self.inverse_l2.cuda()
            self.inverse_l3.cuda()

        self.noise = GaussianNoiseVariant(noise_std, cuda=cuda)    

    def inverse(self, s_t, s_t_plus):
        """
        #TODO: add bias to for
        :param s_t: s(t)
        :param s_t_plus: s(t+1)
        :return: probability of a_t
        """
        concat = torch.cat((s_t, s_t_plus), 1)
        inter = nn.functional.relu(self.inverse_l1(concat))
        inter2 = nn.functional.relu(self.inverse_l2(inter))
        return self.inverse_l3(inter2)
    
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
        if self.training:
            x = self.noise(x)
        return x
