from __future__ import print_function, division, absolute_import

import torch

from .models import *
from .priors import SRLDenseNetwork, SRLConvolutionalNetwork
from .autoencoders import CNNAutoEncoder
from .priors import SRLLinear

from ..preprocessing.preprocess import INPUT_DIM


class BaseForwardModel(BaseModelSRL):
    def __init__(self):
        """
        :param state_dim: (int)
        :param action_dim: (int)
        """
        super(BaseForwardModel, self).__init__()

    def initForwardNet(self, state_dim, action_dim, ratio=1):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.forward_net = nn.Linear(int(state_dim * ratio) + action_dim, state_dim)
        self.ratio = ratio

    def forward(self, x):
        raise NotImplementedError()

    def forwardModel(self, state, action):
        """
        Predict next state given current state and action
        :param state: (th Variable)
        :param action: (th Tensor)
        :return: (th Variable)
        """
        # Predict the delta between the next state and current state
        concat = torch.cat((state[:, :int(self.state_dim*self.ratio)], encodeOneHot(action, self.action_dim)), 1)
        return state + self.forward_net(concat)


class BaseInverseModel(BaseModelSRL):
    def __init__(self):
        """
        :param state_dim: (int)
        :param action_dim: (int)
        """
        super(BaseInverseModel, self).__init__()

    def initInverseNet(self, state_dim, action_dim, ratio=1):
        self.inverse_net = nn.Linear(int(state_dim * ratio) * 2, action_dim)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.ratio = ratio

    def forward(self, x):
        raise NotImplementedError()

    def inverseModel(self, state, next_state):
        """
        Predict action given current state and next state
        :param state: (th Variable)
        :param next_state: (th Variable)
        :return: probability of each action
        """
        return self.inverse_net(th.cat((state[:, :int(self.state_dim*self.ratio)], next_state[:, :int(self.state_dim*self.ratio)]), 1))


class BaseRewardModel(BaseModelSRL):
    def __init__(self, state_dim=2, action_dim=6):
        """
        :param state_dim: (int)
        :param action_dim: (int)
        """
        super(BaseRewardModel, self).__init__()

    def initRewardNet(self, state_dim, action_dim, ratio=1):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.reward_net = nn.Sequential(nn.Linear(int(state_dim *ratio), 2),
                                        nn.ReLU(),
                                        nn.Linear(2, 2),
                                        nn.ReLU(),
                                        nn.Linear(2, 2))
        self.ratio = ratio

    def forward(self, x):
        raise NotImplementedError()

    def rewardModel(self, state):
        """
        Predict reward given current state and action
:        :param state: (th Variable)
        :param action: (th Tensor)
        :return: (th Variable)
        """
        #return self.reward_net(torch.cat((state[:, int(self.state_dim* ( 1 - self.ratio) ):], encodeOneHot(action, self.action_dim), next_state[:, int(self.state_dim * ( 1 - self.ratio) ):]), 1))
        return self.reward_net(state)


class SRLModules(BaseForwardModel, BaseInverseModel, BaseRewardModel):
    def __init__(self, state_dim=2, action_dim=6, ratio=1, cuda=False, model_type="custom_cnn"):
        """
        :param state_dim:
        :param action_dim:
        :param cuda:
        """
        self.model_type = model_type
        BaseForwardModel.__init__(self)
        BaseInverseModel.__init__(self)
        BaseRewardModel.__init__(self)

        self.initForwardNet(state_dim, action_dim, ratio)
        self.initInverseNet(state_dim, action_dim, ratio)
        self.initRewardNet(state_dim, action_dim, ratio)

        # Architecture
        if model_type == "custom_cnn":
            self.model = CustomCNN(state_dim)
        elif model_type == "linear":
            self.model = SRLLinear(input_dim=INPUT_DIM, state_dim=state_dim, cuda=cuda)
        elif model_type == "mlp":
            self.model = SRLDenseNetwork(INPUT_DIM, state_dim, cuda=cuda)
        elif model_type == "resnet":
             self.model = SRLConvolutionalNetwork(state_dim, cuda)
        elif model_type == "ae":
            self.model = CNNAutoEncoder(state_dim)
            self.model.encoder_fc.cuda()
            self.model.decoder_fc.cuda()

        if cuda:
            self.model.cuda()

    def getStates(self, observations):
        """
        :param observations: (PyTorch Variable)
        :return: (PyTorch Variable)
        """
        if self.model_type == "ae":
            return self.model.encode(observations)
        else:
            return self.forward(observations)

    def forward(self, x):
        if self.model_type == "ae":
            return self.model.forward(x)
        if self.model_type == 'linear' or self.model_type == 'mlp':
            x = x.contiguous()
        return self.model(x)
