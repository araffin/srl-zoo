from __future__ import print_function, division, absolute_import

import torch

from .models import *
from .autoencoders import CNNAutoEncoder
from .priors import SRLLinear

from preprocessing.preprocess import INPUT_DIM

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


class SRLInverseAutoEncoder(CNNAutoEncoder, BaseInverseModel):
    def __init__(self, state_dim, action_dim):
        """
        :param state_dim: (int)
        :param action_dim: (int)
        """
        CNNAutoEncoder.__init__(self, state_dim)
        self.initInverseNet(state_dim, action_dim)

class SRLCustomForward(BaseForwardModel, BaseRewardModel):
    def __init__(self, state_dim=2, action_dim=6, cuda=False):
        """
        :param state_dim: (int)
        :param action_dim: (int)
        :param cuda: (bool)
        """
        #super(SRLCustomForward, self).__init__()
        BaseForwardModel.__init__(self)
        BaseRewardModel.__init__(self)

        self.initForwardNet(state_dim, action_dim)
        self.linear= SRLLinear(input_dim=INPUT_DIM, state_dim=state_dim, cuda=cuda)
        self.initRewardNet(state_dim, action_dim)

        if cuda:
            self.linear.cuda()

    def forward(self, x):
        return self.linear(x.contiguous())


class SRLCustomInverse(BaseInverseModel, BaseRewardModel):
    def __init__(self, state_dim=2, action_dim=6, cuda=False):
        """
        :param state_dim:
        :param action_dim:
        :param cuda:
        """
        #super(SRLCustomInverse, self).__init__()
        BaseInverseModel.__init__(self)
        BaseRewardModel.__init__(self)

        self.initInverseNet(state_dim, action_dim)
        self.linear= SRLLinear(input_dim=INPUT_DIM, state_dim=state_dim, cuda=cuda)
        self.initRewardNet(state_dim, action_dim)

        if cuda:
            self.linear.cuda()

    def forward(self, x):
        return self.linear(x.contiguous())


class SRLCustomForwardInverse(BaseForwardModel, BaseInverseModel, BaseRewardModel):
    def __init__(self, state_dim=2, action_dim=6, ratio=1, cuda=False):
        """
        :param state_dim:
        :param action_dim:
        :param cuda:
        """
        BaseForwardModel.__init__(self)
        BaseInverseModel.__init__(self)
        BaseRewardModel.__init__(self)

        self.initForwardNet(state_dim, action_dim, ratio)
        self.initInverseNet(state_dim, action_dim, ratio)
        self.initRewardNet(state_dim, action_dim, ratio)

        self.cnn = CustomCNN(state_dim)

        if cuda:
            self.cnn.cuda()

    def forward(self, x):
        return self.cnn(x)
