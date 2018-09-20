from __future__ import print_function, division, absolute_import

import torch

from .models import *


class BaseForwardModel(BaseModelSRL):
    def __init__(self):
        self.action_dim = None
        self.forward_net = None
        super(BaseForwardModel, self).__init__()

    def initForwardNet(self, state_dim, action_dim):
        self.action_dim = action_dim
        self.forward_net = nn.Linear(state_dim + action_dim, state_dim)

    def forward(self, x):
        raise NotImplementedError()

    def forwardModel(self, state, action):
        """
        Predict next state given current state and action
        :param state: (th.Tensor)
        :param action: (th Tensor)
        :return: (th.Tensor)
        """
        # Predict the delta between the next state and current state
        # by taking as input concatenation of state & action over the 2nd dimension
        concat = torch.cat((state, encodeOneHot(action, self.action_dim)), dim=1)
        return state + self.forward_net(concat)


class BaseInverseModel(BaseModelSRL):
    def __init__(self):
        self.inverse_net = None
        super(BaseInverseModel, self).__init__()

    def initInverseNet(self, state_dim, action_dim, n_hidden=128, model_type="linear"):
        """
        :param state_dim: (th.Tensor)
        :param action_dim: (int)
        :param n_hidden: (int)
        :param model_type: (str)
        :return: (th.Tensor)
        """
        if model_type == "linear":
            self.inverse_net = nn.Linear(state_dim * 2, action_dim)
        elif model_type == "mlp":
            self.inverse_net = nn.Sequential(nn.Linear(state_dim * 2, n_hidden),
                                             nn.ReLU(),
                                             nn.Linear(n_hidden, n_hidden),
                                             nn.ReLU(),
                                             nn.Linear(n_hidden, action_dim)
                                             )
        else:
            raise ValueError("Unknown model_type for inverse model: {}".format(model_type))

    def forward(self, x):
        raise NotImplementedError()

    def inverseModel(self, state, next_state):
        """
        Predict action given current state and next state
        :param state: (th.Tensor)
        :param next_state: (th.Tensor)
        :return: probability of each action
        """
        # input: concatenation of state & next state over the 2nd dimension
        return self.inverse_net(th.cat((state, next_state), dim=1))


class BaseRewardModel(BaseModelSRL):
    def __init__(self):
        self.reward_net = None
        super(BaseRewardModel, self).__init__()

    def initRewardNet(self, state_dim, n_rewards=2, n_hidden=16):
        self.reward_net = nn.Sequential(nn.Linear(2 * state_dim, n_hidden),
                                        nn.ReLU(),
                                        nn.Linear(n_hidden, n_hidden),
                                        nn.ReLU(),
                                        nn.Linear(n_hidden, n_rewards))

    def forward(self, x):
        raise NotImplementedError()

    def rewardModel(self, state, next_state):
        """
        Predict reward given current state and next state
        :param state: (th.Tensor)
        :param next_state: (th.Tensor)
        :return: (th.Tensor)
        """
        return self.reward_net(th.cat((state, next_state), dim=1))
