from __future__ import print_function, division, absolute_import

import torch
import torch.nn.functional as F

from .models import *
from .autoencoders import CNNAutoEncoder


class SRLInverseAutoEncoder(CNNAutoEncoder):
    def __init__(self, state_dim=2, action_dim=6):
        """
        :param state_dim:
        :param action_dim:
        :param cuda:
        :param noise_std:
        :param type:
        """
        super(SRLInverseAutoEncoder, self).__init__(state_dim)
        self.inverse_layer = nn.Linear(state_dim * 2, action_dim)

    def inverse(self, state, next_state):
        """
        :param state:
        :param next_state:
        :return: probability of a_t
        """
        return self.inverse_layer(torch.cat((state, next_state), 1))
