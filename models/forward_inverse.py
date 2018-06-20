from __future__ import print_function, division, absolute_import

import torch

from .models import *
from .triplet import EmbeddingNet
from .priors import SRLDenseNetwork, SRLConvolutionalNetwork, SRLLinear
from .autoencoders import CNNAutoEncoder, DenseAutoEncoder, LinearAutoEncoder
from .vae import CNNVAE, DenseVAE

try:
    from preprocessing.preprocess import INPUT_DIM
except ImportError:
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
        concat = torch.cat((state, encodeOneHot(action, self.action_dim)), 1)
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
        #return self.inverse_net(th.cat((state, next_state), 1))


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
        :param state: (th Variable)
        :param action: (th Tensor)
        :return: (th Variable)
        """
        return self.reward_net(state)


class SRLModules(BaseForwardModel, BaseInverseModel, BaseRewardModel):
    def __init__(self, state_dim=2, action_dim=6, ratio=1, cuda=False, model_type="custom_cnn", losses=None):
        """
        :param state_dim:
        :param action_dim:
        :param cuda:
        """
        self.model_type = model_type
        self.losses = losses
        BaseForwardModel.__init__(self)
        BaseInverseModel.__init__(self)
        BaseRewardModel.__init__(self)

        self.initForwardNet(state_dim, action_dim, ratio)
        self.initInverseNet(state_dim, action_dim, ratio)
        self.initRewardNet(state_dim, action_dim, ratio)

        # Architecture
        if model_type == "custom_cnn":
            if "autoencoder" in losses:
                self.model = CNNAutoEncoder(state_dim)
                self.model.encoder_fc.cuda()
                self.model.decoder_fc.cuda()
            elif "vae" in losses:
                self.model = CNNVAE(state_dim)
                self.model.encoder_fc1.cuda()
                self.model.encoder_fc2.cuda()
                self.model.decoder_fc.cuda()
            else:
                # for losses not depending on specific architecture (supevised, inv, fwd..)
                self.model = CustomCNN(state_dim)

        elif model_type == "mlp":
            if "autoencoder" in losses:
                self.model = DenseAutoEncoder(input_dim=INPUT_DIM, state_dim=state_dim)
                self.model.encoder.cuda()
                self.model.decoder.cuda()
            elif "vae" in losses:
                self.model = DenseVAE(input_dim=INPUT_DIM,
                                      state_dim=state_dim)
                self.model.encoder_fc1.cuda()
                self.model.encoder_fc21.cuda()
                self.model.encoder_fc22.cuda()
                self.model.decoder.cuda()
            else:
                # for losses not depending on specific architecture (supevised, inv, fwd..)
                self.model = SRLDenseNetwork(INPUT_DIM, state_dim, cuda=cuda)

        elif model_type == "linear":
            if "autoencoder" in losses:
                self.model = LinearAutoEncoder(input_dim=INPUT_DIM, state_dim=state_dim)
                self.model.encoder.cuda()
                self.model.decoder.cuda()
            else:
                # for losses not depending on specific architecture (supevised, inv, fwd..)
                self.model = SRLLinear(input_dim=INPUT_DIM, state_dim=state_dim, cuda=cuda)

        elif model_type == "resnet":
            self.model = SRLConvolutionalNetwork(state_dim, cuda)

        if losses is not None and "triplet" in losses:
                # pretrained resnet18 with fixed weights
            self.model = EmbeddingNet(state_dim)

        if cuda:
            self.model.cuda()

    def getStates(self, observations):
        """
        :param observations: (PyTorch Variable)
        :return: (PyTorch Variable)
        """
        if "autoencoder" in self.losses or "vae" in self.losses:
            return self.model.getStates(observations)
        elif "triplet" in self.losses:
            # For inference, the forward pass is done one the positive observation (first view)
            return self.encode(observations[:, :3:, :, :])
        else:
            return self.forward(observations)

    def forward(self, x):
        if self.model_type == 'linear' or self.model_type == 'mlp':
            x = x.contiguous()
        return self.model(x)

    def encode(self, x):
        if "triplet" in self.losses:
            return self.model(x)
        else:
            raise NotImplementedError()

    def forward_triplets(self, anchor, positive, negative):
        """
        Overriding the forward function in the case of Triplet loss
        anchor : observation
        positive : observation
        negative : observation
        """
        return self.model(anchor), self.model(positive), self.model(negative)
