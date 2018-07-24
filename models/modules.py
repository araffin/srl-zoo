from .autoencoders import CNNAutoEncoder, DenseAutoEncoder, LinearAutoEncoder
from .forward_inverse import BaseForwardModel, BaseInverseModel, BaseRewardModel
from .models import CustomCNN
from .priors import SRLConvolutionalNetwork, SRLDenseNetwork, SRLLinear
from .triplet import EmbeddingNet
from .vae import CNNVAE, DenseVAE

# In case of importing into the SRL repository
try:
    from preprocessing.preprocess import getInputDim
# In case of importing material from modules.py into the external Robotics RL repository,
# consider the relative path to the package
except ImportError:
    from ..preprocessing.preprocess import getInputDim


class SRLModules(BaseForwardModel, BaseInverseModel, BaseRewardModel):
    def __init__(self, state_dim=2, action_dim=6, cuda=False, model_type="custom_cnn", losses=None):
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

        self.cuda = cuda

        self.initForwardNet(state_dim, action_dim)
        self.initInverseNet(state_dim, action_dim)
        self.initRewardNet(state_dim, action_dim)

        # Architecture
        if model_type == "custom_cnn":
            if "autoencoder" in losses or "dae" in losses:
                self.model = CNNAutoEncoder(state_dim)
            elif "vae" in losses:
                self.model = CNNVAE(state_dim)
            else:
                # for losses not depending on specific architecture (supervised, inv, fwd..)
                self.model = CustomCNN(state_dim)

        elif model_type == "mlp":
            if "autoencoder" in losses or "dae" in losses:
                self.model = DenseAutoEncoder(input_dim=getInputDim(), state_dim=state_dim)
            elif "vae" in losses:
                self.model = DenseVAE(input_dim=getInputDim(),
                                      state_dim=state_dim)
            else:
                # for losses not depending on specific architecture (supervised, inv, fwd..)
                self.model = SRLDenseNetwork(getInputDim(), state_dim, cuda=cuda)

        elif model_type == "linear":
            if "autoencoder" in losses or "dae" in losses:
                self.model = LinearAutoEncoder(input_dim=getInputDim(), state_dim=state_dim)
            else:
                # for losses not depending on specific architecture (supervised, inv, fwd..)
                self.model = SRLLinear(input_dim=getInputDim(), state_dim=state_dim, cuda=cuda)

        elif model_type == "resnet":
            self.model = SRLConvolutionalNetwork(state_dim, cuda)

        if losses is not None and "triplet" in losses:
            # pretrained resnet18 with fixed weights
            self.model = EmbeddingNet(state_dim)

    def getStates(self, observations):
        """
        :param observations: (PyTorch Tensor)
        :return: (PyTorch Tensor)
        """
        return self.model.getStates(observations)

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