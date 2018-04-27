from __future__ import print_function, division, absolute_import

from .models import *


class EmbeddingNet(nn.Module):
    def __init__(self, state_dim=2, embedding_size=128):
        """
        Resnet18 + FC layer (Embedding to learn a metric)
        input shape : 2 X 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224
        :param state_dim: (int)
        :embedding_size: (int) size of TCN embedding
        """
        super(EmbeddingNet, self).__init__()
        # ResNet 18
        self.conv_layers = models.resnet18(pretrained=True)
        # Freeze params
        for param in self.conv_layers.parameters():
            param.set_grad_enabled(False)
        # Replace the last fully-connected layer
        n_units = self.conv_layers.fc.in_features
        print("{} units in the last layer".format(n_units))
        self.conv_layers.fc = nn.Linear(n_units, embedding_size)
        self.fc = nn.Sequential(nn.PReLU(),
                                nn.Linear(embedding_size, state_dim)
                                )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class TripletNet(BaseModelSRL):
    def __init__(self, state_dim=2):
        super(TripletNet, self).__init__()
        self.embedding = EmbeddingNet(state_dim)

    def getStates(self, observations):
        """
        :param observations: (PyTorch Variable)
        :return: (PyTorch Variable)
        """
        # For inference, the forward pass is done one the positive observation (first view)
        return self.encode(observations[:, :3:, :, :])

    def forward(self, anchor, positive, negative):
        """
        anchor : observation
        positive : observation
        negative : observation
        """
        return self.embedding(anchor), self.embedding(positive), self.embedding(negative)

    def encode(self, x):
        return self.embedding(x)
