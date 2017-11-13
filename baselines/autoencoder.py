from __future__ import print_function, division, absolute_import

import time
import argparse

import numpy as np
import torch as th
import torch.nn as nn
from torch.autograd import Variable
from sklearn.model_selection import train_test_split

from utils import parseDataFolder
from preprocessing.preprocess import INPUT_DIM
from models.base_learner import BaseLearner
from models.models import LinearAutoEncoder, DenseAutoEncoder
from plotting.representation_plot import plot_representation, plt

# Python 2/3 compatibility
try:
    input = raw_input
except NameError:
    pass

DISPLAY_PLOTS = True
EPOCH_FLAG = 1  # Plot every 1 epoch
BATCH_SIZE = 32
NOISE_STD = 1e-3  # Standard deviation of the gaussian noise for AE


class AutoEncoderLearning(BaseLearner):
    """
    :param state_dim: (int)
    :param model_type: (str) one of "cnn" or "mlp"
    :param seed: (int)
    :param learning_rate: (float)
    :param cuda: (bool)
    """

    def __init__(self, state_dim, model_type="cnn", log_folder="logs/default",
                 seed=1, learning_rate=0.001, cuda=False):

        super(AutoEncoderLearning, self).__init__(state_dim, BATCH_SIZE, seed, cuda)

        if model_type == "cnn":
            raise NotImplementedError("CNN Autoencoder not implemented")
            # self.model = ConvolutionalNetwork(self.state_dim, cuda)
        elif model_type == "mlp":
            # self.model = DenseAutoEncoder(INPUT_DIM, self.state_dim, NOISE_STD, cuda)
            self.model = LinearAutoEncoder(INPUT_DIM, self.state_dim, NOISE_STD, cuda)
        else:
            raise ValueError("Unknown model: {}".format(model_type))
        print("Using {} model".format(model_type))

        if cuda:
            self.model.cuda()
        learnable_params = [param for param in self.model.parameters() if param.requires_grad]
        self.optimizer = th.optim.Adam(learnable_params, lr=learning_rate)
        self.log_folder = log_folder

    def _predFn(self, observations, restore_train=True):
        """
        Predict states in test mode given observations
        :param observations: (PyTorch Variable)
        :param restore_train: (bool) whether to restore training mode after prediction
        :return: (numpy tensor)
        """
        # Switch to test mode
        self.model.eval()
        states, _ = self.model(observations)
        if restore_train:
            # Restore training mode
            self.model.train()
        if self.cuda:
            # Move the tensor back to the cpu
            return states.data.cpu().numpy()
        return states.data.numpy()

    def learn(self, observations, rewards):
        """
        Learn a state representation
        :param observations: (numpy tensor)
        :param rewards: (numpy 1D array)
        :return: (numpy tensor) the learned states for the given observations
        """
        # TODO: do not duplicate the data to save memory
        # We assume that observations are already preprocessed
        # that is to say normalized and scaled
        observations = observations.astype(np.float32)

        # Split into train/validation set
        X_train, X_val = train_test_split(observations, test_size=0.33, random_state=self.seed)

        kwargs = {'num_workers': 1, 'pin_memory': False} if self.cuda else {}

        # Convert to torch tensor
        X_train, X_val = th.from_numpy(X_train), th.from_numpy(X_val)

        train_loader = th.utils.data.DataLoader(th.utils.data.TensorDataset(X_train, X_train),
                                                batch_size=self.batch_size, shuffle=True, **kwargs)

        val_loader = th.utils.data.DataLoader(th.utils.data.TensorDataset(X_val, X_val),
                                              batch_size=self.batch_size, shuffle=False, **kwargs)
        # TRAINING -----------------------------------------------------------------------------------------------------
        criterion = nn.MSELoss()
        # criterion = F.smooth_l1_loss
        best_error = np.inf

        self.model.train()
        start_time = time.time()
        for epoch in range(N_EPOCHS):
            # In each epoch, we do a full pass over the training data:
            train_loss, val_loss = 0, 0

            for batch_idx, (obs, _) in enumerate(train_loader):
                if self.cuda:
                    obs = obs.cuda()
                obs = Variable(obs)

                _, decoded = self.model(obs)
                self.optimizer.zero_grad()
                loss = criterion(decoded, obs)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.data[0]

            train_loss /= len(train_loader)

            self.model.eval()
            # Pass on the validation set
            for obs, _ in val_loader:
                if self.cuda:
                    obs = obs.cuda()
                obs = Variable(obs, volatile=True)

                _, decoded = self.model(obs)
                loss = criterion(decoded, obs)
                val_loss += loss.data[0]

            val_loss /= len(val_loader)
            self.model.train()  # Restore train mode

            # Save best model
            if val_loss < best_error:
                best_error = val_loss
                th.save(self.model.state_dict(), "{}/srl_ae_model.pyth.pkl".format(self.log_folder))

            # Then we print the results for this epoch:
            if (epoch + 1) % EPOCH_FLAG == 0:
                print("Epoch {:3}/{}".format(epoch + 1, N_EPOCHS))
                print("train_loss:{:.4f} val_loss:{:.4f}".format(train_loss, val_loss))
                print("{:.2f}s/epoch".format((time.time() - start_time) / (epoch + 1)))
                if DISPLAY_PLOTS:
                    # Optionally plot the current state space
                    plot_representation(self._batchPredStates(observations), rewards, add_colorbar=epoch == 0,
                                        name="Learned State Representation (Training Data)")
        if DISPLAY_PLOTS:
            plt.close("Learned State Representation (Training Data)")

        # TODO: load best model before predicting states
        # return predicted states for training observations
        return self._batchPredStates(observations)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Supervised Learning')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('-bs', '--batch_size', type=int, default=32, help='batch_size (default: 32)')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--no-plots', action='store_true', default=False, help='disables plots')
    parser.add_argument('--model_type', type=str, default="cnn", help='Model architecture (default: "cnn")')
    parser.add_argument('--data_folder', type=str, default="", help='Dataset folder', required=True)
    parser.add_argument('--state_dim', type=int, default=2, help='state dimension (default: 2)')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and th.cuda.is_available()
    DISPLAY_PLOTS = not args.no_plots
    N_EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    args.data_folder = parseDataFolder(args.data_folder)
    log_folder = "logs/{}/baselines".format(args.data_folder)

    print('Log folder: {}'.format(log_folder))

    print('Loading data ... ')
    training_data = np.load("data/{}/preprocessed_data.npz".format(args.data_folder))
    observations = training_data['observations']
    rewards = training_data['rewards']

    # Move the channel dimension to match pretrained model input
    # (batch_size, width, height, n_channels) -> (batch_size, n_channels, height, width)
    observations = np.transpose(observations, (0, 3, 2, 1))
    print("Observations shape: {}".format(observations.shape))
    true_states = np.load("data/{}/ground_truth.npz".format(args.data_folder))['arm_states']
    state_dim = true_states.shape[1]

    print('Learning a state representation ... ')
    srl = AutoEncoderLearning(args.state_dim, model_type=args.model_type, seed=args.seed,
                              log_folder=log_folder, learning_rate=args.learning_rate,
                              cuda=args.cuda)
    learned_states = srl.learn(observations, rewards)

    name = "Learned State Representation - {} \n Autoencoder state_dim={}".format(args.data_folder, args.state_dim)
    path = "{}/learned_states_ae.png".format(log_folder)
    plot_representation(learned_states, rewards, name, add_colorbar=True, path=path)

    if DISPLAY_PLOTS:
        input('\nPress any key to exit.')
