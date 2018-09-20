from __future__ import print_function, division, absolute_import

import argparse
import time

import numpy as np
import torch as th
import torch.nn as nn
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import plotting.representation_plot as plot_script
from models import ConvolutionalNetwork, DenseNetwork, CustomCNN
from models.learner import BaseLearner
from pipeline import saveConfig
from plotting.losses_plot import plotLosses
from plotting.representation_plot import plotRepresentation, plt
from preprocessing.data_loader import SupervisedDataLoader
from preprocessing.preprocess import getInputDim
from train import buildConfig
from utils import parseDataFolder, createFolder, getInputBuiltin, loadData

DISPLAY_PLOTS = True
EPOCH_FLAG = 1  # Plot every 1 epoch
BATCH_SIZE = 32
TEST_BATCH_SIZE = 256


class SupervisedLearning(BaseLearner):

    def __init__(self, state_dim, model_type="resnet", log_folder="logs/default",
                 seed=1, learning_rate=0.001, cuda=False):
        """
        :param state_dim: (int)
        :param model_type: (str) one of "resnet", "custom_cnn" or "mlp"
        :param log_folder: (str
        :param seed: (int)
        :param learning_rate: (float)
        :param cuda: (bool)
        """

        super(SupervisedLearning, self).__init__(state_dim, BATCH_SIZE, seed, cuda)

        if model_type == "resnet":
            self.model = ConvolutionalNetwork(self.state_dim, cuda)
        elif model_type in ["cnn", "custom_cnn"]:
            self.model = CustomCNN(self.state_dim)
        elif model_type == "mlp":
            self.model = DenseNetwork(getInputDim(), self.state_dim)
        else:
            raise ValueError("Unknown model: {}".format(model_type))
        print("Using {} model".format(model_type))

        self.device = th.device("cuda" if th.cuda.is_available() and cuda else "cpu")

        self.model = self.model.to(self.device)
        learnable_params = [param for param in self.model.parameters() if param.requires_grad]
        self.optimizer = th.optim.Adam(learnable_params, lr=learning_rate)
        self.log_folder = log_folder

    def learn(self, true_states, images_path, rewards):
        """
        Learn a state representation
        :param images_path: (numpy 1D array)
        :param true_states: (np.ndarray)
        :param rewards: (numpy 1D array)
        :return: (np.ndarray) the learned states for the given observations
        """
        true_states = true_states.astype(np.float32)
        x_indices = np.arange(len(true_states)).astype(np.int64)

        # Split into train/validation set
        x_train, x_val, y_train, y_val = train_test_split(x_indices, true_states,
                                                          test_size=0.33, random_state=self.seed)

        train_loader = SupervisedDataLoader(x_train, y_train, images_path, batch_size=BATCH_SIZE,
                                            max_queue_len=4, shuffle=True)
        val_loader = SupervisedDataLoader(x_val, y_val, images_path, batch_size=TEST_BATCH_SIZE,
                                          max_queue_len=1, shuffle=False)
        # For plotting
        data_loader = SupervisedDataLoader(x_indices, true_states, images_path, no_targets=True, batch_size=TEST_BATCH_SIZE,
                                           max_queue_len=1, shuffle=False)

        # TRAINING -----------------------------------------------------------------------------------------------------
        criterion = nn.MSELoss()
        # criterion = F.smooth_l1_loss
        best_error = np.inf
        best_model_path = "{}/srl_supervised_model.pth".format(self.log_folder)

        start_time = time.time()
        epoch_train_loss = [[] for _ in range(N_EPOCHS)]
        epoch_val_loss = [[] for _ in range(N_EPOCHS)]
        for epoch in range(N_EPOCHS):
            # In each epoch, we do a full pass over the training data:
            train_loss, val_loss = 0, 0
            pbar = tqdm(total=len(train_loader))
            self.model.train()  # Restore train mode
            for obs, target_states in train_loader:
                obs, target_states = obs.to(self.device), target_states.to(self.device)

                pred_states = self.model(obs)
                self.optimizer.zero_grad()
                loss = criterion(pred_states, target_states.detach())
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                epoch_train_loss[epoch].append(loss.item())
                pbar.update(1)
            pbar.close()

            train_loss /= len(train_loader)

            self.model.eval()
            with th.no_grad():
                # Pass on the validation set
                for obs, target_states in val_loader:
                    obs, target_states = obs.to(self.device), target_states.to(self.device)

                    pred_states = self.model(obs)
                    loss = criterion(pred_states, target_states.detach())
                    val_loss += loss.item()
                    epoch_val_loss[epoch].append(loss.item())

                val_loss /= len(val_loader)

            # Save best model
            if val_loss < best_error:
                best_error = val_loss
                th.save(self.model.state_dict(), best_model_path)

            # Then we print the results for this epoch:
            if (epoch + 1) % EPOCH_FLAG == 0:
                print("Epoch {:3}/{}".format(epoch + 1, N_EPOCHS))
                print("train_loss:{:.4f} val_loss:{:.4f}".format(train_loss, val_loss))
                print("{:.2f}s/epoch".format((time.time() - start_time) / (epoch + 1)))
                if DISPLAY_PLOTS:
                    # Optionally plot the current state space
                    plotRepresentation(self.predStatesWithDataLoader(data_loader), rewards, add_colorbar=epoch == 0,
                                       name="Learned State Representation (Training Data)")
        if DISPLAY_PLOTS:
            plt.close("Learned State Representation (Training Data)")

        # Load best model before predicting states
        self.model.load_state_dict(th.load(best_model_path))
        # save loss
        np.savez(self.log_folder + "/loss.npz", train=epoch_train_loss, val=epoch_val_loss)
        # Save plot
        plotLosses({"train": np.array(epoch_train_loss), "val": np.array(epoch_val_loss)}, self.log_folder)
        # return predicted states for training observations
        with th.no_grad():
            pred_states = self.predStatesWithDataLoader(data_loader)
        return pred_states


def getModelName(args):
    """
    :param args: (parsed args object)
    :return: (str)
    """
    name = "supervised_{}_SEED{}".format(args.model_type, args.seed)
    name += "_EPOCHS{}_BS{}".format(args.epochs, args.batch_size)
    return name


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Supervised Learning')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('-bs', '--batch-size', type=int, default=32, help='batch_size (default: 32)')
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.005, help='learning rate (default: 0.005)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--no-display-plots', action='store_true', default=False, help='disables live plots of the representation learned')
    parser.add_argument('--model-type', type=str, default="resnet", help='Model architecture (default: "resnet")')
    parser.add_argument('--data-folder', type=str, default="", help='Dataset folder', required=True)
    parser.add_argument('--training-set-size', type=int, default=-1,
                        help='Limit size of the training set (default: -1)')
    parser.add_argument('--relative-pos', action='store_true', default=False,
                        help='Use relative position as ground_truth')
    parser.add_argument('--log-folder', type=str, default='', help='Override the default log-folder')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and th.cuda.is_available()
    DISPLAY_PLOTS = not args.no_display_plots
    plot_script.INTERACTIVE_PLOT = DISPLAY_PLOTS
    N_EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    args.data_folder = parseDataFolder(args.data_folder)
    log_folder = args.log_folder

    if log_folder == '':
        name = getModelName(args)
        log_folder = "logs/{}/baselines/{}".format(args.data_folder, name)

    createFolder(log_folder, "supervised folder already exist")

    folder_path = '{}/NearestNeighbors/'.format(log_folder)
    createFolder(folder_path, "NearestNeighbors folder already exist")

    print('Log folder: {}'.format(log_folder))

    print('Loading data ... ')
    training_data, ground_truth, true_states, _ = loadData(args.data_folder)
    rewards, episode_starts = training_data['rewards'], training_data['episode_starts']

    images_path = ground_truth['images_path']
    state_dim = true_states.shape[1]

    if args.training_set_size > 0:
        limit = args.training_set_size
        true_states = true_states[:limit]
        images_path = images_path[:limit]
        rewards = rewards[:limit]

    args.state_dim = state_dim
    args.losses = ["supervised"]
    exp_config = buildConfig(args)
    exp_config["log-folder"] = log_folder
    saveConfig(exp_config, print_config=True)

    print('Learning a state representation ... ')
    srl = SupervisedLearning(state_dim, model_type=args.model_type, seed=args.seed,
                             log_folder=log_folder, learning_rate=args.learning_rate,
                             cuda=args.cuda)

    learned_states = srl.learn(true_states, images_path, rewards)
    srl.saveStates(learned_states, images_path, rewards, log_folder)

    name = "Learned State Representation - {} \n Supervised Learning".format(args.data_folder)
    path = "{}/learned_states.png".format(log_folder)
    plotRepresentation(learned_states, rewards, name, add_colorbar=True, path=path)

    if DISPLAY_PLOTS:
        getInputBuiltin()('\nPress any key to exit.')
