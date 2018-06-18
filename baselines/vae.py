from __future__ import print_function, division, absolute_import

import time
import argparse

import numpy as np
import torch as th
from torch.nn import functional as F
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import plotting.representation_plot as plot_script
from utils import parseDataFolder, createFolder
from preprocessing.data_loader import AutoEncoderDataLoader
from preprocessing.preprocess import INPUT_DIM
from preprocessing.utils import deNormalize
from models.base_learner import BaseLearner
from models import DenseVAE, CNNVAE
from pipeline import saveConfig
from plotting.representation_plot import plotRepresentation, plt, plotImage
from plotting.losses_plot import plotLosses

# Python 2/3 compatibility
try:
    input = raw_input
except NameError:
    pass

DISPLAY_PLOTS = True
EPOCH_FLAG = 1  # Plot every 1 epoch
BATCH_SIZE = 32
NOISE_FACTOR = 0
TEST_BATCH_SIZE = 512


class VAELearning(BaseLearner):
    """
    :param state_dim: (int)
    :param model_type: (str) one of "cnn" or "mlp"
    :param log_folder: (str)
    :param seed: (int)
    :param learning_rate: (float)
    :param cuda: (bool)
    """

    def __init__(self, state_dim, model_type="cnn", log_folder="logs/default",
                 seed=1, learning_rate=0.001, cuda=False, beta=1):

        super(VAELearning, self).__init__(state_dim, BATCH_SIZE, seed, cuda)
        self.beta = beta

        if model_type == "cnn":
            self.model = CNNVAE(self.state_dim)
        elif model_type == "mlp":
            self.model = DenseVAE(INPUT_DIM, self.state_dim)
        else:
            raise ValueError("Unknown model: {}".format(model_type))
        print("Using {} model".format(model_type))

        self.device = th.device("cuda" if th.cuda.is_available() and cuda else "cpu")

        self.model.to(self.device)
        learnable_params = [param for param in self.model.parameters() if param.requires_grad]
        self.optimizer = th.optim.Adam(learnable_params, lr=learning_rate)

        self.log_folder = log_folder

    @staticmethod
    def _lossFunction(decoded, obs, mu, logvar, beta=1):
        """
        Reconstruction + KL divergence losses summed over all elements and batch
        :param decoded: (Pytorch Variable)
        :param obs: (Pytorch Variable)
        :param mu: (Pytorch Variable)
        :param logvar: (Pytorch Variable)
        :param beta: (float) used to weight the KL divergence for disentangling
        :return: (Pytorch Variable)
        """
        generation_loss = F.mse_loss(decoded, obs, size_average=False)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_divergence = -0.5 * th.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return generation_loss + beta * kl_divergence

    def learn(self, images_path, rewards):
        """
        Learn a state representation
        :param images_path: (numpy 1D array)
        :param rewards: (numpy 1D array)
        :return: (numpy tensor) the learned states for the given observations
        """
        x_indices = np.arange(len(images_path)).astype(np.int64)

        # Split into train/validation set
        x_train, x_val = train_test_split(x_indices, test_size=0.33, random_state=self.seed)

        train_loader = AutoEncoderDataLoader(x_train, images_path,
                                             batch_size=self.batch_size,
                                             noise_factor=NOISE_FACTOR)
        val_loader = AutoEncoderDataLoader(x_val, images_path,
                                           batch_size=TEST_BATCH_SIZE,
                                           noise_factor=NOISE_FACTOR, is_training=False)
        # For plotting
        data_loader = AutoEncoderDataLoader(x_indices, images_path, batch_size=TEST_BATCH_SIZE,
                                            no_targets=True, is_training=False)

        # TRAINING -----------------------------------------------------------------------------------------------------
        best_error = np.inf
        best_model_path = "{}/srl_vae_model.pth".format(self.log_folder)
        print("Training...")
        self.model.train()
        start_time = time.time()
        epoch_train_loss = [[] for _ in range(N_EPOCHS)]
        epoch_val_loss = [[] for _ in range(N_EPOCHS)]
        for epoch in range(N_EPOCHS):
            # In each epoch, we do a full pass over the training data:
            train_loss, val_loss = 0, 0
            train_loader.resetAndShuffle()
            pbar = tqdm(total=len(train_loader))
            for batch_idx, (noisy_obs, obs) in enumerate(train_loader):
                noisy_obs, obs = noisy_obs.to(self.device), obs.to(self.device)

                self.optimizer.zero_grad()
                decoded, mu, logvar = self.model(noisy_obs)
                loss = VAELearning._lossFunction(decoded, obs, mu, logvar, self.beta)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                epoch_train_loss[epoch].append(loss.item())
                pbar.update(1)
            pbar.close()

            train_loss /= len(train_loader)

            self.model.eval()
            val_loader.resetIterator()
            # Pass on the validation set
            with th.no_grad():
                for noisy_obs, obs in val_loader:
                    noisy_obs, obs = noisy_obs.to(self.device), obs.to(self.device)

                    decoded, mu, logvar = self.model(noisy_obs)
                    loss = VAELearning._lossFunction(decoded, obs, mu, logvar, self.beta)
                    val_loss += loss.item()
                    epoch_val_loss[epoch].append(loss.item())

                val_loss /= len(val_loader)
                if DISPLAY_PLOTS:
                    # Plot Reconstructed Image
                    plotImage(deNormalize(noisy_obs[0].to(th.device('cpu')).detach().numpy()), "Input Validation Image")
                    plotImage(deNormalize(decoded[0].to(th.device('cpu')).detach().numpy()), "Reconstructed Image")

            self.model.train()  # Restore train mode

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

        # load best model before predicting states
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
    name = "vae_{}_ST_DIM{}_SEED{}_NOISE{}".format(args.model_type, args.state_dim,
                                                   args.seed, args.noise_factor)
    if args.beta != 1.0:
        name += "_BETA{:.2f}".format(args.beta)
    name = name.replace(".", "_")  # replace decimal points by '_' for folder naming
    name += "_EPOCHS{}_BS{}".format(args.epochs, args.batch_size)
    return name


def saveExpConfig(args, log_folder):
    """
    :param args: (parsed args object)
    :param log_folder: (str)
    """
    exp_config = {
        "batch-size": args.batch_size,
        "data-folder": args.data_folder,
        "epochs": args.epochs,
        "learning-rate": args.learning_rate,
        "training-set-size": args.training_set_size,
        "log-folder": log_folder,
        "noise-factor": args.noise_factor,
        "model-type": args.model_type,
        "seed": args.seed,
        "state-dim": args.state_dim,
    }

    saveConfig(exp_config, print_config=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Supervised Learning')
    parser.add_argument('--epochs', type=int, default=25, metavar='N',
                        help='number of epochs to train (default: 25)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('-bs', '--batch-size', type=int, default=32, help='batch_size (default: 32)')
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--no-plots', action='store_true', default=False, help='disables plots')
    parser.add_argument('--model-type', type=str, default="cnn", help='Model architecture (default: "cnn")')
    parser.add_argument('--data-folder', type=str, default="", help='Dataset folder', required=True)
    parser.add_argument('--state-dim', type=int, default=2, help='state dimension (default: 2)')
    parser.add_argument('--noise-factor', type=float, default=0, help='Noise factor for denoising vae')
    parser.add_argument('--training-set-size', type=int, default=-1,
                        help='Limit size of the training set (default: -1)')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='the Beta factor on the KL divergence, higher value means more disentangling.')
    parser.add_argument('--log-folder', type=str, default='', help='Override the default log-folder')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and th.cuda.is_available()
    DISPLAY_PLOTS = not args.no_plots
    plot_script.INTERACTIVE_PLOT = DISPLAY_PLOTS
    N_EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    NOISE_FACTOR = args.noise_factor
    args.data_folder = parseDataFolder(args.data_folder)

    log_folder = args.log_folder
    if log_folder == '':
        name = getModelName(args)
        log_folder = "logs/{}/baselines/{}".format(args.data_folder, name)
    createFolder(log_folder, "vae folder already exist")
    saveExpConfig(args, log_folder)

    folder_path = '{}/NearestNeighbors/'.format(log_folder)
    createFolder(folder_path, "NearestNeighbors folder already exist")

    print('Log folder: {}'.format(log_folder))

    print('Loading data ... ')
    rewards = np.load("data/{}/preprocessed_data.npz".format(args.data_folder))['rewards']
    images_path = np.load("data/{}/ground_truth.npz".format(args.data_folder))['images_path']

    if args.training_set_size > 0:
        limit = args.training_set_size
        images_path = images_path[:limit]
        rewards = rewards[:limit]

    print("{} samples".format(len(images_path)))

    print('Learning a state representation ... ')
    srl = VAELearning(args.state_dim, model_type=args.model_type, seed=args.seed,
                      log_folder=log_folder, learning_rate=args.learning_rate,
                      cuda=args.cuda, beta=args.beta)
    learned_states = srl.learn(images_path, rewards)
    srl.saveStates(learned_states, images_path, rewards, log_folder)

    name = "Learned State Representation - {} \n VAE state_dim={}".format(args.data_folder, args.state_dim)
    path = "{}/learned_states.png".format(log_folder)
    plotRepresentation(learned_states, rewards, name, add_colorbar=True, path=path)

    if DISPLAY_PLOTS:
        input('\nPress any key to exit.')
