from __future__ import print_function, division, absolute_import

from collections import defaultdict
import json
import sys
import numpy as np
import time
import torch as th
from tqdm import tqdm

from losses.losses import LossManager, autoEncoderLoss, roboticPriorsLoss, tripletLoss,rewardModelLoss, \
    rewardPriorLoss, forwardModelLoss, inverseModelLoss, episodePriorLoss, vaeLoss, l1Loss
from losses.utils import findPriorsPairs
from pipeline import NAN_ERROR
from plotting.representation_plot import plotRepresentation, plt, plotImage
from preprocessing.data_loader import CustomDataLoader
from preprocessing.utils import deNormalize
from utils import parseDataFolder, printYellow, createFolder, detachToNumpy, input

from .modules import SRLModules, SRLModulesSplit
from .priors import Discriminator

MAX_BATCH_SIZE_GPU = 512  # For plotting, max batch_size before having memory issues

DISPLAY_PLOTS = True
EPOCH_FLAG = 1  # Plot every 1 epoch
BATCH_SIZE = 256  #
NOISE_STD = 1e-6  # To avoid NaN (states must be different)
VALIDATION_SIZE = 0.2  # 20% of training data for validation
N_WORKERS = 4

# Experimental: episode independent prior
BALANCED_SAMPLING = False  # Whether to do Uniform (default) or balanced sampling



def observationsGenerator(observations, device, batch_size=64):
    """
    Python generator to avoid out of memory issues
    when predicting states for all the observations
    :param observations: (torch tensor)
    :param batch_size: (int)
    :param device: (pytorch device)
    """
    n_minibatches = len(observations) // batch_size + 1
    for i in range(n_minibatches):
        start_idx, end_idx = batch_size * i, batch_size * (i + 1)
        obs_var = observations[start_idx:end_idx].set_grad_enabled(False)
        obs_var = obs_var.to(device)
        yield obs_var


class BaseLearner(object):
    """
    Base class for a method that learn a state representation
    from observations
    :param state_dim: (int)
    :param batch_size: (int)
    :param seed: (int)
    :param cuda: (bool)
    """

    def __init__(self, state_dim, batch_size, seed=1, cuda=False):
        super(BaseLearner, self).__init__()
        self.state_dim = state_dim
        self.batch_size = batch_size
        self.model = None
        self.seed = seed
        # Seed the random generator
        np.random.seed(seed)
        th.manual_seed(seed)
        if cuda:
            th.cuda.manual_seed(seed)

        self.device = th.device("cuda" if th.cuda.is_available() and cuda else "cpu")

    def _predFn(self, observations, restore_train=True):
        """
        Predict states in test mode given observations
        :param observations: (PyTorch Tensor)
        :param restore_train: (bool) whether to restore training mode after prediction
        :return: (numpy tensor)
        """
        # Switch to test mode
        self.model.eval()
        states = self.model.getStates(observations)
        if restore_train:
            # Restore training mode
            self.model.train()
        # Move the tensor back to the cpu
        return detachToNumpy(states)

    def predStates(self, observations):
        """
        Predict states for given observations
        WARNING: you should use _batchPredStates
        if observations tensor is large to avoid memory issues
        :param observations: (numpy tensor)
        :return: (numpy tensor)
        """
        observations = observations.astype(np.float32)
        with th.no_grad():
            obs_var = th.from_numpy(observations)
            obs_var = obs_var.to(self.device)
            states = self._predFn(obs_var, restore_train=False)
        return states

    def _batchPredStates(self, observations):
        """
        Predict states using minibatches to avoid memory issues
        :param observations: (numpy tensor)
        :return: (numpy tensor)
        """
        predictions = []
        for obs_var in observationsGenerator(th.from_numpy(observations), self.device, MAX_BATCH_SIZE_GPU):
            predictions.append(self._predFn(obs_var))
        return np.concatenate(predictions, axis=0)

    def predStatesWithDataLoader(self, data_loader, restore_train=False):
        """
        Predict states using minibatches to avoid memory issues
        :param data_loader: (Baxter Data Loader object)
        :param restore_train: (bool) restore train mode (model + dataLoader) after predicting states
        :return: (numpy tensor)
        """
        # Switch to test mode and reset the iterator
        data_loader.testMode()
        predictions = []
        for obs_var in data_loader:
            obs_var = obs_var.to(self.device)
            predictions.append(self._predFn(obs_var, restore_train))
        # Switch back to train mode
        if restore_train:
            data_loader.trainMode()
        return np.concatenate(predictions, axis=0)

    def learn(self, *args, **kwargs):
        """
        Function called to learn a state representation
        it returns the learned states for the given observations
        """
        raise NotImplementedError("Learn method not implemented")

    @staticmethod
    def saveStates(states, images_path, rewards, log_folder, name=""):
        """
        Save learned states to json and npz files
        :param states: (numpy array)
        :param images_path: ([str])
        :param rewards: (rewards)
        :param log_folder: (str)
        :param name: (str)
        """
        print("Saving image path to state representation")
        image_to_state = {path: list(map(str, state)) for path, state in zip(images_path, states)}
        with open("{}/image_to_state{}.json".format(log_folder, name), 'w') as f:
            json.dump(image_to_state, f, sort_keys=True)
        print("Saving states and rewards")
        states_rewards = {'states': states, 'rewards': rewards}
        np.savez('{}/states_rewards{}.npz'.format(log_folder, name), **states_rewards)


class SRL4robotics(BaseLearner):
    """
    :param state_dim: (int)
    :param model_type: (str) one of "resnet", "mlp" or "custom_cnn"
    :param log_folder: (str)
    :param seed: (int)
    :param learning_rate: (float)
    :param l1_reg: (float)
    :param cuda: (bool)
    :param multi_view: (bool)
    :param losses: ([str])
    :param n_actions: (int)
    :param beta: (float)
    """

    def __init__(self, state_dim, model_type="resnet", log_folder="logs/default",
                 seed=1, learning_rate=0.001, l1_reg=0.0, cuda=False,
                 multi_view=False, losses=None, n_actions=6, beta=1, split_index=-1):

        super(SRL4robotics, self).__init__(state_dim, BATCH_SIZE, seed, cuda)

        self.multi_view = multi_view
        self.losses = losses
        self.dim_action = n_actions
        self.beta = beta
        if model_type in ["linear", "mlp", "resnet", "custom_cnn"] \
                or "autoencoder" in losses or "vae" in losses:
            self.use_forward_loss = "forward" in losses
            self.use_inverse_loss = "inverse" in losses
            self.use_reward_loss = "reward" in losses
            self.no_priors = "priors" not in losses
            self.episode_prior = "episode-prior" in losses
            self.reward_prior = "reward-prior" in losses
            self.use_autoencoder = "autoencoder" in losses
            self.use_vae = "vae" in losses
            self.use_triplets = "triplet" in self.losses
            if split_index > 0:
                self.model = SRLModulesSplit(state_dim=self.state_dim, action_dim=self.dim_action, model_type=model_type,
                                        cuda=cuda, losses=losses, split_index=split_index)
            else:
                self.model = SRLModulesSplit(state_dim=self.state_dim, action_dim=self.dim_action, model_type=model_type,
                                        cuda=cuda, losses=losses)
        else:
            raise ValueError("Unknown model: {}".format(model_type))
        print("Using {} model".format(model_type))

        self.device = th.device("cuda" if th.cuda.is_available() and cuda else "cpu")

        if self.episode_prior:
            self.discriminator = Discriminator(2 * self.state_dim).to(self.device)

        self.model = self.model.to(self.device)

        learnable_params = [param for param in self.model.parameters() if param.requires_grad]

        if self.episode_prior:
            learnable_params += [p for p in self.discriminator.parameters()]

        self.optimizer = th.optim.Adam(learnable_params, lr=learning_rate)
        self.l1_reg = l1_reg
        self.log_folder = log_folder
        self.model_type = model_type

    def learn(self, images_path, actions, rewards, episode_starts):
        """
        Learn a state representation
        :param images_path: (numpy 1D array)
        :param actions: (numpy matrix)
        :param rewards: (numpy 1D array)
        :param episode_starts: (numpy 1D array) boolean array
                                the ith index is True if one episode starts at this frame
        :return: (numpy tensor) the learned states for the given observations
        """

        # PREPARE DATA -------------------------------------------------------------------------------------------------
        # here, we organize the data into minibatches
        # and find pairs for the respective loss terms

        num_samples = images_path.shape[0] - 1  # number of samples

        # indices for all time steps where the episode continues
        indices = np.array([i for i in range(num_samples) if not episode_starts[i + 1]], dtype='int64')
        np.random.shuffle(indices)

        # split indices into minibatches. minibatchlist is a list of lists; each
        # list is the id of the observation preserved through the training
        minibatchlist = [np.array(sorted(indices[start_idx:start_idx + self.batch_size]))
                         for start_idx in range(0, len(indices) - self.batch_size + 1, self.batch_size)]

        if len(minibatchlist[-1]) < self.batch_size:
            printYellow("Removing last minibatch of size {} < batch_size".format(len(minibatchlist[-1])))
            del minibatchlist[-1]

        # Number of minibatches used for validation:
        n_val_batches = np.round(VALIDATION_SIZE * len(minibatchlist)).astype(np.int64)
        val_indices = np.random.permutation(len(minibatchlist))[:n_val_batches]
        # Print some info
        print("{} minibatches for training, {} samples".format(len(minibatchlist) - n_val_batches,
                                                               (len(minibatchlist) - n_val_batches) * BATCH_SIZE))
        print("{} minibatches for validation, {} samples".format(n_val_batches, n_val_batches * BATCH_SIZE))
        assert n_val_batches > 0, "Not enough sample to create a validation set"

        # Stats about actions
        action_set = set(actions)
        n_actions = int(np.max(actions) + 1)
        print("{} unique actions / {} actions".format(len(action_set), n_actions))
        n_pairs_per_action = np.zeros(n_actions, dtype=np.int64)
        n_obs_per_action = np.zeros(n_actions, dtype=np.int64)

        for i in range(n_actions):
            n_obs_per_action[i] = np.sum(actions == i)

        print("Number of observations per action")
        print(n_obs_per_action)

        dissimilar_pairs, same_actions_pairs = None, None
        if not self.no_priors:
            dissimilar_pairs, same_actions_pairs = findPriorsPairs(self.batch_size, minibatchlist, actions, rewards,
                                                                   n_actions, n_pairs_per_action)


        if self.episode_prior:
            idx_to_episode = {idx: episode_idx for idx, episode_idx in enumerate(np.cumsum(episode_starts))}
            minibatch_episodes = [[idx_to_episode[i] for i in minibatch] for minibatch in minibatchlist]

        data_loader = CustomDataLoader(minibatchlist, images_path,
                                       cache_capacity=100, multi_view=self.multi_view, n_workers=N_WORKERS,
                                       use_triplets=self.use_triplets)
        # TRAINING -----------------------------------------------------------------------------------------------------
        loss_history = defaultdict(list)

        loss_manager = LossManager(self.model, self.l1_reg, loss_history)

        best_error = np.inf
        best_model_path = "{}/srl_model.pth".format(self.log_folder)
        self.model.train()
        start_time = time.time()

        for epoch in range(N_EPOCHS):
            # In each epoch, we do a full pass over the training data:
            epoch_loss, epoch_batches = 0, 0
            val_loss = 0
            pbar = tqdm(total=len(minibatchlist))
            data_loader.resetAndShuffle()

            for minibatch_num, (minibatch_idx, obs, next_obs) in enumerate(data_loader):
                obs, next_obs = obs.to(self.device), next_obs.to(self.device)
                self.optimizer.zero_grad()
                loss_manager.resetLosses()

                decoded_obs, decoded_next_obs = None, None

                # Predict states given observations as in Time Contrastive Network (Triplet Loss) [Sermanet et al.]
                if self.use_triplets:
                    states, positive_states, negative_states = self.model.forward_triplets(obs[:, :3:, :, :],
                                                                                           obs[:, 3:6, :, :],
                                                                                           obs[:, 6:, :, :])

                    next_states, next_positive_states, next_negative_states = self.model.forward_triplets(
                        next_obs[:, :3:, :, :],
                        next_obs[:, 3:6, :, :],
                        next_obs[:, 6:, :, :])
                elif self.use_autoencoder:
                    (states, decoded_obs), (next_states, decoded_next_obs) = self.model(obs), self.model(next_obs)
                elif self.use_vae:
                    (decoded_obs, mu, logvar), (next_decoded_obs, next_mu, next_logvar) = self.model(obs), \
                                                                                          self.model(next_obs)
                    states, next_states = self.model.getStates(obs), self.model.getStates(next_obs)
                else:
                    states, next_states = self.model(obs), self.model(next_obs)

                # Actions associated to the observations of the current minibatch
                actions_st = actions[minibatchlist[minibatch_idx]]
                actions_st = th.from_numpy(actions_st).view(-1, 1).requires_grad_(False).to(self.device)

                # L1 regularization
                if loss_manager.l1_coeff > 0:
                    l1Loss(loss_manager.reg_params, loss_manager.l1_coeff, loss_manager)

                if not self.no_priors:
                    roboticPriorsLoss(states, next_states, minibatch_idx=minibatch_idx,
                                        dissimilar_pairs=dissimilar_pairs, same_actions_pairs=same_actions_pairs,
                                        weight=1., loss_manager=loss_manager)

                if self.use_forward_loss:
                    next_states_pred = self.model.forwardModel(states, actions_st)
                    forwardModelLoss(next_states_pred, next_states, weight=1., loss_manager=loss_manager)

                if self.use_inverse_loss:
                    actions_pred = self.model.inverseModel(states, next_states)
                    inverseModelLoss(actions_pred, actions_st, weight=1, loss_manager=loss_manager)

                if self.use_reward_loss:
                    rewards_st = rewards[minibatchlist[minibatch_idx]].copy()
                    # Removing negative reward
                    rewards_st[rewards_st == -1] = 0
                    rewards_st = th.from_numpy(rewards_st).to(self.device)
                    rewards_pred = self.model.rewardModel(states)
                    rewardModelLoss(rewards_pred, rewards_st.long(), weight=2.5, loss_manager=loss_manager)

                if self.use_autoencoder:
                    autoEncoderLoss(obs, decoded_obs, next_obs, decoded_next_obs, weight=1, loss_manager=loss_manager)
                if self.use_vae:
                    vaeLoss(decoded_obs, next_decoded_obs, obs, next_obs, mu, next_mu, logvar, next_logvar, weight=0.5e-6,
                            loss_manager=loss_manager, beta=self.beta)
                if self.reward_prior:
                    rewards_st = rewards[minibatchlist[minibatch_idx]]
                    rewards_st = th.from_numpy(rewards_st).float().view(-1, 1).to(self.device)
                    rewardPriorLoss(states, rewards_st, weight=10., loss_manager=loss_manager)

                if self.episode_prior:
                    episodePriorLoss(minibatch_idx, minibatch_episodes, states, self.discriminator,
                                     BALANCED_SAMPLING, weight=1, loss_manager=loss_manager)
                if self.use_triplets:
                    tripletLoss(states, positive_states, negative_states, weight=1.0, loss_manager=loss_manager, alpha=0.2)
                # Compute weighted average of losses
                loss_manager.updateLossHistory()
                loss = loss_manager.computeTotalLoss()

                # We have to call backward in both train/val
                # to avoid memory error
                loss.backward()
                if minibatch_idx in val_indices:
                    val_loss += loss.item()
                    # We do not optimize on validation data
                    # so optimizer.step() is not called
                else:
                    self.optimizer.step()
                    epoch_loss += loss.item()
                    epoch_batches += 1
                pbar.update(1)
            pbar.close()

            train_loss = epoch_loss / float(epoch_batches)
            val_loss /= float(n_val_batches)
            # Even if loss_history is modified by LossManager
            # we make it explicit
            loss_history = loss_manager.loss_history
            loss_history['train_loss'].append(train_loss)
            loss_history['val_loss'].append(val_loss)
            for key in loss_history.keys():
                if key in ['train_loss', 'val_loss']:
                    continue
                loss_history[key][-1] /= epoch_batches
                if epoch + 1 < N_EPOCHS:
                    loss_history[key].append(0)

            # Save best model
            if val_loss < best_error:
                best_error = val_loss
                th.save(self.model.state_dict(), best_model_path)

            if np.isnan(train_loss):
                print("NaN Loss, consider increasing NOISE_STD in the gaussian noise layer")
                sys.exit(NAN_ERROR)

            # Then we print the results for this epoch:
            if (epoch + 1) % EPOCH_FLAG == 0:
                print("Epoch {:3}/{}, train_loss:{:.4f} val_loss:{:.4f}".format(epoch + 1, N_EPOCHS, train_loss,
                                                                                val_loss))
                print("{:.2f}s/epoch".format((time.time() - start_time) / (epoch + 1)))
                if DISPLAY_PLOTS:
                    with th.no_grad():
                        # Optionally plot the current state space
                        plotRepresentation(self.predStatesWithDataLoader(data_loader, restore_train=True), rewards,
                                           add_colorbar=epoch == 0,
                                           name="Learned State Representation (Training Data)")

                        if self.use_autoencoder or self.use_vae:
                            # Plot Reconstructed Image
                            if obs[0].shape[0] == 3:  # RGB
                                plotImage(deNormalize(detachToNumpy(obs[0])), "Input Image (Train)")
                                plotImage(deNormalize(detachToNumpy(decoded_obs[0])), "Reconstructed Image")

                            elif obs[0].shape[0] % 3 == 0:  # Multi-RGB
                                for k in range(obs[0].shape[0] // 3):
                                    plotImage(deNormalize(detachToNumpy(obs[0][k * 3:(k + 1) * 3, :, :])),
                                              "Input Image {} (Train)".format(k + 1))
                                    plotImage(deNormalize(detachToNumpy(decoded_obs[0][k * 3:(k + 1) * 3, :, :])),
                                              "Reconstructed Image {}".format(k + 1))

        if DISPLAY_PLOTS:
            plt.close("Learned State Representation (Training Data)")

        # Load best model before predicting states
        self.model.load_state_dict(th.load(best_model_path))

        print("Predicting states for all the observations...")
        # return predicted states for training observations
        with th.no_grad():
            pred_states = self.predStatesWithDataLoader(data_loader, restore_train=False)
        return loss_history, pred_states
