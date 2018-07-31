from __future__ import print_function, division, absolute_import

import os
import json
import sys
import time
from collections import defaultdict

import numpy as np
import torch as th
from tqdm import tqdm

from losses.losses import LossManager, autoEncoderLoss, roboticPriorsLoss, tripletLoss, rewardModelLoss, \
    rewardPriorLoss, forwardModelLoss, inverseModelLoss, episodePriorLoss, vaeLoss, l1Loss, l2Loss
from losses.utils import findPriorsPairs
from pipeline import NAN_ERROR
from plotting.representation_plot import plotRepresentation, plt, plotImage
from preprocessing.data_loader import DataLoader
from preprocessing.utils import deNormalize
from utils import printRed, detachToNumpy
from .modules import SRLModules, SRLModulesSplit
from .priors import Discriminator

MAX_BATCH_SIZE_GPU = 256  # For plotting, max batch_size before having memory issues
EPOCH_FLAG = 1  # Plot every 1 epoch
N_WORKERS = 4

# The following variables are defined using arguments of the main script train.py
DISPLAY_PLOTS = True
BATCH_SIZE = 256
N_EPOCHS = 1
VALIDATION_SIZE = 0.2  # 20% of training data for validation
# Experimental: episode independent prior
BALANCED_SAMPLING = False  # Whether to do Uniform (default) or balanced sampling



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
        self.use_dae = False
        # Seed the random generator
        np.random.seed(seed)
        th.manual_seed(seed)
        if cuda:
            # Make CuDNN Determinist
            th.backends.cudnn.deterministic = True
            th.cuda.manual_seed(seed)

        self.device = th.device("cuda" if th.cuda.is_available() and cuda else "cpu")

    def _predFn(self, observations):
        """
        Predict states in test mode given observations
        :param observations: (PyTorch Tensor)
        :return: (numpy tensor)
        """
        # Move the tensor back to the cpu
        return detachToNumpy(self.model.getStates(observations))

    def predStatesWithDataLoader(self, data_loader):
        """
        Predict states using minibatches to avoid memory issues
        :param data_loader: (Baxter Data Loader object)
        :return: (numpy tensor)
        """
        predictions = []
        for (obs_var, obs_noisy_var) in data_loader:

            obs_var = obs_noisy_var.to(self.device) if self.use_dae else obs_var.to(self.device)
            predictions.append(self._predFn(obs_var))

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
        print("Saving image path to state representation (image_to_state{}.json)".format(name))

        image_to_state = {path: list(map(str, state)) for path, state in zip(images_path, states)}

        with open("{}/image_to_state{}.json".format(log_folder, name), 'w') as f:
            json.dump(image_to_state, f, sort_keys=True)

        print("Saving states and rewards (states_rewards{}.npz)".format(name))

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
    :param l2_reg: (float)
    :param cuda: (bool)
    :param multi_view: (bool)
    :param losses: ([str])
    :param n_actions: (int)
    :param beta: (float)
    :param path_denoiser: path to pre-trained DAE when using perceptual loss (str)
    :param occlusion_percentage: max percentage of occlusion when using DAE (float)
    """

    def __init__(self, state_dim, model_type="resnet", log_folder="logs/default",
                 seed=1, learning_rate=0.001, l1_reg=0.0, l2_reg=0.0, cuda=False,
                 multi_view=False, losses=None, losses_weights_dict=None, n_actions=6, beta=1,
                 split_index=-1, path_denoiser=None, occlusion_percentage=None):

        super(SRL4robotics, self).__init__(state_dim, BATCH_SIZE, seed, cuda)

        self.multi_view = multi_view
        self.losses = losses
        self.dim_action = n_actions
        self.beta = beta
        # For splitting representation
        self.split_index = split_index
        self.state_dim_first_split = split_index if split_index > 0 else state_dim
        self.state_dim_second_split = state_dim - split_index if split_index > 0 else state_dim

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
            self.perceptual_similarity_loss = "perceptual" in self.losses
            self.use_dae = "dae" in self.losses
            self.path_denoiser = path_denoiser
            if split_index > 0:
                self.model = SRLModulesSplit(state_dim=self.state_dim, action_dim=self.dim_action,
                                             model_type=model_type,
                                             cuda=cuda, losses=losses, split_index=split_index)
            else:
                self.model = SRLModules(state_dim=self.state_dim, action_dim=self.dim_action, model_type=model_type,
                                        cuda=cuda, losses=losses)
        else:
            raise ValueError("Unknown model: {}".format(model_type))
        print("Using {} model".format(model_type))
        self.cuda = cuda
        self.device = th.device("cuda" if th.cuda.is_available() and cuda else "cpu")

        if self.episode_prior:
            self.discriminator = Discriminator(2 * self.state_dim).to(self.device)

        self.model = self.model.to(self.device)

        learnable_params = [param for param in self.model.parameters() if param.requires_grad]

        if self.episode_prior:
            learnable_params += [p for p in self.discriminator.parameters()]

        self.optimizer = th.optim.Adam(learnable_params, lr=learning_rate)
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.log_folder = log_folder
        self.model_type = model_type

        self.losses_weights_dict = {"forward": 1.0, "inverse": 1.0, "reward": 1.0, "priors": 1.0,
                                    "episode-prior": 1.0, "reward-prior": 10, "triplet": 1.0,
                                    "autoencoder": 1.0, "vae": 1.0, "perceptual": 1e-6, "dae": 1.0}
        self.occlusion_percentage = occlusion_percentage

        if losses_weights_dict is not None:
            self.losses_weights_dict.update(losses_weights_dict)
        print("\nYour are using the following weights for uour lossses: ", self.losses_weights_dict, '\n')
        if self.use_dae and self.occlusion_percentage is not None:
            print("Using a maximum occlusion surface of {}".format(str(self.occlusion_percentage)))


    @staticmethod
    def loadSavedModel(log_folder, valid_models, cuda=True):
        """
        :param log_folder: (str)
        :param valid_models: ([str])
        :param cuda: (bool)
        :return: (SRL4robotics object, dict)
        """
        # Sanity checks
        assert os.path.exists(log_folder), "Error: folder '{}' does not exist".format(log_folder)
        assert os.path.exists(log_folder + "exp_config.json"), \
            "Error: could not find 'exp_config.json' in '{}'".format(log_folder)
        assert os.path.exists(log_folder + "srl_model.pth"), \
            "Error: could not find 'srl_model.pth' in '{}'".format(log_folder)

        with open(log_folder + 'exp_config.json', 'r') as f:
            exp_config = json.load(f)

        state_dim = exp_config['state-dim']
        losses = exp_config['losses']
        n_actions = exp_config['n_actions']
        model_type = exp_config['model-type']
        multi_view = exp_config.get('multi-view', False)
        split_index = exp_config.get('split-index', -1)
        model_path = log_folder + 'srl_model.pth'

        difference = set(losses).symmetric_difference(valid_models)
        assert set(losses).intersection(valid_models) != set(), "Error: Not supported losses " + ", ".join(difference)

        srl_model = SRL4robotics(state_dim, model_type=model_type, cuda=cuda, multi_view=multi_view,
                                 losses=losses, n_actions=n_actions, split_index=split_index)
        srl_model.model.load_state_dict(th.load(model_path))

        return srl_model, exp_config

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

        test_minibatchlist = DataLoader.createTestMinibatchList(len(images_path), MAX_BATCH_SIZE_GPU)

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

        if self.use_vae and self.perceptual_similarity_loss and self.path_denoiser is not None:

            self.denoiser = SRLModules(state_dim=200, action_dim=self.dim_action, model_type="custom_cnn",
                       cuda=self.cuda, losses=["dae"])
            self.denoiser.eval()
            self.device = th.device("cuda" if th.cuda.is_available() and self.cuda else "cpu")
            self.denoiser = self.denoiser.to(self.device)
            self.denoiser.load_state_dict(th.load(self.path_denoiser))
            for param in self.denoiser.parameters():
                param.requires_grad = False

            

        if self.episode_prior:
            idx_to_episode = {idx: episode_idx for idx, episode_idx in enumerate(np.cumsum(episode_starts))}
            minibatch_episodes = [[idx_to_episode[i] for i in minibatch] for minibatch in minibatchlist]

        data_loader = DataLoader(minibatchlist, images_path, n_workers=N_WORKERS, multi_view=self.multi_view,
                                 use_triplets=self.use_triplets, is_training=True, apply_occlusion=self.use_dae,
                                       occlusion_percentage=self.occlusion_percentage)
        test_data_loader = DataLoader(test_minibatchlist, images_path, n_workers=N_WORKERS, multi_view=self.multi_view,
                                      use_triplets=self.use_triplets, max_queue_len=1, is_training=False,
                                      apply_occlusion=self.use_dae, occlusion_percentage=self.occlusion_percentage)
        # TRAINING -----------------------------------------------------------------------------------------------------
        loss_history = defaultdict(list)

        loss_manager = LossManager(self.model, loss_history)

        best_error = np.inf
        best_model_path = "{}/srl_model.pth".format(self.log_folder)
        start_time = time.time()

        for epoch in range(N_EPOCHS):
            # In each epoch, we do a full pass over the training data:
            epoch_loss, epoch_batches = 0, 0
            val_loss = 0
            pbar = tqdm(total=len(minibatchlist))

            for minibatch_num, (minibatch_idx, obs, next_obs, noisy_obs, next_noisy_obs) in enumerate(data_loader):

                validation_mode = minibatch_idx in val_indices
                if validation_mode:
                    self.model.eval()
                else:
                    self.model.train()

                if self.use_dae:
                    noisy_obs = noisy_obs.to(self.device)
                    next_noisy_obs = next_noisy_obs.to(self.device)
                obs, next_obs = obs.to(self.device), next_obs.to(self.device)

                self.optimizer.zero_grad()
                loss_manager.resetLosses()

                decoded_obs, decoded_next_obs = None, None
                states_denoiser = None
                states_denoiser_predicted = None
                next_states_denoiser = None
                next_states_denoiser_predicted = None

                # Predict states given observations as in Time Contrastive Network (Triplet Loss) [Sermanet et al.]
                if self.use_triplets:
                    states, positive_states, negative_states = self.model.forwardTriplets(obs[:, :3:, :, :],
                                                                                          obs[:, 3:6, :, :],
                                                                                          obs[:, 6:, :, :])

                    next_states, next_positive_states, next_negative_states = self.model.forwardTriplets(
                        next_obs[:, :3:, :, :],
                        next_obs[:, 3:6, :, :],
                        next_obs[:, 6:, :, :])
                elif self.use_autoencoder:
                    (states, decoded_obs), (next_states, decoded_next_obs) = self.model(obs), self.model(next_obs)

                elif self.use_dae:
                    (states, decoded_obs), (next_states, decoded_next_obs) = \
                        self.model(noisy_obs), self.model(next_noisy_obs)

                elif self.use_vae:
                    (decoded_obs, mu, logvar), (next_decoded_obs, next_mu, next_logvar) = self.model(obs), \
                                                                                          self.model(next_obs)
                    states, next_states = self.model.getStates(obs), self.model.getStates(next_obs)

                    if self.perceptual_similarity_loss:
                        # Predictions for the perceptual similarity loss as in DARLA
                        # https://arxiv.org/pdf/1707.08475.pdf
                        (states_denoiser, decoded_obs_denoiser), (next_states_denoiser, decoded_next_obs_denoiser) = \
                            self.denoiser(obs), self.denoiser(next_obs)

                        (states_denoiser_predicted, decoded_obs_denoiser_predicted) = self.denoiser(decoded_obs)
                        (next_states_denoiser_predicted,
                         decoded_next_obs_denoiser_predicted) = self.denoiser(next_decoded_obs)
                else:
                    states, next_states = self.model(obs), self.model(next_obs)

                # Actions associated to the observations of the current minibatch
                actions_st = actions[minibatchlist[minibatch_idx]]
                actions_st = th.from_numpy(actions_st).view(-1, 1).requires_grad_(False).to(self.device)

                # L1 regularization
                if self.l1_reg > 0:
                    l1Loss(loss_manager.reg_params, self.l1_reg, loss_manager)

                if self.l2_reg > 0:
                    l2Loss(loss_manager.reg_params, self.l2_reg, loss_manager)

                if not self.no_priors:
                    roboticPriorsLoss(states, next_states, minibatch_idx=minibatch_idx,
                                        dissimilar_pairs=dissimilar_pairs, same_actions_pairs=same_actions_pairs,
                                        weight=self.losses_weights_dict['priors'], loss_manager=loss_manager)

                if self.use_forward_loss:
                    next_states_pred = self.model.forwardModel(states, actions_st)
                    forwardModelLoss(next_states_pred, next_states[:, -self.state_dim_second_split:],
                                     weight=self.losses_weights_dict['forward'],
                                     loss_manager=loss_manager)

                if self.use_inverse_loss:
                    actions_pred = self.model.inverseModel(states, next_states)
                    inverseModelLoss(actions_pred, actions_st, weight=self.losses_weights_dict['inverse'],
                                     loss_manager=loss_manager)

                if self.use_reward_loss:
                    rewards_st = rewards[minibatchlist[minibatch_idx]].copy()
                    # Removing negative reward
                    rewards_st[rewards_st == -1] = 0
                    rewards_st = th.from_numpy(rewards_st).to(self.device)
                    rewards_pred = self.model.rewardModel(states, next_states)
                    rewardModelLoss(rewards_pred, rewards_st.long(), weight=self.losses_weights_dict['reward'],
                                    loss_manager=loss_manager)

                if self.use_autoencoder or self.use_dae:
                    loss_type = "dae" if self.use_dae else "autoencoder"
                    autoEncoderLoss(obs, decoded_obs, next_obs, decoded_next_obs,
                                    weight=self.losses_weights_dict[loss_type], loss_manager=loss_manager)

                if self.use_vae:
                    vaeLoss(decoded_obs, next_decoded_obs, obs, next_obs, mu, next_mu, logvar, next_logvar,
                            weight=self.losses_weights_dict['vae'],
                            loss_manager=loss_manager, beta=self.beta,
                            perceptual_similarity_loss=self.perceptual_similarity_loss,
                            encoded_real=states_denoiser, encoded_prediction=states_denoiser_predicted,
                            next_encoded_real=next_states_denoiser,
                            next_encoded_prediction=next_states_denoiser_predicted,
                            weight_perceptual=self.losses_weights_dict['perceptual'])

                if self.reward_prior:
                    rewards_st = rewards[minibatchlist[minibatch_idx]]
                    rewards_st = th.from_numpy(rewards_st).float().view(-1, 1).to(self.device)
                    rewardPriorLoss(states, rewards_st, weight=self.losses_weights_dict['reward-prior'],
                                    loss_manager=loss_manager)

                if self.episode_prior:
                    episodePriorLoss(minibatch_idx, minibatch_episodes, states, self.discriminator,
                                     BALANCED_SAMPLING, weight=self.losses_weights_dict['episode-prior'],
                                     loss_manager=loss_manager)
                if self.use_triplets:
                    tripletLoss(states, positive_states, negative_states, weight=self.losses_weights_dict['triplet'],
                                loss_manager=loss_manager, alpha=0.2)
                # Compute weighted average of losses
                loss_manager.updateLossHistory()
                loss = loss_manager.computeTotalLoss()

                # We have to call backward in both train/val
                # to avoid memory error
                loss.backward()
                if validation_mode:
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
                printRed("NaN Loss, consider increasing NOISE_STD in the gaussian noise layer")
                sys.exit(NAN_ERROR)

            # Then we print the results for this epoch:
            if (epoch + 1) % EPOCH_FLAG == 0:
                print("Epoch {:3}/{}, train_loss:{:.4f} val_loss:{:.4f}".format(epoch + 1, N_EPOCHS, train_loss,
                                                                                val_loss))
                print("{:.2f}s/epoch".format((time.time() - start_time) / (epoch + 1)))
                if DISPLAY_PLOTS:
                    with th.no_grad():
                        self.model.eval()
                        # Optionally plot the current state space
                        plotRepresentation(self.predStatesWithDataLoader(test_data_loader), rewards,
                                           add_colorbar=epoch == 0,
                                           name="Learned State Representation (Training Data)")

                        if self.use_autoencoder or self.use_vae or self.use_dae:
                            # Plot Reconstructed Image
                            if obs[0].shape[0] == 3:  # RGB
                                plotImage(deNormalize(detachToNumpy(obs[0])), "Input Image (Train)")
                                if self.use_dae:
                                    plotImage(deNormalize(detachToNumpy(noisy_obs[0])), "Noisy Input Image (Train)")
                                if self.perceptual_similarity_loss:
                                    plotImage(deNormalize(detachToNumpy(decoded_obs_denoiser[0])), "Reconstructed Image DAE")
                                    plotImage(deNormalize(detachToNumpy(decoded_obs_denoiser_predicted[0])), "Reconstructed Image predicted DAE")
                                plotImage(deNormalize(detachToNumpy(decoded_obs[0])), "Reconstructed Image")

                            elif obs[0].shape[0] % 3 == 0:  # Multi-RGB
                                for k in range(obs[0].shape[0] // 3):
                                    plotImage(deNormalize(detachToNumpy(obs[0][k * 3:(k + 1) * 3, :, :]), "image_net"),
                                              "Input Image {} (Train)".format(k + 1))
                                    if self.use_dae:
                                        plotImage(deNormalize(detachToNumpy(noisy_obs[0][k * 3:(k + 1) * 3, :, :])),
                                                  "Noisy Input Image (Train)".format(k + 1))

                                    plotImage(deNormalize(detachToNumpy(decoded_obs[0][k * 3:(k + 1) * 3, :, :])),
                                              "Reconstructed Image {}".format(k + 1))

        if DISPLAY_PLOTS:
            plt.close("Learned State Representation (Training Data)")

        # Load best model before predicting states
        self.model.load_state_dict(th.load(best_model_path))

        print("Predicting states for all the observations...")
        # return predicted states for training observations
        self.model.eval()
        with th.no_grad():
            pred_states = self.predStatesWithDataLoader(test_data_loader)
        pairs_loss_weight = [k for k in zip(loss_manager.names, loss_manager.weights)]
        return loss_history, pred_states, pairs_loss_weight
