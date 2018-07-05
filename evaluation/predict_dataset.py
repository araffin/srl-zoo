from __future__ import print_function, division, absolute_import

import argparse

import numpy as np
import torch as th

from models.learner import SRL4robotics, MAX_BATCH_SIZE_GPU
from preprocessing.data_loader import CustomDataLoader

VALID_MODELS = ["forward", "inverse", "reward", "priors", "episode-prior", "reward-prior", "triplet",
                "autoencoder", "vae"]

parser = argparse.ArgumentParser(description="Predict states on a dataset for a trained model")
parser.add_argument('--log-dir', default='', type=str, help='Directory to load model')
parser.add_argument('--no-cuda', default=False, action="store_true", help="Disable CUDA")
parser.add_argument('-n', '--n-samples', type=int, default=-1,
                    help='Limit size (number of samples) for predicting the states (default: -1)')
args = parser.parse_args()


srl_model, exp_config = SRL4robotics.loadSavedModel(args.log_dir, VALID_MODELS, cuda=not args.no_cuda)

images_path = np.load("data/{}/ground_truth.npz".format(exp_config['data-folder']))['images_path']
rewards = np.load("data/{}/preprocessed_data.npz".format(exp_config['data-folder']))['rewards']
limit = args.n_samples if args.n_samples > 0 else len(images_path)

images_path = images_path[:limit]
rewards = rewards[:limit]


indices = np.arange(len(images_path), dtype='int64')
minibatchlist = [np.array(sorted(indices[start_idx:start_idx + MAX_BATCH_SIZE_GPU]))
                 for start_idx in range(0, len(indices) - MAX_BATCH_SIZE_GPU + 1, MAX_BATCH_SIZE_GPU)]

data_loader = CustomDataLoader(minibatchlist, images_path,
                               cache_capacity=0, multi_view=exp_config.get('multi-view', False), n_workers=4,
                               use_triplets='triplet' in exp_config['losses'])


print("Predicting states for {} observations...".format(len(images_path)))
with th.no_grad():
    learned_states = srl_model.predStatesWithDataLoader(data_loader, restore_train=False)

srl_model.saveStates(learned_states, images_path, rewards, args.log_dir, name="_test")

# Print and save some stats
mean_states = np.mean(learned_states, axis=0)
std_states = np.std(learned_states, axis=0)
min_states = np.min(learned_states, axis=0)
max_states = np.max(learned_states, axis=0)

print("Mean:", mean_states)
print("Std:", std_states)
print("Min:", min_states)
print("Max:", max_states)

print("Stats saved (states_stats.npz)")
np.savez(args.log_dir + '/states_stats.npz', **{'mean': mean_states, 'std': std_states, 'min': min_states, 'max': max_states})
