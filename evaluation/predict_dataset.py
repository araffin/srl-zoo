from __future__ import print_function, division, absolute_import

import argparse

import numpy as np
import torch as th

from models.learner import SRL4robotics, MAX_BATCH_SIZE_GPU
from preprocessing.data_loader import DataLoader

VALID_MODELS = ["forward", "inverse", "reward", "priors", "episode-prior", "reward-prior", "triplet",
                "autoencoder", "vae"]

parser = argparse.ArgumentParser(description="Predict states on a dataset for a trained model")
parser.add_argument('-i', '--log-dir', default='', type=str, help='Directory to load model', required=True)
parser.add_argument('--name-suffix', default='_test', type=str, help='Suffix to add to the filename of the output file')
parser.add_argument('--no-cuda', default=False, action="store_true", help="Disable CUDA")
parser.add_argument('-n', '--n-samples', type=int, default=-1,
                    help='Limit size (number of samples) for predicting the states (default: -1)')
args = parser.parse_args()

if not args.log_dir.endswith('/'):
    args.log_dir += '/'


srl_model, exp_config = SRL4robotics.loadSavedModel(args.log_dir, VALID_MODELS, cuda=not args.no_cuda)

images_path = np.load("data/{}/ground_truth.npz".format(exp_config['data-folder']))['images_path']
rewards = np.load("data/{}/preprocessed_data.npz".format(exp_config['data-folder']))['rewards']
limit = args.n_samples if args.n_samples > 0 else len(images_path)

images_path = images_path[:limit]
rewards = rewards[:limit]


minibatchlist = DataLoader.createTestMinibatchList(len(images_path), MAX_BATCH_SIZE_GPU)

data_loader = DataLoader(minibatchlist, images_path, n_workers=4, multi_view=exp_config.get('multi-view', False),
                         use_triplets='triplet' in exp_config['losses'], max_queue_len=1, is_training=False,
                         apply_occlusion=srl_model.use_dae, occlusion_percentage=srl_model.occlusion_percentage)


print("Predicting states for {} observations...".format(len(images_path)))
with th.no_grad():
    learned_states = srl_model.predStatesWithDataLoader(data_loader)

srl_model.saveStates(learned_states, images_path, rewards, args.log_dir, name=args.name_suffix)

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
