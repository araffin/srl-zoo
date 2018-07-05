from __future__ import print_function, division, absolute_import

import argparse
import json
import os

import numpy as np
import torch as th

from models.learner import SRL4robotics, MAX_BATCH_SIZE_GPU
from preprocessing.data_loader import CustomDataLoader

VALID_MODELS = ["forward", "inverse", "reward", "priors", "episode-prior", "reward-prior", "triplet",
                "autoencoder", "vae"]

parser = argparse.ArgumentParser(description="Predict states on a dataset for a trained model")
parser.add_argument('--log-dir', default='', type=str, help='Directory to load model')
parser.add_argument('--no-cuda', default=False, action="store_true", help="Disable CUDA")
parser.add_argument('--n-samples', type=int, default=-1,
                    help='Limit size (number of samples) for predicting the states (default: -1)')
args = parser.parse_args()

# Sanity checks
assert os.path.exists(args.log_dir), "Error: folder '{}' does not exist".format(args.log_dir)
assert os.path.exists(args.log_dir + "exp_config.json"), \
    "Error: could not find 'exp_config.json' in '{}'".format(args.log_dir)
assert os.path.exists(args.log_dir + "srl_model.pth"), \
    "Error: could not find 'srl_model.pth' in '{}'".format(args.log_dir)

with open(args.log_dir + 'exp_config.json', 'r') as f:
    exp_config = json.load(f)

state_dim = exp_config['state-dim']
losses = exp_config['losses']
n_actions = exp_config['n_actions']
model_type = exp_config['model-type']
data_folder = exp_config['data-folder']
multi_view = exp_config.get('multi-view', False)
split_index = exp_config.get('split-index', -1)
model_path = args.log_dir + 'srl_model.pth'


difference = set(losses).symmetric_difference(VALID_MODELS)
assert set(losses).intersection(VALID_MODELS) != set(), "Error: Not supported losses " + ", ".join(difference)


images_path = np.load("data/{}/ground_truth.npz".format(data_folder))['images_path']
rewards = np.load("data/{}/preprocessed_data.npz".format(data_folder))['rewards']
limit = args.n_samples if args.n_samples > 0 else len(images_path)

images_path = images_path[:limit]
rewards = rewards[:limit]


indices = np.arange(len(images_path), dtype='int64')
minibatchlist = [np.array(sorted(indices[start_idx:start_idx + MAX_BATCH_SIZE_GPU]))
                 for start_idx in range(0, len(indices) - MAX_BATCH_SIZE_GPU + 1, MAX_BATCH_SIZE_GPU)]

data_loader = CustomDataLoader(minibatchlist, images_path,
                               cache_capacity=0, multi_view=multi_view, n_workers=4,
                               use_triplets='triplet' in losses)

srl = SRL4robotics(state_dim, model_type=model_type, cuda=not args.no_cuda, multi_view=multi_view,
                   losses=losses, n_actions=n_actions, split_index=split_index)
srl.model.load_state_dict(th.load(model_path))

print("Predicting states for {} observations...".format(len(images_path)))
with th.no_grad():
    learned_states = srl.predStatesWithDataLoader(data_loader, restore_train=False)

print("Saving states to {}")
srl.saveStates(learned_states, images_path, rewards, args.log_dir, name="_test")
