from __future__ import print_function, division, absolute_import

import argparse
import random
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

N_NEIGHBORS_PER_LINE = 5

# Init seaborn
sns.set()

parser = argparse.ArgumentParser(description='KNN plot and KNN MSE')
parser.add_argument('--log_folder', type=str, default="", required=True, help='Path to a log folder')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('-k', '--n_neighbors', type=int, default=5, help='Number of nearest neighbors (default: 5)')
parser.add_argument('-n', '--n_samples', type=int, default=5, help='Number of test samples (default: 10)')
args = parser.parse_args()

n_neighbors = args.n_neighbors
n_samples = args.n_samples
n_lines = (n_neighbors // N_NEIGHBORS_PER_LINE) + 1
random.seed(args.seed)

with open("{}/exp_config.json".format(args.log_folder), 'rb') as f:
    data_folder = json.load(f)['data_folder']

# Load ground truth and images path
ground_truth = np.load('data/{}/ground_truth.npz'.format(data_folder))
states = np.load('{}/states_rewards.npz'.format(args.log_folder))['states']

# TODO: relative states for moving button
true_states = ground_truth['arm_states']
images_path = ground_truth['images_path']

knn_path = '{}/NearestNeighbors'.format(args.log_folder)

print("Computing KNN... with k={}".format(args.n_neighbors))
# Do not consider the reference state as a neighbor
knn = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm='ball_tree').fit(states)
distances, neighbors_indices = knn.kneighbors(states)

print('\nUsing a random test set of images for KNN MSE evaluation...')
print('seed={}\n'.format(args.seed))

# Sample random images
images_indices = np.arange(len(images_path))
data = random.sample(zip(images_path, neighbors_indices, distances, images_indices), n_samples)
# Progressbar
pbar = tqdm(total=n_samples)
n_images, total_error = 0, 0
images_titles = []

for image_path, neigbour_indices, distance, image_idx in data:
    fig = plt.figure()
    fig.set_size_inches(60, 35)

    ref_coord = true_states[image_idx]
    record_folder = images_path[image_idx].split("/")[1]
    frame_name = images_path[image_idx].split("/")[-1].split(".")[0]
    # Add reference image
    # subplot: (i, j, k) i rows, j columns, k^th plot | n_plots: i * j
    ref_image = fig.add_subplot(n_lines + 1, 5, 3)
    img = Image.open("data/{}".format(image_path))
    plt.imshow(img)
    state_str = ", ".join(map(lambda x: '{:.3f}'.format(x), states[image_idx]))
    state_str = "[{}]".format(state_str)

    ref_image.axis('off')
    ref_image.set_title('{}/{}\n {}\n{}'.format(record_folder, frame_name, state_str, ref_coord))
    images_titles.append('{}/{}'.format(record_folder, frame_name))

    for i in range(0, n_neighbors):
        subplot = fig.add_subplot(n_lines + 1, 5, 6 + i)
        # Do not consider the reference state as a neighbor
        neighbor_idx = neigbour_indices[i + 1]
        image_path = images_path[neighbor_idx]
        neighbor_record_folder = image_path.split("/")[1]
        neighbor_frame_name = image_path.split("/")[-1].split(".")[0]

        img = Image.open("data/{}".format(image_path))
        plt.imshow(img)

        dist_str = 'd={:.4f}'.format(distance[i + 1])
        state_str = map(lambda x: '{:.3f}'.format(x), states[neighbor_idx])
        neighbor_coord = true_states[neighbor_idx]
        total_error += np.linalg.norm(neighbor_coord - ref_coord)
        n_images += 1
        title = '{}/{}\n {} {}\n{}'.format(neighbor_record_folder, neighbor_frame_name, state_str, dist_str,
                                           neighbor_coord)
        subplot.set_title(title)
        subplot.axis('off')

    plt.tight_layout()
    output_file = "{}/{}_{}.png".format(knn_path, record_folder, frame_name)

    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    pbar.update(1)

pbar.close()

# Mean MSE Error
mean_error = total_error / n_images
print("KNN MSE: {}".format(mean_error))

result_dict = {'images': images_titles, 'knn_mse': round(mean_error, 5)}

with open("{}/knn_mse.json".format(args.log_folder), 'wb') as f:
    json.dump(result_dict, f)
