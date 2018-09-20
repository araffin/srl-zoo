from __future__ import print_function, division, absolute_import

import argparse
import json
import random
from textwrap import fill

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


MAX_STATE_LENGTH = 60
N_NEIGHBORS_PER_LINE = 5


def formatStateStr(state):
    """
    :param state: ([float])
    :return: (str)
    """
    state_str = ", ".join(map(lambda x: '{:.3f}'.format(x), state))
    state_str = "[{}]".format(state_str)
    return fill(state_str, MAX_STATE_LENGTH)


# Init seaborn
sns.set()

parser = argparse.ArgumentParser(description='KNN plot and KNN MSE')
parser.add_argument('--log-folder', type=str, default="", required=True, help='Path to a log folder')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('-k', '--n-neighbors', type=int, default=5, help='Number of nearest neighbors (default: 5)')
parser.add_argument('-n', '--n-samples', type=int, default=5, help='Number of test samples (default: 5)')
parser.add_argument('--n-to-plot', type=int, default=5, help='Number of samples to plot (default: 5)')
parser.add_argument('--relative-pos', action='store_true', default=False, help='Use relative position as ground_truth')
parser.add_argument('--ground-truth', action='store_true', default=False, help='Compute KNN-MSE for ground truth')
parser.add_argument('--multi-view', action='store_true', default=False, help='To deal with multi view data format')

args = parser.parse_args()

n_neighbors = args.n_neighbors
n_samples = args.n_samples
n_lines = (n_neighbors // N_NEIGHBORS_PER_LINE) + 1
random.seed(args.seed)
n_to_plot = args.n_to_plot

with open("{}/exp_config.json".format(args.log_folder), 'r') as f:
    data_folder = json.load(f)['data-folder']

# Load ground truth and images path
ground_truth = np.load('data/{}/ground_truth.npz'.format(data_folder))

# Backward compatibility with previous name
true_states = ground_truth['ground_truth_states' if 'ground_truth_states' in ground_truth.keys() else 'arm_states']
images_path = ground_truth['images_path']


if args.relative_pos:
    print("Using relative position")
    episode_starts = np.load('data/{}/preprocessed_data.npz'.format(data_folder))['episode_starts']
    # Backward compatibility with previous name
    target_positions = ground_truth['target_positions' if 'target_positions' in ground_truth.keys() else 'button_positions']
    episode_idx = -1
    for i in range(len(episode_starts)):
        if episode_starts[i] == 1:
            episode_idx += 1
        true_states[i] -= target_positions[episode_idx]

if args.ground_truth:
    print("Using ground_truth")
    states = true_states.copy()
else:
    states = np.load('{}/states_rewards.npz'.format(args.log_folder))['states']

knn_path = '{}/NearestNeighbors'.format(args.log_folder)

print("Computing KNN... with k={}".format(args.n_neighbors))
# Do not consider the reference state as a neighbor
knn = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm='ball_tree').fit(states)
distances, neighbors_indices = knn.kneighbors(states)

print('\nUsing a random test set of images for KNN MSE evaluation...')
print('seed={}\n'.format(args.seed))

# Sample random images
images_indices = np.arange(len(images_path))
n_samples = min(len(images_path), n_samples)
data = random.sample(list(zip(images_path, neighbors_indices, distances, images_indices)), n_samples)
# Progressbar
pbar = tqdm(total=n_samples)
n_images, total_error = 0, 0
images_titles = []

for image_path, neigbour_indices, distance, image_idx in data:
    ref_coord = true_states[image_idx]

    record_folder = images_path[image_idx].split("/")[1]
    frame_name = images_path[image_idx].split("/")[-1].split(".")[0]
    images_titles.append('{}/{}'.format(record_folder, frame_name))

    if n_to_plot > 0:
        fig = plt.figure()
        fig.set_size_inches(60, 35)

        # Add reference image
        # subplot: (i, j, k) i rows, j columns, k^th plot | n_plots: i * j

        # Remove trailing .jpg if present
        image_path = image_path.split('.jpg')[0]
        if args.multi_view:
            image_path += '_1.jpg'
        else:
            image_path += '.jpg'

        ref_image = fig.add_subplot(n_lines + 1, 5, 3)
        img = Image.open("data/{}".format(image_path))
        plt.imshow(img)
        state_str = formatStateStr(states[image_idx])
        ref_image.axis('off')
        ref_image.set_title('{}/{}\n {}\n{}'.format(record_folder, frame_name, state_str, ref_coord))

    for i in range(0, n_neighbors):
        neighbor_idx = neigbour_indices[i + 1]
        neighbor_coord = true_states[neighbor_idx]
        total_error += np.linalg.norm(neighbor_coord - ref_coord)**2
        n_images += 1

        if n_to_plot > 0:
            subplot = fig.add_subplot(n_lines + 1, 5, 6 + i)
            # Do not consider the reference state as a neighbor
            image_path = images_path[neighbor_idx]
            neighbor_record_folder = image_path.split("/")[1]
            neighbor_frame_name = image_path.split("/")[-1].split(".")[0]

            # Remove trailing .jpg if present
            image_path = image_path.split('.jpg')[0]
            if args.multi_view:
                image_path += '_1.jpg'
            else:
                image_path += '.jpg'

            img = Image.open("data/{}".format(image_path))
            plt.imshow(img)

            dist_str = 'd={:.4f}'.format(distance[i + 1])
            state_str = formatStateStr(states[neighbor_idx])

            title = '{}/{}\n {} {}\n{}'.format(neighbor_record_folder, neighbor_frame_name, state_str, dist_str,
                                               neighbor_coord)
            subplot.set_title(title)
            subplot.axis('off')

    if n_to_plot > 0:
        plt.tight_layout()
        output_file = "{}/{}_{}.png".format(knn_path, record_folder, frame_name)

        plt.savefig(output_file, bbox_inches='tight')
        plt.close()
        n_to_plot -= 1
    pbar.update(1)

pbar.close()

# Mean MSE Error
mean_error = total_error / n_images
print("KNN MSE: {}".format(mean_error))

result_dict = {'images': images_titles, 'knn_mse': round(mean_error, 5)}

with open("{}/knn_mse.json".format(args.log_folder), 'w') as f:
    json.dump(result_dict, f)
