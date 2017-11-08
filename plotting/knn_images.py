from __future__ import print_function, division, absolute_import

import argparse
import os
import random
import re

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

N_NEIGHBORS_PER_LINE = 5


def detectBasePath(filename, folder_name="srl-robotic-priors-pytorch", default_path=""):
    """
    Try to auto-detect the base path of the project
    :param filename: (str) name of the python script (__file__ constant)
    :param folder_name: (str) name of the root folder
    :param default_path: (str) path used when the detection failed
    :return: (str) detected base path
    """
    regex = r"(.*/" + folder_name + "/).*"
    abs_path = os.path.abspath(filename)
    matches = re.search(regex, abs_path)
    base_path = default_path
    if matches:
        base_path = matches.group(1)
    else:
        print("[ERROR] Base path not found, fallback to default_path: {}".format(default_path))
    return base_path


# Init seaborn
sns.set()

parser = argparse.ArgumentParser(description='KNN plot and KNN MSE')
parser.add_argument('--data_folder', type=str, default="", required=True, help='Path to a dataset folder')
parser.add_argument('--log_folder', type=str, default="", required=True, help='Path to a log folder')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('-k', '--n_neighbors', type=int, default=5, help='Number of nearest neighbors (default: 5)')
parser.add_argument('-n', '--n_samples', type=int, default=5, help='Number of test samples (default: 10)')
args = parser.parse_args()

n_neighbors = args.n_neighbors
n_samples = args.n_samples
n_lines = (n_neighbors // N_NEIGHBORS_PER_LINE) + 1
random.seed(args.seed)

# Load ground truth and images path
ground_truth = np.load('data/{}/ground_truth.npz'.format(args.data_folder))
states = np.load('{}/states_rewards.npz'.format(args.log_folder))['states']
images_path = ground_truth['images_path']
# TODO: relative states for moving button
true_states = ground_truth['arm_states']
base_path = detectBasePath(__file__)
knn_path = '{}/{}/NearestNeighbors'.format(base_path, args.log_folder)

# Do not consider the reference state as a neighbor
knn = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm='ball_tree').fit(states)
distances, neighbors_indices = knn.kneighbors(states)

print('\nUsing a random test set of images for KNN MSE evaluation...')
print('seed={}\n'.format(args.seed))

images_indices = np.arange(len(images_path))
data = random.sample(zip(images_path, neighbors_indices, distances, states, images_indices), n_samples)
# Progressbar
pbar = tqdm(total=n_samples)
n_images, total_error = 0, 0

for img_name, neigbour_indices, dist, state, image_idx in data:
    fig = plt.figure()
    fig.set_size_inches(60, 35)

    ref_coord = true_states[image_idx]
    record_folder = images_path[image_idx].split("/")[1]
    frame_name = images_path[image_idx].split("/")[-1].split(".")[0]
    # Add reference image
    ref_image = fig.add_subplot(n_lines + 1, 5, 3)
    img = Image.open("{}data/{}".format(base_path, img_name))
    imgplot = plt.imshow(img)
    state_str = ", ".join(map(lambda x: '{:.3f}'.format(x), states[image_idx]))
    state_str = "[{}]".format(state_str)

    ref_image.axis('off')
    ref_image.set_title('{}/{}\n {}\n{}'.format(record_folder, frame_name, state_str, ref_coord))

    for i in range(0, n_neighbors):
        # subplot: (i, j, k) ith plot, j rows, k columns
        subplot = fig.add_subplot(n_lines + 1, 5, 6 + i)
        # Do not consider the reference state as a neighbor
        neighbor_idx = neigbour_indices[i + 1]
        img_name = images_path[neighbor_idx]
        neighbor_record_folder = img_name.split("/")[1]
        neighbor_frame_name = img_name.split("/")[-1].split(".")[0]

        img = Image.open("{}/data/{}".format(base_path, img_name))
        plt.imshow(img)

        dist_str = 'd={:.4f}'.format(dist[i + 1])
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
with open("{}/knn_mse.txt".format(args.log_folder), 'w') as f:
    f.write("{:.5f}".format(mean_error))
