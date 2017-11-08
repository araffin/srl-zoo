"""
Preprocessing script to extract actions, rewards, ground truth from text files

TODO: improve image preprocessing speed, reduce memory usage
TODO: normalize with data loaders from pytorch https://github.com/pytorch/examples/blob/42e5b996718797e45c46a25c55b031e6768f8440/imagenet/main.py#L89-L101 as in normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
"""
from __future__ import print_function, division, absolute_import

import argparse
import json
import os

from tqdm import tqdm
import cv2
import pandas as pd
import numpy as np

from .utils import detectBasePath, getActions, findClosestAction, getDataFrame, preprocessInput


base_path = detectBasePath(__file__)
text_files = {
    'is_pressed': 'recorded_button1_is_pressed.txt',
    'button_position': 'recorded_button1_position.txt',
    'joint_states': 'recorded_robot_joint_states.txt',
    'arm_action': 'recorded_robot_limb_left_endpoint_action.txt',
    'arm_state': 'recorded_robot_limb_left_endpoint_state.txt'
}

DELTA_POS = 0.05
N_ACTIONS = 26
# Bound for negative rewards
BOUND_INF = [0.42, -0.1, -0.11]
BOUND_SUP = [0.75, 0.60, 0.35]

# Resized image shape
IMAGE_WIDTH = 224 # in px
IMAGE_HEIGHT = 224 # in px
N_CHANNELS = 3
MAX_RECORDS = 50

def isInBound(coordinate):
    """
    :param coordinate: [float]
    :return: (bool)
    """
    for i, axis in enumerate(coordinate):
        if not (BOUND_INF[i] < axis < BOUND_SUP[i]):
            return False
    return True


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Preprocess extracted ros bags')
    parser.add_argument('--data_folder', type=str, default="", help='Dataset folder name')
    parser.add_argument('--mode', type=str, default="image_net", help='Preprocessing mode: One of "image_net", "tf".')
    parser.add_argument('--no-warnings', action='store_true', default=False,
                        help='disables warnings')
    args = parser.parse_args()

    assert args.data_folder != "", "You must specify a data_folder parameter "
    assert args.mode in ['tf', 'image_net'], "Unknown mode"

    print("Dataset folder: {}".format(args.data_folder))
    print("Mode: {}".format(args.mode))
    print("Resized shape: ({}, {})".format(IMAGE_WIDTH, IMAGE_HEIGHT))
    print("Max records: {}".format(MAX_RECORDS))

    data_folder = args.data_folder
    data_folder = "{}/data/{}/".format(base_path, data_folder)

    if os.path.isfile('{}dataset_config.json'.format(data_folder)):
        print("Loading dataset config...")
        with open('{}dataset_config.json'.format(data_folder), 'rb') as f:
            dataset_config = json.load(f)
        BOUND_INF = dataset_config['bound_inf']
        BOUND_SUP = dataset_config['bound_sup']
    else:
        print("No dataset config file found, using default values")

    record_folders = []
    for item in os.listdir(data_folder):
        path = '{}/{}'.format(data_folder, item)
        if os.path.isdir(path) and "record" in item:
            record_folders.append(item)
    # Sort folders
    record_folders.sort(key=lambda item: int(item.split('_')[1]))

    all_actions, all_rewards, episodes_starts, all_images_path = None, None, None, None
    button_positions, all_arm_states, all_observations = [], None, None
    action_to_idx = getActions(DELTA_POS, N_ACTIONS)

    print("Found {} folder(s)".format(len(record_folders)))
    # Iterate through record folders
    pbar = tqdm(total=len(record_folders))
    for record_folder_name in record_folders[:MAX_RECORDS]:
        record_folder = '{}/{}'.format(data_folder, record_folder_name)
        image_folders = [item for item in os.listdir(record_folder) if os.path.isdir('{}/{}'.format(record_folder, item))]

        assert len(image_folders) == 1, "Multiple image folders are not supported yet"
        # skip time file created by ROS and other unwanted non image files
        images = [item for item in os.listdir('{}/{}/'.format(record_folder, image_folders[0])) if item.endswith(".jpg")]
        images.sort(key=lambda item: int(item.split('.')[0].split('frame')[1]))
        observations = np.zeros((len(images), IMAGE_WIDTH, IMAGE_HEIGHT, N_CHANNELS))
        images_path = []
        for idx, image in enumerate(images):
            # Save only the path starting from the data folder
            images_path.append('{}/{}/{}/{}'.format(args.data_folder, record_folder.split("/")[-1], image_folders[0], image))
            im = cv2.imread('{}/{}/{}'.format(record_folder, image_folders[0], image))
            im = cv2.resize(im, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA)
            obs = preprocessInput(im.astype(np.float32), mode=args.mode)
            obs = np.expand_dims(obs, axis=0)
            observations[idx] = obs

        # Retrieve frame indices where the button was pressed
        df = getDataFrame('{}/{}'.format(record_folder, text_files['is_pressed']))
        rewards = df['value'].values
        # Retrieve button position
        with open('{}/{}'.format(record_folder, text_files['button_position'])) as f:
            button_position = map(float, f.readlines()[1].split(' '))

        # Retrieve arm actions
        df = getDataFrame('{}/{}'.format(record_folder, text_files['arm_action']))
        actions = []
        n_frames = len(df)
        for i in range(n_frames):
            delta_action = map(float, (df.dx[i], df.dy[i], df.dz[i]))
            actions.append(findClosestAction(tuple(delta_action), action_to_idx, not args.no_warnings))
        actions = np.array(actions)

        # Retrieve ground truth states:
        df = getDataFrame('{}/{}'.format(record_folder, text_files['arm_state']))
        arm_states = []
        for i in range(n_frames):
            arm_states.append(map(float, (df.x[i], df.y[i], df.z[i])))
        arm_states = np.array(arm_states)

        # Add negative rewards
        for i in range(n_frames):
            if rewards[i] > 0:
                continue
            if not isInBound(arm_states[i]):
                rewards[i] = -1

        # print('{} positive rewards, {} negative rewards'.format(sum(rewards > 0), sum(rewards < 0)))
        episode_start = np.zeros(len(rewards))
        episode_start[0] = 1
        button_positions.append(button_position)

        if all_actions is None:
            all_actions = actions
            all_rewards = rewards
            episode_starts = episode_start[:]
            all_arm_states = arm_states
            all_observations = [observations]
            all_images_path = [images_path]
        else:
            all_actions = np.concatenate((all_actions, actions), axis=0)
            all_rewards = np.concatenate((all_rewards, rewards), axis=0)
            episode_starts = np.concatenate((episode_starts, episode_start), axis=0)
            all_arm_states = np.concatenate((all_arm_states, arm_states), axis=0)
            all_observations.append(observations)
            all_images_path.append(images_path)
        # Update progressbar
        pbar.update(1)

    all_observations = np.concatenate(all_observations)
    all_images_path = np.concatenate(all_images_path)
    pbar.close()
    # Save Everything
    data = {
        'observations': all_observations,
        'rewards': all_rewards,
        'actions': all_actions.reshape(-1, 1),
        'episode_starts': episode_starts,
    }

    assert len(all_rewards) == len(all_images_path), "n_rewards != n_images: {} != {}".format(len(all_rewards), len(all_images_path))

    print("Saving preprocessed data...")
    np.savez('{}/preprocessed_data.npz'.format(data_folder), **data)

    ground_truth = {
        'button_positions': button_positions,
        'arm_states': all_arm_states,
        'actions_deltas': action_to_idx.keys(),
        'images_path': all_images_path
    }
    np.savez('{}/ground_truth.npz'.format(data_folder), **ground_truth)
