"""
Preprocessing script to extract actions, rewards, ground truth from text files
"""
from __future__ import print_function, division, absolute_import

import argparse
import json
import os

import numpy as np
from tqdm import tqdm

from .utils import getActions, findClosestAction, getDataFrame, samePoint
# Root folder utils file
from utils import parseDataFolder

text_files = {
    'is_pressed': 'recorded_button1_is_pressed.txt',
    'button_position': 'recorded_button1_position.txt',
    'joint_states': 'recorded_robot_joint_states.txt',
    'arm_action': 'recorded_robot_limb_left_endpoint_action.txt',
    'arm_state': 'recorded_robot_limb_left_endpoint_state.txt'
}

DELTA_POS = 0.05  # delta between each action
N_ACTIONS = 26
# Bound for negative rewards (3D limits of the table)
BOUND_INF = [0.42, -0.1, -0.11]  # default for nonStaticButton dataset?
BOUND_SUP = [0.75, 0.60, 0.35]

# Resized image shape
IMAGE_WIDTH = 224  # in px
IMAGE_HEIGHT = 224  # in px
N_CHANNELS = 3
MAX_RECORDS = 5000  # No limit
INPUT_DIM = IMAGE_WIDTH * IMAGE_HEIGHT * N_CHANNELS


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
    parser.add_argument('--data_folder', type=str, default="", help='Dataset folder name', required=True)
    parser.add_argument('--mode', type=str, default="image_net", help='Preprocessing mode: One of "image_net", "tf".')
    parser.add_argument('--no-warnings', action='store_true', default=False,
                        help='disables warnings')
    args = parser.parse_args()

    assert args.mode in ['tf', 'image_net'], "Unknown mode"

    print("Dataset folder: {}".format(args.data_folder))
    print("Mode: {}".format(args.mode))
    print("Resized shape: ({}, {})".format(IMAGE_WIDTH, IMAGE_HEIGHT))
    print("Max records: {}".format(MAX_RECORDS))

    args.data_folder = parseDataFolder(args.data_folder)
    data_folder = "data/{}".format(args.data_folder)

    if os.path.isfile('{}/dataset_config.json'.format(data_folder)):
        print("Loading dataset config...")
        with open('{}/dataset_config.json'.format(data_folder), 'rb') as f:
            dataset_config = json.load(f)
            BOUND_INF = dataset_config['bound_inf']
            BOUND_SUP = dataset_config['bound_sup']
    else:
        print("[WARNING] No dataset config file found, using default values")

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
        image_folders = [item for item in os.listdir(record_folder) if
                         os.path.isdir('{}/{}'.format(record_folder, item))]

        assert len(image_folders) == 1, "Multiple image folders are not supported yet"
        # skip time file created by ROS and other unwanted non image files
        images = [item for item in os.listdir('{}/{}/'.format(record_folder, image_folders[0]))
                  if item.endswith(".jpg")]
        images.sort()

        images_path = []
        for idx, image in enumerate(images):
            # Save only the path starting from the data folder
            image_path = '{}/{}/{}/{}'.format(args.data_folder,
                                              record_folder.split("/")[-1],
                                              image_folders[0], image)
            images_path.append(image_path)

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
            all_images_path = [images_path]
        else:
            all_actions = np.concatenate((all_actions, actions), axis=0)
            all_rewards = np.concatenate((all_rewards, rewards), axis=0)
            episode_starts = np.concatenate((episode_starts, episode_start), axis=0)
            all_arm_states = np.concatenate((all_arm_states, arm_states), axis=0)
            all_images_path.append(images_path)
        # Update progressbar
        pbar.update(1)

    all_images_path = np.concatenate(all_images_path)
    pbar.close()
    # Save Everything
    data = {
        'rewards': all_rewards,
        'actions': all_actions,
        'episode_starts': episode_starts
    }

    assert len(all_rewards) == len(all_images_path), "n_rewards != n_images: {} != {}".format(len(all_rewards),
                                                                                              len(all_images_path))


    ground_truth = {
        'button_positions': button_positions,
        'arm_states': all_arm_states,
        'actions_deltas': action_to_idx.keys(),
        'images_path': all_images_path,
    }

    for key in ['bound_inf', 'bound_sup',
                'fixed_ref_point_threshold', 'fixed_ref_point']:
        if key in dataset_config.keys():
            ground_truth[key] = dataset_config[key]
        else:
            print('Warning: 5th prior will not be able to be executed with this')
            print('dataset because key parameter in .json config is missing: {}'.format(key))

    if 'fixed_ref_point_threshold' in ground_truth.keys():
        # Reference Point Prior (5th prior)
        # Saving a mask to indicate which datapoints
        # corresponds to the reference point (given a threshold)
        ref_point, ref_point_threshold = ground_truth['fixed_ref_point'], ground_truth['fixed_ref_point_threshold']
        print("Searching for reference point:{} with threshold:{}".format(ref_point, ref_point_threshold))
        is_ref_point_list = np.array([samePoint(all_arm_states[i], ref_point, ref_point_threshold)
                                      for i in range(len(all_arm_states))])

        data['is_ref_point_list'] = is_ref_point_list

        assert len(is_ref_point_list) == len(all_rewards), \
            "Length of is_ref_point_list \
                  and all_rewards does not coincide: {}, {}".format(len(is_ref_point_list), len(all_rewards))
    assert len(all_rewards) == len(all_images_path), \
        "n_rewards != n_images: {}, {}".format(len(all_rewards), len(all_images_path))

    print("Saving preprocessed data...")
    np.savez('{}/preprocessed_data.npz'.format(data_folder), **data)
    np.savez('{}/ground_truth.npz'.format(data_folder), **ground_truth)
