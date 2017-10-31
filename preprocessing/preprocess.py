"""
Preprocessing script to extract actions, rewards, ground truth from text files

TODO: negative rewards
TODO: preprocess images ?
"""
from __future__ import print_function, division, absolute_import

import os
from collections import OrderedDict

import pandas as pd
import numpy as np

from .utils import detectBasePath

base_path = detectBasePath(__file__)
experiment = "static_arm_movingButton"
text_files = {
    'is_pressed': 'recorded_button1_is_pressed.txt',
    'button_position': 'recorded_button1_position.txt',
    'joint_states': 'recorded_robot_joint_states.txt',
    'arm_action': 'recorded_robot_limb_left_endpoint_action.txt',
    'arm_state': 'recorded_robot_limb_left_endpoint_state.txt'
}

DELTA_POS = 0.05
N_ACTIONS = 26


def getActions(delta_pos, n_actions):
    """
    :param delta_pos: (float)
    :param n_actions: (int)
    :return: (dict)
    """
    possible_deltas = [i * delta_pos for i in range(-1, 2)]
    actions = []
    for dx in possible_deltas:
        for dy in possible_deltas:
            for dz in possible_deltas:
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                actions.append((dx, dy, dz))

    assert len(actions) == n_actions, "Wrong number of actions: {}".format(len(actions))

    action_to_idx = {action: idx for idx, action in enumerate(actions)}
    # Sort the dictionnary to have a consistent order
    action_to_idx = OrderedDict(sorted(action_to_idx.items(), key=lambda item: item[1]))

    return action_to_idx


def findClosestAction(action, action_to_idx, show_warning=True):
    """
    :param action: ([float])
    :param action_to_idx: (dict)
    :param show_warning: (bool)
    :return: (int)
    """
    action_idx = action_to_idx.get(tuple(action))
    if action_idx is None:
        if show_warning:
            print("[WARNING] {} not found in action dict".format(action))
        distances = [np.linalg.norm(np.array(a) - action) for a in action_to_idx.keys()]
        action_idx = np.argmin(distances)
    return action_idx


def getDataFrame(text_file):
    """
    :param text_file: (str) path to a text file extracted from a ROS bag
    :return: (pandas dataFrame)
    """
    with open(text_file) as f:
        # Read first line and retrieve the column names
        headers = f.readline().strip("#").strip("\n")[1:].split(' ')
    return pd.read_csv(text_file, sep=" ", skiprows=1, names=headers)


if __name__ == '__main__':
    data_folder = "{}/data/{}/".format(base_path, experiment)
    record_folders = os.listdir(data_folder)
    # Sort folders
    record_folders.sort(key=lambda item: int(item.split('_')[1]))
    print("Found {} folder(s)".format(len(record_folders)))
    # Iterate through record folders
    record_folder = '{}/{}'.format(data_folder, record_folders[0])
    image_folders = [item for item in os.listdir(record_folder) if os.path.isdir('{}/{}'.format(record_folder, item))]

    # Retrieve frame indices where the button was pressed
    df = getDataFrame('{}/{}'.format(record_folder, text_files['is_pressed']))
    positive_reward_indices = np.where(df['value'].values == 1)[0]

    # Retrieve button position
    with open('{}/{}'.format(record_folder, text_files['button_position'])) as f:
        button_position = map(float, f.readlines()[1].split(' '))

    action_to_idx = getActions(DELTA_POS, N_ACTIONS)

    # Retrieve arm actions
    df = getDataFrame('{}/{}'.format(record_folder, text_files['arm_action']))
    actions = []
    n_frames = len(df)
    for i in range(n_frames):
        delta_action = map(float, (df.dx[i], df.dy[i], df.dz[i]))
        actions.append(findClosestAction(tuple(delta_action), action_to_idx))
    actions = np.array(actions)

    # Retrieve ground truth states:
    df = getDataFrame('{}/{}'.format(record_folder, text_files['arm_state']))
    states = []
    for i in range(n_frames):
        states.append(map(float, (df.x[i], df.y[i], df.z[i])))
    states = np.array(states)

    # TODO: negative rewards

    # Save Everything
    data = {
        'positive_reward_indices': positive_reward_indices,
        'button_position': button_position,
        'actions': actions,
        'actions_deltas': action_to_idx.keys(),
        'states': states
    }
    print("Saving preprocessed data...")
    np.savez('{}/data.npz'.format(record_folder), **data)
