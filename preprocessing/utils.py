from __future__ import print_function, division

import os
import re
from collections import OrderedDict

import cv2
import numpy as np
import pandas as pd

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


def preprocessInput(x, mode="image_net"):
    """
    :param x: (numpy tensor)
    :param mode: (str) One of "image_net", "tf".
        - image_net: will zero-center each color channel with
            respect to the ImageNet dataset,
            with scaling.
            cf http://pytorch.org/docs/master/torchvision/models.html
        - tf: will scale pixels between -1 and 1,
            sample-wise.
    :return: (numpy tensor)
    """
    x /= 255.
    if mode == "tf":
        x -= 0.5
        x *= 2.
    elif mode == "image_net":
        # Zero-center by mean pixel
        x[..., 0] -= 0.485
        x[..., 1] -= 0.456
        x[..., 2] -= 0.406
        # Scaling
        x[..., 0] /= 0.229
        x[..., 0] /= 0.224
        x[..., 0] /= 0.225
    else:
        raise ValueError("Unknown mode for preprocessing")
    return x
