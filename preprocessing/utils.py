from __future__ import print_function, division

import os
import re
from collections import OrderedDict

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


def samePoint(pos, ref_pos, threshold):
    """
    Return true if the position `pos` is close enough
    to the reference
    :param pos: (numpy array)
    :param ref_pos: (numpy array)
    :return: (bool)
    """
    return np.linalg.norm(ref_pos - pos, 2) <= threshold


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
    Normalize input
    :param x: (numpy tensor) (RGB image with values between [0, 255])
    :param mode: (str) One of "image_net", "tf".
        - image_net: will zero-center each color channel with
            respect to the ImageNet dataset,
            with scaling.
            cf http://pytorch.org/docs/master/torchvision/models.html
        - tf: will scale pixels between -1 and 1,
            sample-wise.
    :return: (numpy tensor)
    """
    assert x.shape[-1] == 3, "Color channel must be at the end of the tensor {}".format(x.shape)
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
        x[..., 1] /= 0.224
        x[..., 2] /= 0.225
    else:
        raise ValueError("Unknown mode for preprocessing")
    return x


def deNormalize(x, mode="image_net"):
    """
    deNormalize data (transform input to [0, 1])
    :param x: (numpy tensor)
    :param mode: (str) One of "image_net", "tf".
    :return: (numpy tensor)
    """
    # Reorder channels when we have only one image
    if x.shape[0] == 3 and len(x.shape) == 3:
        # (n_channels, height, width) -> (width, height, n_channels)
        x = np.transpose(x, (2, 1, 0))
    assert x.shape[-1] == 3, "Color channel must be at the end of the tensor {}".format(x.shape)

    if mode == "tf":
        x /= 2.
        x += 0.5
    elif mode == "image_net":
        # Scaling
        x[..., 0] *= 0.229
        x[..., 1] *= 0.224
        x[..., 2] *= 0.225
        # Undo Zero-center
        x[..., 0] += 0.485
        x[..., 1] += 0.456
        x[..., 2] += 0.406
    else:
        raise ValueError("Unknown mode for deNormalize")
    # Clip to fix numeric imprecision (1e-09 = 0)
    return np.clip(x, 0, 1)
