from __future__ import print_function, division

import os
import re


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
