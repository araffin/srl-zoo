from __future__ import print_function, division, absolute_import

import os
import shutil

TEST_DATA_FOLDER = 'data/kuka_gym_test'
TEST_DATA_FOLDER_DUAL = 'data/kuka_gym_dual_test'
LOG_FOLDER = 'logs/kuka_gym_test'
LOG_FOLDER_DUAL = 'logs/kuka_gym_dual_test'
NUM_EPOCHS = 1
STATE_DIM = 2
TRAINING_SET_SIZE = 2000
KNN_SAMPLES = 1000
SEED = 0


def assertEq(left, right):
    assert left == right, '{} != {}'.format(left, right)


def assertNeq(left, right):
    assert left != right, '{} == {}'.format(left, right)

def removeFolderIfExist(path):
    """
    :param path: (str)
    """
    # cleanup to remove the cluter
    if os.path.exists(path):
        shutil.rmtree(path)
