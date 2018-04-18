from __future__ import print_function, division, absolute_import

import subprocess

from utils import createFolder

TEST_DATA_FOLDER = "data/kuka_gym_test/"
LOG_FOLDER = "logs/kuka_gym_test/test_priors/"
NUM_EPOCHS = 1
STATE_DIM = 2
TRAINING_SET_SIZE = 2000
KNN_SAMPLES = 1000
SEED = 0
MODEL_TYPE = 'mlp'

def assertEq(left, right):
    assert left == right, "{} != {}".format(left, right)

def assertNeq(left, right):
    assert left != right, "{} == {}".format(left, right)

def createFolders():
    createFolder(LOG_FOLDER, "Test log folder already exist")
    folder_path = '{}/NearestNeighbors/'.format(LOG_FOLDER)
    createFolder(folder_path, "NearestNeighbors folder already exist")

def testPriorTrain():
    createFolders()
    args = ['--no-plots', '--data-folder', TEST_DATA_FOLDER,
            '--epochs', NUM_EPOCHS, '--training-set-size', TRAINING_SET_SIZE,
            '--seed', SEED, '--val-size', 0.1, '--log-folder', LOG_FOLDER,
            '--state-dim', STATE_DIM, '--model-type', MODEL_TYPE]
    args = list(map(str, args))

    ok = subprocess.call(['python', 'train.py'] + args)
    assertEq(ok, 0)

def testbaselineTrain():
    createFolders()
    for baseline in ['vae', 'autoencoder']:
        args = ['--no-plots', '--data-folder', TEST_DATA_FOLDER,
                '--epochs', NUM_EPOCHS, '--training-set-size', TRAINING_SET_SIZE,
                '--seed', SEED, '--state-dim', STATE_DIM, '--model-type', MODEL_TYPE]
        args = list(map(str, args))

        ok = subprocess.call(['python', '-m', 'baselines.{}'.format(baseline)] + args)
        assertEq(ok, 0)


def testKnnMSE():
    createFolders()
    args = ['--seed', SEED, '--n-samples', KNN_SAMPLES,
            '--log-folder', LOG_FOLDER,
            '--n-neighbors', 5, '--n-to-plot', 1,
            '--ground_truth']
    args = list(map(str, args))
    ok = subprocess.call(['python', '-m', 'plotting.knn_images'] + args)
    assertEq(ok, 0)
