from __future__ import print_function, division, absolute_import

import subprocess

from utils import createFolder

TEST_DATA_FOLDER = "data/kuka_gym_test"
LOG_FOLDER = "logs/kuka_gym_test/test_priors"
NUM_EPOCHS = 1
STATE_DIM = 2
TRAINING_SET_SIZE = 2000
KNN_SAMPLES = 1000
SEED = 0


def assertEq(left, right):
    assert left == right, "{} != {}".format(left, right)


def assertNeq(left, right):
    assert left != right, "{} == {}".format(left, right)


def createFolders():
    createFolder(LOG_FOLDER, "Test log folder already exist")
    folder_path = '{}/NearestNeighbors/'.format(LOG_FOLDER)
    createFolder(folder_path, "NearestNeighbors folder already exist")


def testPipeLineTest():
    createFolders()
    # Pipeline on test config
    args = ['--data-folder', TEST_DATA_FOLDER,
            '--base-config', 'configs/test_pipeline.json']
    args = list(map(str, args))

    ok = subprocess.call(['python', 'pipeline.py'] + args)
    assertEq(ok, 0)

    # Pipeline on baselines
    args = ['--data-folder', TEST_DATA_FOLDER,
            '--baselines',
            '--base-config', 'configs/test_pipeline.json']
    args = list(map(str, args))

    ok = subprocess.call(['python', 'pipeline.py'] + args)
    assertEq(ok, 0)

def testExtraSRLTrain():
    createFolders()
    for model_type in ['resnet', 'mlp']:
        args = ['--no-plots', '--data-folder', TEST_DATA_FOLDER,
                '--epochs', NUM_EPOCHS, '--training-set-size', TRAINING_SET_SIZE,
                '--seed', SEED, '--val-size', 0.1, '--log-folder', LOG_FOLDER,
                '--state-dim', STATE_DIM, '--model-type', model_type, '-bs', 128,
                '--losses', "forward", "inverse", "reward", "priors", "episode-prior", "reward-prior",
                '--l1-reg', 0.0001]
        args = list(map(str, args))

        ok = subprocess.call(['python', 'train.py'] + args)
        assertEq(ok, 0)

def testExtraBaselineTrain():
    createFolders()
    for baseline in ['vae', 'autoencoder']:
        args = ['--no-plots', '--data-folder', TEST_DATA_FOLDER,
                '--epochs', NUM_EPOCHS, '--training-set-size', TRAINING_SET_SIZE,
                '--seed', SEED, '--val-size', 0.1, '--log-folder', LOG_FOLDER,
                '--state-dim', STATE_DIM, '--model-type', 'mlp', '-bs', 128,
                '--losses', baseline]
        args = list(map(str, args))

        ok = subprocess.call(['python', 'train.py'] + args)
        assertEq(ok, 0)
    # Linear AE
    args = ['--no-plots', '--data-folder', TEST_DATA_FOLDER,
            '--epochs', NUM_EPOCHS, '--training-set-size', TRAINING_SET_SIZE,
            '--seed', SEED, '--val-size', 0.1, '--log-folder', LOG_FOLDER,
            '--state-dim', STATE_DIM, '--model-type', 'linear', '-bs', 128,
            '--losses', 'autoencoder']
    args = list(map(str, args))

    ok = subprocess.call(['python', 'train.py'] + args)
    assertEq(ok, 0)


def testExtraSupervisedTrain():
    createFolders()
    args = ['--no-plots', '--data-folder', TEST_DATA_FOLDER,
            '--epochs', NUM_EPOCHS, '--training-set-size', TRAINING_SET_SIZE,
            '--seed', SEED, '--model-type', 'mlp']
    args = list(map(str, args))

    ok = subprocess.call(['python', '-m', 'baselines.supervised'] + args)
    assertEq(ok, 0)


def testKnnMSE():
    createFolders()
    args = ['--seed', SEED, '--n-samples', KNN_SAMPLES,
            '--log-folder', LOG_FOLDER,
            '--n-neighbors', 5, '--n-to-plot', 1,
            '--ground-truth']
    args = list(map(str, args))
    ok = subprocess.call(['python', '-m', 'plotting.knn_images'] + args)
    assertEq(ok, 0)

def testReport():
    createFolders()
    args = ['-d', 'logs/kuka_gym_test/']
    args = list(map(str, args))
    ok = subprocess.call(['python', 'evaluation/create_report.py'] + args)
    assertEq(ok, 0)
