from __future__ import print_function, division, absolute_import


import subprocess

from .common import *

def testPipeLine():
    removeFolderIfExist(LOG_FOLDER)
    # Pipeline on test config
    args = ['--data-folder', TEST_DATA_FOLDER,
            '--base-config', 'configs/test_pipeline.json']
    args = list(map(str, args))

    ok = subprocess.call(['python', 'pipeline.py'] + args)
    assertEq(ok, 0)


def testPipelineDual():
    removeFolderIfExist(LOG_FOLDER_DUAL)
    args = ['--data-folder', TEST_DATA_FOLDER_DUAL,
            '--base-config', 'configs/test_pipeline_dual.json']
    args = list(map(str, args))

    ok = subprocess.call(['python', 'pipeline.py'] + args)
    assertEq(ok, 0)


def testBaselines():
    # Pipeline on baselines
    args = ['--data-folder', TEST_DATA_FOLDER,
            '--baselines',
            '--base-config', 'configs/test_pipeline.json']
    args = list(map(str, args))

    ok = subprocess.call(['python', 'pipeline.py'] + args)
    assertEq(ok, 0)

    # Test report
    args = ['-i', LOG_FOLDER]
    args = list(map(str, args))
    ok = subprocess.call(['python', '-m', 'evaluation.gather_results'] + args)
    assertEq(ok, 0)


def testExtraSRLTrain():
    for model_type in ['resnet', 'mlp']:
        args = ['--no-display-plots', '--data-folder', TEST_DATA_FOLDER,
                '--epochs', NUM_EPOCHS, '--training-set-size', TRAINING_SET_SIZE,
                '--seed', SEED, '--val-size', 0.1,
                '--state-dim', STATE_DIM, '--model-type', model_type, '-bs', 128,
                '--losses', 'forward', 'inverse', 'reward', 'priors', 'episode-prior', 'reward-prior',
                '--balanced-sampling',
                '--l1-reg', 0.0001]
        args = list(map(str, args))
        ok = subprocess.call(['python', 'train.py'] + args)
        assertEq(ok, 0)

        # Tests for Dual camera
        for model_type in ['custom_cnn', 'mlp']:
            args = ['--no-display-plots', '--data-folder', TEST_DATA_FOLDER_DUAL,
                    '--epochs', NUM_EPOCHS, '--training-set-size', TRAINING_SET_SIZE,
                    '--seed', SEED, '--val-size', 0.1,
                    '--state-dim', STATE_DIM, '--model-type', model_type, '-bs', 32,
                    '--losses', 'forward', 'inverse', 'reward', 'priors', 'episode-prior', 'reward-prior', 'triplet',
                    '--l1-reg', 0.0001,
                    '--multi-view']
            args = list(map(str, args))
            ok = subprocess.call(['python', 'train.py'] + args)
            assertEq(ok, 0)


def testExtraBaselineTrain():
    for baseline in ['vae', 'autoencoder', 'dae']:
        # single camera
        args = ['--no-display-plots', '--data-folder', TEST_DATA_FOLDER,
                '--epochs', NUM_EPOCHS, '--training-set-size', TRAINING_SET_SIZE,
                '--seed', SEED, '--val-size', 0.1,
                '--state-dim', STATE_DIM, '--model-type', 'mlp', '-bs', 128,
                '--losses', baseline]
        args = list(map(str, args))
        ok = subprocess.call(['python', 'train.py'] + args)
        assertEq(ok, 0)

    # Linear AE
    # single camera
    args = ['--no-display-plots', '--data-folder', TEST_DATA_FOLDER,
            '--epochs', NUM_EPOCHS, '--training-set-size', TRAINING_SET_SIZE,
            '--seed', SEED, '--val-size', 0.1,
            '--state-dim', STATE_DIM, '--model-type', 'linear', '-bs', 128,
            '--losses', 'autoencoder']
    args = list(map(str, args))
    ok = subprocess.call(['python', 'train.py'] + args)
    assertEq(ok, 0)


def testExtraBaselineDualTrain():
    for baseline in ['vae', 'autoencoder']:
        # dual camera
        args = ['--no-display-plots', '--data-folder', TEST_DATA_FOLDER_DUAL,
                '--epochs', NUM_EPOCHS, '--training-set-size', TRAINING_SET_SIZE,
                '--seed', SEED, '--val-size', 0.1, '-lr', 0.0001,
                '--state-dim', STATE_DIM, '--model-type', 'mlp', '-bs', 16,
                '--losses', baseline,
                '--multi-view']
        args = list(map(str, args))
        ok = subprocess.call(['python', 'train.py'] + args)
        assertEq(ok, 0)
    # Linear AE
    # dual camera
    args = ['--no-display-plots', '--data-folder', TEST_DATA_FOLDER_DUAL,
            '--epochs', NUM_EPOCHS, '--training-set-size', TRAINING_SET_SIZE,
            '--seed', SEED, '--val-size', 0.1,
            '--state-dim', STATE_DIM, '--model-type', 'linear', '-bs', 16,
            '--losses', 'autoencoder',
            '--multi-view']
    args = list(map(str, args))
    ok = subprocess.call(['python', 'train.py'] + args)
    assertEq(ok, 0)


def testExtraSupervisedTrain():
    args = ['--no-display-plots', '--data-folder', TEST_DATA_FOLDER,
            '--epochs', NUM_EPOCHS, '--training-set-size', TRAINING_SET_SIZE,
            '--seed', SEED, '--model-type', 'mlp']
    args = list(map(str, args))
    ok = subprocess.call(['python', '-m', 'srl_baselines.supervised'] + args)
    assertEq(ok, 0)

def testPredictReward():
    ok = subprocess.call(['python', '-m', 'evaluation.predict_reward', '--data-folder', TEST_DATA_FOLDER, '--epochs', str(1)])
    assertEq(ok, 0)
