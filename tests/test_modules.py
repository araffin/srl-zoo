from __future__ import print_function, division, absolute_import


import subprocess

from .common import *

def testStackedModels():
    args = ['--no-display-plots', '--data-folder', TEST_DATA_FOLDER,
            '--epochs', NUM_EPOCHS, '--training-set-size', TRAINING_SET_SIZE,
            '--seed', SEED, '--val-size', 0.1,
            '--state-dim', 100, '--model-type', 'custom_cnn', '-bs', 128,
            '--losses', 'dae:1:20', 'reward:1:-1', 'forward:1:60', 'inverse:5:20',
            '--inverse-model-type', 'mlp',
            '--occlusion-percentage', 0.3,
            '--l2-reg', 0.0001]
    args = list(map(str, args))
    ok = subprocess.call(['python', 'train.py'] + args)
    assertEq(ok, 0)

    # Test predict dataset on last trained model
    # Get Latest edited folder
    path = max([LOG_FOLDER + "/" + d for d in os.listdir(LOG_FOLDER) if not d.startswith('baselines')], key=os.path.getmtime)

    ok = subprocess.call(['python', '-m', 'evaluation.predict_dataset', '-n', str(10), '--log-dir', path + "/"])
    assertEq(ok, 0)

    ok = subprocess.call(['python', '-m', 'evaluation.predict_reward', '--data-folder', TEST_DATA_FOLDER, '--epochs', str(1),
        '-i', path + "/states_rewards.npz"])
    assertEq(ok, 0)
