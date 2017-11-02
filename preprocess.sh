#!/bin/bash

experiment_name=75_episodes_250_frames_movingButton_staticDelta_staticRightArm
mode=tf

python -m preprocessing.preprocess --experiment $experiment_name --no-warnings --mode $mode
