#!/bin/bash

experiment_name=staticButtonSimplest
mode=image_net

python -m preprocessing.preprocess --experiment $experiment_name --no-warnings --mode $mode
