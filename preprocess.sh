#!/bin/bash

experiment_name=staticButtonSimplest
mode=image_net

# Remove Warnings:
python -m preprocessing.preprocess --experiment $experiment_name --no-warnings --mode $mode
# With warnings:
# python -m preprocessing.preprocess --experiment $experiment_name --mode $mode
