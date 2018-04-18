#!/bin/bash

# Params
data_folder=kuka_gym_test

# State Representation Learning with Priors (grid search)
python pipeline.py --data-folder data/$data_folder --base-config configs/default.json
# python pipeline.py --data-folder data/$data_folder --base-config configs/ref_prior.json
python pipeline.py --data-folder data/$data_folder --base-config configs/original_priors.json
# Baselines (grid search)
python pipeline.py --baselines --data-folder data/$data_folder --base-config configs/default.json
# Create Report
python evaluation/create_report.py -d logs/$data_folder
