#!/bin/bash

# Params
data_folder=staticButtonSimplest

# State Representation Learning with Priors (grid search)
python pipeline.py --data_folder data/$data_folder --base_config configs/default.json
python pipeline.py --data_folder data/$data_folder --base_config configs/original_priors.json
# Baselines (grid search)
python pipeline.py --baselines --data_folder data/$data_folder --base_config configs/default.json
# Create Report
python evaluation/create_report.py -d logs/$data_folder
