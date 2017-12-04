#!/bin/bash

data_folder=complexData
# State Representation Learning with Priors (grid search)
python pipeline.py --data_folder data/$data_folder --base_config configs/default.json
python pipeline.py --data_folder data/$data_folder --base_config configs/original_priors.json
# Baselines
python -m baselines.supervised --data_folder data/$data_folder --no-plots
python -m baselines.autoencoder --data_folder data/$data_folder --no-plots --state_dim 3
# KNN MSE for baselines
python plotting/knn_images.py --log_folder logs/$data_folder/baselines/supervised/ --seed 1 -k 5 -n 5
python plotting/knn_images.py --log_folder logs/$data_folder/baselines/autoencoder/ --seed 1 -k 5 -n 5
