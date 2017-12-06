#!/bin/bash

# Params
data_folder=movingButton_staticDelta_staticRightArm
knn_seed=1
n_neighbors=5
knn_samples=15

# State Representation Learning with Priors (grid search)
python pipeline.py --data_folder data/$data_folder --base_config configs/default.json
python pipeline.py --data_folder data/$data_folder --base_config configs/original_priors.json
# Baselines
python -m baselines.supervised --data_folder data/$data_folder --no-plots
python -m baselines.autoencoder --data_folder data/$data_folder --no-plots --state_dim 3
# KNN MSE for baselines
python plotting/knn_images.py --log_folder logs/$data_folder/baselines/supervised/ --seed $knn_seed -k $n_neighbors -n $knn_samples
python plotting/knn_images.py --log_folder logs/$data_folder/baselines/autoencoder/ --seed $knn_seed -k $n_neighbors -n $knn_samples
# Create Report
python evaluation/create_report.py -d logs/$data_folder
