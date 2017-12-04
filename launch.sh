#!/bin/bash

python pipeline.py --data_folder data/complexData/ --base_config configs/default.json
python pipeline.py --data_folder data/complexData/ --base_config configs/original_priors.json
