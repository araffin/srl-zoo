#!/bin/bash


# -----------------DATASETS AVAILABLE:  (in order of model robustness and reliability so far)
# --=================================================
# MOBILE_ROBOT = 'mobileRobot'
# ----Baxter datasets:

# STATIC_BUTTON_SIMPLEST = 'staticButtonSimplest'
# COMPLEX_DATA = 'complexData'
# COLORFUL75 = 'colorful75'-- a smaller version half size of colorful
#       For larger, 150 seqs, use # COLORFUL = 'colorful'-- 150 data recording sequences
# NONSTATIC_BUTTON = 'nonStaticButton'

data_folder=staticButtonSimplest
mode=image_net

python -m preprocessing.preprocess --data_folder $data_folder --no-warnings --mode $mode
