# State Representation Learning Zoo with PyTorch

A collection of State Representation Learning (SRL) methods for Reinforcement Learning, written using PyTorch.

Available methods:

- SRL with Robotic Priors + extensions (stereovision, additional priors)
- Denoising Autoencoder (DAE)
- Variational Autoencoder (VAE) and beta-VAE
- PCA
- Supervised Learning
- Forward, Inverse Models
- Triplet Network (for stereovision only)
- Reward losses
- Combination and stacking of methods

Related papers:
- "State Representation Learning for Control: An Overview" (Lesort et al., 2018), link: [https://arxiv.org/pdf/1802.04181.pdf](https://arxiv.org/pdf/1802.04181.pdf)
- "Learning State Representations with Robotic Priors" (Jonschkowski and Brock, 2015), link: [http://tinyurl.com/gly9sma](http://tinyurl.com/gly9sma)



Table of Contents
=================
* [Installation](#installation)
  * [Using Anaconda](#using-anaconda)
    * [Python 3](#python-3)
    * [Python 2](#python-2)
  * [Using Docker](#using-docker)
  * [Using Requirements.txt](#using-requirementstxt)
* [Learning a State Representation](#learning-a-state-representation)
  * [Examples](#examples)
  * [Stacking/Splitting Models Instead of Combining Them](#stackingsplitting-models-instead-of-combining-them)
  * [Predicting States on the Whole Dataset](#predicting-states-on-the-whole-dataset)
  * [Predicting Reward Using a Trained Model](#predicting-reward-using-a-trained-model)
* [Multiple Cameras](#multiple-cameras)
  * [Stacked Observations](#stacked-observations)
  * [Triplets of Observations](#triplets-of-observations)
* [Evaluation and Plotting](#evaluation-and-plotting)
* [Learned Space Visualization](#learned-space-visualization)
  * [Create a report](#create-a-report)
  * [Plot a Learned Representation](#plot-a-learned-representation)
  * [Interactive Plot](#interactive-plot)
  * [Create a KNN Plot and Compute KNN-MSE](#create-a-knn-plot-and-compute-knn-mse)
* [Baselines](#baselines)
  * [Supervised Learning](#supervised-learning)
  * [Principal Components Analysis](#principal-components-analysis)
* [Config Files](#config-files)
  * [Base Config](#base-config)
  * [Dataset config](#dataset-config)
  * [Experiment Config](#experiment-config)
* [Dataset Format](#dataset-format)
* [Launch script](#launch-script)
* [Pipeline Script](#pipeline-script)
  * [Examples](#examples-1)
* [SRL Server for Reinforcement Learning](#srl-server-for-reinforcement-learning)
* [Running Tests](#running-tests)
* [Example Data](#example-data)
* [Troubleshooting](#troubleshooting)
  * [CUDA out of memory error](#cuda-out-of-memory-error)




## Installation

Recommended configuration: Ubuntu 16.04 with python >=3.5 (or python 2.7)

### Using Anaconda

#### Python 3
Please use `environment.yml` file from [https://github.com/araffin/robotics-rl-srl](https://github.com/araffin/robotics-rl-srl)
To create a conda environment from this file:

```
conda env create -f environment.yml
```

#### Python 2

Create the new environment `srl` from `environment.yml` file:
```
conda env create -f environment.yml
```

Then activate it using:
```
source activate srl
```

### Using Docker

We provide docker images to work with our repository, please read *Installation using docker** from [https://github.com/araffin/robotics-rl-srl](https://github.com/araffin/robotics-rl-srl) for more information.

### Using Requirements.txt
Alternatively, you can use requirements.txt file:
```
pip install -r requirements.txt
```
In that case, you will need to install OpenCV too (cf below).

- OpenCV (version >= 2.4)
```
conda install -c menpo opencv
```
or
```
sudo apt-get install python-opencv (opencv 2.4 - python2)
```


## Learning a State Representation

To learn a state representation, you need to impose constrains on the representation using one or more losses. For example, to train an autoencoder, you need to use a reconstruction loss.
Most losses are not exclusive, that means you can combine them.

All losses are defined in `losses/losses.py`. The available losses are:

- autoencoder: reconstruction loss, using current and next observation
- vae: (beta)-VAE loss (reconstruction + kullback leiber divergence loss)
- inverse: predict the action given current and next state
- forward: predict the next state given current state and taken action
- reward: predict the reward (positive or not) given current and next state
- priors: robotic priors losses (see "Learning State Representations with Robotic Priors")
- triplet: triplet loss for multi-cam setting (see *Multiple Cameras* section)
- reward-prior [Experimental] Maximise correlation between states and reward (does not make sense for sparse reward)

All possible arguments can be display using `python train.py --help`. You can limit the training set size (`--training-set-size` argument), change the minibatch size (`-bs`), number of epochs (`--epochs`), ...


### Examples

Train an inverse model:
```
python train.py --data-folder data/path/to/dataset --losses inverse
```

Train an autoencoder:
```
python train.py --data-folder data/path/to/dataset --losses autoencoder
```

Combining an autoencoder with an inverse model is as easy as:
```
python train.py --data-folder data/path/to/dataset --losses autoencoder inverse
```

### Stacking/Splitting Models Instead of Combining Them

Because losses do not optimize the same objective and can be opposed, it may make sense to stack representations learned with different objectives, instead of combining them. For instance, you can stack an autoencoder (with a state dimension of 20) with an inverse model (of dimension 2) using:

```
python train.py --data-folder data/path/to/dataset --losses autoencoder inverse --state-dim 22 --split-index 20
```

The details of how models are splitted can be found inside the `SRLModulesSplit` class, defined in `models/modules.py`. All models share the same *encoder* or *features extractor*, that maps observations to states.


### Predicting States on the Whole Dataset

If you trained your model on a subset of a dataset, you can predict states for the whole dataset (or on a subset) using:
```
python -m evaluation.predict_dataset --log-dir logs/path/to/log_folder/
```
use  `-n 1000` to predict on the first 1000 samples only.

### Predicting Reward Using a Trained Model

If you want to predict the reward (train a classifier for positive or null reward) using ground truth states or learned states, you can use `evaluation/predict_reward.py` script.
Ground Truth:
```
python -m evaluation.predict_reward --data-folder data/dataset_name/ --training-set-size 50000
```

On Learned States:
```
python -m evaluation.predict_reward --data-folder data/dataset_name/ -i log/path/to/states_rewards.npz
```


## Multiple Cameras

### Stacked Observations

Using the `custom_cnn` and `mlp` architecture, it is possible to pass pairs of images from different views stacked along the channels' dimension i.e of dim (224,224,6).

To use this functionality to perform state representation learning, enable `--multi-view` (see usage of script train.py),
and use a dataset generated for the purpose.


### Triplets of Observations

Similarly, it is possible to learn representation of states using a dataset of triplets, i.e tuples made of an anchor, a positive and a negative observation.

The anchor and the positive observation are views of the scene at the same time step, but from different cameras.

The negative example is an image from the same camera as the anchor but at a different time step selected randomly among images in the same record.

In our case, enable `triplet` as a loss (`--losses`) to use the TCN-like architecture made of a pre-trained ResNet with an extra fully connected layer (embedding).

To use this functionality also enable `--multi-view`, and use a dataset generated for the purpose.
Related papers:
- "Time-Contrastive Networks: Self-Supervised Learning from Video" (P. Sermanet et al., 2017), paper: [https://arxiv.org/abs/1704.06888](https://arxiv.org/abs/1704.06888)

## Evaluation and Plotting

## Learned Space Visualization

To view the learned state and play with the latent space of a trained model, you may use:
```bash
python -m enjoy.enjoy_latent --log-dir logs/nameOfTheDataset/nameOfTheModel
```

### Create a report
After a report you can create a csv report file using:
```
python evaluation/create_report.py -d logs/nameOfTheDataset/
```

### Plot a Learned Representation

```
usage: representation_plot.py [-h] [-i INPUT_FILE] [--data-folder DATA_FOLDER]
                              [--color-episode] [--plot-against]
                              [--correlation] [--projection]

Plotting script for representation

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT_FILE, --input-file INPUT_FILE
                        Path to a npz file containing states and rewards
  --data-folder DATA_FOLDER
                        Path to a dataset folder, it will plot ground truth
                        states
  --color-episode       Color states per episodes instead of reward
  --plot-against        Plot against each dimension
  --correlation         Plot the Pearson Matrix of correlation between the Ground truth and learned states.
  --projection          Plot 1D projection of predicted state on ground truth

```
You can plot a learned representation with:
```
python -m plotting.representation_plot -i path/to/states_rewards.npz
```

You can also plot ground truth states with:
```
python -m plotting.representation_plot --data-folder path/to/datasetFolder/
```

To have a different color per episode, you have to pass `--data-folder` argument along with `--color-episode`.

Correlation plot with ground truth states:

```
python -m plotting.representation_plot -i path/to/states_rewards.npz --data-folder path/to/datasetFolder/ --correlation
```

Plotting each dimension of the state representation against another:

```
python -m plotting.representation_plot -i path/to/states_rewards.npz --plot-against
```

### Interactive Plot

You can have an interactive plot of a learned representation using:
```
python -m plotting.interactive_plot --data-folder path/to/datasetFolder/ -i path/to/states_rewards.npz
```
When you click on a state in the representation plot (left click for 2D, **right click for 3D plots**!), it shows the corresponding image along with the reward and the coordinates in the space.

Pass `--multi-view` as argument to visualize in case of multiple cameras.

You can also plot ground truth states when you don't specify a npz file:
```
python -m plotting.interactive_plot --data-folder path/to/datasetFolder/
```

### Create a KNN Plot and Compute KNN-MSE

Usage:
```
python evaluation/knn_images.py [-h] --log-folder LOG_FOLDER [--seed SEED]
                     [-k N_NEIGHBORS] [-n N_SAMPLES] [--n-to-plot N_TO_PLOT]
                     [--relative-pos] [--ground-truth] [--multi-view]

KNN plot and KNN MSE

optional arguments:
  -h, --help            show this help message and exit
  --log-folder LOG_FOLDER
                        Path to a log folder
  --seed SEED           random seed (default: 1)
  -k N_NEIGHBORS, --n-neighbors N_NEIGHBORS
                        Number of nearest neighbors (default: 5)
  -n N_SAMPLES, --n-samples N_SAMPLES
                        Number of test samples (default: 5)
  --n-to-plot N_TO_PLOT
                        Number of samples to plot (default: 5)
  --relative-pos        Use relative position as ground_truth
  --ground-truth        Compute KNN-MSE for ground truth
  --multi-view          To deal with multi view data format


```

Example:
```
python plotting/knn_images.py --log-folder path/to/an/experiment/log/folder
```

## Baselines

Baseline models are saved in `logs/nameOfTheDataset/baselines/` folder.

### Supervised Learning

Example:
```
python -m baselines.supervised --data-folder path/to/data/folder
```

### Principal Components Analysis

PCA:
```
python -m baselines.pca --data-folder path/to/data/folder --state-dim 3
```


## Config Files

### Base Config
Config common to all dataset can be found in [configs/default.json](configs/default.json).

### Dataset config
All datasets must be placed in the `data/` folder.
Each dataset must contain a `dataset_config.json` file, an example can be found [here](configs/example_dataset_config.json).
This config file describes specific variables to this dataset.


### Experiment Config
Experiment config file is generated by the `pipeline.py` script. An example can be found [here](configs/example_exp_config.json).

## Dataset Format

In order to use SRL methods on a dataset, this dataset must be preprocessed and formatted as in the [example dataset](https://drive.google.com/open?id=154qMJHgUnzk0J_Hxmr2jCnV1ipS7o1D5).
We recommend you downloading this example dataset to have a concrete and working example of what a preprocessed dataset looks like.

NOTE: If you use data generated with the [RL Repo](https://github.com/araffin/robotics-rl-srl), the dataset will be already preprocessed, so you don't need to bother about this step.

The dataset format is as follows:

0. You must provide a dataset config file (see previous section) that contains at least if the ground truth is the relative position or not
1. Images are grouped by episode in different folders (`record_{03d}/` folders)
2. At the root of the dataset folder, preprocessed_data.npz contains numpy arrays ('episode_starts', 'rewards', 'actions')
3. At the root of the dataset folder, ground_truth.npz contains numpy arrays ('target_positions', 'ground_truth_states', 'images_path')

The exact format for each numpy array can be found in the example dataset (or in the [RL Repo](https://github.com/araffin/robotics-rl-srl)).
Note: the variables 'arm_states' and 'button_positions' were renamed 'ground_truth_states' and 'target_positions'


## Launch script
Located [here](launch.sh), it is a shell script that launches multiple grid searches, trains the baselines and calls the report script.
You have to edit `$data_folder` and make sure of the parameters for knn evaluation before running it:
```
./launch.sh
```

## Pipeline Script
It learns state representations and evaluates them using knn-mse.

To generate data for Kuka and Mobile Robot environment, please see the RL repo: [https://github.com/araffin/robotics-rl-srl](https://github.com/araffin/robotics-rl-srl).

Baxter data used in the paper are not public yet. However you can generate new data using [Baxter Simulator](https://github.com/araffin/arm_scenario_simulator) and [Baxter Experiments](https://github.com/NataliaDiaz/arm_scenario_experiments)

```
python pipeline.py [-h] [-c EXP_CONFIG] [--data-folder DATA_FOLDER]
                   [--base_config BASE_CONFIG]
-c EXP_CONFIG, --exp-config EXP_CONFIG
                     Path to an experiment config file
--data-folder DATA_FOLDER
                     Path to a dataset folder
--base_config BASE_CONFIG
                     Path to overall config file, it contains variables
                     independent from datasets (default:
                     /configs/default.json)
```

### Examples

Grid search:
```
python pipeline.py --data-folder data/staticButtonSimplest/
```

Reproducing an experiment:
```
python pipeline.py -c path/to/exp_config.json
```


## SRL Server for Reinforcement Learning

This feature is currently experimental. It will launch a server that will learn a srl model and send a response to the RL client when  it is ready.
```
python server.py
```

## Running Tests

Download the test datasets [kuka_gym_test](https://drive.google.com/open?id=154qMJHgUnzk0J_Hxmr2jCnV1ipS7o1D5) and [kuka_gym_dual_test](https://drive.google.com/open?id=15Fhqr4-kai4b8qQWiq2mEAWW5ZqH5qID) and put it in `data/` folder.
```
./run_tests.sh
```

## Example Data
You can reproduce Rico Jonschkowski's results by downloading npz files from the original [github repository](https://github.com/tu-rbo/learning-state-representations-with-robotic-priors) and placing them in the `data/` folder.

It was tested with the following commit (checkout this one to be sure it will work): [https://github.com/araffin/srl-zoo/commit/5175b88a891c240f393b717dd1866435c73ebbda](https://github.com/araffin/srl-zoo/commit/5175b88a891c240f393b717dd1866435c73ebbda)

You have to do:
```
git checkout 5175b88a891c240f393b717dd1866435c73ebbda
```

Then run (for example):
```
python main.py --path data/slot_car_task_train.npz
```


## Troubleshooting

### CUDA out of memory error

1.  python train.py --data-folder data/staticButtonSimplest
```
RuntimeError: cuda runtime error (2) : out of memory at /b/wheel/pytorch-src/torch/lib/THC/generic/THCStorage.cu:66
```

SOLUTION 1: Decrease the batch size, e.g. 32-64 in GPUs with little memory.

SOLUTION 2 Use simple 2-layers neural network model
python train.py --data-folder data/staticButtonSimplest --model-type mlp
