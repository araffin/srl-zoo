from __future__ import print_function, division, absolute_import

import argparse
import time

import numpy as np
import torch as th
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

from models.forward_inverse import BaseRewardModel
from utils import parseDataFolder, detachToNumpy, loadData

parser = argparse.ArgumentParser(description='Predict Reward from Ground Truth')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs to train (default: 10)')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('-bs', '--batch-size', type=int, default=32, help='batch_size (default: 256)')
parser.add_argument('--training-set-size', type=int, default=-1,
                    help='Limit size (number of samples) of the training set (default: -1)')
parser.add_argument('-lr', '--learning-rate', type=float, default=0.005, help='learning rate (default: 0.005)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--data-folder', type=str, default="", help='Dataset folder', required=True)
parser.add_argument('-i', '--input-file', type=str, default="",
                    help='Path to a npz file containing states and rewards')

args = parser.parse_args()
args.cuda = not args.no_cuda and th.cuda.is_available()
device = th.device('cuda') if args.cuda else th.device('cpu')
args.data_folder = parseDataFolder(args.data_folder)

print('Loading data ... ')
training_data, ground_truth, true_states, target_positions = loadData(args.data_folder)
rewards, episode_starts = training_data['rewards'], training_data['episode_starts']

# Predict only positive or null rewards
rewards[rewards < 0] = 0

if args.input_file != "":
    print("Loading {}...".format(args.input_file))
    states = np.load(args.input_file)['states']
else:
    print("Using ground truth")
    states = true_states

state_dim = states.shape[1]

if args.training_set_size > 0:
    limit = min(args.training_set_size, len(states))
    rewards = rewards[:limit]
    states = states[:limit]
    target_positions = target_positions[:limit]
    episode_starts = episode_starts[:limit]

num_samples = rewards.shape[0] - 1  # number of samples
print("{} samples".format(num_samples))

# indices for all time steps where the episode continues
indices = np.array([i for i in range(num_samples) if not episode_starts[i + 1]], dtype='int64')

model = BaseRewardModel()
model.initRewardNet(state_dim, n_rewards=2, n_hidden=4)
model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = th.optim.Adam(model.parameters(), lr=args.learning_rate)
num_epochs = args.epochs

# Seed the random generator
np.random.seed(args.seed)
th.manual_seed(args.seed)
if args.cuda:
    th.cuda.manual_seed(args.seed)

X_train, X_val = train_test_split(indices, test_size=0.8, random_state=args.seed)
X_train, X_val = th.from_numpy(X_train), th.from_numpy(X_val)

mean_val = np.mean(states, axis=0, keepdims=True)

datasets = {'train': TensorDataset(X_train), 'val': TensorDataset(X_val)}
train_loader = DataLoader(datasets['train'], batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(datasets['val'], batch_size=args.batch_size, shuffle=False)
dataloaders = {'train': train_loader, 'val': val_loader}

start_time = time.time()
best_acc = 0

for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch + 1, num_epochs - 1))
    print('-' * 10)
    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()  # Set model to training mode
        else:
            model.eval()  # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0
        corrects = {0: 0, 1: 0}
        n_per_class = {0: 0, 1: 0}

        # Iterate over data.
        for idx in dataloaders[phase]:
            idx = idx[0]
            index = detachToNumpy(idx)
            labels = th.from_numpy(rewards[index]).to(device)
            inputs = th.Tensor(states[idx, :]).float().to(device)
            next_inputs = th.Tensor(states[idx + 1, :]).float().to(device)

            # Reshape input in case `inputs` is a vector
            if len(inputs.shape) == 1:
                inputs = inputs[None]
                next_inputs = next_inputs[None]

            if len(next_inputs) < args.batch_size:
                continue

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with th.set_grad_enabled(phase == 'train'):
                outputs = model.rewardModel(inputs, next_inputs)
                _, preds = th.max(outputs, dim=1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += th.sum(preds == labels.data)
            for class_idx in range(2):
                corrects[class_idx] += th.sum((preds == labels.data) * (preds == class_idx).byte())
                n_per_class[class_idx] += th.sum(labels.data == class_idx).item()

        epoch_loss = running_loss / len(datasets[phase])
        epoch_acc = running_corrects.double() / len(datasets[phase])
        for class_idx in range(2):
            corrects[class_idx] = detachToNumpy(corrects[class_idx])
            corrects[class_idx] = corrects[class_idx] / (n_per_class[class_idx])

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            phase, epoch_loss, epoch_acc))
        print("Accuracy per class: {:.4f} {:.4f}".format(corrects[0], corrects[1]))

        # deep copy the model
        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc

    print()

time_elapsed = time.time() - start_time
print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
print('Best val Acc: {:4f}'.format(best_acc))
