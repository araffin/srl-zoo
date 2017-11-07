# srl-robotic-priors-pytorch
State Representation Learning with Robotic Priors in PyTorch


### Preprocessing
```
preprocess.sh

```
calls to:
```
python -m preprocessing.preprocess --data_folder $dataset_folder_in_data --mode $mode
```
e.g.
```
python -m preprocessing.preprocess --data_folder staticButtonSimplest
```
or if small GPU memory, try setting one of the other GPU:
```
CUDA_VISIBLE_DEVICES=1,2 python -m preprocessing.preprocess --data_folder $dataset_folder_in_data --mode $mode
```
### Training model

python main.py --path path_to_preprocessed_npz_data --epoch E --state_dim S -lr LR
```
e.g.
```
python main.py --path ../learning-state-representations-with-robotic-priors/slot_car_task_train.npz
```



## Library Requirements
For Ubuntu 14: OpenCV3
-  import cv2 includes both opencv 2 and 3, the easiest way to install opencv is
pip install opencv-python

Otherwise:  https://www.pyimagesearch.com/2015/06/22/install-opencv-3-0-and-python-2-7-on-ubuntu/




## Data Dependencies

Jonschkowski's dataset slot car: git clone in an external folder
git clone git@github.com:tu-rbo/learning-state-representations-with-robotic-priors.git





## Potential issues

1.  python main.py --path ../learning-state-representations-with-robotic-priors/slot_car_task_train.npz
...  File "/home/seurin/.local/lib/python2.7/site-packages/torch/nn/modules/module.py", line 206, in __call__
    result = self.forward(*input, **kwargs)
  File "/home/seurin/.local/lib/python2.7/site-packages/torch/nn/modules/batchnorm.py", line 43, in forward
    self.training, self.momentum, self.eps)
  File "/home/seurin/.local/lib/python2.7/site-packages/torch/nn/functional.py", line 463, in batch_norm
    return f(input, weight, bias)
RuntimeError: cuda runtime error (2) : out of memory at /b/wheel/pytorch-src/torch/lib/THC/generic/THCStorage.cu:66

SOLUTION 1: CUDA_VISIBLE_DEVICES – Masking GPUs

CUDA_VISIBLE_DEVICES=0,1	Devices 0 and 1 will be visible  
NOTE: DO NOT USE 2 if you only have two devices, because it will use your CPU and freeze all your computer!

SOLUTION: change batchsize to 60 in GPUs with little memory. TODO:  a solution could be using both gpu 0 and 1  at the same time


2.  File "main.py", line 240, in learn
    loss = criterion(states, next_states, diss, same)
  File "/home/seurin/.local/lib/python2.7/site-packages/torch/nn/modules/module.py", line 206, in __call__
    result = self.forward(*input, **kwargs)
  File "main.py", line 124, in forward
    causality_loss = similarity(states[dissimilar_pairs[:, 0]],
IndexError: trying to index 2 dimensions of a 0 dimensional tensor

SOLUTION: the BATCH_SIZE is not large enough and as a watchdog, no equal actions/states are found to apply the priors. Increase your BATCH_SIZE or memory!
