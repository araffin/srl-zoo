# srl-robotic-priors-pytorch
State Representation Learning with Robotic Priors in PyTorch


### Preprocessing
```
preprocess.sh

```
calls to:
```
python -m preprocessing.preprocess --experiment $dataset_folder_in_data --mode $mode
```

### Traiing model

python main.py --path slot_car_task_train.npz --epoch 50 --state_dim 3 -lr 0.005


## Requirements
For Ubuntu 14: OpenCV3
-  import cv2 includes both opencv 2 and 3, the easiest way to install opencv is
pip install opencv-python

Otherwise:  https://www.pyimagesearch.com/2015/06/22/install-opencv-3-0-and-python-2-7-on-ubuntu/
