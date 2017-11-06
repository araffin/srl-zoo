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

### Traiing model

python main.py --path path_to_preprocessed_npz_data --epoch E --state_dim S -lr LR
```
e.g.
```
python main.py --path ../learning-state-representations-with-robotic-priors/slot_car_task_train.npz
```



## Requirements
For Ubuntu 14: OpenCV3
-  import cv2 includes both opencv 2 and 3, the easiest way to install opencv is
pip install opencv-python

Otherwise:  https://www.pyimagesearch.com/2015/06/22/install-opencv-3-0-and-python-2-7-on-ubuntu/
