from __future__ import print_function, division, absolute_import

import argparse
import json
from collections import OrderedDict

import cv2
import numpy as np
import torch as th
from sklearn.neighbors import KNeighborsClassifier

from models.learner import SRL4robotics
from preprocessing.utils import deNormalize
from utils import detachToNumpy

VALID_MODELS = ["forward", "inverse", "reward", "priors", "episode-prior", "reward-prior", "triplet",
                "autoencoder", "vae", "dae", "random"]
AUTOENCODERS = ['autoencoder', 'vae', 'dae']


def getImage(srl_model, state, device):
    """
    Gets an image by using the decoder of a SRL model
    (when available)

    :param srl_model: (Pytorch model)
    :param state: ([float]) the state vector from latent space
    :param device: (pytorch device)
    :return: ([float])
    """
    with th.no_grad():
        state = th.from_numpy(np.array(state).reshape(1, -1)).float()
        state = state.to(device)

        net_out = srl_model.decode(state)
        img = detachToNumpy(net_out)[0].T

    img = deNormalize(img, mode="image_net")
    return img[:, :, ::-1]


def createFigureAndSlider(name, state_dim):
    """
    Creating a window for the latent space visualization, an another for the slider to control it
    :param name: name of model (str)
    :param state_dim: (int)
    :return:
    """
    # opencv gui setup
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, 500, 500)
    cv2.namedWindow('slider for ' + name)
    # add a slider for each component of the latent space
    for i in range(state_dim):
        # the sliders MUST be between 0 and max, so we placed max at 100, and start at 50
        # So that when we substract 50 and divide 10 we get [-5,5] for each component
        cv2.createTrackbar(str(i), 'slider for ' + name, 50, 100, (lambda a: None))


def main():
    parser = argparse.ArgumentParser(description="latent space enjoy")
    parser.add_argument('--log-dir', default='', type=str, help='directory to load model')
    parser.add_argument('--no-cuda', default=False, action="store_true")

    args = parser.parse_args()
    use_cuda = not args.no_cuda
    device = th.device("cuda" if th.cuda.is_available() and use_cuda else "cpu")

    srl_model, exp_config = SRL4robotics.loadSavedModel(args.log_dir, VALID_MODELS, cuda=use_cuda)
    # Retrieve the pytorch model
    srl_model = srl_model.model
    losses = exp_config['losses']
    state_dim = exp_config['state-dim']

    split_dimensions = exp_config.get('split-dimensions')
    loss_dims = OrderedDict()
    n_dimensions = 0
    if split_dimensions is not None and isinstance(split_dimensions, OrderedDict):
        for loss_name, loss_dim in split_dimensions.items():
            print(loss_name, loss_dim)
            if loss_dim > 0 or len(split_dimensions) == 1:
                loss_dims[loss_name] = loss_dim

    if len(loss_dims) == 0:
        print(losses)
        loss_dims = {losses[0]: state_dim}

    # Load all the states and images
    data = json.load(open(args.log_dir + 'image_to_state.json'))
    X = np.array(list(data.values())).astype(float)
    y = list(data.keys())

    bound_max, bound_min, fig_names = {}, {}, {}
    start_indices, end_indices = {}, {}
    start_idx = 0

    for loss_name, loss_dim in loss_dims.items():
        # TODO: correct names (when sharing dimensions)
        start_indices[loss_name] = start_idx
        end_indices[loss_name] = start_idx + loss_dim

        if loss_name in AUTOENCODERS:
            fig_name = "Decoder for {}".format(loss_name)
        else:
            srl_model_knn = KNeighborsClassifier()
            # Find bounds and train KNN model
            srl_model_knn.fit(X[:, start_indices[loss_name]:end_indices[loss_name]], np.arange(X.shape[0]))
            fig_name = "KNN on " + ", ".join([item + " " for item in losses])[:-1]

        bound_min[loss_name] = np.min(X[:, start_indices[loss_name]:end_indices[loss_name]], axis=0)
        bound_max[loss_name] = np.max(X[:, start_indices[loss_name]:end_indices[loss_name]], axis=0)

        fig_names[loss_name] = fig_name
        start_idx += loss_dim
        createFigureAndSlider(fig_name, loss_dim)

    should_exit = False
    while not should_exit:
        # stop if escape is pressed
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        for loss_name, loss_dim in loss_dims.items():
            state = []
            for i in range(loss_dim):
                state.append(cv2.getTrackbarPos(str(i), 'slider for ' + fig_names[loss_name]))
            # Rescale the values to fit the bounds of the representation
            state = (np.array(state) / 100) * (bound_max[loss_name] - bound_min[loss_name]) + bound_min[loss_name]

            # Mask all the irrelevant dimensions with zeros
            full_state = np.zeros(state_dim)
            full_state[start_indices[loss_name]:end_indices[loss_name]] = state

            if loss_name in AUTOENCODERS:
                img = getImage(srl_model.model, full_state, device)
            else:
                img_path = y[srl_model_knn.predict([state])[0]]
                # Remove trailing .jpg if present
                img_path = img_path.split('.jpg')[0]
                img = cv2.imread("data/" + img_path + ".jpg")

            # stop if user closed a window
            if (cv2.getWindowProperty(fig_names[loss_name], 0) < 0) or (cv2.getWindowProperty('slider for ' + fig_names[loss_name], 0) < 0):
                should_exit = True
                break
            cv2.imshow(fig_names[loss_name], img)

    # gracefully close
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
