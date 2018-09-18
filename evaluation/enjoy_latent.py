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
                "autoencoder", "vae", "dae"]
AUTOENCODERS = ['autoencoder', 'vae', 'dae']


def getImage(srl_model, mu, device):
    """
    Gets an image for a chosen mu value using the srl_model
    :param srl_model: (Pytorch model)
    :param mu: ([float]) the mu vector from latent space
    :param device: (pytorch device)
    :return: ([float])
    """
    with th.no_grad():
        mu = th.from_numpy(np.array(mu).reshape(1, -1)).float()
        mu = mu.to(device)

        net_out = srl_model.decode(mu)

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

    # "split-dimensions": {"autoencoder": 198, "reward": -1, "inverse": 2}
    split_dimensions = exp_config.get('split-dimensions')
    # TODO: Fix issue where split_dimensions should be loaded as an OrderedDict
    split_dimensions = OrderedDict(split_dimensions)
    loss_dims = OrderedDict()
    for loss_name, loss_dim in split_dimensions.items():
        print(loss_name, loss_dim)
        if loss_dim > 0 or len(split_dimensions) == 1:
            loss_dims[loss_name] = loss_dim

    # Load all the states and images
    data = json.load(open(args.log_dir + 'image_to_state.json'))
    X = np.array(list(data.values())).astype(float)
    y = list(data.keys())

    bound_max, bound_min, fig_names = {}, {}, {}

    for loss_name, loss_dim in loss_dims.items():
        # TODO: correct indices
        # TODO: support for n_splits > 2
        if loss_name in AUTOENCODERS:
            fig_name = "Decoder for {}".format(loss_name)
        else:
            srl_model_knn = KNeighborsClassifier()
            # Find bounds and train KNN model
            srl_model_knn.fit(X[:, :], np.arange(X.shape[0]))
            fig_name = "KNN on " + ", ".join([item + " " for item in losses])[:-1]

        bound_min[loss_name] = np.min(X[:, :], axis=0)
        bound_max[loss_name] = np.max(X[:, :], axis=0)
        fig_names[loss_name] = fig_name
        createFigureAndSlider(fig_name, state_dim)

    should_exit = False
    while not should_exit:
        # stop if escape is pressed
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        for loss_name, loss_dim in loss_dims.items():
            # TODO: correct indices
            mu = []
            for i in range(state_dim):
                mu.append(cv2.getTrackbarPos(str(i), 'slider for ' + fig_names[loss_name]))
            # Rescale the values to fit the bounds of the representation
            state = (np.array(mu) / 100) * (bound_max[loss_name] - bound_min[loss_name]) + bound_min[loss_name]
            if loss_name in AUTOENCODERS:
                img = getImage(srl_model.model, state, device)
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
