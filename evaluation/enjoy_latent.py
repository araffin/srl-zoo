from __future__ import print_function, division, absolute_import

import argparse
import json

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

    split_dimensions = exp_config.get('split-dimensions')
    # backward compatibility
    if split_dimensions is None:
        split_indices = exp_config.get('split-index', [-1])
        if not isinstance(split_indices, list):
            split_indices = [split_indices]

        # Compute the number of dimensions for each method
        if split_indices[0] > 0:
            split_dimensions = [split_indices[0]]
            for i in range(len(split_indices) - 1):
                split_dimensions.append(split_indices[i + 1] - split_indices[i])

            split_dimensions.append(state_dim - split_indices[-1])

    # model param and info
    is_auto_encoder = False
    for ae_model in AUTOENCODERS:
        if ae_model in losses:
            ae_type = ae_model
            is_auto_encoder = True
            break

    # Load all the states and images
    data = json.load(open(args.log_dir + 'image_to_state.json'))
    X = np.array(list(data.values())).astype(float)
    y = list(data.keys())

    if is_auto_encoder:
        state_dim_ae = state_dim
        # Boundaries for the AE slider
        min_x_ae = np.min(X[:, :state_dim_ae], axis=0)
        max_x_ae = np.max(X[:, :state_dim_ae], axis=0)
        createFigureAndSlider(ae_type, state_dim_ae)

    # Note: the enjoy_latent does not work yeat for n_splits > 2
    if not is_auto_encoder or len(losses) > 1:
        state_dim_second_split = state_dim

        srl_model_knn = KNeighborsClassifier()

        # Find bounds and train KNN model
        srl_model_knn.fit(X[:, -state_dim_second_split:], np.arange(X.shape[0]))

        min_X = np.min(X[:, -state_dim_second_split:], axis=0)
        max_X = np.max(X[:, -state_dim_second_split:], axis=0)

        fig_name = "KNN on " + ", ".join([item + " " for item in losses])[:-1]
        createFigureAndSlider(fig_name, state_dim_second_split)

    # run the param through the network
    while True:
        # stop if escape is pressed
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        # make the image
        if is_auto_encoder:
            mu_ae = []
            for i in range(state_dim_ae):
                mu_ae.append(cv2.getTrackbarPos(str(i), 'slider for ' + ae_type))
            # TODO: Mask dimensions
            # mu_ae = maskStates(mu_ae, split_dimensions, ae_type)
            # Rescale the values to fit the bounds of the representation
            mu_ae = (np.array(mu_ae) / 100) * (max_x_ae - min_x_ae) + min_x_ae
            img_ae = getImage(srl_model.model, mu_ae, device)

            # stop if user closed a window
            if (cv2.getWindowProperty(ae_type, 0) < 0) or (cv2.getWindowProperty('slider for ' + ae_type, 0) < 0):
                break

            cv2.imshow(ae_type, img_ae)

        if not is_auto_encoder or len(losses) > 1:
            mu = []
            for i in range(state_dim_second_split):
                mu.append(cv2.getTrackbarPos(str(i), 'slider for ' + fig_name))
            # rescale for the bounds of the priors representation, and find nearest image
            img_path = y[srl_model_knn.predict([(np.array(mu) / 100) * (max_X - min_X) + min_X])[0]]
            # Remove trailing .jpg if present
            img_path = img_path.split('.jpg')[0]
            img = cv2.imread("data/" + img_path + ".jpg")

            # stop if user closed a window
            if (cv2.getWindowProperty(fig_name, 0) < 0) or (cv2.getWindowProperty('slider for ' + fig_name, 0) < 0):
                break

            cv2.imshow(fig_name, img)

    # gracefully close
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
