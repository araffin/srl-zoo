from __future__ import print_function, division, absolute_import

import argparse
import json
import os

import numpy as np
import cv2
import torch
from sklearn.neighbors import KNeighborsClassifier

from preprocessing.utils import deNormalize
from models import SRLModules, SRLModulesSplit
from utils import detachToNumpy

VALID_MODELS = ["forward", "inverse", "reward", "priors", "episode-prior", "reward-prior", "triplet",
               "autoencoder", "vae"]


def getImage(srl_model, mu, device):
    """
    Gets an image for a chosen mu value using the srl_model
    :param srl_model: (Pytorch model)
    :param mu: ([float]) the mu vector from latent space
    :param device: (pytorch device)
    :return: ([float])
    """
    with torch.no_grad():
        mu = torch.from_numpy(np.array(mu).reshape(1, -1)).float()
        mu = mu.to(device)

        net_out = srl_model.decode(mu)

        img = detachToNumpy(net_out)[0].T

    img = deNormalize(img)
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
    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")

    # making sure you chose the right folder
    assert os.path.exists(args.log_dir), "Error: folder '{}' does not exist".format(args.log_dir)

    with open(args.log_dir + 'exp_config.json', 'r') as f:
        exp_config = json.load(f)

    state_dim = exp_config['state-dim']
    loss_type = exp_config['losses']
    n_actions = exp_config['n_actions']
    model_type = exp_config['model-type']
    split_index = exp_config.get('split-index', -1)

    # is this a valid model ?
    difference = set(loss_type).symmetric_difference(VALID_MODELS)
    assert set(loss_type).intersection(VALID_MODELS) != set(), "Error: Not supported losses " + ", ".join(difference)

    if os.path.exists(args.log_dir + 'srl_model.pth'):
        model_path = args.log_dir + 'srl_model.pth'

    # model param and info
    is_auto_encoder = 'autoencoder' in loss_type or 'vae' in loss_type
    if is_auto_encoder:
        assert os.path.exists(
            args.log_dir + "exp_config.json"), "Error: could not find 'exp_config.json' in '{}'".format(
            args.log_dir)

        if split_index > 0:
            srl_model = SRLModulesSplit(state_dim=state_dim, action_dim=n_actions, model_type=model_type,
                                   cuda=use_cuda, losses=loss_type, split_index=split_index)
        else:
            srl_model = SRLModules(state_dim=state_dim, action_dim=n_actions, model_type=model_type,
                                   cuda=use_cuda, losses=loss_type)
        srl_model.eval()
        srl_model.load_state_dict(torch.load(model_path))
        srl_model = srl_model.to(device)

        ae_type = 'autoencoder' if 'autoencoder' in loss_type else 'vae'
        state_dim_first_split = split_index if split_index > 0 else state_dim
        createFigureAndSlider(ae_type, state_dim_first_split)

    if not is_auto_encoder or len(loss_type) > 1:
        state_dim_second_split = state_dim - split_index if split_index > 0 else state_dim
        data = json.load(open(args.log_dir + 'image_to_state.json'))
        srl_model_knn = KNeighborsClassifier()

        # Load all the points and images, find bounds and train KNN model
        X = np.array(list(data.values())).astype(float)
        y = list(data.keys())
        print(X[:, -state_dim_second_split:].shape)
        srl_model_knn.fit(X[:, -state_dim_second_split:], np.arange(X.shape[0]))


        min_X = np.min(X[:, -state_dim_second_split:], axis=0)
        max_X = np.max(X[:, -state_dim_second_split:], axis=0)

        fig_name = "KNN on " + ", ".join([item + " " for item in loss_type])[:-1]
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
            for i in range(state_dim_first_split):
                mu_ae.append(cv2.getTrackbarPos(str(i), 'slider for ' + ae_type))
            mu_ae = (np.array(mu_ae) - 50) / 10
            img_ae = getImage(srl_model.model, mu_ae, device)

            # stop if user closed a window
            if (cv2.getWindowProperty(ae_type, 0) < 0) or (cv2.getWindowProperty('slider for ' + ae_type, 0) < 0):
                break

            cv2.imshow(ae_type, img_ae)

        if not is_auto_encoder or len(loss_type) > 1:
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
