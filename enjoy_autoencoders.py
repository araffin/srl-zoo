import argparse
import json
import os

import numpy as np
import cv2
import torch
from torch.autograd import Variable

from models import CNNAutoEncoder, CNNVAE

VALID_MODEL = ['vae', 'autoencoder']


def getImage(srl_model, mu, cuda=True):
    """
    Gets an image for a chosen mu value using the srl_model
    :param srl_model: (Pytorch model)
    :param mu: ([float]) the mu vector from latent space
    :param cuda: (bool) if the gpu should be used or not (default: True)
    :return: ([float])
    """
    mu = Variable(torch.from_numpy(np.array(mu).reshape(1, -1))).float()
    if cuda:
        mu = mu.cuda()
        srl_model = srl_model.cuda()

    net_out = srl_model.decode(mu)

    if cuda:
        net_out = net_out.cpu()
    img = net_out.data.numpy()[0].T

    # Undo scale
    img[..., 0] *= 0.229
    img[..., 1] *= 0.224
    img[..., 2] *= 0.225
    # Undo Zero-center
    img[..., 0] += 0.485
    img[..., 1] += 0.456
    img[..., 2] += 0.406
    return img[:, :, ::-1]


def main():
    parser = argparse.ArgumentParser(description="autoencoder enjoy")
    parser.add_argument('--log-dir', default='', type=str, help='directory to load model')
    parser.add_argument('--no-cuda', default=False, action="store_true")

    args = parser.parse_args()

    # making sure you chose the right folder
    assert os.path.exists(args.log_dir), "Error: folder '{}' does not exist".format(args.log_dir)
    assert os.path.exists(args.log_dir + "exp_config.json"), "Error: could not find 'exp_config.json' in '{}'".format(
        args.log_dir)

    # model param and info
    data = json.load(open(args.log_dir + 'exp_config.json'))
    state_dim = data["state_dim"]
    srl_model_type = data["log_folder"].split("/")[-1].split("_")[0]

    # is this a valid model
    assert srl_model_type in VALID_MODEL, "Error: '{}' model is not supported."

    # setup the loading location
    if srl_model_type == 'vae':
        model_path = args.log_dir + 'srl_vae_model.pth'
    elif srl_model_type == 'autoencoder':
        model_path = args.log_dir + 'srl_ae_model.pth'

    # making sure the model actually exists
    assert os.path.exists(model_path), "Error: could not find '{}' in '{}'".format(model_path.split("/")[-1],
                                                                                   args.log_dir)

    # loading the model
    if srl_model_type == "autoencoder":
        srl_model = CNNAutoEncoder(state_dim)
    elif srl_model_type == "vae":
        srl_model = CNNVAE(state_dim)
    srl_model.eval()
    srl_model.load_state_dict(torch.load(model_path))
    if not args.no_cuda:
        srl_model.cuda()

    # opencv gui setup
    cv2.namedWindow(srl_model_type, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(srl_model_type, 500, 500)
    cv2.namedWindow('sliders')
    for i in range(state_dim):
        cv2.createTrackbar(str(i), 'sliders', 50, 100, (lambda a: None))

    # run the param through the network
    while 1:
        # stop if escape is pressed
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        # make the image 
        mu = []
        for i in range(state_dim):
            mu.append(cv2.getTrackbarPos(str(i), 'sliders'))
        mu = (np.array(mu) - 50) / 10
        img = getImage(srl_model, mu)

        # stop if user closed a window
        if (cv2.getWindowProperty(srl_model_type, 0) < 0) or (cv2.getWindowProperty('sliders', 0) < 0):
            break

        cv2.imshow(srl_model_type, img)

    # gracefully close
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
