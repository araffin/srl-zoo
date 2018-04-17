import time
import argparse
import json
import os

import numpy as np
import cv2
import torch
from torch.autograd import Variable

from state_representation.models import SRLNeuralNetwork


def getImage(srl_model, mu, cuda=True):
    mu = Variable(torch.from_numpy(np.array(mu).reshape(1,-1))).float()
    logvar = Variable(torch.from_numpy(np.array(np.random.normal(size=mu.shape)))).float()
    if cuda:
        mu = mu.cuda()
        logvar = logvar.cuda()

    if srl_model.model_type != 'vae':
        net_out = srl_model.model.decode(mu)
    else:
        net_out = srl_model.model.decode(mu, logvar)

    if cuda:
        net_out = net_out.cpu()
    img = net_out.data.numpy()[0].T

    img[..., 0] *= 0.229
    img[..., 1] *= 0.224
    img[..., 2] *= 0.225
    # Undo Zero-center
    img[..., 0] += 0.485
    img[..., 1] += 0.456
    img[..., 2] += 0.406
    return img[:,:,::-1]


def main():
    parser = argparse.ArgumentParser(description="autoencoder enjoy")
    parser.add_argument('--log-dir', default='', type=str, help='directory to load model')
    parser.add_argument('--no-cuda', default=False, action="store_true")

    args = parser.parse_args()

    assert os.path.exists(args.log_dir), "Error: folder '{}' does not exist".format(args.log_dir)
    assert os.path.exists(args.log_dir + "exp_config.json"), "Error: could not find 'exp_config.json' in '{}'".format(args.log_dir)

    data = json.load(open(args.log_dir + 'exp_config.json'))
    state_dim = data["state_dim"]
    srl_model_type = data["log_folder"].split("/")[-1].split("_")[0]

    if srl_model_type == 'vae':
        model_path = args.log_dir + 'srl_vae_model.pth'
    else:
        model_path = args.log_dir + 'srl_ae_model.pth'

    assert os.path.exists(model_path), "Error: could not find '{}' in '{}'".format(model_path.split("/")[-1], args.log_dir)

    srl_model = SRLNeuralNetwork(state_dim=state_dim, cuda=(not args.no_cuda), model_type=srl_model_type)
    srl_model.load(model_path)

    cv2.namedWindow(srl_model_type, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(srl_model_type, 500,500)

    cv2.namedWindow('sliders')
    for i in range(state_dim):
        cv2.createTrackbar(str(i), 'sliders', 50, 100, (lambda a: None))

    while(1):
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        mu = []
        for i in range(state_dim):
            mu.append(cv2.getTrackbarPos(str(i), 'sliders'))
        mu = (np.array(mu) - 50)/10

        cv2.imshow(srl_model_type ,getImage(srl_model, mu))

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()