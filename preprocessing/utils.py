from __future__ import print_function, division

import numpy as np


def preprocessInput(x, mode="image_net"):
    """
    Normalize input
    :param x: (np.ndarray) (RGB image with values between [0, 255])
    :param mode: (str) One of "image_net", "tf".
        - image_net: will zero-center each color channel with
            respect to the ImageNet dataset,
            with scaling.
            cf http://pytorch.org/docs/master/torchvision/models.html
        - tf: will scale pixels between -1 and 1,
            sample-wise.
    :return: (np.ndarray)
    """
    assert x.shape[-1] == 3, "Color channel must be at the end of the tensor {}".format(x.shape)
    x /= 255.
    if mode == "tf":
        x -= 0.5
        x *= 2.
    elif mode == "image_net":
        # Zero-center by mean pixel
        x[..., 0] -= 0.485
        x[..., 1] -= 0.456
        x[..., 2] -= 0.406
        # Scaling
        x[..., 0] /= 0.229
        x[..., 1] /= 0.224
        x[..., 2] /= 0.225
    else:
        raise ValueError("Unknown mode for preprocessing")
    return x


def deNormalize(x, mode="image_net"):
    """
    deNormalize data (transform input to [0, 1])
    :param x: (np.ndarray)
    :param mode: (str) One of "image_net", "tf".
    :return: (np.ndarray)
    """
    # Reorder channels when we have only one image
    if x.shape[0] == 3 and len(x.shape) == 3:
        # (n_channels, height, width) -> (width, height, n_channels)
        x = np.transpose(x, (2, 1, 0))
    assert x.shape[-1] == 3, "Color channel must be at the end of the tensor {}".format(x.shape)

    if mode == "tf":
        x /= 2.
        x += 0.5
    elif mode == "image_net":
        # Scaling
        x[..., 0] *= 0.229
        x[..., 1] *= 0.224
        x[..., 2] *= 0.225
        # Undo Zero-center
        x[..., 0] += 0.485
        x[..., 1] += 0.456
        x[..., 2] += 0.406
    else:
        raise ValueError("Unknown mode for deNormalize")
    # Clip to fix numeric imprecision (1e-09 = 0)
    return np.clip(x, 0, 1)
