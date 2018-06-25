"""
Preprocessing constants
"""
from __future__ import print_function, division, absolute_import

# Resized image shape
IMAGE_WIDTH = 224  # in px
IMAGE_HEIGHT = 224  # in px
N_CHANNELS = 3
INPUT_DIM = IMAGE_WIDTH * IMAGE_HEIGHT * N_CHANNELS


def getNChannels():
    return N_CHANNELS

def getInputDim():
    return IMAGE_WIDTH * IMAGE_HEIGHT * N_CHANNELS