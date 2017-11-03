from __future__ import print_function, division

import unittest
from PIL import Image

import matplotlib.pyplot as plt
import os
import os.path

from utils import get_immediate_files_in_path

import imageio

test = unittest.TestCase('__init__')

"""
Example to run this program for a given trained model:
python generateNNImages.py 5 5 Log/modelY2017_D24_M06_H06M19S10_staticButtonSimplest_resnet_cont_MCD0_8_S0_4
IMPORTANT: In order to run it with a non random fixed test set of images,
call it with only one argument (the number of neigbours to generate for each
image in the test set and it will assess the test set of 50 images defined in Const.lua and utils.py)

"""

def create_GIF_from_imgs_in_folder(folder_rel_path, output_folder, output_file_name):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    filenames = get_immediate_files_in_path(folder_rel_path, '.jpg')
    # For longer movies, use the streaming approach 'I'
    output_full_path = output_folder + '/' + output_file_name
    with imageio.get_writer(output_full_path, mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
    print('Made GIF from images in ', folder_rel_path, ' saved in ', output_file_name)


def create_1_row_mosaic(path_to_folder_with_input_images, output_mosaic_path, titles, title):
    # Generate mosaics
    input_images = get_immediate_files_in_path(path_to_folder_with_input_images, '.jpg')
    input_images.sort()
    rows = 1
    columns = len(input_images)
    output_file_name = 'DatasetsMosaic.png'
    print('Computing mosaic for images ', input_images)

    fig = plt.figure()
    fig.set_size_inches(60, 35)
    a = fig.add_subplot(rows, columns, 2)  # subplot(nrows, ncols, plot_number)
    # a.axis('off')
    # img = mpimg.imread(img_name) # image on top
    # img = Image.open(input_reference_img_to_show_on_top)
    # imgplot = plt.imshow(img)
    if len(title) > 0:
        a.set_title(title, fontsize=60)

    for i in range(0, len(input_images)):
        a = fig.add_subplot(rows, columns, i + 1)
        img_name = input_images[i]
        img = Image.open(img_name)
        plt.imshow(img)

        if len(titles) > 0:
            a.set_title(titles[i], fontsize=70)
        a.axis('off')

    plt.tight_layout()
    output_mosaic_path = path_to_folder_with_input_images + output_file_name
    plt.savefig(output_mosaic_path, bbox_inches='tight')
    print('Saved mosaic in ', output_mosaic_path)
    plt.close()  # efficiency: avoids keeping all images into RAM


path_to_folder_with_input_images = './data_minisample/'
titles = ['Mobile Robot', 'Static-Button-Distractors\n3D', 'Complex-Data-Distractors\n(3D Cont.Act. 2 Arms)', 'Colorful75']
# BENCHMARK_DATASETS ;titles.reverse()
#  INTERESTING! reverse returns nil, as it is inplace method!
print(titles)
# create_1_row_mosaic(path_to_folder_with_input_images, path_to_folder_with_input_images, titles, 'Datasets')

# TODO
# create_4_row_mosaic(path_to_folder_with_input_images, path_to_folder_with_input_images, titles, 'Datasets')
create_GIF_from_imgs_in_folder(path_to_folder_with_input_images, 'GIF_MOVIES', 'test.gif')
