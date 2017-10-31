import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from sklearn.neighbors import NearestNeighbors
import shutil
import random
import sys
import pandas as pd
from PIL import Image
import os, os.path
import subprocess

from Utils import ALL_STATE_FILE, LEARNED_REPRESENTATIONS_FILE, LAST_MODEL_FILE, GLOBAL_SCORE_LOG_FILE, IMG_TEST_SET, COMPLEX_TEST_SET, STATIC_BUTTON_SIMPLEST, COMPLEX_DATA, MOBILE_ROBOT, ROBOT_TEST_SET, SUPERVISED, DEFAULT_DATASET, COLORFUL75, COMPLEX_DATA_MOVIE_TEST_SET, COLORFUL75_MOVIE_TEST_SET, STATIC_BUTTON_SIMPLEST_MOVIE_TEST_SET, COLORFUL_MOVIE_TEST_SET, MOBILE_ROBOT_MOVIE_TEST_SET, FOLDER_NAME_FOR_KNN_GIF_SEQS
from Utils import get_data_folder_from_model_name, file2dict, parse_repr_file, parse_true_state_file, get_test_set_for_data_folder, get_movie_test_set_for_data_folder
from Utils import BENCHMARK_DATASETS, get_immediate_subdirectories_path, get_immediate_files_in_path
import unittest
test = unittest.TestCase('__init__')
import imageio


"""
NOTE, if sklearn.neighbours import fails, remove  and install:
Either use conda (in which case all your installed packages would be in ~/miniconda/ or pip install --user don't mix the two and do not use -U, nor sudo.
 Removing either
rm -rf ~/.local/lib/python2.7/site-packages/sklearn or your ~/miniconda folder and reinstalling it cleanly should fix this.
sudo rm -rf scikit_learn-0.18.1.egg-info/
pip uninstall sklearn
and
1)  pip install --user scikit-learn
or 2) conda install -c anaconda scikit-learn=0.18.1
If needed, also do
pip install --user numpy
pip install --user scipy

NOTE: Q: if this error is obtained: _tkinter.TclError: no display name and no $DISPLAY environment variable
A: Instead of ssh account@machine, do: ssh -X

Example to run this program for a given trained model:
python generateNNImages.py 5 5 Log/modelY2017_D24_M06_H06M19S10_staticButtonSimplest_resnet_cont_MCD0_8_S0_4
IMPORTANT: In order to run it with a non random fixed test set of images,
call it with only one argument (the number of neigbours to generate for each
image in the test set and it will assess the test set of 50 images defined in Const.lua and Utils.py)

"""
# fig = plt.figure()
#     fig.set_size_inches(60,35)
#     a=fig.add_subplot(rows_in_mosaic, columns_in_mosaic, 2) # subplot(nrows, ncols, plot_number)
#     a.axis('off')
#     # img = mpimg.imread(img_name)
#     img = Image.open(input_reference_img_to_show_on_top)
#     imgplot = plt.imshow(img)

#     if len(top_title)>0:
#         a.set_title(top_title, fontsize = 60) 

#     # DRAW BELOW ALL MODELS IMAGES (KNN)
#     for i in range(0, len(list_of_input_imgs)):
#         a=fig.add_subplot(rows_in_mosaic, columns_in_mosaic, 4+i)#2+i)
#         img_name= list_of_input_imgs[i]
#         # img = mpimg.imread(img_name)
#         img = Image.open(img_name)
#         imgplot = plt.imshow(img)

#         if len(titles)>0:
#             a.set_title(titles[i], fontsize = 40) 
#         a.axis('off')

def create_GIF_from_imgs_in_folder(folder_rel_path, output_folder, output_file_name):
    if not os.path.exists(output_folder):
		os.mkdir(output_folder)
    filenames = get_immediate_files_in_path(folder_rel_path, '.jpg')
    #For longer movies, use the streaming approach 'I'
    output_full_path = output_folder+'/'+output_file_name
    with imageio.get_writer(output_full_path, mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
    print 'Made GIF from images in ', folder_rel_path, ' saved in ',output_file_name


def create_1_row_mosaic(path_to_folder_with_input_images, output_mosaic_path, titles, title ):
	#Generate mosaics
	input_images = get_immediate_files_in_path(path_to_folder_with_input_images, '.jpg')
	input_images.sort()
	rows = 1
	columns = len(input_images)
	output_file_name = 'DatasetsMosaic.png'
	print 'Computing mosaic for images ', input_images
	
	fig = plt.figure()
	fig.set_size_inches(60,35)
	a=fig.add_subplot(rows, columns, 2)	# subplot(nrows, ncols, plot_number)
	#a.axis('off')
	# img = mpimg.imread(img_name) # image on top
	# img = Image.open(input_reference_img_to_show_on_top)
	# imgplot = plt.imshow(img)
	if len(title)>0:
		a.set_title(title, fontsize = 60) 

	for i in range(0, len(input_images)):
		a=fig.add_subplot(rows, columns, i+1)
		img_name= input_images[i]
		img = Image.open(img_name)
		imgplot = plt.imshow(img)

		if len(titles)>0:
			a.set_title(titles[i], fontsize = 70)
		a.axis('off')

	plt.tight_layout()
	output_mosaic_path = path_to_folder_with_input_images+output_file_name
	plt.savefig(output_mosaic_path, bbox_inches='tight')
	print 'Saved mosaic in ', output_mosaic_path
	plt.close() # efficiency: avoids keeping all images into RAM


path_to_folder_with_input_images = './data_minisample/'
titles = ['Mobile Robot', 'Static-Button-Distractors\n3D', 'Complex-Data-Distractors\n(3D Cont.Act. 2 Arms)', 'Colorful75'] # BENCHMARK_DATASETS ;titles.reverse() # INTERESTING! reverse returns nil, as it is inplace method!
print titles
#create_1_row_mosaic(path_to_folder_with_input_images, path_to_folder_with_input_images, titles, 'Datasets')

# TODO
#create_4_row_mosaic(path_to_folder_with_input_images, path_to_folder_with_input_images, titles, 'Datasets')
create_GIF_from_imgs_in_folder(path_to_folder_with_input_images, 'GIF_MOVIES','test.gif')