# coding: utf-8
from Utils import library_versions_tests, get_data_folder_from_model_name, produceRelevantImageStatesPlotMovie, get_movie_test_set_for_data_folder
from Utils import LEARNED_REPRESENTATIONS_FILE, SKIP_RENDERING, MOBILE_ROBOT, GIF_MOVIES_PATH, ALL_KNN_MOVIE_TEST_SETS, BENCHMARK_DATASETS, FOLDER_NAME_FOR_KNN_GIF_SEQS,PATH_TO_MOSAICS
from Utils import create_GIF_from_imgs_in_folder, get_immediate_subdirectories_path, get_immediate_files_in_path, create_mosaic_img_and_save
import numpy as np
import sys
import os.path
import subprocess
from sklearn.decomposition import PCA  # with some version of sklearn fails with ImportError: undefined symbol: PyFPE_jbuf
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.neighbors import NearestNeighbors
from PIL import Image

# ++++ This program assumes a folder containing all models where we have run  generate_neighbors_for_all_models_movie.sh
use_ground_truth = False


FOLDER_CONTAINING_ALL_MODELS = './Log/ALL_MODELS_KNNS'
datasets = BENCHMARK_DATASETS


print"\n\n >> Running makeMovieFromComparingKNNAcrossModels.py... FOLDER_CONTAINING_ALL_MODELS: ", FOLDER_CONTAINING_ALL_MODELS#, '\nALL_KNN_MOVIE_TEST_SETS: ', ALL_KNN_MOVIE_TEST_SETS
model_name = ''
# from glob import glob
# paths = glob(FOLDER_CONTAINING_ALL_MODELS)
# print paths

if len(sys.argv) == 2:
    datasets = [sys.argv[1]]
    ALL_KNN_MOVIE_TEST_SETS =[get_movie_test_set_for_data_folder(sys.argv[1])]
    print 'Using dataset only: ', datasets
# Some parameters

print 'Using datasets and KNN_TEST_SETS: ', datasets, ALL_KNN_MOVIE_TEST_SETS
for data_folder, test_set in zip(datasets, ALL_KNN_MOVIE_TEST_SETS):
    models_for_a_dataset = get_immediate_subdirectories_path(FOLDER_CONTAINING_ALL_MODELS+'/'+data_folder)#list_only_directories_in_path(FOLDER_CONTAINING_ALL_MODELS)
    GT_imgs = []
    AE_imgs = []
    supervised_imgs = []
    priors_imgs = []
    print 'Computing mosaics for models: ', models_for_a_dataset
    for model_folder in models_for_a_dataset:
        path_to_neighbors = FOLDER_CONTAINING_ALL_MODELS+'/'+data_folder+'/'+model_folder+FOLDER_NAME_FOR_KNN_GIF_SEQS
        #print 'Processing model ', model_folder, ' from dataset ', data_folder, '\n path_to_neighbors: ', path_to_neighbors
        if 'Supervised' in model_folder or 'supervised' in model_folder:
            supervised_imgs = get_immediate_files_in_path(path_to_neighbors, containing_pattern_in_name='_frame')
            if len(supervised_imgs)>0:
                if len(supervised_imgs)!=len(test_set):
                    print 'Sizes Model KNNs and Input KNNs: ', len(supervised_imgs), ' and ', len(test_set)
                    sys.exit("The size of the image sets in each model's folder should coincide! It does not for model_folder "+model_folder+' and supervised_imgs')
        elif 'AE' in model_folder:
            AE_imgs = get_immediate_files_in_path(path_to_neighbors, containing_pattern_in_name='_frame')
            if len(AE_imgs) >0:
                if len(AE_imgs) !=len(test_set):
                    print 'Sizes Model KNNs and Input KNNs:', len(AE_imgs), ' and ', len(test_set)
                    sys.exit("The size of the image sets in each model's folder should coincide! It does not for model_folder "+model_folder+' and AE_imgs')
        elif 'GT' in model_folder or 'ground_truth' in model_folder:
            GT_imgs = get_immediate_files_in_path(path_to_neighbors, containing_pattern_in_name='_frame')
            if len(GT_imgs)>0:
                if len(GT_imgs) !=len(test_set):
                    print 'Sizes Model KNNs and Input KNNs:', len(GT_imgs), ' and ', len(test_set)
                    sys.exit("The size of the image sets in each model's folder should coincide! It does not for model_folder "+model_folder+' and GT_imgs')
        else:
            if model_folder != '' and len(model_folder) >0: #     # TODO elif 'Priors' in model_folder: #                print 'Processi           print 'Processing Robotics priors model: ', path_to_neighbors
                priors_imgs = get_immediate_files_in_path(path_to_neighbors, containing_pattern_in_name='_frame') #list_only_files_in_path(path_to_neighbors, containing_pattern_in_name='_frame')
                if len(priors_imgs)>0:
                    if len(priors_imgs) !=len(test_set):
                        print 'Sizes Model KNNs and Input KNNs:', len(priors_imgs), ' and ', len(test_set)
                        sys.exit("The size of the image sets in each model's folder should coincide! It does not for model_folder "+model_folder+' and priors_imgs')
            else:
                sys.exit('Missing model for robotic priors in folder: '+data_folder)


    # create mosaics
    # FOR IT TO LOOK COHERENT! THEY NEED TO BE SORTED!! listing  files in Ubuntu does not return them sorted!! and they will be sampled randomly!!!!!!
    # SO, SORTING THEM (here, inline) IS CRUCIAL!
    test_set.sort()
    GT_imgs.sort()
    supervised_imgs.sort()
    priors_imgs.sort()
    AE_imgs.sort()
    if not os.path.exists(PATH_TO_MOSAICS):
        os.mkdir(PATH_TO_MOSAICS)
    if len(test_set)>0 and len(AE_imgs)>0 and len(supervised_imgs)>0 and len(priors_imgs)>0 :
        print 'Stitching images into a mosaic for dataset ',data_folder, ' Using models: ', models_for_a_dataset
        index =0
        if use_ground_truth:
            if use_ground_truth and len(GT_imgs) ==0:
                sys.exit('If you want to also plot ground truth neighbors, you should run first generate_neighbors_for_all_models_movie.sh and include a GT folder model there')      
            for input_img_test, gt, superv, prior, ae in zip(test_set, GT_imgs, supervised_imgs, priors_imgs, AE_imgs):
                path_to_mosaic_images = PATH_TO_MOSAICS+data_folder+'/'
                create_mosaic_img_and_save(input_img_test, [gt, superv, prior, ae], path_to_mosaic_images, 'mosaic_'+str(index)+'.jpg', top_title=get_data_folder_from_model_name(input_img_test), titles=['Ground Truth', 'Supervised (Robot Hand Position)', 'Robotic Priors', 'Denoising Auto-Encoder'])
                index +=1
        else:
            for input_img_test, superv, prior, ae in zip(test_set, supervised_imgs, priors_imgs, AE_imgs):
                path_to_mosaic_images = PATH_TO_MOSAICS+data_folder+'/'
                create_mosaic_img_and_save(input_img_test, [superv, prior, ae], path_to_mosaic_images, 'mosaic_'+str(index)+'.jpg', top_title=get_data_folder_from_model_name(input_img_test), titles=['Supervised (Robot Hand Position)', 'Robotic Priors', 'Denoising Auto-Encoder'])
                index +=1
        #TODO : AT THE MOMENT THE SPEED IS TOO FAST, SO using GIFMAKER.ME INSTEAD
        #create_GIF_from_imgs_in_folder(path_to_mosaic_images, './DEMO_GIFs/'+data_folder+'_KNN.gif')
    else:
        print 'Missing models for dataset (must be 4 at least): ', data_folder, ' Skipping this dataset for now'


