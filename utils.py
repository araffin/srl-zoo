#!/usr/bin/python
# coding: utf-8
import matplotlib
#matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import sys
import numpy as np
import os, os.path, errno
import matplotlib
import seaborn as sns
from PIL import Image
import json
/home/seurin/baxter_representation_learning_3D/Utils.py

"""
Documentation for colorblind-supported plots: #http://seaborn.pydata.org/introduction.html
"""

SKIP_RENDERING = True  # Make True when running remotely via ssh for the batch/grid_search programs to save the plots and KNN figures folder
#DATASETS AVAILABLE:  NOTE: when adding a new dataset, add also to ALL_DATASETS for stats and logging consistency purposes
BABBLING = 'babbling'
MOBILE_ROBOT = 'mobileRobot'
SIMPLEDATA3D = 'simpleData3D'
PUSHING_BUTTON_AUGMENTED = 'pushingButton3DAugmented'
STATIC_BUTTON_SIMPLEST = 'staticButtonSimplest'
COMPLEX_DATA = 'complexData'
COLORFUL = 'colorful'  # 150 data recording sequences
COLORFUL75 = 'colorful75' # a smaller version half size of colorful
NONSTATIC_BUTTON = 'nonStaticButton'

ALL_DATASETS = [BABBLING, MOBILE_ROBOT, SIMPLEDATA3D, PUSHING_BUTTON_AUGMENTED, STATIC_BUTTON_SIMPLEST,COMPLEX_DATA, COLORFUL75, COLORFUL, NONSTATIC_BUTTON] #COLORFUL not in use yet due to memory issues
SUPERVISED = 'Supervised'
DEFAULT_DATASET = NONSTATIC_BUTTON  #COLORFUL75  # needs to be set for running all Python scripts in AE, GT? and Supervised modes
DIMENSIONS_OUT = 3 # default

# 2 options of plotting:
LEARNED_REPRESENTATIONS_FILE = "saveImagesAndRepr.txt"
GLOBAL_SCORE_LOG_FILE = 'globalScoreLog.csv'
MODELS_CONFIG_LOG_FILE  = 'modelsConfigLog.csv'
ALL_STATE_FILE = 'allStatesGT.txt'
LAST_MODEL_FILE = 'lastModel.txt'
ALL_STATS_FILE ='allStats.csv'
CONFIG = 'config.json' # not used yet, TODO
PATH_TO_LINEAR_MODEL = 'disentanglementLinearModels/'
GIF_MOVIES_PATH = 'GIF_MOVIES/'  # used for states plot movie
FOLDER_NAME_FOR_KNN_GIF_SEQS =  '/KNN_GIF_Seqs/'
PATH_TO_MOSAICS = './Mosaics/'
CONFIG_JSON_FILE = 'Config.json'
FILENAME_FOR_BUTTON_POSITION = 'recorded_button1_position.txt'  # content example: # x y z     # 0.599993419271 0.29998631216 -0.160117283495
USING_BUTTONS_RELATIVE_POSITION = False  # by default

# Priors
REP = "Rep"
CAUS = "Caus"
PROP = "Prop"
TEMP = "Temp"
BRING_CLOSER_REWARD = "Reward_closer"
BRING_CLOSER_REF_POINT = "Fixed_point"
REWARD_PREDICTION_CRITERION= 'Prediction Reward'

def save_config_to_file(config_dict, filename):
    """
    Saves config into json file for only one file to include important constans
    to be read by whole learning pipeline of lua and python scripts
    """
    print 'Saving config ',config_dict
    json.dump(config_dict, open(filename, 'wb'))

def read_config(filename=CONFIG_JSON_FILE):
    # load the data from json file into a dictionary
    CONFIG_DICT = json.load(open(filename, 'rb'))
    USING_BUTTONS_RELATIVE_POSITION = CONFIG_DICT['USING_BUTTONS_RELATIVE_POSITION']
    STATES_DIMENSION = CONFIG_DICT['STATES_DIMENSION']
    DATA_FOLDER = CONFIG_DICT['DATA_FOLDER']
    PRIORS_CONFIGS_TO_APPLY = CONFIG_DICT['PRIORS_CONFIGS_TO_APPLY']
    return CONFIG_DICT

### LOADING CONFIG FILE
if os.path.exists(CONFIG_JSON_FILE) and  os.path.isfile(CONFIG_JSON_FILE):
    CONFIG_DICT = read_config(CONFIG_JSON_FILE)
else:  # Default values
    CONFIG_DICT = {
        'DATA_FOLDER': DEFAULT_DATASET,
        'STATES_DIMENSION': DIMENSIONS_OUT,
        'PRIORS_CONFIGS_TO_APPLY': [PROP, TEMP, CAUS, REP, BRING_CLOSER_REF_POINT],
        'USING_BUTTONS_RELATIVE_POSITION': USING_BUTTONS_RELATIVE_POSITION
    }
    save_config_to_file(CONFIG_DICT, CONFIG_JSON_FILE)

# DEFINING A SET OF PREDEFINED IMAGES WE WANT ITS CORRESPONDING STATES FOR:
# they represent left up, right up, down right, down left corner and pushing button images (clockwise hand movement. Used by makeMovieFromPlotStates.py
REPRESENTATIVE_DIFFERENT_IMAGES = {COLORFUL75: ['colorful75/record_008/recorded_cameras_head_camera_2_image_compressed/frame00012.jpg',
'colorful75/record_008/recorded_cameras_head_camera_2_image_compressed/frame00087.jpg',
'colorful75/record_008/recorded_cameras_head_camera_2_image_compressed/frame00149.jpg',
'colorful75/record_008/recorded_cameras_head_camera_2_image_compressed/frame00011.jpg',
'colorful75/record_008/recorded_cameras_head_camera_2_image_compressed/frame00234.jpg'],
COLORFUL: ['colorful75/record_008/recorded_cameras_head_camera_2_image_compressed/frame00012.jpg',
'colorful75/record_008/recorded_cameras_head_camera_2_image_compressed/frame00087.jpg',
'colorful75/record_008/recorded_cameras_head_camera_2_image_compressed/frame00149.jpg',
'colorful75/record_008/recorded_cameras_head_camera_2_image_compressed/frame00011.jpg',
'colorful75/record_008/recorded_cameras_head_camera_2_image_compressed/frame00234.jpg'],
COMPLEX_DATA: ['complexData/record_008/recorded_cameras_head_camera_2_image_compressed/frame00001.jpg',
'complexData/record_008/recorded_cameras_head_camera_2_image_compressed/frame00070.jpg',
'complexData/record_008/recorded_cameras_head_camera_2_image_compressed/frame00103.jpg',
'complexData/record_008/recorded_cameras_head_camera_2_image_compressed/frame00176.jpg',
'complexData/record_008/recorded_cameras_head_camera_2_image_compressed/frame00109.jpg'],
STATIC_BUTTON_SIMPLEST:['staticButtonSimplest/record_043/recorded_camera_top/frame00000.jpg',
'staticButtonSimplest/record_043/recorded_camera_top/frame00020.jpg',
'staticButtonSimplest/record_043/recorded_camera_top/frame00071.jpg',
'staticButtonSimplest/record_043/recorded_camera_top/frame00028.jpg',
'staticButtonSimplest/record_043/recorded_camera_top/frame00050.jpg',
'staticButtonSimplest/record_043/recorded_camera_top/frame00009.jpg'],
MOBILE_ROBOT: ['mobileRobot/record_008/recorded_camera_top/frame00001.jpg',
'mobileRobot/record_008/recorded_camera_top/frame00072.jpg',
'mobileRobot/record_008/recorded_camera_top/frame00063.jpg',
'mobileRobot/record_008/recorded_camera_top/frame00011.jpg',
'mobileRobot/record_008/recorded_camera_top/frame00048.jpg',
'mobileRobot/record_008/recorded_camera_top/frame00090.jpg']}
# NEW DATASET AFTER ICRA18
NONSTATIC_BUTTON = []







def library_versions_tests():
    if not matplotlib.__version__.startswith('2.'):
        print "Using a too old matplotlib version (can be critical for properly plotting reward colours, otherwise the colors are difficult to see), to update, you need to do it via Anaconda: "
        print "Min version required is 2.0.0. Current version: ", matplotlib.__version__
        print "Option 1) (Preferred)\n - pip install --upgrade matplotlib (In general, prefer pip install --user (WITHOUT SUDO) to anaconda"
        print "2) To install anaconda (WARNING: can make sklearn PCA not work by installing a second version of numpy): \n -wget https://repo.continuum.io/archive/Anaconda2-4.4.0-Linux-x86_64.sh  \n -bash Anaconda2-4.4.0-Linux-x86_64.sh  \n -Restart terminal \n -conda update matplotlib"
        sys.exit(-1)

    numpy_versions_installed = np.__path__
    print "numpy_versions_installed: ", numpy_versions_installed
    if len(numpy_versions_installed)>1:
        print "Probably you have installed numpy with and without Anaconda, so there is a conflict because two numpy versions can be used."
        print "Remove non-Anaconda numpy:\n 1) pip uninstall numpy \n and if needed, install 2.1) pip install --user numpy  \n "
        print "2.2) If 1 does not work: last version in: \n -https://anaconda.org/anaconda/numpy"

    # common error, check your version: from sklearn.neighbors import NearestNeighbors
    # File "/home/seurin/anaconda2/lib/python2.7/site-packages/sklearn/__init__.py", line 134, in <module>
    #     from .base import clone
    #   File "/home/seurin/anaconda2/lib/python2.7/site-packages/sklearn/base.py", line 11, in <module>
    #     from scipy import sparse
    #   File "/home/seurin/.local/lib/python2.7/site-packages/scipy/__init__.py", line 114, in <module>
    #     from scipy._lib._ccallback import LowLevelCallable
    #   File "/home/seurin/.local/lib/python2.7/site-packages/scipy/_lib/_ccallback.py", line 1, in <module>
    #     from . import _ccallback_c
    # ImportError: /home/seurin/.local/lib/python2.7/site-packages/scipy/_lib/_ccallback_c.so: undefined symbol: PyFPE_jbuf
    import sklearn; print sklearn.__version__
    # If fail, try install a stable one: e.g. '0.16.1' does not work, but 0.16 does.      conda install -c anaconda scikit-learn=0.18.1

# def create_GIF_from_imgs_in_folder(folder_rel_path, output_folder, output_file_name):
#     # TODO: does not work properly, plus needs imageio installation, GIF created has to be rendered slower, other ideas in https://stackoverflow.com/questions/24688802/saving-an-animated-gif-in-pillow
#     # or https://stackoverflow.com/questions/753190/programmatically-generate-video-or-animated-gif-in-python
#     # For now gifmaker.me works better up to 300 frames, and if not, ffmpeg:
#     # ffmpeg -framerate 2 -f image2 -i "%*.jpg" -q:v 2  test.mp4
#     if not os.path.exists(output_folder):
#         os.mkdir(output_folder)
#     filenames = get_immediate_files_in_path(folder_rel_path, '.jpg')
#     #For longer movies, use the streaming approach 'I'
#     output_full_path = output_folder+'/'+output_file_name
#     with imageio.get_writer(output_full_path, mode='I') as writer:
#         for filename in filenames:
#             image = imageio.imread(filename)
#             writer.append_data(image)
#     print 'Made GIF from images in ', folder_rel_path, ' saved in ',output_file_name


def get_data_folder_from_model_name(model_name):

    if BABBLING in model_name:
        return BABBLING
    elif MOBILE_ROBOT in model_name:
        return MOBILE_ROBOT
    elif SIMPLEDATA3D in model_name:
        return SIMPLEDATA3D
    elif PUSHING_BUTTON_AUGMENTED in model_name:
        return PUSHING_BUTTON_AUGMENTED
    elif STATIC_BUTTON_SIMPLEST in model_name:
        return STATIC_BUTTON_SIMPLEST
    elif COMPLEX_DATA in model_name or 'complex' in model_name:
        return COMPLEX_DATA
    elif COLORFUL75 in model_name:  # VERY IMPORTANT THE ORDER! TO NOT PROCESS THE WRONG SUPER LARGE DATASET WHEN RESOURCES NOT AVAILABLE!
        return COLORFUL75
    elif COLORFUL in model_name:
        return COLORFUL
    elif NONSTATIC_BUTTON in model_name:
        return NONSTATIC_BUTTON
    else:
        print model_name
        sys.exit("get_data_folder_from_model_name: Unsupported dataset! model_name must contain one of the official datasets defined in Utils.py, input is: "+ model_name)


# def get_visible_states_for_images(specific_images):
#     return REPRESENTATIVE_DIFFERENT_IMAGES[]

# def produceEvolvingRelevantImageStatesPlotMovie(images2states, mode, rewards, toplot, model_name, axes_labels = ['State Dimension 1','State Dimension 2','State Dimension 3'], title='Learned Representations-Rewards Distribution\n'):
#     list_of_colors = [(0.3,0.3,0.3), (0.0,0.0,1.0), (1,0,0)]
#     #list_of_colors = [(0.0, 0.0, 1.0),(1,0,0), (0.3, 0.3, 0.3), (0,1,0), (1,0.5,0), (0.5, 0, 0.5)]
#     data_folder = get_data_folder_from_model_name(model_name)
#     specific_images = REPRESENTATIVE_DIFFERENT_IMAGES[data_folder]
#     #TODO toPlot = get_states_for_images(specific_images)
#     #FIX
#     #toplot = toplot[:3]
#     plot_path = GIF_MOVIES_PATH+model_name+'/'
#     # if not os.path.exists(plot_path):
#     #     os.makedirs(plot_path)
#     try:
#         os.makedirs(plot_path)
#     except OSError as e:
#         if e.errno != errno.EEXIST:
#             raise
#     plot_path = plot_path+'StatesDemo'+model_name+'_*.png'
#     for n_states_in_plot in range(1, len(toplot)):
#         # save one plot per point to make a GIF movie with increasing number of states being represented
#         statesToPlot = toplot[:n_states_in_plot]
#         rewardsToPlot = rewards[:n_states_in_plot]
#         images_states_to_plot = specific_images[:n_states_in_plot]
#         colors_to_use = list_of_colors[:n_states_in_plot]
#         if not (0.0, 0.0, 1.0) in colors_to_use:
#             colors_to_use.append((0.0, 0.0, 1.0))
#         if not (1,0,0) in colors_to_use:
#             colors_to_use.append((1,0,0))
#         print statesToPlot, rewardsToPlot, images_states_to_plot, colors_to_use
#         plotStates(mode, rewardsToPlot, statesToPlot, plot_path.replace('*', str(n_states_in_plot)), dataset=data_folder, specific_images=images_states_to_plot, list_of_colors = colors_to_use)
#     create_movie_from_folder(plot_path, model_name)

def stitch_images_into_one_and_save(input_folder, output_folder, file_name):
    print 'stitch_images_into_one_and_save wrote GIF movie to file: ', output_folder

def create_mosaic_img_and_save(input_reference_img_to_show_on_top, list_of_input_imgs, path_to_image_directory, output_file_name, top_title='', titles=[]):
    print "Creating mosaic from input reference image ", input_reference_img_to_show_on_top, '\nUsing images: ', list_of_input_imgs, 'saving it to ', path_to_image_directory

    if not os.path.exists(path_to_image_directory):
        os.mkdir(path_to_image_directory)

    rows_in_mosaic = 2 #len(list_of_input_imgs) +1   # number of rows to show in the image mosaic
    columns_in_mosaic = 3 #1
    with_title = True

    # DRAW FIRST REFERENCE INPUT IMAGE FIRST
    fig = plt.figure()
    fig.set_size_inches(60,35)
    a=fig.add_subplot(rows_in_mosaic, columns_in_mosaic, 2) # subplot(nrows, ncols, plot_number)
    a.axis('off')
    # img = mpimg.imread(img_name)
    img = Image.open(input_reference_img_to_show_on_top)
    imgplot = plt.imshow(img)

    if len(top_title)>0:
        a.set_title(top_title, fontsize = 60)

    # DRAW BELOW ALL MODELS IMAGES (KNN)
    for i in range(0, len(list_of_input_imgs)):
        a=fig.add_subplot(rows_in_mosaic, columns_in_mosaic, 4+i)#2+i)
        img_name= list_of_input_imgs[i]
        # img = mpimg.imread(img_name)
        img = Image.open(img_name)
        imgplot = plt.imshow(img)

        if len(titles)>0:
            a.set_title(titles[i], fontsize = 40)
        a.axis('off')

    plt.tight_layout()
    output_file = path_to_image_directory+output_file_name

    plt.savefig(output_file, bbox_inches='tight')
    plt.close() # efficiency: avoids keeping all images into RAM
    print 'Created mosaic in ', output_file

def produceRelevantImageStatesPlotMovie(mode, rewards, toplot, img_paths2repr, model_name, axes_labels = ['State Dimension 1','State Dimension 2','State Dimension 3'], title='Learned Representations-Rewards Distribution\n'):
    # Produces full (all states) static plot GIF while the Evolving corresponding method provides an evolving (!= nr of states at each plot generated, where axis scale and labelling changes and squeezes the axes
    colors_to_use = [(0.3,0.3,0.3), (0.0,0.0,1.0), (1,0,0)]
    model_category = ''
    #colors_to_use = [(0.0, 0.0, 1.0),(1,0,0), (0.3, 0.3, 0.3), (0,1,0), (1,0.5,0), (0.5, 0, 0.5)]
    data_folder = get_data_folder_from_model_name(model_name)
    specific_images_to_plot = REPRESENTATIVE_DIFFERENT_IMAGES[data_folder]
    specific_images_to_plot.sort()
    statesToPlot = []; rewardsToPlot = []; images = []
    axis_limits = [[np.min(toplot[:,0]), np.max(toplot[:,0])],
    [np.min(toplot[:,0]), np.max(toplot[:,0])],
    [np.min(toplot[:,0]), np.max(toplot[:,0])]]

    #for img in range(len(specific_images_to_plot)):
    for img in specific_images_to_plot:
        statesToPlot.append(img_paths2repr[img][0])
        rewardsToPlot.append(img_paths2repr[img][1])
        images.append(img)
    print axis_limits
    print images
    #rewardsToPlot = np.array(rewardsToPlot) #map(float, rewardsToPlot)
    # else:
    #     toplot_invisible.append(toplot[i])
    #     rewards_invisible.append(rewards[i])
    #TODO toPlot = get_states_for_images(specific_images)
    #FIX
    #toplot = toplot[:3]
    plot_path = GIF_MOVIES_PATH+model_name+'/'
    # if not os.path.exists(plot_path):
    #     os.makedirs(plot_path)
    try:
        os.makedirs(plot_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    plot_filename_template = plot_path+model_name+'_GIF_frame*.png'
    if not (0.0, 0.0, 1.0) in colors_to_use:
        colors_to_use.append((0.0, 0.0, 1.0))
    if not (1,0,0) in colors_to_use:
        colors_to_use.append((1,0,0))

    for n_states_in_plot in range(1, len(statesToPlot)+1):
        # save one plot per point to make a GIF movie with increasing number of states being represented   states = np.zeros((n_states_in_plot, len(toplot[0])))
        #rewards_visible = []#np.zeros((n_states_in_plot, 1))
        for s in range(1, n_states_in_plot+1):
            states = np.zeros((s, len(statesToPlot[0])))
            for i in range(len(states)):
                states[i] = np.array(statesToPlot[i]) #            states = statesToPlot[:s] #np.array(statesToPlot[s])
        rewards_visible = rewardsToPlot[:n_states_in_plot] #.append(rewardsToPlot[s]) #np.array(rewardsToPlot[s]) #rewardsToPlot[:s]
        colors_visible = colors_to_use[:n_states_in_plot]  #n_states_in_plot>0
        # images_states_to_plot = specific_images[:n_states_in_plot] #        colors_to_use = list_of_colors[:n_states_in_plot]
        #https://stackoverflow.com/questions/30686157/python-matplotlib-invisible-point  , alpha=0.5  ('2D', rewards, toplot, img_paths2repr, model_name)
        plot_filename = plot_filename_template.replace('*', str(n_states_in_plot-1))
        print 'Plotting rewards and states: ', rewards_visible, states, rewards_visible[-1]
        if n_states_in_plot:
            plotStates(mode, rewards_visible, states, plot_filename, dataset=data_folder, title='State Representation Learned Space: '+model_category+'\nDataset: '+data_folder, axis_limits= axis_limits, one_reward_value=rewards_visible[-1])#, list_of_colors = colors_visible)
        else:
            sys.exit('ERROR in produceRelevantImageStatesPlotMovie! Make sure you have a representative image set to make the gift, defined in REPRESENTATIVE_DIFFERENT_IMAGES for this dataset!')
        #plotStates(mode, rewards_invisible, statesToPlot, plot_path.replace('*', str(n_states_in_plot)), dataset=data_folder, specific_images=images_states_to_plot, list_of_colors = colors_to_use)
    # draw all finally
    plotStates(mode, rewards, toplot, plot_filename_template.replace('*',str(n_states_in_plot)), dataset=model_name, title='State Representation Learned Space: '+model_category+'\nDataset: '+data_folder, axis_limits= axis_limits)
    #create_GIF_from_imgs_in_folder(plot_path, output_folder, plot_file_name)


"""
Use this function if rewards need to be visualized, use plot_3D otherwise
"""
def plotStates(mode, rewards, toplot, plot_path, axes_labels = ['State Dimension 1','State Dimension 2','State Dimension 3'], title='Learned Representations-Rewards Distribution\n', dataset='', list_of_colors = [], axis_limits = [[]], one_reward_value=''):
    # INPUT: mode: '2d" or '3D'
    # rewards: an array of  rewards in string shape
    # toplot: an np.ndarray of states (n_states, dim_of_states)
    # Plots states either learned or the ground truth
    # Useful documentation: https://matplotlib.org/examples/mplot3d/scatter3d_demo.html
    # Against colourblindness: https://chrisalbon.com/python/seaborn_color_palettes.html
    # TODO: add vertical color bar for representing reward values  https://matplotlib.org/examples/api/colorbar_only.html
    reward_values = set(rewards)
    rewards_cardinal = len(reward_values)
    print'plotStates ',mode,' for rewards cardinal: ',rewards_cardinal,' (', reward_values,' ',type(reward_values),': ',reward_values,'), states: ', type(toplot)#, toplot.shape

    rewards = map(float, rewards)

    # custom Red Gray Blue colormap
    if len(list_of_colors) == 0: # default for 3 values of reward possible
        print "Using 3 default colors"
        list_of_colors = [(0.3,0.3,0.3), (0.0,0.0,1.0), (1,0,0)]
        cmap = colors.ListedColormap(['gray', 'blue', 'red'])     # print "cmap: ",type(cmap)
    else:
        cmap = colors.ListedColormap(['gray', 'blue', 'red', 'Pastel1', 'Pastel2', 'Paired', 'Accent','Dark2'])#'orange', 'purple'])
    if 'State Representation Learned Space' in title:
        #plot only one state and therefore concatenating the dataset is redundant
        n_bins = 3 # TODO make function based on rewards_cardinal AND dataset
    else:
        n_bins = 100
    cmap_name = 'rgrayb'
    print 'Using list_of_colors ',list_of_colors, type(list_of_colors)
    cm = LinearSegmentedColormap.from_list(cmap_name, list_of_colors, n_bins)

    colorblind_palette = sns.color_palette("colorblind", rewards_cardinal)  # 3 is the number of different colours to use
    #print(type(colorblind_palette))
    #cubehelix_pal_cmap = sns.cubehelix_palette(as_cmap=True)
    #print(type(cubehelix_pal_cmap))

    #sns.palplot(sns.color_palette("colorblind", 10))
    #sns.color_palette()
    #sns.set_palette('colorblind')

    colorblind_cmap = ListedColormap(colorblind_palette) # not used
    colormap = cmap
    bounds=[-1,0,9,15]
    norm = colors.BoundaryNorm(bounds, colormap.N)
    # TODO: for some reason, no matther if I use cmap=cmap or make cmap=colorblind_palette work, it prints just 2 colors too similar for a colorblind person

    fig = plt.figure()
    if mode =='2D':
        ax = fig.add_subplot(111)
        # colors_markers = [('r', 'o', -10, 0.5), ('b', '^', 0.5, 10)]
        # for c, m, zlow, zhigh in colors_markers:
        #     ax.scatter(toplot[:,0], toplot[:,1], c=c, marker=m)
        cax = ax.scatter(toplot[:,0], toplot[:,1], c=rewards, cmap=cmap, norm=norm, marker=".")#,fillstyle=None)
    elif mode == '3D':
        ax = fig.add_subplot(111, projection='3d')
        # for c, m, zlow, zhigh in colors_markers:
        #     ax.scatter(toplot[:,0], toplot[:,1], toplot[:,2], c=c, marker=m)
        cax = ax.scatter(toplot[:,0], toplot[:,1], toplot[:,2], c=rewards, cmap=cm, marker=".")#, linestyle="")#, color = 'b')#,fillstyle=None)
        ax.set_zlabel(axes_labels[2])
    else:
        sys.exit("only mode '2D' and '3D' plot supported")

    ax.set_xlabel(axes_labels[0])
    ax.set_ylabel(axes_labels[1])

    if len(axis_limits[0]) >0:
        ax.set_xlim(axis_limits[0][0], axis_limits[0][1])
        ax.set_ylim(axis_limits[1][0], axis_limits[1][1])
        ax.set_zlim(axis_limits[2][0], axis_limits[2][1])
    if 'GroundTruth' in plot_path:
        ax.set_title(title.replace('Learned Representations','Ground Truth')+dataset)
    else:
        if 'Representation Learned Space' in title:
            #plot only one state and therefore concatenating the dataset is redundant
            ax.set_title(title+'\n                                                                            Reward: '+str(one_reward_value).replace('recorded_cameras_head_camera_2_image_compressed/',''))
        else:
            ax.set_title(title+dataset)

    # adding color bar
    cbar = fig.colorbar(cax, ticks=[-1, 0, 1])
    cbar.ax.set_yticklabels(['-1', '0', '1'])  # vertically oriented colorbar
    plt.savefig(plot_path)
    #plt.colorbar()  #TODO WHY IT DOES NOT SHOW AND SHOWS A PALETTE INSTEAD?
    if not SKIP_RENDERING:  # IMPORTANT TO SAVE BEFORE SHOWING SO THAT IMAGES DO NOT BECOME BLANK!
        plt.show()
    print('\nSaved plot to '+plot_path)


"""
Use this function if rewards DO NOT need to be visualized, use plotStates otherwise
"""
def plot_3D(x =[1,2,3,4,5,6,7,8,9,10], y =[5,6,2,3,13,4,1,2,4,8], z =[2,3,3,3,5,7,9,11,9,10], axes_labels = ['U','V','W'], title='Learned representations-rewards distribution\n', dataset=''):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z, c='r', marker='o')  # 'r' : red

    ax.set_xlabel(axes_labels[0])
    ax.set_ylabel(axes_labels[1])
    ax.set_zlabel(axes_labels[2])
    ax.set_title(title+dataset)


def file2dict(file): # DO SAME FUNCTIONS IN LUA and call at the end of set_hyperparams() method SKIP_VISUALIZATIOn, USE_CUDA and all the other params to used them in the subprocess subroutine.
    d = {}
    with open(file) as f:
        for line in f:
            if line[0]!='#':
               key_and_values = line.split()
               key, values = key_and_values[0], key_and_values[1:]
               d[key] = map(float, values)
    return d

def parse_true_state_file(dataset):
    true_states = {}
    if USING_BUTTONS_RELATIVE_POSITION:
        all_states_file = ALL_STATE_FILE.replace('.txt', ('_'+dataset+'RelativePos.txt'))
    else:
        all_states_file = ALL_STATE_FILE.replace('.txt', ('_'+dataset+'.txt'))
    file_state = open(all_states_file, "r")

    for line in file_state:
        if line[0]!='#':
            words = line.split()
            true_states[words[0]] = np.array(map(float,words[1:]))
    print "parse_true_state_file: ",all_states_file," returned #true_states: ",len(true_states)
    if len(true_states) == 0:
        sys.exit('parse_true_state_file could not find any states file!')
    return true_states

def parse_repr_file(learned_representations_file):
    images=[]
    representations=[]

    #reading data
    file_representation = open(learned_representations_file, "r")
    for line in file_representation:
        if line[0]!='#':
            words = line.split()
            images.append(words[0])
            representations.append(words[1:])
    print "parse_repr_file: ",learned_representations_file," returned #representations: ",len(representations)
    if len(images) == 0:
        sys.exit('parse_repr_file could not find any images !')
    if len(representations) == 0:
        sys.exit('parse_repr_file could not find any representations file!: ',learned_representations_file)
    return images, representations

def get_test_set_for_data_folder(data_folder):
    # Returns a dictionary (notice, with unique keys) of test images. Used to create movie from scatterplot
    # TODO : extend for other datasets for comparison, e.g. babbling.
    if data_folder == STATIC_BUTTON_SIMPLEST:
        return IMG_TEST_SET
    elif data_folder == COMPLEX_DATA:
        return COMPLEX_TEST_SET
    elif data_folder == COLORFUL75:
        return COLORFUL75_TEST_SET
    elif data_folder == COLORFUL:
        return COLORFUL_TEST_SET
    elif data_folder == MOBILE_ROBOT:
        return ROBOT_TEST_SET
    elif data_folder == NONSTATIC_BUTTON:
        return NONSTATIC_BUTTON_TEST_SET
    elif SUPERVISED in data_folder or 'supervised' in data_folder:
        return SUPERVISED
    else:
        sys.exit('get_test_set_for_data_folder has not defined a set for: {}'.format(data_folder))

def get_movie_test_set_for_data_folder(data_folder):
    # returns the ordered sequence of first 50 frames in the test sequence for movie smoothness creation purpose
    # Used to create movie from KNN mosaics
    if data_folder == STATIC_BUTTON_SIMPLEST:
        return STATIC_BUTTON_SIMPLEST_MOVIE_TEST_SET
    elif data_folder == COMPLEX_DATA:
        return COMPLEX_DATA_MOVIE_TEST_SET
    elif data_folder == COLORFUL75:
        return COLORFUL75_MOVIE_TEST_SET
    elif data_folder == COLORFUL:
        return COLORFUL_MOVIE_TEST_SET #'not yet'
    elif data_folder == MOBILE_ROBOT:
        return MOBILE_ROBOT_MOVIE_TEST_SET # not yet
    elif data_folder == NONSTATIC_BUTTON:
        return NONSTATIC_BUTTON_TEST_SET  # TODO create NONSTATIC_BUTTON_MOVIE_TEST_SET
    elif 'supervised' in data_folder or SUPERVISED in data_folder:
        return DEFAULT_DATASET
    else:
        sys.exit('get_movie_test_set_for_data_folder has not defined a set for: {}'.format(data_folder))

def remove_dataset_name(string):
    for dataset in ALL_DATASETS:
        if string.endswith(dataset):
            return string[:(len(string)-len(dataset))]

def get_immediate_subdirectories_path(given_path, containing_pattern_in_name = ''):  # TODO add param for relative path vs just folder names
    return [name for name in os.listdir(given_path)
            if os.path.isdir(os.path.join(given_path, name)) and containing_pattern_in_name in os.path.join(given_path, name)]

def get_immediate_files_in_path(given_path, containing_pattern_in_name = ''):
    return [os.path.join(given_path, name) for name in os.listdir(given_path)
            if os.path.isfile(os.path.join(given_path, name)) and containing_pattern_in_name in os.path.join(given_path, name)]



# 50 lines, 49 images (1 repeated by error) IMAGES TEST SET HANDPICKED TO SHOW VISUAL VARIABILITY
IMG_TEST_SET = {
'staticButtonSimplest/record_000/recorded_cameras_head_camera_2_image_compressed/frame00000.jpg',
'staticButtonSimplest/record_000/recorded_cameras_head_camera_2_image_compressed/frame00012.jpg',
'staticButtonSimplest/record_000/recorded_cameras_head_camera_2_image_compressed/frame00015.jpg',
'staticButtonSimplest/record_000/recorded_cameras_head_camera_2_image_compressed/frame00042.jpg',
'staticButtonSimplest/record_000/recorded_cameras_head_camera_2_image_compressed/frame00039.jpg',
'staticButtonSimplest/record_000/recorded_cameras_head_camera_2_image_compressed/frame00065.jpg',
'staticButtonSimplest/record_000/recorded_cameras_head_camera_2_image_compressed/frame00048.jpg',
'staticButtonSimplest/record_000/recorded_cameras_head_camera_2_image_compressed/frame00080.jpg',
'staticButtonSimplest/record_000/recorded_cameras_head_camera_2_image_compressed/frame00004.jpg',
'staticButtonSimplest/record_000/recorded_cameras_head_camera_2_image_compressed/frame00078.jpg',

'staticButtonSimplest/record_008/recorded_cameras_head_camera_2_image_compressed/frame00056.jpg',
'staticButtonSimplest/record_008/recorded_cameras_head_camera_2_image_compressed/frame00047.jpg',
'staticButtonSimplest/record_008/recorded_cameras_head_camera_2_image_compressed/frame00033.jpg',
'staticButtonSimplest/record_008/recorded_cameras_head_camera_2_image_compressed/frame00005.jpg',
'staticButtonSimplest/record_008/recorded_cameras_head_camera_2_image_compressed/frame00026.jpg',
'staticButtonSimplest/record_008/recorded_cameras_head_camera_2_image_compressed/frame00056.jpg',

'staticButtonSimplest/record_011/recorded_cameras_head_camera_2_image_compressed/frame00003.jpg',
'staticButtonSimplest/record_011/recorded_cameras_head_camera_2_image_compressed/frame00056.jpg',
'staticButtonSimplest/record_011/recorded_cameras_head_camera_2_image_compressed/frame00063.jpg',
'staticButtonSimplest/record_011/recorded_cameras_head_camera_2_image_compressed/frame00035.jpg',
'staticButtonSimplest/record_011/recorded_cameras_head_camera_2_image_compressed/frame00015.jpg',

'staticButtonSimplest/record_019/recorded_cameras_head_camera_2_image_compressed/frame00009.jpg',
'staticButtonSimplest/record_019/recorded_cameras_head_camera_2_image_compressed/frame00074.jpg',
'staticButtonSimplest/record_019/recorded_cameras_head_camera_2_image_compressed/frame00049.jpg',

'staticButtonSimplest/record_022/recorded_cameras_head_camera_2_image_compressed/frame00039.jpg',
'staticButtonSimplest/record_022/recorded_cameras_head_camera_2_image_compressed/frame00085.jpg',
'staticButtonSimplest/record_022/recorded_cameras_head_camera_2_image_compressed/frame00000.jpg',

'staticButtonSimplest/record_031/recorded_cameras_head_camera_2_image_compressed/frame00000.jpg',
'staticButtonSimplest/record_031/recorded_cameras_head_camera_2_image_compressed/frame00007.jpg',
'staticButtonSimplest/record_031/recorded_cameras_head_camera_2_image_compressed/frame00070.jpg',

'staticButtonSimplest/record_036/recorded_cameras_head_camera_2_image_compressed/frame00085.jpg',
'staticButtonSimplest/record_036/recorded_cameras_head_camera_2_image_compressed/frame00023.jpg',
'staticButtonSimplest/record_036/recorded_cameras_head_camera_2_image_compressed/frame00036.jpg',

'staticButtonSimplest/record_037/recorded_cameras_head_camera_2_image_compressed/frame00053.jpg',
'staticButtonSimplest/record_037/recorded_cameras_head_camera_2_image_compressed/frame00083.jpg',
'staticButtonSimplest/record_037/recorded_cameras_head_camera_2_image_compressed/frame00032.jpg',

'staticButtonSimplest/record_040/recorded_cameras_head_camera_2_image_compressed/frame00045.jpg',
'staticButtonSimplest/record_040/recorded_cameras_head_camera_2_image_compressed/frame00003.jpg',
'staticButtonSimplest/record_040/recorded_cameras_head_camera_2_image_compressed/frame00080.jpg',

'staticButtonSimplest/record_048/recorded_cameras_head_camera_2_image_compressed/frame00034.jpg',
'staticButtonSimplest/record_048/recorded_cameras_head_camera_2_image_compressed/frame00059.jpg',
'staticButtonSimplest/record_048/recorded_cameras_head_camera_2_image_compressed/frame00089.jpg',
'staticButtonSimplest/record_048/recorded_cameras_head_camera_2_image_compressed/frame00030.jpg',

'staticButtonSimplest/record_050/recorded_cameras_head_camera_2_image_compressed/frame00064.jpg',
'staticButtonSimplest/record_050/recorded_cameras_head_camera_2_image_compressed/frame00019.jpg',
'staticButtonSimplest/record_050/recorded_cameras_head_camera_2_image_compressed/frame00008.jpg',

'staticButtonSimplest/record_052/recorded_cameras_head_camera_2_image_compressed/frame00000.jpg',
'staticButtonSimplest/record_052/recorded_cameras_head_camera_2_image_compressed/frame00008.jpg',
'staticButtonSimplest/record_052/recorded_cameras_head_camera_2_image_compressed/frame00068.jpg',
'staticButtonSimplest/record_052/recorded_cameras_head_camera_2_image_compressed/frame00025.jpg'}
#print(len(IMG_TEST_SET))

# 50 unique images
COMPLEX_TEST_SET = {
'complexData/record_025/recorded_cameras_head_camera_2_image_compressed/frame00030.jpg',
'complexData/record_025/recorded_cameras_head_camera_2_image_compressed/frame00003.jpg',
'complexData/record_025/recorded_cameras_head_camera_2_image_compressed/frame00021.jpg',
'complexData/record_025/recorded_cameras_head_camera_2_image_compressed/frame00025.jpg',
'complexData/record_025/recorded_cameras_head_camera_2_image_compressed/frame00014.jpg',
'complexData/record_025/recorded_cameras_head_camera_2_image_compressed/frame00027.jpg',
'complexData/record_025/recorded_cameras_head_camera_2_image_compressed/frame00034.jpg',
'complexData/record_025/recorded_cameras_head_camera_2_image_compressed/frame00016.jpg',
'complexData/record_025/recorded_cameras_head_camera_2_image_compressed/frame00001.jpg',
'complexData/record_025/recorded_cameras_head_camera_2_image_compressed/frame00026.jpg',
'complexData/record_025/recorded_cameras_head_camera_2_image_compressed/frame00015.jpg',
'complexData/record_025/recorded_cameras_head_camera_2_image_compressed/frame00011.jpg',
'complexData/record_025/recorded_cameras_head_camera_2_image_compressed/frame00047.jpg',
'complexData/record_025/recorded_cameras_head_camera_2_image_compressed/frame00020.jpg',
'complexData/record_025/recorded_cameras_head_camera_2_image_compressed/frame00012.jpg',
'complexData/record_025/recorded_cameras_head_camera_2_image_compressed/frame00029.jpg',
'complexData/record_025/recorded_cameras_head_camera_2_image_compressed/frame00045.jpg',
'complexData/record_025/recorded_cameras_head_camera_2_image_compressed/frame00049.jpg',
'complexData/record_025/recorded_cameras_head_camera_2_image_compressed/frame00039.jpg',
'complexData/record_025/recorded_cameras_head_camera_2_image_compressed/frame00038.jpg',
'complexData/record_025/recorded_cameras_head_camera_2_image_compressed/frame00032.jpg',
'complexData/record_025/recorded_cameras_head_camera_2_image_compressed/frame00028.jpg',
'complexData/record_025/recorded_cameras_head_camera_2_image_compressed/frame00037.jpg',
'complexData/record_025/recorded_cameras_head_camera_2_image_compressed/frame00005.jpg',
'complexData/record_025/recorded_cameras_head_camera_2_image_compressed/frame00004.jpg',
'complexData/record_025/recorded_cameras_head_camera_2_image_compressed/frame00040.jpg',
'complexData/record_025/recorded_cameras_head_camera_2_image_compressed/frame00017.jpg',
'complexData/record_025/recorded_cameras_head_camera_2_image_compressed/frame00008.jpg',
'complexData/record_025/recorded_cameras_head_camera_2_image_compressed/frame00006.jpg',
'complexData/record_025/recorded_cameras_head_camera_2_image_compressed/frame00031.jpg',
'complexData/record_025/recorded_cameras_head_camera_2_image_compressed/frame00035.jpg',
'complexData/record_025/recorded_cameras_head_camera_2_image_compressed/frame00042.jpg',
'complexData/record_025/recorded_cameras_head_camera_2_image_compressed/frame00000.jpg',
'complexData/record_025/recorded_cameras_head_camera_2_image_compressed/frame00036.jpg',
'complexData/record_025/recorded_cameras_head_camera_2_image_compressed/frame00002.jpg',
'complexData/record_025/recorded_cameras_head_camera_2_image_compressed/frame00044.jpg',
'complexData/record_025/recorded_cameras_head_camera_2_image_compressed/frame00018.jpg',
'complexData/record_025/recorded_cameras_head_camera_2_image_compressed/frame00041.jpg',
'complexData/record_025/recorded_cameras_head_camera_2_image_compressed/frame00013.jpg',
'complexData/record_025/recorded_cameras_head_camera_2_image_compressed/frame00033.jpg',
'complexData/record_025/recorded_cameras_head_camera_2_image_compressed/frame00048.jpg',
'complexData/record_025/recorded_cameras_head_camera_2_image_compressed/frame00009.jpg',
'complexData/record_025/recorded_cameras_head_camera_2_image_compressed/frame00024.jpg',
'complexData/record_025/recorded_cameras_head_camera_2_image_compressed/frame00010.jpg',
'complexData/record_025/recorded_cameras_head_camera_2_image_compressed/frame00022.jpg',
'complexData/record_025/recorded_cameras_head_camera_2_image_compressed/frame00043.jpg',
'complexData/record_025/recorded_cameras_head_camera_2_image_compressed/frame00007.jpg',
'complexData/record_025/recorded_cameras_head_camera_2_image_compressed/frame00023.jpg',
'complexData/record_025/recorded_cameras_head_camera_2_image_compressed/frame00019.jpg',
'complexData/record_025/recorded_cameras_head_camera_2_image_compressed/frame00046.jpg'
}
#print(len(COMPLEX_TEST_SET))


# 56 Images
ROBOT_TEST_SET = {
'mobileRobot/record_005/recorded_camera_top/frame00001.jpg',
'mobileRobot/record_005/recorded_camera_top/frame00002.jpg',
'mobileRobot/record_005/recorded_camera_top/frame00003.jpg',
'mobileRobot/record_005/recorded_camera_top/frame00004.jpg',
'mobileRobot/record_005/recorded_camera_top/frame00005.jpg',
'mobileRobot/record_005/recorded_camera_top/frame00006.jpg',
'mobileRobot/record_005/recorded_camera_top/frame00007.jpg',
'mobileRobot/record_005/recorded_camera_top/frame00008.jpg',
'mobileRobot/record_005/recorded_camera_top/frame00009.jpg',
'mobileRobot/record_005/recorded_camera_top/frame00010.jpg',
'mobileRobot/record_005/recorded_camera_top/frame00011.jpg',
'mobileRobot/record_005/recorded_camera_top/frame00012.jpg',
'mobileRobot/record_005/recorded_camera_top/frame00013.jpg',
'mobileRobot/record_005/recorded_camera_top/frame00014.jpg',
'mobileRobot/record_005/recorded_camera_top/frame00015.jpg',
'mobileRobot/record_005/recorded_camera_top/frame00016.jpg',
'mobileRobot/record_005/recorded_camera_top/frame00017.jpg',
'mobileRobot/record_005/recorded_camera_top/frame00018.jpg',
'mobileRobot/record_005/recorded_camera_top/frame00019.jpg',
'mobileRobot/record_005/recorded_camera_top/frame00020.jpg',
'mobileRobot/record_005/recorded_camera_top/frame00021.jpg',
'mobileRobot/record_005/recorded_camera_top/frame00022.jpg',
'mobileRobot/record_000/recorded_camera_top/frame00048.jpg',
'mobileRobot/record_000/recorded_camera_top/frame00049.jpg',
'mobileRobot/record_000/recorded_camera_top/frame00050.jpg',
'mobileRobot/record_000/recorded_camera_top/frame00051.jpg',
'mobileRobot/record_000/recorded_camera_top/frame00052.jpg',
'mobileRobot/record_000/recorded_camera_top/frame00053.jpg',
'mobileRobot/record_000/recorded_camera_top/frame00054.jpg',
'mobileRobot/record_000/recorded_camera_top/frame00055.jpg',
'mobileRobot/record_000/recorded_camera_top/frame00056.jpg',
'mobileRobot/record_000/recorded_camera_top/frame00057.jpg',
'mobileRobot/record_000/recorded_camera_top/frame00058.jpg',
'mobileRobot/record_000/recorded_camera_top/frame00059.jpg',
'mobileRobot/record_000/recorded_camera_top/frame00060.jpg',
'mobileRobot/record_000/recorded_camera_top/frame00061.jpg',
'mobileRobot/record_000/recorded_camera_top/frame00062.jpg',
'mobileRobot/record_000/recorded_camera_top/frame00063.jpg',
'mobileRobot/record_000/recorded_camera_top/frame00064.jpg',
'mobileRobot/record_000/recorded_camera_top/frame00065.jpg',
'mobileRobot/record_000/recorded_camera_top/frame00066.jpg',
'mobileRobot/record_004/recorded_camera_top/frame00010.jpg',
'mobileRobot/record_004/recorded_camera_top/frame00011.jpg',
'mobileRobot/record_004/recorded_camera_top/frame00012.jpg',
'mobileRobot/record_004/recorded_camera_top/frame00013.jpg',
'mobileRobot/record_004/recorded_camera_top/frame00014.jpg',
'mobileRobot/record_004/recorded_camera_top/frame00015.jpg',
'mobileRobot/record_004/recorded_camera_top/frame00016.jpg',
'mobileRobot/record_004/recorded_camera_top/frame00017.jpg',
'mobileRobot/record_004/recorded_camera_top/frame00018.jpg',
'mobileRobot/record_004/recorded_camera_top/frame00019.jpg',
'mobileRobot/record_004/recorded_camera_top/frame00020.jpg',
'mobileRobot/record_004/recorded_camera_top/frame00021.jpg',
'mobileRobot/record_004/recorded_camera_top/frame00022.jpg',
'mobileRobot/record_004/recorded_camera_top/frame00023.jpg',
'mobileRobot/record_004/recorded_camera_top/frame00024.jpg'
}

# 50 Images: NOTE: IMPORTANT: RECORD_150 is a special one created with multi colors domain randomization WITHIN the same sequence (other sequences are not)
# in order to have a varied dataset in the test set.
COLORFUL_TEST_SET = {
'colorful/record_150/recorded_cameras_head_camera_2_image_compressed/frame00030.jpg',
'colorful/record_150/recorded_cameras_head_camera_2_image_compressed/frame00003.jpg',
'colorful/record_150/recorded_cameras_head_camera_2_image_compressed/frame00021.jpg',
'colorful/record_150/recorded_cameras_head_camera_2_image_compressed/frame00025.jpg',
'colorful/record_150/recorded_cameras_head_camera_2_image_compressed/frame00014.jpg',
'colorful/record_150/recorded_cameras_head_camera_2_image_compressed/frame00027.jpg',
'colorful/record_150/recorded_cameras_head_camera_2_image_compressed/frame00034.jpg',
'colorful/record_150/recorded_cameras_head_camera_2_image_compressed/frame00016.jpg',
'colorful/record_150/recorded_cameras_head_camera_2_image_compressed/frame00001.jpg',
'colorful/record_150/recorded_cameras_head_camera_2_image_compressed/frame00026.jpg',
'colorful/record_150/recorded_cameras_head_camera_2_image_compressed/frame00015.jpg',
'colorful/record_150/recorded_cameras_head_camera_2_image_compressed/frame00011.jpg',
'colorful/record_150/recorded_cameras_head_camera_2_image_compressed/frame00047.jpg',
'colorful/record_150/recorded_cameras_head_camera_2_image_compressed/frame00020.jpg',
'colorful/record_150/recorded_cameras_head_camera_2_image_compressed/frame00012.jpg',
'colorful/record_150/recorded_cameras_head_camera_2_image_compressed/frame00029.jpg',
'colorful/record_150/recorded_cameras_head_camera_2_image_compressed/frame00045.jpg',
'colorful/record_150/recorded_cameras_head_camera_2_image_compressed/frame00049.jpg',
'colorful/record_150/recorded_cameras_head_camera_2_image_compressed/frame00039.jpg',
'colorful/record_150/recorded_cameras_head_camera_2_image_compressed/frame00038.jpg',
'colorful/record_150/recorded_cameras_head_camera_2_image_compressed/frame00032.jpg',
'colorful/record_150/recorded_cameras_head_camera_2_image_compressed/frame00028.jpg',
'colorful/record_150/recorded_cameras_head_camera_2_image_compressed/frame00037.jpg',
'colorful/record_150/recorded_cameras_head_camera_2_image_compressed/frame00005.jpg',
'colorful/record_150/recorded_cameras_head_camera_2_image_compressed/frame00004.jpg',
'colorful/record_150/recorded_cameras_head_camera_2_image_compressed/frame00040.jpg',
'colorful/record_150/recorded_cameras_head_camera_2_image_compressed/frame00017.jpg',
'colorful/record_150/recorded_cameras_head_camera_2_image_compressed/frame00008.jpg',
'colorful/record_150/recorded_cameras_head_camera_2_image_compressed/frame00006.jpg',
'colorful/record_150/recorded_cameras_head_camera_2_image_compressed/frame00031.jpg',
'colorful/record_150/recorded_cameras_head_camera_2_image_compressed/frame00035.jpg',
'colorful/record_150/recorded_cameras_head_camera_2_image_compressed/frame00042.jpg',
'colorful/record_150/recorded_cameras_head_camera_2_image_compressed/frame00000.jpg',
'colorful/record_150/recorded_cameras_head_camera_2_image_compressed/frame00036.jpg',
'colorful/record_150/recorded_cameras_head_camera_2_image_compressed/frame00002.jpg',
'colorful/record_150/recorded_cameras_head_camera_2_image_compressed/frame00044.jpg',
'colorful/record_150/recorded_cameras_head_camera_2_image_compressed/frame00018.jpg',
'colorful/record_150/recorded_cameras_head_camera_2_image_compressed/frame00041.jpg',
'colorful/record_150/recorded_cameras_head_camera_2_image_compressed/frame00013.jpg',
'colorful/record_150/recorded_cameras_head_camera_2_image_compressed/frame00033.jpg',
'colorful/record_150/recorded_cameras_head_camera_2_image_compressed/frame00048.jpg',
'colorful/record_150/recorded_cameras_head_camera_2_image_compressed/frame00009.jpg',
'colorful/record_150/recorded_cameras_head_camera_2_image_compressed/frame00024.jpg',
'colorful/record_150/recorded_cameras_head_camera_2_image_compressed/frame00010.jpg',
'colorful/record_150/recorded_cameras_head_camera_2_image_compressed/frame00022.jpg',
'colorful/record_150/recorded_cameras_head_camera_2_image_compressed/frame00043.jpg',
'colorful/record_150/recorded_cameras_head_camera_2_image_compressed/frame00007.jpg',
'colorful/record_150/recorded_cameras_head_camera_2_image_compressed/frame00023.jpg',
'colorful/record_150/recorded_cameras_head_camera_2_image_compressed/frame00019.jpg',
'colorful/record_150/recorded_cameras_head_camera_2_image_compressed/frame00046.jpg'
}

COLORFUL75_TEST_SET = {
'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00030.jpg',
'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00003.jpg',
'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00021.jpg',
'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00025.jpg',
'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00014.jpg',
'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00027.jpg',
'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00034.jpg',
'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00016.jpg',
'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00001.jpg',
'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00026.jpg',
'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00015.jpg',
'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00011.jpg',
'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00047.jpg',
'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00020.jpg',
'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00012.jpg',
'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00029.jpg',
'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00045.jpg',
'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00049.jpg',
'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00039.jpg',
'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00038.jpg',
'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00032.jpg',
'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00028.jpg',
'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00037.jpg',
'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00005.jpg',
'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00004.jpg',
'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00040.jpg',
'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00017.jpg',
'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00008.jpg',
'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00006.jpg',
'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00031.jpg',
'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00035.jpg',
'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00042.jpg',
'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00000.jpg',
'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00036.jpg',
'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00002.jpg',
'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00044.jpg',
'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00018.jpg',
'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00041.jpg',
'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00013.jpg',
'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00033.jpg',
'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00048.jpg',
'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00009.jpg',
'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00024.jpg',
'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00010.jpg',
'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00022.jpg',
'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00043.jpg',
'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00007.jpg',
'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00023.jpg',
'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00019.jpg',
'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00046.jpg'
}

NONSTATIC_BUTTON_TEST_SET = {
'nonStaticButton/record_025/recorded_cameras_head_camera_2_image_compressed/frame00030.jpg',
'nonStaticButton/record_025/recorded_cameras_head_camera_2_image_compressed/frame00003.jpg',
'nonStaticButton/record_025/recorded_cameras_head_camera_2_image_compressed/frame00021.jpg',
'nonStaticButton/record_025/recorded_cameras_head_camera_2_image_compressed/frame00025.jpg',
'nonStaticButton/record_025/recorded_cameras_head_camera_2_image_compressed/frame00014.jpg',
'nonStaticButton/record_025/recorded_cameras_head_camera_2_image_compressed/frame00027.jpg',
'nonStaticButton/record_025/recorded_cameras_head_camera_2_image_compressed/frame00034.jpg',
'nonStaticButton/record_025/recorded_cameras_head_camera_2_image_compressed/frame00016.jpg',
'nonStaticButton/record_025/recorded_cameras_head_camera_2_image_compressed/frame00001.jpg',
'nonStaticButton/record_025/recorded_cameras_head_camera_2_image_compressed/frame00026.jpg',
'nonStaticButton/record_025/recorded_cameras_head_camera_2_image_compressed/frame00015.jpg',
'nonStaticButton/record_025/recorded_cameras_head_camera_2_image_compressed/frame00011.jpg',
'nonStaticButton/record_025/recorded_cameras_head_camera_2_image_compressed/frame00047.jpg',
'nonStaticButton/record_025/recorded_cameras_head_camera_2_image_compressed/frame00020.jpg',
'nonStaticButton/record_025/recorded_cameras_head_camera_2_image_compressed/frame00012.jpg',
'nonStaticButton/record_025/recorded_cameras_head_camera_2_image_compressed/frame00029.jpg',
'nonStaticButton/record_025/recorded_cameras_head_camera_2_image_compressed/frame00045.jpg',
'nonStaticButton/record_025/recorded_cameras_head_camera_2_image_compressed/frame00049.jpg',
'nonStaticButton/record_025/recorded_cameras_head_camera_2_image_compressed/frame00039.jpg',
'nonStaticButton/record_025/recorded_cameras_head_camera_2_image_compressed/frame00038.jpg',
'nonStaticButton/record_025/recorded_cameras_head_camera_2_image_compressed/frame00032.jpg',
'nonStaticButton/record_025/recorded_cameras_head_camera_2_image_compressed/frame00028.jpg',
'nonStaticButton/record_025/recorded_cameras_head_camera_2_image_compressed/frame00037.jpg',
'nonStaticButton/record_025/recorded_cameras_head_camera_2_image_compressed/frame00005.jpg',
'nonStaticButton/record_025/recorded_cameras_head_camera_2_image_compressed/frame00004.jpg',
'nonStaticButton/record_025/recorded_cameras_head_camera_2_image_compressed/frame00040.jpg',
'nonStaticButton/record_025/recorded_cameras_head_camera_2_image_compressed/frame00017.jpg',
'nonStaticButton/record_025/recorded_cameras_head_camera_2_image_compressed/frame00008.jpg',
'nonStaticButton/record_025/recorded_cameras_head_camera_2_image_compressed/frame00006.jpg',
'nonStaticButton/record_025/recorded_cameras_head_camera_2_image_compressed/frame00031.jpg',
'nonStaticButton/record_025/recorded_cameras_head_camera_2_image_compressed/frame00035.jpg',
'nonStaticButton/record_025/recorded_cameras_head_camera_2_image_compressed/frame00042.jpg',
'nonStaticButton/record_025/recorded_cameras_head_camera_2_image_compressed/frame00000.jpg',
'nonStaticButton/record_025/recorded_cameras_head_camera_2_image_compressed/frame00036.jpg',
'nonStaticButton/record_025/recorded_cameras_head_camera_2_image_compressed/frame00002.jpg',
'nonStaticButton/record_025/recorded_cameras_head_camera_2_image_compressed/frame00044.jpg',
'nonStaticButton/record_025/recorded_cameras_head_camera_2_image_compressed/frame00018.jpg',
'nonStaticButton/record_025/recorded_cameras_head_camera_2_image_compressed/frame00041.jpg',
'nonStaticButton/record_025/recorded_cameras_head_camera_2_image_compressed/frame00013.jpg',
'nonStaticButton/record_025/recorded_cameras_head_camera_2_image_compressed/frame00033.jpg',
'nonStaticButton/record_025/recorded_cameras_head_camera_2_image_compressed/frame00048.jpg',
'nonStaticButton/record_025/recorded_cameras_head_camera_2_image_compressed/frame00009.jpg',
'nonStaticButton/record_025/recorded_cameras_head_camera_2_image_compressed/frame00024.jpg',
'nonStaticButton/record_025/recorded_cameras_head_camera_2_image_compressed/frame00010.jpg',
'nonStaticButton/record_025/recorded_cameras_head_camera_2_image_compressed/frame00022.jpg',
'nonStaticButton/record_025/recorded_cameras_head_camera_2_image_compressed/frame00043.jpg',
'nonStaticButton/record_025/recorded_cameras_head_camera_2_image_compressed/frame00007.jpg',
'nonStaticButton/record_025/recorded_cameras_head_camera_2_image_compressed/frame00023.jpg',
'nonStaticButton/record_025/recorded_cameras_head_camera_2_image_compressed/frame00019.jpg',
'nonStaticButton/record_025/recorded_cameras_head_camera_2_image_compressed/frame00046.jpg'
}


#########
# IMPORTANT NOTE: MOVIE TEST SETS MUST BE AN ARRAY TO PRESERVE ORDER IN A SMOOTH TRANSITION IN THE MOVIE:
COLORFUL75_MOVIE_TEST_SET = get_immediate_files_in_path('colorful75/record_031/recorded_cameras_head_camera_2_image_compressed', containing_pattern_in_name='frame')
COLORFUL75_MOVIE_TEST_SET.sort()

# THE FOLLOWING SEQUENCE (N 150) IS AD-HOC CREATED WITH  MULTIMPLE NON_SMOOTH LOOKING COLOUR CHANGING IN BETWEEN THE SEQUENCE. FOR MAKING MOVIE< USE THE ABOVE ONE
# COLORFUL75_MOVIE_TEST_SET = [
# 'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00000.jpg',
# 'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00001.jpg',
# 'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00002.jpg',
# 'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00003.jpg',
# 'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00004.jpg',
# 'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00005.jpg',
# 'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00006.jpg',
# 'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00007.jpg',
# 'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00008.jpg',
# 'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00009.jpg',
# 'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00010.jpg',
# 'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00011.jpg',
# 'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00012.jpg',
# 'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00013.jpg',
# 'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00014.jpg',
# 'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00015.jpg',
# 'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00016.jpg',
# 'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00017.jpg',
# 'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00018.jpg',
# 'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00019.jpg',
# 'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00020.jpg',
# 'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00021.jpg',
# 'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00022.jpg',
# 'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00023.jpg',
# 'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00024.jpg',
# 'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00025.jpg',
# 'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00026.jpg',
# 'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00027.jpg',
# 'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00028.jpg',
# 'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00029.jpg',
# 'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00030.jpg',
# 'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00031.jpg',
# 'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00032.jpg',
# 'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00033.jpg',
# 'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00034.jpg',
# 'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00035.jpg',
# 'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00036.jpg',
# 'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00037.jpg',
# 'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00038.jpg',
# 'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00039.jpg',
# 'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00040.jpg',
# 'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00041.jpg',
# 'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00042.jpg',
# 'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00043.jpg',
# 'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00044.jpg',
# 'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00045.jpg',
# 'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00046.jpg',
# 'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00047.jpg',
# 'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00048.jpg',
# 'colorful75/record_150/recorded_cameras_head_camera_2_image_compressed/frame00049.jpg']

COMPLEX_DATA_MOVIE_TEST_SET = get_immediate_files_in_path('complexData/record_024/recorded_cameras_head_camera_2_image_compressed', containing_pattern_in_name='frame')
COMPLEX_DATA_MOVIE_TEST_SET.sort()

# former dataset: record_011
STATIC_BUTTON_SIMPLEST_MOVIE_TEST_SET = get_immediate_files_in_path('staticButtonSimplest/record_022/recorded_cameras_head_camera_2_image_compressed', containing_pattern_in_name='frame')
STATIC_BUTTON_SIMPLEST_MOVIE_TEST_SET.sort()

# NOTICE index starts from 1 here (only for mobileRobot dataset)
MOBILE_ROBOT_MOVIE_TEST_SET = get_immediate_files_in_path('mobileRobot/record_004/recorded_camera_top', containing_pattern_in_name='frame')
MOBILE_ROBOT_MOVIE_TEST_SET.sort()


COLORFUL_MOVIE_TEST_SET = COLORFUL75_MOVIE_TEST_SET

# used for GIF movie demo for all discrete actions DREAM DEMO
# NOTE THAT THE ORDER MUST COINCIDE PER DATASET PER ITS CORRECT USAGE BY makeMovieComparingKNNAcrossModels.py
ALL_KNN_MOVIE_TEST_SETS = [COLORFUL75_MOVIE_TEST_SET, COMPLEX_DATA_MOVIE_TEST_SET, STATIC_BUTTON_SIMPLEST_MOVIE_TEST_SET, MOBILE_ROBOT_MOVIE_TEST_SET]#, COLORFUL_MOVIE_TEST_SET]
BENCHMARK_DATASETS = [COLORFUL75, COMPLEX_DATA, STATIC_BUTTON_SIMPLEST, MOBILE_ROBOT, NONSTATIC_BUTTON]




#### Tests

library_versions_tests()
# save_config_to_file(CONFIG_DICT, CONFIG_JSON_FILE)
# read_config(CONFIG_JSON_FILE)
