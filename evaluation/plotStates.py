# coding: utf-8
from Utils import library_versions_tests, get_data_folder_from_model_name, plotStates, read_config
from Utils import MOBILE_ROBOT, LEARNED_REPRESENTATIONS_FILE, SKIP_RENDERING, DEFAULT_DATASET, SUPERVISED
import numpy as np
import sys
import os.path
import subprocess
from sklearn.decomposition import PCA  # with some version of sklearn fails with ImportError: undefined symbol: PyFPE_jbuf
import unittest
test = unittest.TestCase('__init__')

# PLOTTING GROUND TRUTH OR LEARNED STATES
#####################
# True if we plot ground truth observed states, and false to plot the learned state representations
plotGroundTruthStates = False
with_title = False #do you want the title on your plots or nop ? Not implemented at the moment

library_versions_tests()
print"\n\n >> Running plotStates.py....plotGroundTruthStates: ",plotGroundTruthStates, " SKIP_RENDERING = ", SKIP_RENDERING
CONFIG_DICT = read_config()
STATES_DIMENSION = CONFIG_DICT['STATES_DIMENSION']

model_name = ''

if len(sys.argv) < 2:  # regular pipeline in gridsearch script
    lastModelFile = open('lastModel.txt')
    path = lastModelFile.readline()[:-1]+'/'
    model_name = path.split('/')[1]
    # ONLY FOR FAST TESTING !!:   model_name = MOBILE_ROBOT#STATIC_BUTTON_SIMPLEST#'pushingButton3DAugmented' #TODO REMOVE-testing  model_name = MOBILE_ROBOT
    data_folder = get_data_folder_from_model_name(model_name)
    if data_folder == SUPERVISED:
        data_folder = DEFAULT_DATASET
    if plotGroundTruthStates:
        state_file_str = 'allStatesGT_'+data_folder+'.txt'
        print "*********************\nPLOTTING GROUND TRUTH (OBSERVED) STATES for model: ", model_name#(Baxter left wrist position for 3D PUSHING_BUTTON_AUGMENTED dataset, or grid 2D position for MOBILE_ROBOT dataset)
        plot_path = path+'GroundTruthStatesPlot_'+model_name+'.png'
    else:
        state_file_str = path+ LEARNED_REPRESENTATIONS_FILE
        print "*********************\nPLOTTING LEARNT STATES for model: ", model_name #(3D for Baxter PUSHING_BUTTON_AUGMENTED dataset, or 2D position for MOBILE_ROBOT dataset): ", state_file_str
        plot_path = path+'LearnedStatesPlot_'+model_name+'.png'
    lastModelFile.close()

else:
    state_file_str = sys.argv[1]
    path_list = state_file_str.split('/')[-2:]

    print "path_list",path_list , ' Warning: plotting particular case, should not be the general pipeline case...'

    if path_list[0][:2] == 'co':
        data_folder = COMPLEX_DATA
    elif path_list[0][:2].lower() == '3d':
        data_folder = STATIC_BUTTON_SIMPLEST
    else:
        data_folder = 'mobileData' # NOTICE, not, MOBILE_ROBOT

reward_file_str = 'allRewardsGT_'+data_folder+'.txt'
if not os.path.isfile(state_file_str): 
    print('Calling subprocess to write to file all GT states: create_plotStates_file_in file and for dataset: ',state_file_str, data_folder)
    subprocess.call(['th','create_plotStates_file_for_all_seq.lua','-use_cuda','-use_continuous','-data_folder', data_folder])  # TODO: READ CMD LINE ARGS FROM FILE INSTEAD (and set accordingly here) TO NOT HAVING TO MODIFY INSTEAD train_predict_plotStates and the python files
if not os.path.isfile(reward_file_str): 
    print('Calling subprocess to write to file all GT rewards: create_all_reward in file and for dataset: ',reward_file_str, data_folder)
    subprocess.call(['th','create_all_reward.lua', '-use_cuda','-use_continuous','-data_folder', data_folder])


total_rewards = 0
total_states = 0
states_l=[]
rewards_l=[]

if 'recorded_robot' in state_file_str :
    print 'Plotting ', MOBILE_ROBOT,' observed states and rewards in ',state_file_str
    for line in state_file:
            if line[0]!='#':
                words=line.split(' ')
                states_l.append([ float(words[0]),float(words[1])] )
    states=np.asarray(states_l)
else: # general case
    print 'GT states file name: ', state_file_str
    with open(state_file_str) as f:
        for line in f:
            if line[0]!='#':
                # Saving each image file and its learned representations
                words=line.split(' ')
                states_l.append((words[0], list(map(float,words[1:-1]))))
                total_states += 1

    states_l.sort(key= lambda x : x[0])
    states = np.zeros((len(states_l), len(states_l[0][1])))

    for i in range(len(states_l)):
        states[i] = np.array(states_l[i][1])


# Reading rewards
with open(reward_file_str) as f:
    for line in f:
        if line[0]!='#':
            words=line.split(' ')
            rewards_l.append(words[0])
            total_rewards+= 1

rewards=rewards_l
toplot=states
print "Ploting total states and total rewards: ",total_states, " ", total_rewards," in files: ",state_file_str," and ", reward_file_str
test.assertEqual(total_rewards, total_states, "Datapoints size discordance! Length of rewards and state files should be equal, and it is "+str(len(rewards))+" and "+str(len(toplot))+" Run first create_all_reward.lua and create_plotStates_file_for_all_seq.lua")

REPRESENTATIONS_DIMENSIONS = len(states[0])
test.assertEqual(REPRESENTATIONS_DIMENSIONS, STATES_DIMENSION, "REPRESENTATIONS_DIMENSIONS and STATES_DIMENSION should coincide, set your current configuration in const.lua and either a) provide the model as argument or b) if  running plotStates.py without arguments, make sure you have trained a model (saved in lastModel.txt) and run imagesAndReprToTxt.lua previously (i.e. make sure you are running the train_predict_plotStates pipeline which saves the last model for which representations have been learned). Values are: "+str(REPRESENTATIONS_DIMENSIONS)+' '+str(STATES_DIMENSION))

PLOT_DIMENSIONS = 3

if REPRESENTATIONS_DIMENSIONS >3:
    print "[Applying PCA to visualize the ",REPRESENTATIONS_DIMENSIONS,"D learnt representations space (PLOT_DIMENSIONS = ", PLOT_DIMENSIONS,")"
    pca = PCA(n_components=PLOT_DIMENSIONS) # default to 3
    pca.fit(states)
    toplot = pca.transform(states)
elif REPRESENTATIONS_DIMENSIONS==2:
    PLOT_DIMENSIONS = 2 #    print "[PCA not applied since learnt representations' dimensions are not larger than 2]"
else:
    PLOT_DIMENSIONS = 3  # Default, if mobileData used, we plot just 2
#print "\n REPRESENTATIONS_DIMENSIONS =", REPRESENTATIONS_DIMENSIONS


if PLOT_DIMENSIONS == 2:
    plotStates('2D', rewards, toplot, plot_path, dataset=model_name)
elif PLOT_DIMENSIONS ==3:
    plotStates('3D', rewards, toplot, plot_path, dataset=model_name)
else:
    print " PLOT_DIMENSIONS other than 2 or 3 not supported"


# def parse_arguments(): # TODO in future
#     skipRendering = False
#     import argparse

#     parser = argparse.ArgumentParser(description='Process some integers.')
#     parser.add_argument('integers', metavar='N', type=int, nargs='+',
#                        help='an integer for the accumulator')
#     parser.add_argument('--sum', dest='accumulate', action='store_const',
#                        const=sum, default=max,
#                        help='sum the integers (default: find the max)')

#     args = parser.parse_args()
#     print(args.accumulate(args.integers))
#     print "This is the name of the script: ", sys.argv[0]
#     print "Number of arguments: ", len(sys.argv)
#     print "The arguments are: " , str(sys.argv)
#     print "\n\n >> RUNNING plotStates.py  -skipRendering: ", skipRendering
#     print parser.parse_args()
#     return skipRendering
