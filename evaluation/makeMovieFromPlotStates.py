# coding: utf-8
from Utils import library_versions_tests, get_data_folder_from_model_name, produceRelevantImageStatesPlotMovie
from Utils import LEARNED_REPRESENTATIONS_FILE, SKIP_RENDERING, MOBILE_ROBOT, GIF_MOVIES_PATH
#from Utils import STATIC_BUTTON_SIMPLEST, COMPLEX_DATA, COLORFUL75, COLORFUL, MOBILE_ROBOT
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
print"\n\n >> Running makeMovieFromPlotStates.py... plotGroundTruthStates: ",plotGroundTruthStates, " SKIP_RENDERING = ", SKIP_RENDERING

model_name = ''

if len(sys.argv) !=2:  # regular pipeline in gridsearch script    
    print 'Provide as argument a folder path (ending in "/") containing the model and reprsentations file'
    exit("Please provide the path to the model's learned representations within the Log folder, e.g. of program run: \n python plotStatesGivenImages.py Log/PredictRewPriormodelY2017_D24_M08_H19M18S22_mobileRobot_resnet_ProTemCauRep")

else:
    state_file_str = sys.argv[1]
    path = state_file_str
    model_name = path.split('/')[-1]
    if plotGroundTruthStates:
        state_file_str = 'allStatesGT_'+data_folder+'.txt'
        print "*********************\nPLOTTING GROUND TRUTH (OBSERVED) STATES for model: ", model_name#(Baxter left wrist position for 3D PUSHING_BUTTON_AUGMENTED dataset, or grid 2D position for MOBILE_ROBOT dataset)
        plot_path = path+'GroundTruthStatesPlot_'+model_name+'.png'
    else:
        if not state_file_str.endswith('/'):
            state_file_str = path+'/'+LEARNED_REPRESENTATIONS_FILE
        else:
            state_file_str = path+ LEARNED_REPRESENTATIONS_FILE
        print "*********************\nPLOTTING LEARNT STATES for model: ", model_name #(3D for Baxter PUSHING_BUTTON_AUGMENTED dataset, or 2D position for MOBILE_ROBOT dataset): ", state_file_str
        plot_path = path+'LearnedStatesPlot_'+model_name+'.png'

    data_folder = get_data_folder_from_model_name(model_name)  
    print 'state_file_str', state_file_str, '\n model name: ', model_name, 'data_folder: ', data_folder

reward_file_str = 'allRewardsGT_'+data_folder+'.txt'
print "state file ",state_file_str
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
img_paths = []

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
                img_paths.append(words[0])
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
print type(states), 'states'
img_paths2repr = dict()
for i in range(len(img_paths)):
    img_paths2repr[img_paths[i]] = [states[i], rewards[i]]

print "Ploting total states and total rewards: ",total_states, " ", total_rewards," in files: ",state_file_str," and ", reward_file_str
test.assertEqual(total_rewards, total_states, "Datapoints size discordance! Length of rewards and state files should be equal, and it is "+str(len(rewards))+" and "+str(len(toplot))+" Run first create_all_reward.lua and create_plotStates_file_for_all_seq.lua")

REPRESENTATIONS_DIMENSIONS = len(states[0])
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
    produceRelevantImageStatesPlotMovie('2D', rewards, toplot, img_paths2repr, model_name)
elif PLOT_DIMENSIONS ==3:
    produceRelevantImageStatesPlotMovie('3D', rewards, toplot, img_paths2repr, model_name)
else:
    print " PLOT_DIMENSIONS other than 2 or 3 not supported"



if not os.path.isfile(reward_file_str): 
    print('Calling subprocess to write to create KNN images for dataset: ', data_folder)
    subprocess.call(['python','generateNNimages.py', '-1', '-data_folder', data_folder])
