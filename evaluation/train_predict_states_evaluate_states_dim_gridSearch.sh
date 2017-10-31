#!/bin/bash

# CONFIG OPTIONS:
# -use_cuda
# -use_continuous
# -params.sigma  is CONTINUOUS_ACTION_SIGMA
# -params.mcd is MAX_COS_DIST_AMONG_ACTIONS_THRESHOLD
# -params.states_dimension is the states dimensionality to be learned in script.lua
# -data_folder options: DATA_FOLDER (Dataset to use):
#          staticButtonSimplest, mobileRobot, simpleData3D, pushingButton3DAugmented, babbling, nonStaticButton)
#data= staticButtonSimplest, mobileRobot, complexData colorful  #staticButtonSimplest https://stackoverflow.com/questions/2459286/unable-to-set-variables-in-bash-script  #"$data"='staticButtonSimplest'

function has_command_finished_correctly {
    if [ "$?" -ne "0" ]
    then
        exit
    else
        return 0
    fi
}


data_folder='staticButtonSimplest' #nonStaticButton' #colorful75' #staticButtonSimplest' #mobileRobot' #'nonStaticButton' #'complexData' #colorful75'  #'mobileRobot' # 'complexData' #'colorful'  #
for states_dimension in 4 5 6 7 8 9 10 15 20 50 100 200 500 1000
  do    # qlua or th
        echo " ********** Running pipeline for finetuning states dimension to be learned: $states_dimension and sigma: $s *************"
        th script.lua  -use_cuda -states_dimension $states_dimension  -data_folder $data_folder
        has_command_finished_correctly

        th imagesAndReprToTxt.lua  -use_cuda -data_folder $data_folder
        has_command_finished_correctly

        python generateNNImages.py 10
        #   ----- Note: includes the call to:
        #                th create_all_reward.lua
        #                th create_plotStates_file_for_all_seq.lua
        has_command_finished_correctly

        python plotStates.py
        has_command_finished_correctly

        python report_results.py
        has_command_finished_correctly
done
