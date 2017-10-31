# coding: utf-8

import pandas as pd
import numpy as np
import math
import sys, os, os.path

# coding: utf-8
from Utils import library_versions_tests, get_data_folder_from_model_name, plotStates
#from Utils import BABBLING, MOBILE_ROBOT, SIMPLEDATA3D, PUSHING_BUTTON_AUGMENTED, STATIC_BUTTON_SIMPLEST,LEARNED_REPRESENTATIONS_FILE
from Utils import GLOBAL_SCORE_LOG_FILE, MODELS_CONFIG_LOG_FILE, ALL_STATS_FILE, ALL_DATASETS


################

####   MAIN program

############ PLOT ALL EXPERIMENTS SCORES
print"\n\n >> Running report_results.py...."

header = ['Model','KNN_MSE','DATA_FOLDER','MODEL_ARCHITECTURE_FILE','MAX_COS_DIST_AMONG_ACTIONS_THRESHOLD','CONTINUOUS_ACTION_SIGMA','STATES_DIMENSION']

def plot_all_config_performance(df):
    # Plot all MSE_KNN scores for each experiment
    print "Reporting all experiments KNN_MSE scores for a varying number of MAX_COS_DIST_AMONG_ACTIONS_THRESHOLD and  CONTINUOUS_ACTION_SIGMA "

def print_leader_board(df, datasets):
    print "\n****************************************\n LEADERBOARD OF WINNING MODELS PER DATASET: \n****************************************"
    for dataset in datasets:
        #sub_dataframe = df[df['KNN_MSE'].notnull() & df[df['DATA_FOLDER'].notnull() & (df['DATA_FOLDER']== dataset)] #df.loc[df['DATA_FOLDER'] == dataset]  #df[['DATA_FOLDER'] == dataset]
        sub_dataframe = df[df['DATA_FOLDER'].notnull() & (df['DATA_FOLDER']== dataset)]
        best_KNN_MSE = sub_dataframe['KNN_MSE'].min()
        if not pd.isnull(best_KNN_MSE):
            best_model_name = sub_dataframe[sub_dataframe['KNN_MSE'] == best_KNN_MSE].Model.values[0]
            print "\nDATASET ", dataset, " Min KNN_MSE: ", best_KNN_MSE, ": ", best_model_name
        else:
            print "\nDATASET ", dataset, '[No data available yet/all KNN_MSE were nan, delete old files, and run train_predict_plotStates.sh again]'
            #  (see first cat globalScoreLog.csv  ; cat modelsConfigLog.csv  ; cat allStats.csv )

# writing scores to global log for plotting and reporting
if os.path.isfile(GLOBAL_SCORE_LOG_FILE):
    if os.path.isfile(MODELS_CONFIG_LOG_FILE):
        mse_df = pd.read_csv(GLOBAL_SCORE_LOG_FILE, usecols=['Model','KNN_MSE','STATES_DIMENSION'])# columns = header)
        models_df = pd.read_csv(MODELS_CONFIG_LOG_FILE, usecols=['Model','DATA_FOLDER','MODEL_ARCHITECTURE_FILE','MAX_COS_DIST_AMONG_ACTIONS_THRESHOLD','CONTINUOUS_ACTION_SIGMA'])#, columns = header)   print mse_df.head()    print models_df.head()
        #print "Initial log files contain ", len(mse_df), ' MSE datapoints and ', len(models_df), ' models experimented '
        #all_scores_logs = mse_df.merge(models_df, on='Model', how='left') #all_scores_logs = mse_df.set_index('Model').join(models_df.set_index('Model'))
        all_scores_logs = mse_df.fillna(np.nan).dropna().merge(models_df.fillna(np.nan).dropna(),on='Model',how='outer')
        final = all_scores_logs[header]
        final.sort_values(by='Model', inplace=True )
        print "Latest scores logged so far: \n", final.tail(5)
        print_leader_board(final, ALL_DATASETS)
        final.to_csv(ALL_STATS_FILE, header = header)

    else:
        print 'Error: the following files must exist to plot MSE over configuration values: ',MODELS_CONFIG_LOG_FILE
        sys.exit(-1)
else:
    print 'Error: the following files must exist to plot MSE over configuration values: ',GLOBAL_SCORE_LOG_FILE
    sys.exit(-1)
