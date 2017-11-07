from const import LOG_FOLDER, MODEL_APPROACH, USE_CONTINUOUS, MAX_COS_DIST_AMONG_ACTIONS_THRESHOLD
from const import CONTINUOUS_ACTION_SIGMA, STATES_DIMENSION, PRIORS_CONFIGS_TO_APPLY,  PROP, TEMP, CAUS, REP, BRING_CLOSER_REF_POINT


def addLeadingZero(number):
    # Returns a string with a leading zero of the number if the number has only one digit (for model logging and sorting purposes)
    return "0"+str(number) if len(str(number)) >=0 and len(str(number)) <2   else str(number)

def priorsToString(listOfPriors):
    string = '_'
    for index, priors in enumerate(listOfPriors) :
        string = string+ priors[0:3]
    return string

def create_experiment_model_name(data_folder, architecture_name):
    import datetime
    now = datetime.datetime.now()

    if USE_CONTINUOUS:
        DAY = 'Y'+str(now.year)+'_M'+addLeadingZero(now.month)+'_D'+addLeadingZero(now.day)+'_H'+addLeadingZero(now.hour)+'M'+addLeadingZero(now.minute)+'S'+addLeadingZero(now.second)+'_'+data_folder+'_'+architecture_name+'_cont'+'_MCD'+str(MAX_COS_DIST_AMONG_ACTIONS_THRESHOLD)+'_S'+str(CONTINUOUS_ACTION_SIGMA)+priorsToString(PRIORS_CONFIGS_TO_APPLY)+'_ST_DIM'+str(STATES_DIMENSION)
        DAY = DAY.replace(".", "_")  # replace decimal points by '_' for folder naming
    else:
        DAY = 'Y'+str(now.year)+'_M'+addLeadingZero(now.month)+'_D'+addLeadingZero(now.day)+'_H'+addLeadingZero(now.hour)+'M'+addLeadingZero(now.minute)+'S'+addLeadingZero(now.second)+'_'+data_folder+'_'+architecture_name+priorsToString(PRIORS_CONFIGS_TO_APPLY)+'_ST_DIM'+str(STATES_DIMENSION)

    if len(MODEL_APPROACH) != '': #to add an extra keyword  to the model name for fwd models
        NAME_SAVE= 'model'+DAY+'_'+MODEL_APPROACH
    else:
        NAME_SAVE= 'model'+str(DAY)
    SAVED_MODEL_PATH = LOG_FOLDER+ experiment_name+'/'+NAME_SAVE
    print(SAVED_MODEL_PATH)
    #TODO create folder for experiment and models
    return SAVED_MODEL_PATH

print  priorsToString(PRIORS_CONFIGS_TO_APPLY)
data_folder = 'staticButtonSimplest' #nonStaticButton' #colorful75' #staticButtonSimplest' #mobileRobot' #'nonStaticButton' #'complexData' #colorful75'  #'mobileRobot' # 'complexData'
experiment_name = 'Experiment1'#ToBeTakenFromPARAM_LINE
EXPERIMENT_MODEL_PATH = create_experiment_model_name(data_folder, experiment_name)

print ('Running pipeline and saving model to {}'.format(EXPERIMENT_MODEL_PATH))
