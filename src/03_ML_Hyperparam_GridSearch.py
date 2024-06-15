#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script performing hyperparameter testing using GridSearch
for a selected algorithm and data set type:
    - based on routines implemented in class "PSD_2K_ML" 
    - reading PSD data from csv-file (with standardised column naming)
    - determining grain size diameters (d10, d50, d60 etc)
    - Hyperparameter testing with GridSearch
    - Algorithm Performance with optimal Parameters

Author: A. Zech
"""

import PSD_2K_ML
import os
import warnings
warnings.filterwarnings("ignore")

## --------------------------------------------------------------------------
### Key words to specify modus of script:
### --------------------------------------------------------------------------
    
### name of algorithms to test (from listed options)
algorithm ='LR' # 'DT' #'RF' #'XG' # 'LR'   #'SVR'  #'ANN'  

### type of data set (top-all, top-sand, top-silt, top-clay, top-por)
soil_type = 'topall' #'sand' #'silt'# 'clay' # 'por' 

verbose = True #False #

### Combination of feature and target variables:
feature = 'dX' #'PSD' #'dX_por' #
target = 'Kf' #'por' # 

print('\n################################################')
print('   Hyper parameter tuning for algorithm {}'.format(algorithm))
print('##################################################\n')

### ===========================================================================
### Set file pathes and names
### ===========================================================================

file_data = "../data/data_PSD_Kf_por_props.csv"
file_results_GS = "Hyper_{}_{}_GS.csv"#.format(soil_type,algorithm)

dir_results_HP = "../results/Hyperparameter_tuning/HP_tuning_{}_{}/".format(feature,target)
if not os.path.exists(dir_results_HP):
    os.makedirs(dir_results_HP)
path_results_GS = dir_results_HP+file_results_GS
 
### ===========================================================================
### Initialize Analysis and load in data
### ===========================================================================

Analysis = PSD_2K_ML.PSD_2K_ML(
                        algorithm = algorithm,
                        feature = feature,
                        target = target,                            
                        )
data_PSD = Analysis.prepare_data(filename=file_data,
                      soil_type = soil_type, 
                      remove_outlier = False,
                      verbose = verbose,      
                      )

### ===========================================================================
### Speficy Algorithm and set target and feature variables
### ===========================================================================

### specify AI algorithm
Analysis.set_algorithm(verbose = verbose)

### specifying feature (input) and target (output) variables
Analysis.set_feature_variables()
Analysis.set_target_variables()

### split data for training and train 
Analysis.data_split()#verbose = verbose)

### ===========================================================================
###   Hyperparameter testing 
### ===========================================================================

### perform hyperparameter testings (includes specification of AI algorithm)
Analysis.hyperparameter_GS(verbose = verbose,
                           file_results=path_results_GS,   
                           )
Analysis.training()
Analysis.prediction(verbose = verbose)
Analysis.prediction(
    x_pred = 'training_set',
    verbose = verbose)
Analysis.prediction(
    x_pred = 'testing_set',
    verbose = verbose)

