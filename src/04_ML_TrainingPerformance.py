#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 11:24:52 2023

@author: zech0001
"""

# import numpy as np
# import pandas as pd
import PSD_2K_ML

algorithm ='SVR' # 'LR' #'RF' #'ANN' #'LR' #'DT' #
soil_type = 'all'
verbose = True #False #

print('\n#################################')
print('  Training Performance Evaluation')
print('#################################\n')

### ===========================================================================
### Set file pathes and names
### ===========================================================================
file_AI_data = "../data/AI_data.csv"
#file_psd_props = "../data/PSD_properties.csv"
# file_Kemp_all = "../results/Kemp_all.csv"

Analysis = PSD_2K_ML.PSD_2K_ML()

# Analysis.read_data(file_AI_data) # read in data
# Analysis.remove_outliers(verbose = True)
# Analysis.sub_sample_soil_type(soil_type = soil_type,
#                           inplace = True,
#                           filter_props = False,
#                           verbose = verbose
#                           )

data_PSD = Analysis.prepare_data(filename=file_AI_data,
                      soil_type = soil_type, 
                      remove_outlier = False,
                      verbose = verbose,      
                      )

### ===========================================================================
### Speficy Algorithm and set target and feature variables
### ===========================================================================

### specify AI algorithm
Analysis.set_algorithm(algorithm = algorithm,
                       verbose = verbose)

### specifying feature (input) and target (output) variables
Analysis.set_feature_variables()
Analysis.set_target_variables()

### split data for training and train 
Analysis.data_split(verbose = verbose)

### ===========================================================================
###   Hyperparameter testing 
### ===========================================================================

### perform hyperparameter testings (includes specification of AI algorithm)
# Analysis.hyperparameter_GS(verbose = verbose)
#results = Analysis.hyperparameter_skopt(verbose = verbose)

### ===========================================================================
###   Algorithm Performance with optimal Parameters
### ===========================================================================

Analysis.training()
Analysis.prediction(verbose = verbose)
Analysis.prediction(
    x_pred = 'training_set',
    verbose = verbose)
Analysis.prediction(
    x_pred = 'testing_set',
    verbose = verbose)
test = Analysis.psd
