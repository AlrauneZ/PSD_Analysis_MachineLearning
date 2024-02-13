#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 11:24:52 2023

@author: zech0001
"""

# import numpy as np
# import pandas as pd
import PSD_2K_ML

algorithm ='XG' # 'ANN'  #'SVR' # 'LR' #'RF' # 'DT' #####  
soil_type = 'silt' #'sand' #'full'#  'clay' #
verbose = True #False #

print('\n#################################')
print('  Training Performance Evaluation')
print('#################################\n')

### ===========================================================================
### Set file pathes and names
### ===========================================================================
file_application_data = "../data/data_PSD_Kf_props.csv"
#file_application_data = "../data/data_PSD_Kf.csv"

Analysis = PSD_2K_ML.PSD_2K_ML()
data_PSD = Analysis.prepare_data(filename=file_application_data,
                      soil_type = soil_type, 
                      remove_outlier = False, #True , #
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
