#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script evaluating performance of a selected ML algorithms after training based on 
routines implemented in class "PSD_2K_ML":
    - specifying settings, such as feature/target-variable combination and 
        type of data(sub)set
    - reading data from csv-file (with standardised column naming)
    - train algorithm with optimal hyperparameters (determined separately)
    - evaluate algorithm performance and save results to file

Author: A. Zech
"""

# import numpy as np
# import pandas as pd
import PSD_2K_ML

### ===========================================================================
### Key words to specify modus of script:
### ===========================================================================

algorithm ='XG' # 'DT' #LR' # 'ANN'  # 'RF' #'SVR' # ####  
soil_type = 'topall'#'sand' #  'clay' #'silt' #'por' # 
feature = 'PSD' #'dX_por' #'dX' #
target = 'Kf' #'por' # #
verbose = True #False #

print('\n#################################')
print('  Training Performance Evaluation')
print('   Algorithm {}'.format(algorithm))
print('   Feature variables: {}'.format(feature))
print('   Target variable: {}'.format(target))

print('#################################\n')

### ===========================================================================
### Set file pathes and names
### ===========================================================================

file_data = "../data/data_PSD_Kf_por.csv"

### ===========================================================================
### Initialize Analysis
### ===========================================================================

Analysis = PSD_2K_ML.PSD_2K_ML(
                        algorithm = algorithm,
                        feature = feature,
                        target = target,                            
                        )

data_PSD = Analysis.prepare_data(filename=file_data,
                      soil_type = soil_type, 
                      remove_outlier = False, #True , #
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
