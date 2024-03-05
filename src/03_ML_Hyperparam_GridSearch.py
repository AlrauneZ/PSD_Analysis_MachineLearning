#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""

import PSD_2K_ML
import matplotlib.pyplot as plt
# import numpy as np

import warnings
warnings.filterwarnings("ignore")
plt.close('all')


algorithm =  'XG' ##### # 'SVR'  #'LR' # 'RF'  # #'DT' #'ANN'#
soil_type = 'clay' #'silt'# 'sand' # 'full' #
verbose = True #False #


test = 2

print('\n################################################')
print('   Hyper parameter tuning for algorithm {}'.format(algorithm))
print('##################################################\n')

### ===========================================================================
### Set file pathes and names
### ===========================================================================

file_data = "../data/data_PSD_Kf_por.csv"

### ===========================================================================
### Initialize Analysis and load in data
### ===========================================================================

Analysis = PSD_2K_ML.PSD_2K_ML()
data_PSD = Analysis.prepare_data(filename=file_data,
                      soil_type = soil_type, 
                      remove_outlier = False,
                      verbose = verbose,      
                      )

### ===========================================================================
### Speficy Algorithm and set target and feature variables
### ===========================================================================

### specify AI algorithm
Analysis.set_algorithm(algorithm = algorithm,verbose = verbose)

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
                           file_results = "../results/HP_tuning/Hyper_GS_{}_{}_"+str(test)+".csv"
                           )
Analysis.training()
Analysis.prediction(verbose = verbose)
Analysis.prediction(
    x_pred = 'training_set',
    verbose = verbose)
Analysis.prediction(
    x_pred = 'testing_set',
    verbose = verbose)

