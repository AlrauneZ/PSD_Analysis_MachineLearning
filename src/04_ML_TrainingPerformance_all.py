#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script evaluating performance of all six ML algorithms after training based on 
routines implemented in class "PSD_2K_ML":
    - specifying settings, such as feature/target-variable combination and 
        type of data(sub)set
    - reading data from csv-file (with standardised column naming)
    - train algorithms with optimal hyperparameters (determined separately)
    - evaluate algorithm performance and save results to file

Author: A. Zech
"""

# import numpy as np
import pandas as pd
import PSD_2K_ML

### ===========================================================================
### Key words to specify modus of script:
### ===========================================================================

soil_type = 'topall' #'por'# 'clay' #'silt' #'sand' ### ''full' #
feature ='PSD' # 'dX_por' #'dX' #
target = 'Kf' # 'por' # 

algorithms = ['DT','RF','XG','LR','SVR','ANN']
sets = ['training_set','testing_set','full_set']

verbose = True #False #
save_to_file = True #False #

print('\n#################################')
print('  Training Performance Evaluation')
print('   Feature variables: {}'.format(feature))
print('   Target variable: {}'.format(target))
print('#################################\n')

### ===========================================================================
### Set file pathes and names
### ===========================================================================

file_data = "../data/data_PSD_Kf_por_props.csv" # path to input data
### pathes + file names to store performance measures to
file_AI_performance_r2 = "../results/ML_performance/Performance_{}_{}_{}_r2.csv".format(feature,target,soil_type)
file_AI_performance_mse = "../results/ML_performance/Performance_{}_{}_{}_mse.csv".format(feature,target,soil_type)

### ===========================================================================
### Initialize Analysis
### ===========================================================================

Analysis = PSD_2K_ML.PSD_2K_ML(
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
results_r2 = pd.DataFrame(columns = algorithms,index=sets)
results_mse = pd.DataFrame(columns = algorithms,index=sets)

for algorithm in algorithms:
    ### specify AI algorithm
    Analysis.set_algorithm(algorithm = algorithm,
                            verbose = verbose)
    
    ### specifying feature (input) and target (output) variables
    Analysis.set_feature_variables()
    Analysis.set_target_variables()
    
    ### split data for training and train 
    Analysis.data_split(verbose = verbose)  
    Analysis.training()

    for x_pred in sets:
        Analysis.prediction(
            x_pred = x_pred,
            verbose = verbose)
        results_r2[algorithm][x_pred] = Analysis.r2
        results_mse[algorithm][x_pred] = Analysis.mse

if save_to_file:
    results_r2.to_csv(file_AI_performance_r2)
    results_mse.to_csv(file_AI_performance_mse)