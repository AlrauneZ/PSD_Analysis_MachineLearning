#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 12:32:43 2023

@author: alraune
"""

# import numpy as np
import pandas as pd
import PSD_2K_ML

verbose = True #False #
save_to_file = True

soil_type = 'all'
algorithms = ['DT','RF','XG','LR','SVR','ANN']
sets = ['full_set','training_set','testing_set']

print('\n#################################')
print('  Training Performance Evaluation')
print('#################################\n')

### ===========================================================================
### Set file pathes and names
### ===========================================================================

file_AI_data = "../data/AI_data.csv"
file_AI_performance_r2 = "../results/Performance_r2_{}.csv".format(soil_type)
file_AI_performance_mse = "../results/Performance_mse_{}.csv".format(soil_type)

Analysis = PSD_2K_ML.PSD_2K_ML()

data_PSD = Analysis.prepare_data(filename=file_AI_data,
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