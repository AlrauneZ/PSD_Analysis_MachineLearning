#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 08:42:04 2023

@author: zech0001
"""

import matplotlib.pyplot as plt
# import numpy as np
import skopt.plots
import PSD_2K_ML
plt.close('all')

algs = ['DT','RF']#,['XG','SVR','ANN']#,'LR' #'DT',#'RF',

# algorithm = 'LR' #'RF' #'ANN' #'DT' #'SVR' #'LR' #
soil_type = 'full' # 'silt'#'sand' #'clay' # 
verbose = True #False #

print('\n#################################')
print('   Hyper parameter tuning')
print('#################################\n')

### ===========================================================================
### Set file pathes and names // Plot specifications
### ===========================================================================

file_data = "../data/data_PSD_Kf_por.csv"
textsize = 10

### ===========================================================================
### Initialize Analysis and prepare data
### ===========================================================================

Analysis = PSD_2K_ML.PSD_2K_ML()
data_PSD = Analysis.prepare_data(filename=file_data,
                      soil_type = soil_type, 
                      remove_outlier = False,
                      verbose = verbose,      
                      )

### Loop over all tested algorithms
for algorithm in algs:

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
    Analysis.hyperparameter_GS(verbose = verbose)
    results = Analysis.hyperparameter_skopt(verbose = verbose)

    # =============================================================================
    plt.close('all')
    fig1 = skopt.plots.plot_evaluations(results)
    
    if algorithm == 'LR':
        fig1.tick_params(axis="both",which="major",labelsize=textsize-1) 
        fig1.set_xlabel(fig1.get_xlabel(),fontsize = textsize)
        fig1.set_ylabel(fig1.get_ylabel(),fontsize = textsize)
    else:
        for i in range(fig1.shape[0]):
            for j in range(fig1.shape[1]):
                fig1[i,j].tick_params(axis="both",which="major",labelsize=textsize-1) 
                fig1[i,j].set_xlabel(fig1[i,j].get_xlabel(),fontsize = textsize)
                fig1[i,j].set_ylabel(fig1[i,j].get_ylabel(),fontsize = textsize)
    # plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.3)
    
    plt.savefig('../results/HP_tuning/HP_skopt_eval_{}_{}.png'.format(Analysis.soil_type,Analysis.algorithm),dpi=300, bbox_inches = 'tight')
    plt.savefig('../results/HP_tuning/HP_skopt_eval_{}_{}.pdf'.format(Analysis.soil_type,Analysis.algorithm),bbox_inches = 'tight')
    
    # # =============================================================================
    
    # plt.figure(2)
    fig2 = skopt.plots.plot_objective(results)
    if algorithm == 'LR':
        fig2.tick_params(axis="both",which="major",labelsize=textsize-1) 
        fig2.set_xlabel(fig2.get_xlabel(),fontsize = textsize)
        fig2.set_ylabel(fig2.get_ylabel(),fontsize = textsize)
    else:
        for i in range(fig2.shape[0]):
            for j in range(fig2.shape[1]):
                fig2[i,j].tick_params(axis="both",which="major",labelsize=textsize-1) 
                fig2[i,j].set_xlabel(fig2[i,j].get_xlabel(),fontsize = textsize)
                fig2[i,j].set_ylabel(fig2[i,j].get_ylabel(),fontsize = textsize)
    # plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.3)
    plt.savefig('../results/HP_tuning/HP_skopt_obj_{}_{}.png'.format(Analysis.soil_type,Analysis.algorithm),dpi = 300, bbox_inches = 'tight')
    plt.savefig('../results/HP_tuning/HP_skopt_obj_{}_{}.pdf'.format(Analysis.soil_type,Analysis.algorithm), bbox_inches = 'tight')
    
 
    # np.savetxt('../results/HP_tuning/HP_scopt_{}_{}.txt'.format(Analysis.soil_type,Analysis.algorithm),results['x'])

# # =============================================================================

### ===========================================================================
###   Algorithm Performance with optimal Parameters
### ===========================================================================

# Analysis.training()
# Analysis.prediction(verbose = verbose)
# Analysis.prediction(
#     x_pred = 'training_set',
#     verbose = verbose)
# Analysis.prediction(
#     x_pred = 'testing_set',
#     verbose = verbose)
