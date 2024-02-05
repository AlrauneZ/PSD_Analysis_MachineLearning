#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 08:42:04 2023

@author: zech0001
"""

import PSD_2K_ML
import matplotlib.pyplot as plt
# import numpy as np
import skopt.plots

import warnings
warnings.filterwarnings("ignore")
plt.close('all')


algorithm = 'SVR' #'DT' #'LR' #'XG' #'ANN' #'RF'  #
soil_type = 'all'
verbose = True #False #

hp_GS = True
hp_skopt = True

# hp_GS = True
# hp_skopt = False

# hp_GS = False
# hp_skopt = True


test = 2

print('\n################################################')
print('   Hyper parameter tuning for algorithm {}'.format(algorithm))
print('##################################################\n')

### ===========================================================================
### Set file pathes and names
### ===========================================================================
file_AI_data = "../data/AI_data.csv"
#file_psd_props = "../data/PSD_properties.csv"
# file_Kemp_all = "../results/Kemp_all.csv"

textsize= 10

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
                      # verbose = verbose,      
                      )

### ===========================================================================
### Speficy Algorithm and set target and feature variables
### ===========================================================================

### specify AI algorithm
Analysis.set_algorithm(algorithm = algorithm,
                       # verbose = verbose
                       )

### specifying feature (input) and target (output) variables
Analysis.set_feature_variables()
Analysis.set_target_variables()

### split data for training and train 
Analysis.data_split()#verbose = verbose)

### ===========================================================================
###   Hyperparameter testing 
### ===========================================================================

if hp_GS:
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

if hp_skopt:
    results = Analysis.hyperparameter_skopt(verbose = verbose,
                                            file_results = "../results/HP_tuning/Hyper_Skopt_{}_{}_"+str(test)+".csv"
                                            )
    
    
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
    plt.savefig('../results/HP_tuning/HP_scopt_obj_{}_{}_{}.png'.format(Analysis.soil_type,Analysis.algorithm,test),dpi = 300, bbox_inches = 'tight')
    
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
