#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""

import PSD_2K_ML
import matplotlib.pyplot as plt
# import numpy as np
import skopt.plots
import os
import warnings

warnings.filterwarnings("ignore")
plt.close('all')


algorithms = ['RF'] #'ANN'  #'XG' #'SVR'  #'RF' 
#algorithms = ['DT','RF','XG','LR','SVR','ANN']
soil_type = 'topall' #'silt'#'sand' # 'clay' # 'por' #'clay' #
verbose = True #False #
feature = 'dX' #'PSD' #'dX_por' #
target = 'Kf' #'por' # 
test = 0

file_data = "../data/data_PSD_Kf_por_props.csv"
# file_data = "../data/data_PSD_Kf_por.csv"
dir_results = "../results/HP_tuning_{}_{}/"#.format(feature,target)
file_results_GS = "Hyper_{}_{}_"+str(test)+"_GS.csv"
file_results_skopt = "Hyper_{}_{}_"+str(test)+"_Skopt.csv"
file_figure_skopt = 'HP_{}_{}_{}_skopt.png'#.format(soil_type,algorithm,test)
 

for algorithm in algorithms:
    print('\n################################################')
    print('   Hyper parameter tuning for algorithm {}'.format(algorithm))
    print('   Feature variables: {}'.format(feature))
    print('   Target variable: {}'.format(target))
    print('##################################################\n')
    
    ### ===========================================================================
    ### Set file pathes and names
    ### ===========================================================================   
    
    dir_results_HP=dir_results.format(feature,target)
    if not os.path.exists(dir_results_HP):
        os.makedirs(dir_results_HP)
    path_results_GS = dir_results_HP+file_results_GS
    path_results_skopt = dir_results_HP+file_results_skopt
    path_figure_skopt = dir_results_HP+file_figure_skopt.format(soil_type,algorithm,test)
            
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
    data_feat = Analysis.set_feature_variables() #feature = feature)
    data_target = Analysis.set_target_variables() #target = target)
    
    ### split data for training and train 
    Analysis.data_split()#verbose = verbose)
    
    ### ===========================================================================
    ###   Hyperparameter testing GridSearch
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
    
    ### ===========================================================================
    ###   Hyperparameter testing Skopt
    ### ===========================================================================
    
    results = Analysis.hyperparameter_skopt(verbose = verbose,
                                            file_results=path_results_skopt,                
                                            )  
    
    textsize= 10
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
    plt.savefig(path_figure_skopt,dpi = 300, bbox_inches = 'tight')
    
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
    
