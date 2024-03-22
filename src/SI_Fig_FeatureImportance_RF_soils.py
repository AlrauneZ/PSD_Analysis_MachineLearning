#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 14:57:23 2024

@author: alraune
"""

import PSD_2K_ML
import numpy as np
import matplotlib.pyplot as plt

plt.close('all')


# soil_types = ['all','sand','silt','clay']
soil_types = ['topall','sand','silt','clay']
feature = 'PSD' #'dX_por' #'dX' #
target = 'Kf' # 'por' #
verbose = True #False #
algorithm = 'RF' 

### ===========================================================================
### Set file pathes and names
### plot specifications
### ===========================================================================
  
file_data = "../data/data_PSD_Kf_por_props.csv"
file_fig = '../results/SI_Fig_FeatureImportance_RF_soils'

textsize = 8
figure_text = ['a','b','c','d']
colors = ['C0','C1','C2','C3','C4','C5'] #mcolors.TABLEAU_COLORS #['lightblue', 'blue', 'purple', 'red', 'black']


# =============================================================================
# Load Data and perform Algorithm fitting to produce predictions
# =============================================================================
print("Training and Prediction of algorithm")
print("###########################################")

### ===========================================================================
### Speficy Algorithm and set target and feature variables, run training
### ===========================================================================

# =============================================================================
### plot specifications
# Create a subplot for each model's comparison plot

# fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(7.5, 9), 
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(7.5,5), sharex = True) #, sharey = True)##, layout='constrained')
axs = axs.ravel()

# Plot the actual and predicted values for each model
for i,soil_type in enumerate(soil_types):
# for i,algorithm in enumerate(['LR']):
    print("\n###########################################")
    print("Training and Prediction of {}".format(algorithm))
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
    
    Analysis.set_target_variables()
    soil_class_names,soil_class_sample = Analysis.soil_class_specification()
    k_min,k_max = np.min(Analysis.target_var),np.max(Analysis.target_var)
    
    ### specify AI algorithm
    Analysis.set_algorithm(
        # algorith = algorithm, 
        verbose = verbose)
    
    ### specifying feature (input) and target (output) variables
    Analysis.set_feature_variables()#scale = False)
    Analysis.data_split()
    
    Analysis.training(verbose = verbose)
    ### determine prediction data on trained algorithm for specified data set
    Analysis.prediction(x_pred = 'full_set',verbose = verbose)
    

# for i,algorithm in enumerate(algs):
    importances_mean,importances_std = Analysis.feature_importance()

    importances_mean.plot.bar(yerr= importances_std, color = colors, ax=axs[i])

    if i in [0,2]:
        axs[i].set_ylabel("Mean accuracy decrease",fontsize=textsize)
    axs[i].set_xlabel("Features: Sieve size ranges in mm",fontsize=textsize)
    # axs[1].grid(True, zorder = 1)
    axs[i].tick_params(axis="y",which="major",labelsize=textsize)
    axs[i].tick_params(axis="x",which="major",labelsize=textsize-1)
    # axs[i].text(0.05,0.9,'({}) {}'.format(figure_text[i],soil_type),
    #             fontsize=textsize, transform=axs[i].transAxes,
    #             bbox = dict(boxstyle='round', facecolor='white'))

    axs[i].text(0.8,0.9,'Top-{}'.format(soil_types[i]),
            fontsize=textsize, transform=axs[i].transAxes,
            bbox = dict(boxstyle='round', facecolor='antiquewhite', alpha=0.5))

plt.tight_layout()
# plt.savefig(file_fig+'.png',dpi = 300)
plt.savefig(file_fig+'.pdf')
