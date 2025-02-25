#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script reproducing Figure of the supporting information on the feature 
importance of the random forest algorithm comparing impact of each PSD size 
category on the algorithm performance to predict Kf. Subplots show feature
importance for the different data(sub)sets.
Feature importance evaluation based on the method implemented in the class
PSD_2K_ML which is based on the routine "permutation_importance" from sklearn.

Author: A. Zech
"""

import PSD_2K_ML
import numpy as np
import matplotlib.pyplot as plt
plt.close('all')

### ===========================================================================
### Key words to specify modus of script:
### ===========================================================================

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
file_fig = '../results/Figures_SI/SI_Fig_FeatureImportance_RF_soils2'

textsize = 8
figure_text = ['a','b','c','d']
colors = ['C0','C1','C2','C3','C4','C5'] 

# =============================================================================
# Load Data and perform Algorithm fitting to produce predictions
# =============================================================================
print("Training and Prediction of algorithm")
print("###########################################")

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(7.5,5), sharex = True) 
axs = axs.ravel()
for i,soil_type in enumerate(soil_types):
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
    Analysis.set_algorithm(verbose = verbose)
    Analysis.set_feature_variables()
    Analysis.data_split()
    Analysis.training(verbose = verbose)
    Analysis.prediction(x_pred = 'full_set',verbose = verbose)
    
    importances_mean,importances_std = Analysis.feature_importance()
    # modify index name to remove "F" infront of each sieve size range
    importances_mean.index = [text[1:] for text in importances_mean.index.values]
    importances_mean.plot.bar(yerr= importances_std, color = colors, ax=axs[i])

    if i in [0,2]:
        axs[i].set_ylabel("Feature importance mean",fontsize=textsize)
    axs[i].set_xlabel("Sieve size ranges in $\mu$m",fontsize=textsize)
    axs[i].tick_params(axis="y",which="major",labelsize=textsize)
    axs[i].tick_params(axis="x",which="major",labelsize=textsize-1)

    axs[i].text(0.8,0.9,'Top-{}'.format(soil_types[i]),
            fontsize=textsize, transform=axs[i].transAxes,
            bbox = dict(boxstyle='round', facecolor='antiquewhite', alpha=0.5))

plt.tight_layout()
plt.savefig(file_fig+'_new.pdf')
