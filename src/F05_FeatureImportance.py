#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script reproducing Figure 5 of the manuscripts containing bar plot 
on the feature importance of the random forest algorithm comparing 
impact of each PSD size category on the algorithm performance to predict Kf.
Feature importance evaluation based on the method implemented in the class
PSD_2K_ML which is based on the routine "permutation_importance" from sklearn.

Author: A. Zech
"""

# import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
import PSD_2K_ML
plt.close('all')

### ===========================================================================
### Key words to specify modus of script:
### ===========================================================================

algorithm = 'RF' #'SVR' #LR' #'XG' # 'DT' # 'ANN'  #  ####  
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
fig_results = '../results/Figures_paper/Fig05_Feature_importance_{}_{}'.format(algorithm,soil_type)
textsize = 8
colors = ['C0','C1','C2','C3','C4','C5'] 

### ===========================================================================
### Speficy Algorithm and set target and feature variables
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

Analysis.set_algorithm(verbose = verbose)
Analysis.set_feature_variables()
Analysis.set_target_variables()
Analysis.data_split(verbose = verbose)

Analysis.training()
Analysis.prediction(verbose = verbose)
Analysis.prediction(
    x_pred = 'training_set',
    verbose = verbose)
Analysis.prediction(
    x_pred = 'testing_set',
    verbose = verbose)

### ===========================================================================
###   Plotting Feature Importance  based on feature permutation
### ===========================================================================

### Feature Importance from permutation importances 
importances_mean,importances_std = Analysis.feature_importance()
# modify index name to remove "F" infront of each sieve size range
importances_mean.index = [text[1:] for text in importances_mean.index.values]

fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(3.75, 2.8))##, layout='constrained')
importances_mean.plot.bar(yerr= importances_std, color = colors, ax=axs)

axs.set_ylabel("Feature importance mean",fontsize=textsize)
axs.set_xlabel(r"Sieve size ranges in $\mu$m",fontsize=textsize)
axs.tick_params(axis="y",which="major",labelsize=textsize)
axs.tick_params(axis="x",which="major",labelsize=textsize-1)

fig.tight_layout()
fig.savefig(fig_results+'.pdf')
