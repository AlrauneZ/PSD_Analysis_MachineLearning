#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 11:24:52 2023

@author: zech0001
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PSD_2K_ML
from sklearn.inspection import permutation_importance

soil_type = 'topall'#'sand' #  'clay' #'silt' #'por' #
feature = 'PSD' #'dX_por' #'dX' #
target = 'Kf' #'por' # #

verbose = True #False #
algorithm ='RF' # fix - only works for this algorithm!

fig_results = '../results/Fig05_Feature_importance_RF_MDI_{}'.format(soil_type)


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

### ===========================================================================
###   Extracting Feature Importance Information from algorithm
### ===========================================================================

feature_names = list(Analysis.feature_var)
feature_names[0] = 'F0.01-0.1'
feature_names[1] = 'F0.1-0.2'
feature_names[2] = 'F0.2-0.5'
feature_names[3] = 'F0.4-1'

###Tree’s Feature Importance from Mean Decrease in Impurity
importances = Analysis.AI.feature_importances_

importances_MDI = pd.Series(importances, index=feature_names)
std_MDI = np.std([tree.feature_importances_ for tree in Analysis.AI.estimators_], axis=0)

###Tree’s Feature Importance from permutation importances 
result = permutation_importance(
    Analysis.AI, Analysis.x_test, Analysis.y_test, n_repeats=10, random_state=42, n_jobs=2)
importances_PI = pd.Series(result.importances_mean, index=feature_names)

### ===========================================================================
###   Plotting Feature Importance based on mean decrease in impurity
### ===========================================================================
#import matplotlib.colors as mcolors

plt.close('all')
textsize = 8
colors = ['C0','C1','C2','C3','C4','C5'] #mcolors.TABLEAU_COLORS #['lightblue', 'blue', 'purple', 'red', 'black']

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(7.5, 3.0))##, layout='constrained')
axs = axs.ravel()

importances_MDI.plot.bar(yerr=std_MDI, color = colors, ax=axs[0])

#axs[0].set_title("Feature importances using MDI")
axs[0].set_ylabel("Mean decrease in impurity",fontsize=textsize)
axs[0].set_xlabel("Features: Sieve size ranges in mm",fontsize=textsize)
axs[0].tick_params(axis="y",which="major",labelsize=textsize)
axs[0].tick_params(axis="x",which="major",labelsize=textsize-1)

### ===========================================================================
###   Plotting Feature Importance  based on feature permutation
### ===========================================================================

#fig, ax = plt.subplots()
importances_PI.plot.bar(yerr=result.importances_std, 
                            color = colors,
                            ax=axs[1])
#ax.set_title("Feature importances using permutation on full model")
axs[1].set_ylabel("Mean accuracy decrease",fontsize=textsize)
axs[1].set_xlabel("Features: Sieve size ranges in mm",fontsize=textsize)
# axs[1].grid(True, zorder = 1)
axs[1].tick_params(axis="y",which="major",labelsize=textsize)
axs[1].tick_params(axis="x",which="major",labelsize=textsize-1)


fig.tight_layout()
fig.savefig(fig_results+'_MDI.pdf')
