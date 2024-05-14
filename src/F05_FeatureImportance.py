#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 11:24:52 2023

@author: zech0001
"""

# import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
import PSD_2K_ML
# from sklearn.inspection import permutation_importance


algorithm = 'RF' #'SVR' #LR' #'XG' # 'DT' # 'ANN'  #  ####  
soil_type = 'topall'#'sand' #  'clay' #'silt' #'por' #
feature = 'PSD' #'dX_por' #'dX' #
target = 'Kf' #'por' # #

verbose = True #False #

fig_results = '../results/Fig05_Feature_importance_{}_{}'.format(algorithm,soil_type)


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
###   Plotting Feature Importance  based on feature permutation
### ===========================================================================

plt.close('all')
textsize = 8
colors = ['C0','C1','C2','C3','C4','C5'] #mcolors.TABLEAU_COLORS #['lightblue', 'blue', 'purple', 'red', 'black']

### ===========================================================================
###   Extracting Feature Importance Information from algorithm
### ===========================================================================

# ###Tree’s Feature Importance from permutation importances 
# from sklearn.inspection import permutation_importance
# import pandas as pd
# feature_names = list(Analysis.psd)
# feature_names[0] = 'F0.01-0.1'
# feature_names[1] = 'F0.1-0.2'
# feature_names[2] = 'F0.2-0.5'
# feature_names[3] = 'F0.4-1'

# result = permutation_importance(
#     Analysis.AI, Analysis.x_test, Analysis.y_test, n_repeats=10, random_state=42, n_jobs=2)
# importances_mean = pd.Series(result.importances_mean, index=feature_names)
# importances_std = result.importances_std

### Feature Importance from permutation importances 
importances_mean,importances_std = Analysis.feature_importance()

fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(3.75, 2.8))##, layout='constrained')
importances_mean.plot.bar(yerr= importances_std, color = colors, ax=axs)

#ax.set_title("Feature importances using permutation on full model")
axs.set_ylabel("Mean accuracy decrease",fontsize=textsize)
axs.set_xlabel("Features: Sieve size ranges in mm",fontsize=textsize)
# axs[1].grid(True, zorder = 1)
axs.tick_params(axis="y",which="major",labelsize=textsize)
axs.tick_params(axis="x",which="major",labelsize=textsize-1)
# axs[i].text(0.1,0.9,'NSE = {:.2f}'.format(r2_Barr_measured),
#             fontsize=textsize, transform=axs[i].transAxes,
#             bbox = dict(boxstyle='round', facecolor='white'))

fig.tight_layout()
#fig.savefig(fig_results+'.pdf')
