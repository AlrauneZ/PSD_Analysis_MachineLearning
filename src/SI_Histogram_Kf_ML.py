#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script producing a histogram of estimated Kf values for selected algorithm, 
dataset and feature/target variable combination 

Author: A. Zech
"""

import PSD_2K_ML
import matplotlib.pyplot as plt
plt.close('all')

### ===========================================================================
### Key words to specify modus of script:
### ===========================================================================

algorithm = 'RF' #'ANN' #'DT' #'SVR' #'LR' #'LR' #
soil_type = 'topall'
feature = 'PSD' #'dX_por' #'dX' #
target = 'Kf' # 'por' #
verbose = True #False #
data_set =  'full_set'

### ===========================================================================
### Set file pathes and names
### ===========================================================================
     
file_data = "../data/data_PSD_Kf_por_props.csv"
textsize = 12 #8

### ===========================================================================
### Speficy Algorithm and set target and feature variables, run training
### ===========================================================================

print("Training and Prediction of {}".format(algorithm))
print("###############################")

# Instance of PSD analysis with AI
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
Analysis.prediction(x_pred =data_set, verbose = verbose)

### ===========================================================================
### plot
### ===========================================================================

fig, ax = plt.subplots(figsize=(3.5, 2.5))
hist = plt.hist(Analysis.y_pred,density = True, bins = 30,rwidth=0.9,color='goldenrod')#,zorder = 3)
ax.set_xlabel("$\log_{10}(K_{obs}$)",fontsize=textsize)
ax.set_title(r'$K$ - distribution from {}'.format(algorithm),fontsize=textsize)
ax.grid(True, zorder = 1)
ax.set_xlim([-6.8,2.2])
ax.tick_params(axis="both",which="major",labelsize=textsize)
plt.tight_layout()
# plt.savefig('../results/Fig_Histogram_{}.png'.format(algorithm),dpi=300)
#plt.savefig('../results/Fig_Histogram_{}.pdf'.format(algorithm))
