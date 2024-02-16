#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 16:39:32 2023

@author: alraune
"""

import numpy as np
import pandas as pd
import PSD_2K_ML
import matplotlib.pyplot as plt
plt.close('all')

algorithm ='RF' #'ANN' #'LR' #'DT' #'SVR' # 'LR' #
soil_type = 'all' #'por' 'silt' #'full'#'sand' #  'clay' #
feature = 'PSD' #'dX_por' #'dX' #
target = 'Kf' #'por' # #
verbose = True #False #

print('\n#############################################')
print('  Training and Application to additional data')
print('###############################################\n')

### ===========================================================================
### Set file pathes and names
### ===========================================================================
file_AI_data = "../data/AI_data.csv"
file_application_data = "../data/application_data.csv"

#Analysis = PSD_2K_ML.PSD_2K_ML()
Analysis = PSD_2K_ML.PSD_2K_ML(
                        algorithm = algorithm,
                        feature = feature,
                        target = target,                            
                        )

data_PSD = Analysis.prepare_data(filename=file_AI_data,
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
Analysis.set_feature_variables()
Analysis.set_target_variables()
### split data for training and train 
Analysis.data_split(verbose = verbose)


### ===========================================================================
###   Algorithm Performance with optimal Parameters
### ===========================================================================

Analysis.training()
Analysis.prediction(verbose = verbose)

data_app = pd.read_csv(file_application_data)
data_app['logK'] = np.log10(data_app["Kf"].values)
y_app_obs = data_app['logK']

sieve_classes = data_app.columns[[x.startswith("F") for x in data_app.columns]]
x_app_pred = pd.DataFrame(data_app, columns=sieve_classes)#.values

results = Analysis.application(
    x_app_pred,
    y_app = y_app_obs,
    verbose = False)

y_app_pred = results

# =============================================================================
fig, ax = plt.subplots(figsize=(3.5, 2.5))
# fig, ax = plt.subplots(figsize=(3.75, 3.))
# fig, ax = plt.subplots(figsize=(7.5, 6))
textsize = 12 #8
#soil_class_name, soil_class_sample = Analysis.soil_class_specification()

hist = plt.hist(y_app_pred,density = True, bins = 30,rwidth=0.9,color='forestgreen')#,zorder = 3)

ax.set_xlabel("$\log_{10}(K_{obs}$)",fontsize=textsize)
#ax.set_ylabel("density",fontsize=textsize)
#ax.set_title(r'$K$ - distribution from {}'.format(algorithm),fontsize=textsize)
ax.grid(True, zorder = 1)

ax.set_xlim([-5,2.2])
ax.tick_params(axis="both",which="major",labelsize=textsize)

plt.tight_layout()
plt.savefig('../results/Fig_Application_Histogram_{}.png'.format(algorithm),dpi=300)

# # =============================================================================
# ### scatter plot of predicted against observed K-values

# plt.figure(2)
# fig, ax = plt.subplots(figsize=(3.1, 3.1))
# soil_class_name, soil_class_sample = Analysis.soil_class_specification()

# scatter = ax.scatter(
#     x = y_app_obs,
#     y = y_app_pred, 
#     c = 'forestgreen',
#     # c = soil_class_sample, 
#     # cmap= 'coolwarm', #'Spectral', 
#     marker='.', 
#     s= 10,
#     zorder = 2)

# ### one-by-one line of 
# ax.plot([-6.8,2.2],[-6.8,2.2], c="grey", linestyle = "dotted")
# #ax.plot(Analysis.y_test,Analysis.y_test, c="0.3", ls = ':',lw = 3,zorder = 3)

# ax.set_xlabel("$\log_{10}(K_{obs}$)",fontsize=textsize)
# ax.set_ylabel("$\log_{10}(K_{ML}$)",fontsize=textsize)
# #ax.set_ylabel("$\log_{10}(K_{pred}$)",fontsize=textsize)
# ax.set_title('Application of {}'.format(algorithm),fontsize=textsize)
# #ax.set_title('Linear Regression',fontsize=textsize)
# #ax.set_title('Random Forest',fontsize=textsize)
# ax.grid(True, zorder = 1)

# ax.set_xlim([-6.8,2.2])
# ax.set_ylim([-6.8,2.2])
# ax.set_xticks([-6,-4,-2,0,2])
# ax.set_yticks([-6,-4,-2,0,2])
# ax.tick_params(axis="both",which="major",labelsize=textsize)
# #ax.axis("equal")

# plt.tight_layout()
# plt.savefig('../results/Fig_Application_Scatter_{}.png'.format(algorithm),dpi = 300)
# # plt.savefig('../results/Fig_Scatter_{}.pdf'.format(algorithm))
