#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 15:02:31 2024

@author: alraune
"""

import PSD_2K_ML
import matplotlib.pyplot as plt
plt.close('all')

algorithm ='LR' # 'RF' #'ANN' #'DT' #'SVR' #
algorithms = ['LR'] #'ANN'  #'XG' #'SVR'  #'RF' 
#algorithms = ['DT','RF','XG','LR','SVR','ANN']
soil_type = 'topall' #'silt'#'sand' # 'clay' # 'por' #'clay' #
feature = 'PSD' #'dX_por' #'dX' #
target = 'Kf' #'por' # 
verbose = True #False #

# =============================================================================
### plot specifications
textsize = 8

# =============================================================================
# Load Data and perform Algorithm fitting to produce predictions
# =============================================================================
      
print("Training and Prediction of {}".format(algorithm))
print("###############################")


### ===========================================================================
### Set file pathes and names
### ===========================================================================
file_data = "../data/data_PSD_Kf_por_props.csv"
file_fig = '../results/Fig_Scatter_{}'.format(algorithm)

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
### Speficy Algorithm and set target and feature variables, run training
### ===========================================================================

### specify AI algorithm
Analysis.set_algorithm(verbose = verbose)

### specifying feature (input) and target (output) variables
Analysis.set_feature_variables()
Analysis.set_target_variables()

### split data for training and train 
Analysis.data_split(verbose = verbose)
Analysis.training()


### ===========================================================================
###   Algorithm Performance with optimal Parameters
### ===========================================================================

### determine prediction data on trained algorithm for specified data set
# Analysis.prediction(x_pred = 'testing_set', verbose = verbose)
# Analysis.prediction(x_pred = 'training_set',verbose = verbose)
Analysis.prediction(x_pred = 'full_set', verbose = verbose)

### calculate percentiles for plot
bc5,pc5 = Analysis.quantiles_4_plot(bins=10,nth=5)
bc95,pc95 = Analysis.quantiles_4_plot(bins=10,nth=95)

# =============================================================================
### plot specifications
# fig, ax = plt.subplots(figsize=(3.75, 3.75)) # for paper
# textsize = 8

fig, ax = plt.subplots(figsize=(0.33*7.5,2.5))
soil_class_name, soil_class_sample = Analysis.soil_class_specification()

# =============================================================================

### scatter plot of predicted against observed K-values
scatter = ax.scatter(
    x = Analysis.y_obs,
    y = Analysis.y_pred, 
    # c = 'goldenrod',
    c = soil_class_sample, 
    cmap= 'coolwarm', #'Spectral', 
    marker='.', 
    s= 10,
    zorder = 2)

# ### Plotting the 5th and 95th percentile range of fit
plt.plot(bc5,pc5,'--',c = 'k',zorder=3)
plt.plot(bc95,pc95,'--', c = 'k', zorder = 3)

### one-by-one line of 
ax.plot(Analysis.y_test,Analysis.y_test, c="grey", linestyle = "dotted")
#ax.plot(Analysis.y_test,Analysis.y_test, c="0.3", ls = ':',lw = 3,zorder = 3)

ax.set_xlabel("$\log_{10}(K_{obs}$ [m/d])",fontsize=textsize)
ax.set_ylabel("$\log_{10}(K_{ML}$ [m/d])",fontsize=textsize)
#ax.set_ylabel("$\log_{10}(K_{pred}$)",fontsize=textsize)
#ax.set_title('{}'.format(algorithm),fontsize=textsize)
ax.set_title('Linear Regression',fontsize=textsize)
#ax.set_title('Random Forest',fontsize=textsize)
ax.grid(True, zorder = 1)

ax.set_xlim([-6.8,2.2])
ax.set_ylim([-6.8,2.2])
ax.set_xticks([-6,-4,-2,0,2])
ax.set_yticks([-6,-4,-2,0,2])
ax.tick_params(axis="both",which="major",labelsize=textsize)
#ax.axis("equal")

#plt.legend(handles=scatter.legend_elements()[0], labels = soil_class_name, #title = "soil class", 
#            loc = 'lower right', ncol=5, prop={'size': textsize-1},columnspacing=1.0)#,fontsize=textsize)

# ax.set_xlim([-7,5])
# plt.legend(handles=scatter.legend_elements()[0], labels = soil_class_name, title = "soil class", 
#             loc = 'lower right', ncol=2, prop={'size': textsize},columnspacing=1.0)#,fontsize=textsize)

#plt.legend(handles=scatter.legend_elements()[0], labels = soil_class_name, title = "soil class", 
#            loc = 'best', ncol=4, prop={'size': textsize})#,fontsize=textsize)

plt.tight_layout()
plt.savefig(file_fig+'.png',dpi = 300)
plt.savefig(file_fig+'.pdf')
