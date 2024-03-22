#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 14:57:23 2024

@author: alraune
"""

import PSD_2K_ML
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
# from matplotlib.transforms import Transform
plt.close('all')

### algorithms to plot (and their order)
algs = ["DT", "RF", "XG", "LR", "SVR", "ANN"]
soil_type ='por' #'sand' # 'topall' #  'clay' #'silt'#
feature = 'PSD' #'dX_por' #'dX' #
target =  'por' #'Kf' #
verbose = True #False #

### ===========================================================================
### Set file pathes and names
### plot specifications
### ===========================================================================
  
file_data = "../data/data_PSD_Kf_por_props.csv"
if feature == 'PSD' and target == 'Kf': 
    file_fig = '../results/SI_Fig_Scatter_Measured_{}'.format(soil_type)
    text = 'Top - {}'.format(soil_type)
else:
    file_fig = '../results/SI_Fig_Scatter_Measured_{}_{}'.format(feature,target)
    text = '{} --> {}'.format(feature,target)

textsize = 8
markersize = 2
cmap = cm.get_cmap('Spectral')
figure_text = ['a','b','c','d','e','f']


# =============================================================================
# Load Data and perform Algorithm fitting to produce predictions
# =============================================================================
print("Training and Prediction of all 6 algorithms")
print("###########################################")

### ===========================================================================
### Speficy Algorithm and set target and feature variables, run training
### ===========================================================================

# =============================================================================
### plot specifications
# Create a subplot for each model's comparison plot

# fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(7.5, 9), 
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(7.5, 5.25), 
                        sharex = True, sharey = True)##, layout='constrained')
axs = axs.ravel()

# Plot the actual and predicted values for each model
for i,algorithm in enumerate(algs):
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
    
    ## calculate percentiles for plot
    bc5,pc5 = Analysis.quantiles_4_plot(bins=10,nth=5)
    bc95,pc95 = Analysis.quantiles_4_plot(bins=10,nth=95)

# Plot the actual and predicted values for each model
# for i,algorithm in enumerate(algs):

    scatter = axs[i].scatter(
        x = Analysis.y_obs,
        y = Analysis.y_pred, 
        c = soil_class_sample, 
        cmap= cmap, 
        marker='.', 
        s= markersize,
        label=algorithm,
        zorder = 2)

    if target == 'por':
        axs[i].set_xlabel(r"$\theta_{obs}$)",fontsize=textsize)
        axs[i].set_ylabel(r"$\theta_{pred}$)",fontsize=textsize)
    else:
        axs[i].set_xlabel("$\log_{10}(K_{obs}$ [m/d])",fontsize=textsize)
        axs[i].set_ylabel("$\log_{10}(K_{pred}$ [m/d])",fontsize=textsize)
    # axs[i].set_title('{}'.format(algorithm),fontsize=textsize)
    axs[i].grid(True, zorder = 1)
    axs[i].tick_params(axis="both",which="major",labelsize=textsize)

    ### Plotting the 5th and 95th percentile range of fit
    axs[i].plot(bc5,pc5,'--',c = 'k',zorder=3)
    axs[i].plot(bc95,pc95,'--', c = 'k', zorder = 3)
    ### one-by-one line of 
    axs[i].plot(Analysis.y_test,Analysis.y_test,':', c="grey")
    axs[i].set_xlim([k_min-0.01,k_max+0.01])
    axs[i].set_ylim([k_min-0.01,k_max+0.01])
    axs[i].text(0.1,0.89,'({}) {}'.format(figure_text[i],algorithm),
                fontsize=textsize, transform=axs[i].transAxes,
                bbox = dict(boxstyle='round', facecolor='white'))

    axs[i].text(0.55,0.09,'NSE = {:.2f}'.format(Analysis.r2),
                fontsize=textsize, transform=axs[i].transAxes,
                bbox = dict(boxstyle='round', facecolor='white'))


axs[0].text(-0.05,1.1,text,
# axs[0].text(-0.05,1.1,'Top - {}'.format(soil_type),
            fontsize=textsize+1, transform=axs[0].transAxes,
            bbox = dict(boxstyle='round', facecolor='antiquewhite', alpha=0.5))

fig.subplots_adjust(bottom=.16)
fig.legend(handles=scatter.legend_elements()[0], 
            labels=list(soil_class_names), 
            loc='lower center', 
            ncol=7, 
            # bbox_to_anchor=(1, 0.1),             
            prop={'size': textsize},#,fontsize=textsize,
            bbox_transform=fig.transFigure,
#            columnspacing=1.0,
#            title = "soil classes",
            )

### plt.tight_layout()
# plt.savefig(file_fig+'.png',dpi = 300)
plt.savefig(file_fig+'.pdf')
