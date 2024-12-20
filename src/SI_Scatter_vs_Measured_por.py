#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script reproducing Figure of the supporting information containing scatter 
plots comparing porosity estimates of all 6 algorithms to measured porosity
for those samples where available (Top-por).

Author: A. Zech
"""

import PSD_2K_ML
import numpy as np
import matplotlib.pyplot as plt

### ===========================================================================
### Set file pathes and Plot specifications 
### ===========================================================================

algs = ["DT", "RF", "XG", "LR", "SVR", "ANN"]
soil_type ='por'
feature = 'PSD' 
target =  'por' 
verbose = True #False #

### ===========================================================================
### Set file pathes and names & lot specifications
### ===========================================================================
  
file_data = "../data/data_PSD_Kf_por_props.csv"
file_fig = '../results/Figures_SI/SI_Fig_Scatter_Measured_{}_{}'.format(feature,target)

textsize = 8
markersize = 2
figure_text = ['a','b','c','d','e','f']

### ===========================================================================
### Speficy Algorithm and set target and feature variables, run training
### ===========================================================================

print("Training and Prediction of all 6 algorithms")
print("###########################################")

fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(7.5, 4.9), 
                        sharex = True, sharey = True)##, layout='constrained')
axs = axs.ravel()

for i,algorithm in enumerate(algs):
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
    soil_class_names,soil_class_sample = Analysis.soil_class_specification(sort = True)
    k_min,k_max = np.min(Analysis.target_var),np.max(Analysis.target_var)
    Analysis.set_algorithm(verbose = verbose)
    Analysis.set_feature_variables()#scale = False)
    Analysis.data_split()
    Analysis.training(verbose = verbose)
    Analysis.prediction(x_pred = 'full_set',verbose = verbose)
    bc5,pc5 = Analysis.quantiles_4_plot(bins=10,nth=5)
    bc95,pc95 = Analysis.quantiles_4_plot(bins=10,nth=95)

    scatter = axs[i].scatter(
        x = Analysis.y_obs,
        y = Analysis.y_pred, 
        c = 'chocolate', 
        marker='.', 
        s= markersize,
        label=algorithm,
        zorder = 2)

    axs[i].set_xlabel(r"$\theta_{obs}$",fontsize=textsize)
    axs[i].set_ylabel(r"$\theta_{pred}$",fontsize=textsize)
    axs[i].grid(True, zorder = 1)
    axs[i].tick_params(axis="both",which="major",labelsize=textsize)
    axs[i].plot(bc5,pc5,'--',c = 'k',zorder=3)
    axs[i].plot(bc95,pc95,'--', c = 'k', zorder = 3)
    axs[i].plot(Analysis.target_var,Analysis.target_var,':', c="grey")
    axs[i].set_xlim([k_min-0.01,k_max+0.01])
    axs[i].set_ylim([k_min-0.01,k_max+0.01])
    axs[i].text(0.1,0.89,'({}) {}'.format(figure_text[i],algorithm),
                fontsize=textsize, transform=axs[i].transAxes,
                bbox = dict(boxstyle='round', facecolor='white'))

    axs[i].text(0.55,0.09,'NSE = {:.2f}'.format(Analysis.r2),
                fontsize=textsize, transform=axs[i].transAxes,
                bbox = dict(boxstyle='round', facecolor='white'))

axs[0].text(-0.05,1.1,'{} --> {}'.format(feature,target),
            fontsize=textsize+1, transform=axs[0].transAxes,
            bbox = dict(boxstyle='round', facecolor='antiquewhite', alpha=0.5))

plt.tight_layout()
plt.savefig(file_fig+'.pdf')
