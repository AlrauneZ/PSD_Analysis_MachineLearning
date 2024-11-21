#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script reproducing Figure 4 of the manuscripts containing scatter plots 
comparing Kf values: estimates of Kf with empirical Barr method to measured Kf
and estimates of Kf with empirical Barr method to estimates of the Random
Forest ML algorithm for standard feature/target variable combination.

Author: A. Zech
"""

import matplotlib.pyplot as plt
import numpy as np
import PSD_2K_ML
from sklearn.metrics import r2_score
plt.close('all')

### ===========================================================================
### Key words to specify modus of script:
### ===========================================================================

algorithm = "RF"
soil_type ='topall' 
feature = 'PSD' 
target = 'Kf' 
verbose = True #False #

### ===========================================================================
### Set file pathes and names & plot specifications
### ===========================================================================
  
file_data = "../data/data_PSD_Kf_por_props_Kemp.csv"
fig_results = '../results/Figures_paper/Fig04_Scatter_RF_Barr'
textsize = 8
markersize = 10

### ===========================================================================
### Load Data and perform Algorithm fitting to produce predictions
### ===========================================================================

print("\n###########################################")
print("Training and Prediction of {}".format(algorithm))
Analysis = PSD_2K_ML.PSD_2K_ML(
                        algorithm = algorithm,
                        feature = feature,
                        target = target,                            
                        )
data = Analysis.prepare_data(filename=file_data,
                      soil_type = soil_type, 
                      remove_outlier = False,
                      verbose = verbose,      
                      )

Analysis.set_target_variables()
Analysis.set_algorithm(verbose = verbose)
Analysis.set_feature_variables()#scale = False)
Analysis.data_split()
Analysis.training(verbose = verbose)
Analysis.prediction(x_pred = 'full_set',verbose = verbose)

soil_class_names,soil_class_sample = Analysis.soil_class_specification(sort = True)
k_min,k_max = np.min(Analysis.target_var),np.max(Analysis.target_var)

logKf_Barr = np.log10(100*data['K_Barr'])
r2_Barr_RF = r2_score(Analysis.y_pred,logKf_Barr)
r2_Barr_measured = r2_score(Analysis.target_var,logKf_Barr)
print("NSE measured vs Barr:", r2_Barr_measured)
print("NSE RF vs Barr:", r2_Barr_RF)
print("NSE RF vs measured:", Analysis.r2)

### ===========================================================================
### Plot the actual and predicted values for each model
### ===========================================================================

fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(3.75, 6.5), sharex = True)
axs = axs.ravel()

### ===========================================================================
### right figure
i = 1
scatter = axs[i].scatter(
    x = Analysis.y_pred, 
    y = np.log10(100*data['K_Barr']), 
    c = soil_class_sample, 
    cmap= 'Spectral', 
    marker='.', 
    s= markersize,
    label=algorithm,
    zorder = 2)

axs[i].set_xlabel("$\log_{10}(K_{RF}$ [m/d])",fontsize=textsize)
axs[i].set_ylabel("$\log_{10}(K_{Barr}$ [m/d])",fontsize=textsize)
axs[i].grid(True, zorder = 1)
axs[i].set_yticks([2,0,-2,-4,-6])
axs[i].tick_params(axis="both",which="major",labelsize=textsize)
axs[i].plot([k_min-0.01,k_max+0.01],[k_min-0.01,k_max+0.01],':', c="grey")
axs[i].set_xlim([0.98*k_min,1.05*k_max])
axs[i].set_ylim([0.98*k_min,1.05*k_max])
axs[i].text(0.1,0.9,'NSE = {:.2f}'.format(r2_Barr_RF),
            fontsize=textsize, transform=axs[i].transAxes,
            bbox = dict(boxstyle='round', facecolor='white'))
axs[i].text(-0.1,-0.1,'(b)', fontsize=textsize, transform=axs[i].transAxes) 

### ===========================================================================
### left figure
i = 0
scatter = axs[i].scatter(
    x = np.log10(data['Kf']),
    y = np.log10(100*data['K_Barr']), 
    c = soil_class_sample, 
    cmap= 'Spectral', 
    marker='.', 
    s= markersize,
    zorder = 2)

axs[i].set_xlabel("$\log_{10}(K_{obs}$ [m/d])",fontsize=textsize)
axs[i].set_ylabel("$\log_{10}(K_{Barr}$ [m/d])",fontsize=textsize)
axs[i].grid(True, zorder = 1)
axs[i].set_yticks([2,0,-2,-4,-6])
axs[i].tick_params(axis="both",which="major",labelsize=textsize)
axs[i].plot([k_min-0.01,k_max+0.01],[k_min-0.01,k_max+0.01],':', c="grey")
axs[i].set_xlim([0.98*k_min,1.05*k_max])
axs[i].set_ylim([0.98*k_min,1.05*k_max])
axs[i].text(0.1,0.9,'NSE = {:.2f}'.format(r2_Barr_measured),
            fontsize=textsize, transform=axs[i].transAxes,
            bbox = dict(boxstyle='round', facecolor='white'))
axs[i].text(-0.1,-0.1,'(a)', fontsize=textsize, transform=axs[i].transAxes)
# axs[i].text(-0.05,1.05,'Top - All',
#             fontsize=textsize+1, transform=axs[0].transAxes,
#             bbox = dict(boxstyle='round', facecolor='antiquewhite', alpha=0.5))

fig.subplots_adjust(right=.78)
soil_class_names = [r'$Z_{s1}$', r'$Z_{s2}$', r'$Z_{s3}$', r'$Z_{s4}$', r'$Z_{k}$',
                    r'$L_{z1}$', r'$L_{z3}$', r'$K_{s4}$', 
                    r'$K_{z3}$', r'$K_{z2}$', r'$K_{z1}$', r'$K_{s3}$', r'$K_{s2}$', r'$K_{s1}$',
                    'peat']
fig.legend(handles=scatter.legend_elements(num=len(soil_class_names))[0], 
            labels=list(soil_class_names), 
            loc='center right', 
            ncol=1, 
            prop={'size': textsize},#,fontsize=textsize,
            bbox_transform=fig.transFigure,
            title = "Litho- \nclasses",
            )

fig.savefig(fig_results+'.pdf')
