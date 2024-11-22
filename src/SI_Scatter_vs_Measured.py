#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script reproducing Figure in the supporting information containing scatter 
plots comparing Kf estimates of all 6 algorithms to measured Kf for data-subset
(sand, silt, clay) as well as other feature/target-variable combinations 
linked to Top-por data set.

Author: A. Zech
"""

import PSD_2K_ML
import numpy as np
import matplotlib.pyplot as plt
plt.close('all')

### ===========================================================================
### Set file pathes and Plot specifications 
### ===========================================================================

algs = ["DT", "RF", "XG", "LR", "SVR", "ANN"]
soil_type = 'clay' # 'por' #'sand' #'silt'#
feature = 'PSD' # 'dX_por'#
target =  'Kf'# 
verbose = True

### ===========================================================================
### Set file pathes and names & plot specifications
### ===========================================================================

file_data = "../data/data_PSD_Kf_por_props.csv"
if feature == 'PSD' and target == 'Kf': 
    file_fig = '../results/Figures_SI/SI_Fig_Scatter_Measured_{}'.format(soil_type)
    text = 'Top - {}'.format(soil_type)
else:
    file_fig = '../results/Figures_SI/SI_Fig_Scatter_Measured_{}_{}'.format(feature,target)
    text = '{} --> {}'.format(feature,target)

textsize = 8
markersize = 2
figure_text = ['a','b','c','d','e','f']

lithoclasses = dict(
    por  = [r'$Z_{s1}$', r'$Z_{s2}$', r'$Z_{s3}$', r'$Z_{s4}$', r'$Z_{k}$'],
    sand = [r'$Z_{s1}$', r'$Z_{s2}$', r'$Z_{s3}$', r'$Z_{s4}$', r'$Z_{k}$'],
    silt = [r'$L_{z1}$', r'$L_{z3}$', r'$K_{s4}$'], 
    clay = [r'$K_{z3}$', r'$K_{z2}$', r'$K_{z1}$', r'$K_{s3}$', r'$K_{s2}$', r'$K_{s1}$', r'peat'],
    )

### ===========================================================================
### Speficy Algorithm and set target and feature variables, run training
### ===========================================================================

print("Training and Prediction of all 6 algorithms")
print("###########################################")

fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(7.5, 5.25), 
                        sharex = True, sharey = True)
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
    Analysis.set_feature_variables()
    Analysis.data_split()    
    Analysis.training(verbose = verbose)
    Analysis.prediction(x_pred = 'full_set',verbose = verbose)
    bc5,pc5 = Analysis.quantiles_4_plot(bins=10,nth=5)
    bc95,pc95 = Analysis.quantiles_4_plot(bins=10,nth=95)

    scatter = axs[i].scatter(
        x = Analysis.y_obs,
        y = Analysis.y_pred, 
        c = soil_class_sample, 
        cmap= 'Spectral',
        vmin = 0, 
        vmax = 14,
        marker='.', 
        s= markersize,
        label=algorithm,
        zorder = 2)

    axs[i].set_xlabel("$\log_{10}(K_{obs}$ [m/d])",fontsize=textsize)
    axs[i].set_ylabel("$\log_{10}(K_{pred}$ [m/d])",fontsize=textsize)
    axs[i].grid(True, zorder = 1)
    axs[i].tick_params(axis="both",which="major",labelsize=textsize)

    axs[i].plot(bc5,pc5,'--',c = 'k',zorder=3)
    axs[i].plot(bc95,pc95,'--', c = 'k', zorder = 3)
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
            fontsize=textsize+1, transform=axs[0].transAxes,
            bbox = dict(boxstyle='round', facecolor='antiquewhite', alpha=0.5))

fig.subplots_adjust(bottom=.16)
fig.legend(handles=scatter.legend_elements()[0], 
            labels = lithoclasses[soil_type], 
            loc='lower center', 
            ncol=7, 
            prop={'size': textsize},
            bbox_transform=fig.transFigure,
            )

#plt.savefig(file_fig+'.pdf')
