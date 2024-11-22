#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script reproducing Figure 3 of the manuscripts containing scatter plots 
of all 6 algorithms comparing algorithm estimate of Kf to measured Kf
for the standard feature/target variable combination.

Author: A. Zech
"""

import PSD_2K_ML
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
plt.close('all')

### ===========================================================================
### Key words to specify modus of script:
### ===========================================================================
algs = ["DT", "RF", "XG", "LR", "SVR", "ANN"]
soil_type ='topall' #  'clay' #'silt'#'sand' #'por' # 
feature = 'PSD' #'dX_por' #'dX' #
target = 'Kf' #'por' # 
verbose = True #False #

### ===========================================================================
### Set file pathes and names & plot specifications
### ===========================================================================
  
file_data = "../data/data_PSD_Kf_por_props.csv"
file_fig = '../results/Figures_paper/Fig03_Scatter_Measured_{}'.format(soil_type)

textsize = 8
markersize = 2
cmap = cm.get_cmap('Spectral')
figure_text = ['a','b','c','d','e','f']

# =============================================================================
# Setup Figure, load data, train algorithm and visualize results
# =============================================================================

print("Training and Prediction of all 6 algorithms")
print("###########################################")

fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(7.5, 5.5), 
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
        cmap= cmap, 
        marker='.', 
        s= markersize,
        label=algorithm,
        zorder = 2)

    axs[i].set_xlabel("$\log_{10}(K_{obs}$ [m/d])",fontsize=textsize)
    axs[i].set_ylabel("$\log_{10}(K_{pred}$ [m/d])",fontsize=textsize)
    axs[i].grid(True, zorder = 1)
    axs[i].tick_params(axis="both",which="major",labelsize=textsize)

    ### Plotting the 5th and 95th percentile range of fit
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

soil_class_names = [r'$Z_{s1}$', r'$Z_{s2}$', r'$Z_{s3}$', r'$Z_{s4}$', r'$Z_{k}$',
                    r'$L_{z1}$', r'$L_{z3}$', r'$K_{s4}$', 
                    r'$K_{z3}$', r'$K_{z2}$', r'$K_{z1}$', r'$K_{s3}$', r'$K_{s2}$', r'$K_{s1}$',
                    r'peat']

fig.subplots_adjust(bottom=.16)
fig.legend(handles=scatter.legend_elements(num=len(soil_class_names))[0], 
            labels=list(soil_class_names), 
            loc='lower center', 
            ncol=8, 
            prop={'size': textsize},
            bbox_transform=fig.transFigure,
            )

#plt.savefig(file_fig+'.pdf')
