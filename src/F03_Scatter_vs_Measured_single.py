#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script reproducing individual subplot of Figure 3 of the manuscripts 
containing scatter plots of ML algorithm comparing algorithm estimate of Kf 
to measured Kf for the standard feature/target variable combination.

Author: A. Zech
"""


import PSD_2K_ML
import matplotlib.pyplot as plt
plt.close('all')

### ===========================================================================
### Key words to specify modus of script:
### ===========================================================================

algorithm ='LR' # 'RF' #'ANN' #'DT' #'SVR' # 'RF'
soil_type = 'topall' #'silt'#'sand' # 'clay' # 'por' #
feature = 'PSD' #'dX_por' #'dX' #
target = 'Kf' #'por' # 
verbose = True #False #

### ===========================================================================
### Set file pathes and names & plot specifications
### ===========================================================================
file_data = "../data/data_PSD_Kf_por_props.csv"
file_fig = '../results/Figures_paper/Fig03_Scatter_{}'.format(algorithm)
textsize = 8

print("Training and Prediction of {}".format(algorithm))
print("###############################")

# =============================================================================
# Load Data and perform Algorithm fitting to produce predictions
# =============================================================================

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
Analysis.prediction(x_pred = 'full_set', verbose = verbose)
bc5,pc5 = Analysis.quantiles_4_plot(bins=10,nth=5)
bc95,pc95 = Analysis.quantiles_4_plot(bins=10,nth=95)


### ==========================================================================
### Plotting
### ==========================================================================

fig, ax = plt.subplots(figsize=(0.33*7.5,2.5))
soil_class_names, soil_class_sample = Analysis.soil_class_specification(sort = True)

### scatter plot of predicted against observed K-values
scatter = ax.scatter(
    x = Analysis.y_obs,
    y = Analysis.y_pred, 
    c = soil_class_sample, 
    cmap= 'Spectral', 
    marker='.', 
    s= 10,
    zorder = 2)

### Plotting the 5th and 95th percentile range of fit
ax.plot(bc5,pc5,'--',c = 'k',zorder=3)
ax.plot(bc95,pc95,'--', c = 'k', zorder = 3)
ax.plot(Analysis.y_test,Analysis.y_test, c="grey", linestyle = "dotted")
ax.set_xlabel("$\log_{10}(K_{obs}$ [m/d])",fontsize=textsize)
ax.set_ylabel("$\log_{10}(K_{ML}$ [m/d])",fontsize=textsize)
ax.set_title('Linear Regression',fontsize=textsize)
ax.grid(True, zorder = 1)

ax.set_xlim([-6.8,2.2])
ax.set_ylim([-6.8,2.2])
ax.set_xticks([-6,-4,-2,0,2])
ax.set_yticks([-6,-4,-2,0,2])
ax.tick_params(axis="both",which="major",labelsize=textsize)

fig.legend(handles=scatter.legend_elements(num=len(soil_class_names))[0], 
            labels=list(soil_class_names), 
            loc='upper right', 
            ncol=1, 
            prop={'size': textsize},
            bbox_transform=fig.transFigure,
            )

plt.tight_layout()
plt.savefig(file_fig+'4.pdf')
