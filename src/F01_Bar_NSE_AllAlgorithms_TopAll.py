#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script reproducing Figure 1 of the manuscripts containing bar-plots comparing
algorithm performance measures NSE/R2 of all 6 ML methods for the test 
data (20%), training data (80%) and the complete data set (100%) for the 
standard feature/target variable combination 

Author: A. Zech
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
plt.close('all')

### ===========================================================================
### Key words to specify modus of script:
### ===========================================================================

soil_type = 'topall' #'clay' #'silt'#'sand' #  'por' #
feature = 'PSD' #'dX_por' #'dX' #
target = 'Kf' #'por' # #

algorithms = ['DT','RF','XG','LR','SVR','ANN']
data_sets =['full_set','training_set','testing_set' ] 
titles = ['full set (100%)','training set (80%)','testing set (20%)' ]

print('\n###############################################')
print(' Visualization of Training and Test Performance')
print('#################################################\n')

### ===========================================================================
### Set file pathes and names & Plot specifications 
### ===========================================================================

file_AI_performance_r2 = "../results/ML_performance/Performance_{}_{}_{}_r2.csv".format(feature,target,soil_type)
fig_results = '../results/Figures_paper/Fig01_Bar_NSE_{}_{}_{}'.format(feature,target,soil_type)

textsize = 8 #  12 #
### Define a color dictionary for the bar charts
color_dict = {'DT': 'tab:brown', 
              'RF': 'tab:green', 
              'XG': 'tab:blue', 
              'LR': 'tab:purple', 
              'SVR': 'tab:orange',
              'ANN': 'tab:red'
              }

# =============================================================================
# Read in data and plot results
# =============================================================================

results_r2 = pd.read_csv(file_AI_performance_r2,index_col=0)
fig, ax = plt.subplots(figsize=(7.5, 2),ncols=len(data_sets))

for j,data_set in enumerate(data_sets):
    argsort = np.argsort(results_r2.loc[data_set].values)[::-1]
    full_bar = ax[j].bar(np.array(algorithms)[argsort], 
                         results_r2.loc[data_set].iloc[argsort], 
                         width = 0.7,
                         color=[color_dict[i] for i in np.array(algorithms)[argsort]],
                         zorder = 2)#
    
    ### Annotate each bar with its value
    for i,bar in enumerate(full_bar):
        val = results_r2.loc[data_set].iloc[argsort[i]]
        ax[j].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                '{:.2f}'.format(val), ha='center', va='bottom', fontsize=textsize,zorder=3)    
    
    ax[j].set_ylim([0,1.1])
    ax[j].tick_params(axis="both",which="major",labelsize=textsize)
    ax[j].set_title("{}".format(titles[j]),fontsize=textsize)

# ax[0].text(-0.05,1.1,'{}'.format(soil_type),
#             fontsize=textsize, transform=ax[0].transAxes,
#             bbox = dict(boxstyle='round', facecolor='antiquewhite', alpha=0.5))

ax[0].set_ylabel(r"$NSE$",fontsize=textsize)
plt.tight_layout()
fig.savefig(fig_results+'.pdf')