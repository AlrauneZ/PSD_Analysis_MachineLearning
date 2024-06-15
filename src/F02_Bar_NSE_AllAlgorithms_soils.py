#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script reproducing Figure 2 of the manuscripts containing bar-plots comparing
algorithm performance measures NSE/R2 of all 6 ML methods for the three
data sub-sets based on main lithology: Top-Sand, Top-Silt, Top-Clay 
Evaluation measures NSE/R2 is taken for performance on all samples 
(i.e. test data + training data) and the standard feature/target variable 
combination.

Author: A. Zech
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
plt.close('all')

### ===========================================================================
### Key words to specify modus of script:
### ===========================================================================

soil_types = ['sand','silt','clay']
feature = 'PSD' #'dX_por' #'dX' #
target = 'Kf' #'por' # #
algorithms = ['DT','RF','XG','LR','SVR','ANN']
data_set = 'full_set'
# verbose = True #False #

print('\n###############################################')
print(' Visualization of Training and Test Performance')
print('#################################################\n')

### ===========================================================================
### Set file pathes and names & Plot specifications 
### ===========================================================================

file_AI_performance_r2 = "../results/ML_performance/Performance_{}_{}_{}_r2.csv"#.format(feature,target,soil_type)
fig_results = '../results/Figures_paper/Fig02_Bar_NSE_{}_{}_soiltypes2'.format(feature,target)

textsize = 8 #  12 #
color_dict = {'DT': 'tab:brown', 
              'RF': 'tab:green', 
              'XG': 'tab:blue', 
              'LR': 'tab:purple', 
              'SVR': 'tab:orange',
              'ANN': 'tab:red'
              }

# =============================================================================
# Read in data
# =============================================================================

data  = []
for soil_type in soil_types:
    results_r2 = pd.read_csv(
        file_AI_performance_r2.format(feature,target,soil_type),
        index_col=0)
    data.append(results_r2)
    
### ===========================================================================
### Prepare plot
### ===========================================================================

fig, ax = plt.subplots(figsize=(7.5, 2),ncols=len(data))
for j,results_r2 in enumerate(data):
    argsort = np.argsort(results_r2.loc[data_set].values)[::-1]
    full_bar = ax[j].bar(np.array(algorithms)[argsort], 
                         results_r2.loc[data_set].iloc[argsort], 
                         width = 0.7,
                         color=[color_dict[i] for i in np.array(algorithms)[argsort]],
                         zorder = 2)
    
    # =============================================================================
    ### Annotate each bar with its value
    for i,bar in enumerate(full_bar):
        val = results_r2.loc[data_set].iloc[argsort[i]]
        ax[j].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                '{:.2f}'.format(val), ha='center', va='bottom', fontsize=textsize,zorder=3)   
    
    ax[j].set_ylim([0,1.05])
    ax[j].tick_params(axis="both",which="major",labelsize=textsize)
    ax[j].set_title("{}".format(soil_types[j]),fontsize=textsize)

ax[0].set_ylabel(r"$NSE$",fontsize=textsize)
plt.tight_layout()
fig.savefig(fig_results+'.pdf')