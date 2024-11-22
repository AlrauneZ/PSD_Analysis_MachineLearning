#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script reproducing Figure 7 of the manuscripts containing bar-plots comparing
algorithm performances of all 6 ML methods for different feature/target variable
combinations for the data(sub)set "Top-por" (containing also porosity measurement)
Evaluation measures NSE/R2 is taken for performance on all samples 
(i.e. test data + training data)

Author: A. Zech
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
plt.close('all')

### ===========================================================================
### Key words to specify modus of script:
### ===========================================================================

algorithms = ['DT','RF','XG','LR','SVR','ANN']
data_set = 'full_set'
# verbose = True #False #

print('\n###############################################')
print(' Visualization of Training and Test Performance')
print('#################################################\n')

### ===========================================================================
### Set file pathes and names
### ===========================================================================

file_AI_performance_r2 = "../results/ML_performance/Performance_{}_{}_{}_r2.csv"#.format(feature,target,soil_type)
fig_results = '../results/Figures_paper/Fig07_Bar_NSE_features'

### ===========================================================================
### Plot and setting specifications 
### ===========================================================================
textsize = 8 #  12 #

### Define a color dictionary for the bar charts
color_dict = {'DT': 'tab:brown', 
              'RF': 'tab:green', 
              'XG': 'tab:blue', 
              'LR': 'tab:purple', 
              'SVR': 'tab:orange',
              'ANN': 'tab:red'
              }

sets = dict(
    # s1 = dict(
    #     soil_type ='topall', 
    #     feature = 'dX', 
    #     target = 'Kf',
    #     text = r"$d_X$ $\rightarrow$ $K_f$  (Top-all)"
    #     ),
    s2 = dict(
        soil_type ='por', 
        feature = 'dX_por', 
        target = 'Kf', 
        text = r"$d_X$ & por $\rightarrow$ $K_f$  (Top-por)"
        ),
    s3 = dict(
        soil_type ='por',
        feature = 'PSD' ,
        target = 'Kf',  
        text = r"PSD $\rightarrow$ $K_f$  (Top-por)"
        ),
    s4 = dict(
        soil_type ='por',
        feature = 'PSD',
        target = 'por',  
        text = r"PSD $\rightarrow$ por  (Top-por)"
        ),
    )

### ===========================================================================
### Read in data
### ===========================================================================

data  = []
title_text = []
for i in sets.keys():
    results_r2 = pd.read_csv(file_AI_performance_r2.format(sets[i]['feature'],sets[i]['target'],sets[i]['soil_type']),index_col=0)
    data.append(results_r2)
    title_text.append(sets[i]['text'])

### ===========================================================================
### Create Plot 
### ===========================================================================
fig, ax = plt.subplots(figsize=(7.5, 2),ncols=len(data), sharey = True)

for j,results_r2 in enumerate(data):
    argsort = np.argsort(results_r2.loc[data_set].values)[::-1]
    full_bar = ax[j].bar(np.array(algorithms)[argsort], 
                         results_r2.loc[data_set].iloc[argsort], 
                         width = 0.7,
                         color=[color_dict[i] for i in np.array(algorithms)[argsort]],
                         zorder = 2)#, label='Full')
    
    # =============================================================================
    ### Annotate each bar with its value
    for i,bar in enumerate(full_bar):
        val = results_r2.loc[data_set].iloc[argsort[i]]
        ax[j].text(bar.get_x() + bar.get_width() / 2, max(0,bar.get_height()),
                '{:.2f}'.format(val), ha='center', va='bottom', fontsize=textsize,zorder=3)
    
    
    ax[j].set_ylim([0,1.08])
    ax[j].tick_params(axis="both",which="major",labelsize=textsize)
    ax[j].set_title(title_text[j],fontsize=textsize)
 
ax[0].set_ylabel(r"$NSE$",fontsize=textsize)
plt.tight_layout()

#fig.savefig(fig_results+'2.pdf')