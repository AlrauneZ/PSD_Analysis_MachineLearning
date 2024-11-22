#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script reproducing sub-plot of Figure 1 of the manuscripts containing 
bar-plots comparing algorithm performance measures NSE/R2 of all 6 ML methods 
for either the test data (20%), training data (80%) or the complete data set (100%) 
for the standard feature/target variable combination 

Author: A. Zech
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

### ===========================================================================
### Key words to specify modus of script:
### ===========================================================================

soil_type = 'topall'
feature = 'PSD' #'dX_por' #'dX' #
target = 'Kf' #'por' # #
data_set = 'training_set' # 'full_set' #'testing_set' #, 
algorithms = ['DT','RF','XG','LR','SVR','ANN']

### ===========================================================================
### Set file pathes and names
### ===========================================================================

file_AI_performance_r2 = "../results/ML_performance/Performance_{}_{}_{}_r2.csv"#.format(feature,target,soil_type)
fig_results = '../results/Figures_paper/Fig01_Bar_NSE_{}_{}_{}_{}'.format(feature,target,soil_type,data_set)

textsize = 10 # for presentations
color_dict = {'DT': 'tab:brown', 
              'RF': 'tab:green', 
              'XG': 'tab:blue', 
              'LR': 'tab:purple', 
              'SVR': 'tab:orange',
              'ANN': 'tab:red'
              }

### ===========================================================================
### Read in data and prepape plot
### ===========================================================================

print('\n###############################################')
print(' Visualization of Training and Test Performance')
print('#################################################\n')

results_r2 = pd.read_csv(file_AI_performance_r2.format(feature,target,soil_type),index_col=0)
argsort = np.argsort(results_r2.loc[data_set].values)[::-1]

fig, ax = plt.subplots(figsize=(3.25, 2.25)) # for presentations/poster
full_bar = ax.bar(np.array(algorithms)[argsort], 
                  results_r2.loc[data_set].iloc[argsort], 
                  color=[color_dict[i] for i in np.array(algorithms)[argsort]],
                  zorder = 2)

## Annotate each bar with its value
for i,bar in enumerate(full_bar):
    val = results_r2.loc[data_set].iloc[argsort[i]]
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
            '{:.2f}'.format(val), ha='center', va='bottom', fontsize=textsize,zorder=3)

plt.ylim([0,1.1])
plt.yticks([0,0.2,0.4,0.6,0.8,1])
plt.ylabel(r"$R^2$",fontsize=textsize)
plt.tick_params(axis="both",which="major",labelsize=textsize)
plt.title("Performance for {}".format(data_set),fontsize=textsize)

plt.tight_layout()
#plt.savefig(fig_results+'.pdf')
