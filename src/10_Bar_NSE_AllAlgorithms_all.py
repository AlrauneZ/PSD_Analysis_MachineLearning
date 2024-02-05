# -*- coding: utf-8 -*-
"""
@author: Alraune Zech
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# from matplotlib.lines import Line2D
plt.close('all')

# verbose = True #False #

soil_type = 'all'
algorithms = ['DT','RF','XG','LR','SVR','ANN']
data_sets = ['testing_set', 'training_set','full_set']

print('\n###############################################')
print(' Visualization of Training and Test Performance')
print('#################################################\n')

### ===========================================================================
### Set file pathes and names
### ===========================================================================

file_AI_performance_r2 = "../results/Performance_r2_{}.csv".format(soil_type)
#file_AI_performance_mse = "../results/Performance_mse_{}.csv".format(soil_type)

### Read in results from algorithm results files
results_r2 = pd.read_csv(file_AI_performance_r2,index_col=0)


# =============================================================================
# Plot specifications 
# =============================================================================
fig, ax = plt.subplots(figsize=(7.5, 2),ncols=3)
textsize = 8

### Define a color dictionary for the bar charts
color_dict = {'DT': 'tab:brown', 
              'RF': 'tab:red', 
              'XG': 'tab:orange', 
              'LR': 'tab:purple', 
              'SVR': 'tab:green',
              'ANN': 'tab:blue'
              }

for j,data_set in enumerate(data_sets):
    argsort = np.argsort(results_r2.loc[data_set].values)[::-1]
    full_bar = ax[j].bar(np.array(algorithms)[argsort], results_r2.loc[data_set].iloc[argsort], color=[color_dict[i] for i in np.array(algorithms)[argsort]],zorder = 2)#, label='Full')
    
    # =============================================================================
    ### Annotate each bar with its value
    for i,bar in enumerate(full_bar):
        val = results_r2.loc[data_set].iloc[argsort[i]]
        ax[j].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                '{:.2f}'.format(val), ha='center', va='bottom', fontsize=textsize,zorder=3)
    
    
    ax[j].set_ylim([0,1.1])
    #plt.grid(True)
    ax[j].tick_params(axis="both",which="major",labelsize=textsize)
    ax[j].set_title("{}".format(data_set),fontsize=textsize)

ax[0].set_ylabel(r"$R^2$",fontsize=textsize)
plt.tight_layout()
# plt.savefig('../results/Fig_Bar_R2_{}.png'.format(soil_type),dpi=300)
# plt.savefig('../results/Fig_Bar_R2_{}.pdf'.format(soil_type))
    
