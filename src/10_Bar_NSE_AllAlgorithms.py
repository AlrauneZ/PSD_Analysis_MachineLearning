# -*- coding: utf-8 -*-
"""
@author: Alraune Zech
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.close('all')

soil_type = 'all'
feature = 'PSD' #'dX_por' #'dX' #
target = 'Kf' #'por' # #

soil_type = 'all'
algorithms = ['DT','RF','XG','LR','SVR','ANN']
data_sets =['full_set','training_set','testing_set' ] #['full_set'] #
# verbose = True #False #

print('\n###############################################')
print(' Visualization of Training and Test Performance')
print('#################################################\n')

### ===========================================================================
### Set file pathes and names
### ===========================================================================

file_AI_performance_r2 = "../results/Performance_{}_{}_{}_r2.csv".format(feature,target,soil_type)
fig_results = '../results/Fig_Bar_R2_{}_{}_{}'.format(feature,target,soil_type)

# =============================================================================
# Read in data
# =============================================================================
results_r2 = pd.read_csv(file_AI_performance_r2.format(feature,target,soil_type),index_col=0)

# # =============================================================================
# # Plot specifications 
# # =============================================================================
textsize = 8 #  12 #
# # textsize = 8

### Define a color dictionary for the bar charts
color_dict = {'DT': 'tab:brown', 
              'RF': 'tab:green', 
              'XG': 'tab:blue', 
              'LR': 'tab:purple', 
              'SVR': 'tab:orange',
              'ANN': 'tab:red'
              }


# fig, ax = plt.subplots(figsize=(3.25, 2.25),ncols=len(data_sets)) # for presentations/poster
fig, ax = plt.subplots(figsize=(7.5, 2),ncols=len(data_sets))

for j,data_set in enumerate(data_sets):
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
        ax[j].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                '{:.2f}'.format(val), ha='center', va='bottom', fontsize=textsize,zorder=3)
    
    
    ax[j].set_ylim([0,1.1])
    #plt.grid(True)
    ax[j].tick_params(axis="both",which="major",labelsize=textsize)
    ax[j].set_title("{}".format(data_set),fontsize=textsize)

ax[0].set_ylabel(r"$R^2$",fontsize=textsize)
plt.tight_layout()
# plt.savefig(fig_results+'.png',dpi=300)
fig.savefig(fig_results+'.pdf')