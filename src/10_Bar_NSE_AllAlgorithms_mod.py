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
data_set = 'testing_set' #, 'training_set','full_set']
algorithms = ['DT','RF','XG','LR','SVR','ANN']

print('\n###############################################')
print(' Visualization of Training and Test Performance')
print('#################################################\n')

### ===========================================================================
### Set file pathes and names
### ===========================================================================

file_AI_performance_r2 = "../results/Performance_{}_{}_{}_r2.csv" .format(feature,target,soil_type)
fig_results = '../results/Fig_Bar_R2_{}_{}_{}_{}'.format(feature,target,soil_type,data_set)

# =============================================================================
# Read in data
# =============================================================================

results_r2 = pd.read_csv(file_AI_performance_r2.format(feature,target,soil_type),index_col=0)

# # =============================================================================
# # Plot specifications 
# # =============================================================================
textsize = 10 # for presentations
### Define a color dictionary for the bar charts
color_dict = {'DT': 'tab:brown', 
              'RF': 'tab:green', 
              'XG': 'tab:blue', 
              'LR': 'tab:purple', 
              'SVR': 'tab:orange',
              'ANN': 'tab:red'
              }

fig, ax = plt.subplots(figsize=(3.25, 2.25)) # for presentations/poster

argsort = np.argsort(results_r2.loc[data_set].values)[::-1]
full_bar = ax.bar(np.array(algorithms)[argsort], 
                  results_r2.loc[data_set].iloc[argsort], 
                  color=[color_dict[i] for i in np.array(algorithms)[argsort]],
                  zorder = 2)#, label='Full')

##=============================================================================
## Annotate each bar with its value
for i,bar in enumerate(full_bar):
    val = results_r2.loc[data_set].iloc[argsort[i]]
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
            '{:.2f}'.format(val), ha='center', va='bottom', fontsize=textsize,zorder=3)

plt.ylim([0,1.1])
plt.yticks([0,0.2,0.4,0.6,0.8,1])
#plt.grid(True)
plt.ylabel(r"$R^2$",fontsize=textsize)
plt.tick_params(axis="both",which="major",labelsize=textsize)
plt.title("Performance for {}".format(data_set),fontsize=textsize)
# plt.title("$R^2$ for test data set (20%)",fontsize=textsize)
# plt.title("$R^2$ for full data set (100%)",fontsize=textsize)
plt.tight_layout()
plt.savefig(fig_results+'.png',dpi=300)
# plt.savefig(fig_results+'.pdf')
