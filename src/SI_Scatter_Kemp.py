#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script reproducing Figure of the supporting information containing scatter 
plots comparing Kf estimates of empirical methods to measured Kf
for standard feature/target variable combination.

Author: A. Zech
"""

import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

import numpy as np
import pandas as pd
plt.close('all')

### ===========================================================================
### Set file pathes and Plot specifications 
### ===========================================================================

file_data_Kemp = "../data/data_PSD_Kf_por_props_Kemp.csv"
file_fig = '../results/Figures_SI/SI_Fig_Scatter_Kemp'

textsize = 8
lw = 2
markersize = 10 #2
figure_text = ['a','b','c','d']

### =============================================================================
### Load Data and extract information
### =============================================================================

data = pd.read_csv(file_data_Kemp,index_col = 0)

### column names provide empirical methods
Kemp_cols = data.columns[[x.startswith("K_") for x in data.columns]] # condition picking methods applicable to all samples
Kemp_values = pd.DataFrame(data, columns=Kemp_cols)

### samples with number coded soil class name for color specification in plots
soil_class_sample = data.soil_class.astype('category').cat.codes
soil_class_names = np.unique(data.soil_class)

soil_class_names_sort = ['zs1','zs2','zs3','zs4','zk','kz3','kz2','kz1','lz1','lz3','ks4', 'ks3','ks2', 'ks1','p' ]
map_list = [soil_class_names_sort.index(o) for o in soil_class_names]

soil_class_sample = [map_list[i] for i in soil_class_sample]#objects = [object_map[id] for id in ids]
soil_class_names = soil_class_names_sort

k_min,k_max = np.min(np.log10(data['Kf'])),np.max(np.log10(data['Kf']))

### ===========================================================================
### Prepare plot 
### ===========================================================================

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(7.5, 7.5), 
                        sharex = True, sharey = True)
axs = axs.ravel()

for i in range(4):    

    print("Scatter plot for empirical method: {}".format(Kemp_values.columns[i+1][2:]))
    logKf_emp = np.log10(864*Kemp_values.iloc[:,i+1]) 
    r2_Kemp_measured = r2_score(np.log10(data['Kf']),logKf_emp)  

    scatter = axs[i].scatter(
        x = np.log10(data['Kf']),
        y = np.log10(100*Kemp_values.iloc[:,i+1]), 
        c = soil_class_sample, 
        cmap= 'Spectral', 
        marker='.', 
        s= markersize,
        zorder = 2)

    axs[i].set_xlabel("$\log_{10}(K_{obs}$ [m/d])",fontsize=textsize)
    axs[i].set_ylabel("$\log_{10}(K_{emp}$ [m/d])",fontsize=textsize)
    axs[i].grid(True, zorder = 1)
    axs[i].tick_params(axis="both",which="major",labelsize=textsize)
    axs[i].plot([k_min-0.01,k_max+0.01],[k_min-0.01,k_max+0.01],':', c="grey")
    axs[i].set_xlim([-7,1.05*k_max+0.01])
    axs[i].set_ylim([-7,1.05*k_max+0.01])

    axs[i].text(0.1,0.9,'({}) {}'.format(figure_text[i],Kemp_values.columns[i+1][2:]),
                fontsize=textsize, transform=axs[i].transAxes,
                bbox = dict(boxstyle='round', facecolor='white'))

    axs[i].text(0.7,0.09,'NSE = {:.2f}'.format(r2_Kemp_measured),
                fontsize=textsize, transform=axs[i].transAxes,
                bbox = dict(boxstyle='round', facecolor='white'))

fig.subplots_adjust(bottom=.12)
fig.legend(handles=scatter.legend_elements(num=len(soil_class_names))[0], 
            labels=list(soil_class_names), 
            loc='lower center', 
            ncol=8, 
            prop={'size': textsize},
            bbox_transform=fig.transFigure,
            )

plt.savefig(file_fig+'.pdf')

