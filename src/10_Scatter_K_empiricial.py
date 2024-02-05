#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 09:15:40 2023

@author: alraune
"""

import matplotlib.pyplot as plt
#import numpy as np
import pandas as pd
plt.close('all')

# =============================================================================
# Load Data and perform Algorithm fitting to produce predictions
# =============================================================================

file_Kemp_app = "../results/Kemp_app.csv"
file_AI_data = "../data/AI_data.csv"

Kemp_data = pd.read_csv(file_Kemp_app,index_col = 0)
AI_data = pd.read_csv(file_AI_data)

# =============================================================================
### extract plot specific information:
# =============================================================================

### column names provide empirical methods
Kemp_methods = Kemp_data.columns.to_list()
### samples with number coded soil class name for color specification in plots
soil_class_sample = AI_data.soil_class.astype('category').cat.codes

# =============================================================================
# Plot specifications 
# =============================================================================
fig = plt.figure(figsize=(3.75, 3))
textsize = 8
lw = 2

### select empirical method
i=0
print("Scatter plot for empirical method: {}".format(Kemp_methods[i]))
### scatter plot of predicted against observed K-values
scatter = plt.scatter(
    x = AI_data['Kf'],
    y = 100*Kemp_data.iloc[:,i], 
    c = soil_class_sample, 
    cmap= 'Spectral', 
    marker='.', 
    s= 10,
    zorder = 2)

### one-by-one line of 
plt.plot([1e-8,2e2],[1e-8,2e2], c="grey", linestyle = "dotted")

plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"$K_{obs}$",fontsize=textsize)
plt.ylabel(r"$K_{emp}$)",fontsize=textsize)
# plt.xlabel("$\log_{10}(K_{obs}$)",fontsize=textsize)
# plt.ylabel("$\log_{10}(K_{pred}$)",fontsize=textsize)
plt.title('Empirical method: {}'.format(Kemp_methods[i][2:]),fontsize=textsize)
plt.grid(True, zorder = 1)
plt.tick_params(axis="both",which="major",labelsize=textsize)
plt.xlim([2e-8,2e2])
plt.ylim([2e-8,2e2])
#plt.axis("equal")

plt.tight_layout()
#plt.savefig('../results/Fig_Scatter_Kemp_{}.png'.format(algorithm),dpi=300)
#plt.savefig('../results/Fig_Scatter_Kemp_{}.pdf'.format(algorithm))

