#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 10:28:14 2024

@author: alraune
"""

import PSD_Analysis
import numpy as np
import matplotlib.pyplot as plt
plt.close('all')

### ===========================================================================
### Set file pathes and names
### ===========================================================================

# file_data = "../data/data_PSD_Kf_por.csv"
file_data = "../data/data_PSD_Kf_por_props.csv"

soil_types = ['all','silt','sand','clay']
colors = ['C0','C3','goldenrod','C2']

# soil_types = ['all','silt','sand','clay']
# colors = ['goldenrod','C2','C0','C3']

# soil_types = ['all','sand','silt','clay']
# colors = ['goldenrod','C0','C2','C3']

# =============================================================================
# Load Data and perform data analysis
# =============================================================================

### initiate analysis
Analysis = PSD_Analysis.PSD_Analysis() 

### read in data through in-class routine
data = Analysis.read_data(file_data)


# =============================================================================
### plot specifications

plt.close('all')
textsize = 8

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(7.5, 4))#, sharey = True,sharex = True)##, layout='constrained')
axs = axs.ravel()

kmin = np.min(Analysis.data['logK'])
kmax = np.max(Analysis.data['logK'])
bins = np.linspace(kmin, kmax,40)
# print(kmin,kmax)

for i in range(0,4):
# for soil_type in ['sand','silt','clay']:
   
    if i ==0:
        data_filtered  = Analysis.data
    else:
        data_filtered =  Analysis.sub_sample_litho(
                                soil_type = soil_types[i],
                                inplace = False)                


    hist = axs[i].hist(data_filtered['logK'],density = True, bins = bins,rwidth=0.85,color=colors[i])#,zorder = 3)

    if i>=2:
        axs[i].set_xlabel("$\log_{10}(K_f$ [m/d])",fontsize=textsize)
    if i in [0,2]:
        axs[i].set_ylabel("relative frequency",fontsize=textsize)
    # # axs[i].set_title('{}'.format(algorithm),fontsize=textsize)
    # axs[i].grid(True, zorder = 1)
    axs[i].tick_params(axis="both",which="major",labelsize=textsize)
    axs[i].set_xlim([kmin-0.05,kmax+0.05])
    axs[i].text(0.03,0.9,'Top-{}'.format(soil_types[i]),
            fontsize=textsize, transform=axs[i].transAxes,
            bbox = dict(boxstyle='round', facecolor='antiquewhite', alpha=0.5))
            
plt.tight_layout()
# # plt.savefig('../results/SI_Fig_Histogram_Kf.png',dpi=300)
plt.savefig('../results/SI_Fig_Histogram_Kf.pdf')
