#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script reproducing Figure of histogram of measured Kf values for the 
Top-integraal data set and subsets: Top-all, Top-sand, Top-silt, Top-Clay 

Author: A. Zech
"""

import PSD_Analysis
import numpy as np
import matplotlib.pyplot as plt
plt.close('all')


### ===========================================================================
### Key words to specify modus of script:
### ===========================================================================
soil_types = ['all','silt','sand','clay']

### ===========================================================================
### Set file pathes and names & plot specifications
### ===========================================================================

file_data = "../data/data_PSD_Kf_por_props.csv"
fig_results = '../results/Figures_SI/SI_Fig_Histogram_Kf'

colors = ['C0','C3','goldenrod','C2']
textsize = 8

# =============================================================================
# Load Data and perform data analysis
# =============================================================================

Analysis = PSD_Analysis.PSD_Analysis() 
data = Analysis.read_data(file_data)

### ===========================================================================
### Prepare plot
### ===========================================================================

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(7.5, 4))#, sharey = True,sharex = True)##, layout='constrained')
axs = axs.ravel()

kmin = np.min(Analysis.data['logK'])
kmax = np.max(Analysis.data['logK'])
bins = np.linspace(kmin, kmax,40)

for i in range(0,len(soil_types)):
  
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
    axs[i].tick_params(axis="both",which="major",labelsize=textsize)
    axs[i].set_xlim([kmin-0.05,kmax+0.05])
    axs[i].text(0.03,0.9,'Top-{}'.format(soil_types[i]),
            fontsize=textsize, transform=axs[i].transAxes,
            bbox = dict(boxstyle='round', facecolor='antiquewhite', alpha=0.5))
            
plt.tight_layout()
plt.savefig(fig_results+'.pdf')
