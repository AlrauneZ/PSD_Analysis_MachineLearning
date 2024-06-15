#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cript reproducing Figure of the supporting information on particle size
distributions (PSDs) for a selection of samples including extreme values.
s
Author: A. Zech
"""

import PSD_Analysis
import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
plt.close('all')

print('\n######################################')
print(' Plot selected PSD in cumulative form')

### ===========================================================================
### Load Data & Plot specifications  
###============================================================================

# file_data = "../data/data_PSD_Kf_por_props.csv"
file_data = "../data/data_PSD_Kf_por.csv"
fig_results = '../results/Figures_SI/SI_Fig_PSDs'

textsize = 8
lw = 2

### ===========================================================================
### Prepare data for plot  
###============================================================================
### initiate analysis
Analysis = PSD_Analysis.PSD_Analysis() 
Analysis.read_data(file_data)

psd = Analysis.psd
psd_cum = np.cumsum(psd,axis = 1)
sieve_diam = Analysis.sieve_diam[1:]

### identify a few extreme cdfs
i1 = np.argmax(psd_cum['F105-125'])
i2 = np.argmin(psd_cum['F105-125'])
i3 = np.argmax(psd_cum['F850-1000'])
i4 = np.argmin(psd_cum['F850-1000'])
i5 = np.argmax(psd_cum['F4-8'])
i6 = np.argmin(psd_cum['F4-8'])

# ### ===========================================================================
# ### Prepare plot
# ### ===========================================================================

fig = plt.figure(figsize=(3.75, 2.25))
for i in range(50,700,100):
    plt.plot(sieve_diam,psd_cum.iloc[i,:],lw = lw)

plt.plot(sieve_diam,psd_cum.iloc[i2,:],lw = lw)
plt.plot(sieve_diam,psd_cum.iloc[i1,:],lw = lw)

for i in range(0,10,1):
    plt.plot(sieve_diam,psd_cum.iloc[i,:],lw = lw)

plt.plot(sieve_diam,psd_cum.iloc[i3,:],lw = lw)
plt.plot(sieve_diam,psd_cum.iloc[i4,:],lw = lw)
plt.plot(sieve_diam,psd_cum.iloc[i5,:],lw = lw)
plt.plot(sieve_diam,psd_cum.iloc[i6,:],lw = lw)

plt.xlim([2e-4,2])
plt.xlabel('Sieve diameter [mm]',fontsize=textsize)
plt.ylabel('PSD [%]',fontsize=textsize)
plt.xscale('log')
plt.grid(True)
plt.tick_params(axis="both",which="major",labelsize=textsize)

plt.tight_layout()
plt.savefig(fig_results+'.pdf')
