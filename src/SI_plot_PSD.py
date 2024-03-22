#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 09:22:49 2023

@author: alraune
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
plt.close('all')
print('\n######################################')
print(' Plot selected PSD in cumulative form')

# =============================================================================
# Load Data 
# =============================================================================

file_AI_data = "../data/AI_data.csv"

AI_data = pd.read_csv(file_AI_data,index_col=0)
sieve_diam = [.00001,0.0001,0.0002,0.0005,.001,.002,.004,.008,.016,.025,.035,.05,.063,.075,.088,.105,.125,.150,.177,.21,.25,.3,.354,.42,.5,.6,.707,.85,1.,1.190,1.41,1.68,2] # in mm

### specify max-sieve diameters and compute cumulative PSD from data 
sieve_diam = np.array(sieve_diam[1:])
AI_cfd = AI_data.iloc[:,:len(sieve_diam)].cumsum(axis = 1)

### identify a few extreme cdfs
i1 = np.argmax(AI_cfd['F105-125'])
i2 = np.argmin(AI_cfd['F105-125'])
i3 = np.argmax(AI_cfd['F850-1000'])
i4 = np.argmin(AI_cfd['F850-1000'])
i5 = np.argmax(AI_cfd['F4-8'])
i6 = np.argmin(AI_cfd['F4-8'])

# =============================================================================
# Plot specifications 
# =============================================================================
# fig = plt.figure(figsize=(5, 2.75))
fig = plt.figure(figsize=(3.75, 2.25))
textsize = 8
lw = 2

for i in range(50,700,100):
    plt.plot(sieve_diam,AI_cfd.iloc[i,:],lw = lw)

plt.plot(sieve_diam,AI_cfd.iloc[i2,:],lw = lw)
plt.plot(sieve_diam,AI_cfd.iloc[i1,:],lw = lw)

for i in range(0,10,1):
    plt.plot(sieve_diam,AI_cfd.iloc[i,:],lw = lw)

plt.plot(sieve_diam,AI_cfd.iloc[i3,:],lw = lw)
plt.plot(sieve_diam,AI_cfd.iloc[i4,:],lw = lw)
plt.plot(sieve_diam,AI_cfd.iloc[i5,:],lw = lw)
plt.plot(sieve_diam,AI_cfd.iloc[i6,:],lw = lw)

plt.xlim([2e-4,2])
plt.xlabel('Sieve diameter [mm]',fontsize=textsize)
plt.ylabel('PSD [%]',fontsize=textsize)
#plt.ylabel('Cumulative frequency [%]',fontsize=textsize)
plt.xscale('log')
plt.grid(True)
plt.tick_params(axis="both",which="major",labelsize=textsize)

plt.tight_layout()
# plt.savefig('../results/SI_Fig_PSDs.png',dpi = 300)
plt.savefig('../results/SI_Fig_PSDs.pdf')
