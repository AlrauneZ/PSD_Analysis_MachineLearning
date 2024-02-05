#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zech0001
"""

import numpy as np
#import pandas as pd
import PSD_Analysis

write_to_file = True

print('\n#################################')
print('   Input data Analysis')

# =============================================================================
# Load Data and perform Algorithm fitting to produce predictions
# =============================================================================


file_AI_data = "../data/AI_data.csv"
file_psd_props = "../data/PSD_properties.csv"
file_data_stats = "../results/Data_{}_stats.csv"
     
### initiate analysis
Analysis = PSD_Analysis.PSD_Analysis() 

### read in data through in-class routine
data = Analysis.read_data(file_AI_data)
psd = Analysis.psd

### read in data here and set psd and sieve-diameter values separately:
# sieve_diam = [.00001,0.0001,0.0002,0.0005,.001,.002,.004,.008,.016,.025,.035,.05,.063,.075,.088,.105,.125,.150,.177,.21,.25,.3,.354,.42,.5,.6,.707,.85,1.,1.190,1.41,1.68,2]
# data = pd.read_csv(file_AI_data)
# psd = pd.DataFrame(data, columns=data.columns[[x.startswith("F") for x in data.columns]])
# Analysis.set_psd(psd,sieve_diam)

### perform data analysis on psd
Analysis.calc_psd_diameters()
Analysis.calc_psd_soil_class()
Analysis.calc_NEN5104_classification()
# #Analysis.calc_psd_folk()

stats2save = ['d10','d50','perc_lutum','perc_silt','perc_sand']

stats_data = Analysis.psd_properties[stats2save].copy()
stats_data['logK'] = np.log10(Analysis.data.Kf)
stats = stats_data.describe()

if write_to_file:
    stats.to_csv(file_data_stats.format('all'))
print(stats)

for soil_type in ['sand','silt','clay']:
    print('\n################################# \n')
    Analysis.sub_sample_soil_type(soil_type)

    stats_data = Analysis.psd_properties_filtered[stats2save].copy()
    stats_data['logK'] = np.log10(Analysis.data_filtered.Kf)
    # test = np.log10(Analysis.data.Kf.values)
    stats = stats_data.describe()
    if write_to_file:
        stats.to_csv(file_data_stats.format(soil_type))
    print(stats)

