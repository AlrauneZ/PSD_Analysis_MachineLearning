#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zech0001
"""

import numpy as np
import PSD_Analysis
#import pandas as pd

write_to_file = False #True

print('###########################\n   Input data Analysis\n###########################')

# =============================================================================
# Load Data and perform Algorithm fitting to produce predictions
# =============================================================================

file_application_data = "../data/data_PSD_Kf.csv"
file_application_data_extended = "../data/data_PSD_Kf_props.csv"
#file_application_data = "../data/AI_data.csv"
file_psd_props = "../results/data_PSD_props.csv"
file_data_stats = "../results/data_{}_stats.csv"
     
### initiate analysis
Analysis = PSD_Analysis.PSD_Analysis() 

### read in data through in-class routine
data = Analysis.read_data(file_application_data)
#psd = Analysis.psd

### read in data here and set psd and sieve-diameter values separately:
# sieve_diam = [.00001,0.0001,0.0002,0.0005,.001,.002,.004,.008,.016,.025,.035,.05,.063,.075,.088,.105,.125,.150,.177,.21,.25,.3,.354,.42,.5,.6,.707,.85,1.,1.190,1.41,1.68,2]
# data = pd.read_csv(file_application_data)
# Analysis.set_data(data,sieve_diam)

### perform data analysis on psd
Analysis.calc_psd_diameters()
Analysis.calc_psd_soil_class()
Analysis.calc_NEN5104_classification(write_ext_data = file_application_data_extended)
# #Analysis.calc_psd_folk()
Analysis.psd_properties_to_csv(file_psd_props)

### --- write filtered data to file --- ###
###########################################
# data_extend = pd.concat([data, Analysis.psd_properties], axis=1)
# data_extend.to_csv(file_application_data_extended)
# print("\nPDS data file with extended properties saved to file: ",file_application_data_extended)


print('\n#################################\n   Input data: full\n')

stats2save = ['d10','d50','perc_lutum','perc_silt','perc_sand']
stats_data = Analysis.psd_properties[stats2save].copy()
stats_data['logK'] = np.log10(Analysis.data.Kf)
stats = stats_data.describe()

if write_to_file:
    stats.to_csv(file_data_stats.format('full'))
print(stats)

for soil_type in ['sand','silt','clay']:
    print('\n#################################')
    Analysis.sub_sample_soil_type(soil_type)

    stats_data = Analysis.psd_properties_filtered[stats2save].copy()
    stats_data['logK'] = np.log10(Analysis.data_filtered.Kf)
    # test = np.log10(Analysis.data.Kf.values)
    stats = stats_data.describe()
    if write_to_file:
        stats.to_csv(file_data_stats.format(soil_type))
    print(stats)
 
