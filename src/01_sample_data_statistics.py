#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zech0001
"""

import numpy as np
import PSD_Analysis
#import pandas as pd

data_set = "Top-Por" #"Top-All" #
write_to_file = True #False #

print('###########################\n   Input data Analysis \n   for data set {} \n###########################'.format(data_set))

# =============================================================================
# Load Data and perform Algorithm fitting to produce predictions
# =============================================================================

if data_set == "Top-All":
    file_application_data = "../data/data_PSD_Kf.csv"
    file_application_data_extended = "../data/data_PSD_Kf_props.csv"
    file_psd_props = "../results/data_PSD_props.csv"
    file_data_stats = "../results/data_{}_stats.csv"
elif data_set == "Top-Por":
    ### for Top-Por (data subset with samples also having porosity value)
    file_application_data = "../data/data_PSD_por_Kf.csv"
    file_application_data_extended = "../data/data_PSD_por_Kf_props.csv"
    file_psd_props = "../results/data_PSD_por_props.csv"
    file_data_stats = "../results/data_por_{}_stats.csv"

     
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

print('\n#################################\n   Input data: full\n')
stats2save = ['d10','d50','perc_lutum','perc_silt','perc_sand']
test = Analysis.psd_properties
stats_data = Analysis.psd_properties[stats2save].copy()
stats_data['logK'] = np.log10(Analysis.data.Kf)
if data_set == "Top-Por":
    stats_data['por'] = Analysis.data.por
stats = stats_data.describe()

if write_to_file:
    stats.to_csv(file_data_stats.format('full'))
print(stats)


if data_set == "Top-All":
    for soil_type in ['sand','silt','clay']:
        print('\n#################################')
        Analysis.sub_sample_soil_type(soil_type,
                                      filter_props = True
                                      )
    
        stats_data = Analysis.psd_properties_filtered[stats2save].copy()
        stats_data['logK'] = np.log10(Analysis.data_filtered.Kf)
        # test = np.log10(Analysis.data.Kf.values)
        stats = stats_data.describe()
        if write_to_file:
            stats.to_csv(file_data_stats.format(soil_type))
        print(stats)


