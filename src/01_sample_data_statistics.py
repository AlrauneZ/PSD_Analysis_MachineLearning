#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script performing data analysis of PSD:
    - reading PSD data from csv-file (with standardised column naming)
    - determining grain size diameters (d10, d50, d60 etc)
    - determine lithoclass and sort samples to sub-datasets
    - perform statistical analysis (mean, std, median etc)
    - saving result in csv-files
Author: A. Zech
"""

import numpy as np
import PSD_Analysis
# import pandas as pd

write_to_file = True #False # 

print('###########################\n   Input data Analysis \n ###########################')

# =============================================================================
# Load Data and perform Algorithm fitting to produce predictions
# =============================================================================

file_data = "../data/data_PSD_Kf_por.csv"
file_data_extended = "../data/data_PSD_Kf_por_props.csv"
file_psd_props = "../results/Data_analysis/data_PSD_props.csv"
file_data_stats = "../results/Data_analysis/data_{}_stats.csv"
 
### initiate analysis
Analysis = PSD_Analysis.PSD_Analysis() 

### read in data through in-class routine
Analysis.read_data(file_data)

### read in data here and set psd and sieve-diameter values separately:
# sieve_diam = [.00001,0.0001,0.0002,0.0005,.001,.002,.004,.008,.016,.025,.035,.05,.063,.075,.088,.105,.125,.150,.177,.21,.25,.3,.354,.42,.5,.6,.707,.85,1.,1.190,1.41,1.68,2]
# data = pd.read_csv(file_application_data)
# Analysis.set_data(data,sieve_diam)

### perform data analysis on psd
Analysis.calc_psd_diameters()
Analysis.calc_psd_soil_class()
Analysis.calc_NEN5104_classification(treat_peat = True)
Analysis.filter_litho(treat_peat = True, 
                      verbose = True,
                      write_ext_data = file_data_extended)
##Analysis.calc_psd_folk()
Analysis.psd_properties_to_csv(file_psd_props)

### Geometric mean of selected psd properties
kg_d10 = np.exp(np.mean(np.log(Analysis.psd_properties['d10'])))
kg_d5 = np.exp(np.mean(np.log(Analysis.psd_properties['d5'])))
kg_d50 = np.exp(np.mean(np.log(Analysis.psd_properties['d50'])))
print("Geometric mean of d10: ",kg_d10)
print("Geometric mean of d5: ",kg_d5)
print("Geometric mean of d50: ",kg_d50)

stats = Analysis.stats_data(
    other_stats2save = ['logK','porosity'],  
    file_data_stats = file_data_stats.format('full'))

Analysis.sub_sample_por(
    filter_props = True, 
    inplace = False)

Analysis.stats_data(
    other_stats2save = ['logK','porosity'],  
    filter_props = True, 
    file_data_stats = file_data_stats.format('por'))

for soil_type in ['sand','silt','clay']:
    print('\n#################################')
    Analysis.sub_sample_litho(soil_type,
                                  filter_props = True,
                                  inplace = False,
                                  )
    Analysis.stats_data(
        other_stats2save = ['logK','porosity'],  
        filter_props = True,
        file_data_stats = file_data_stats.format(soil_type))

