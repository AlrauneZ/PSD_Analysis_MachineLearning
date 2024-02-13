#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for loading in TopIntegraal dataset from Excel file and process it for
extracting the information needed for this study
    - PSD sieve data (including filtering on NAN values)
    - Kf + hydraulic conductivity measurments (including quality check) and 
            filtering out samples not matching quality check

Writing condensed data to csv file (with standardised column naming) for further use in the study

Author: A. Zech
"""

import pandas as pd
#import numpy as np

### --- File names for input and output data ---###
###################################################

file_psd_data = "../data/TopInt_meetgegevens_DL&KGVparameters_18-01-2024.csv"
file_application_data = "../data/data_PSD_Kf.csv"

### list of PSD filter sizes (predefined)
sieve_diam = [.00001,0.0001,0.0002,0.0005,.001,.002,.004,.008,.016,.025,.035,.05,.063,.075,.088,.105,.125,.150,.177,.21,.25,.3,.354,.42,.5,.6,.707,.85,1.,1.190,1.41,1.68,2]

### column names in excel file for relevant information:   
name_K = 'K (m/d 10C)'
name_lithoclass = 'Lithoklasse_gemeten'
#name_por = 'Lithoklasse_gemeten'

### --- Load Data from excel file and perform filtering of samples 
##################################################################
data = pd.read_csv(file_psd_data)   # read in data as panda data frame


### --- Quality Check ---###
############################
# name_Kquality = 'Kwaliteit_monster'         # quality check of K measurement
### filter for K-data to be used (fulfilling quality check)
# filter_q = data[name_Kquality].isin(['OK','OK(G)','OK(M)','OK(N)','OK(V)','OK(Z)','OK(G)(M)','OK(V)(G)'])
# print("Number of samples after quality check:", len(filter_q.values))
# data = data[filter_q]

### --- extract PSD from data-frame --- ###
###########################################
sieve_classes = data.columns[[x.startswith("F") for x in data.columns]]
if len(sieve_diam)-1 != len(sieve_classes.values):
     print("WARNING: number of sieve classes does not match to pre-specified list of sieve diameters.")
data_app = pd.DataFrame(data, columns=sieve_classes)#.values

### --- extract Kf and lithoclass from data-frame --- ###
#########################################################
data_app['Kf'] = data[name_K]
data_app['lithoclass'] = data[name_lithoclass]
print("Number of available samples:", len(data_app.index))

### --- drop samples with NAN values (in Kf and sieve samples)
##############################################################
data_app.dropna(inplace = True)
data_app.reset_index(drop=True,inplace=True) # reset index in data frame
print("Total number of applied samples:", len(data_app.index))

### --- write filtered data to file --- ###
###########################################
data_app.to_csv(file_application_data,index = False)

