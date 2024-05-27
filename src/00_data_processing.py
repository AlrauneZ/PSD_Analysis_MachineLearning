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
import numpy as np

### --- File names for input and output data ---###
###################################################

# file_psd_data = "../data/TopInt_meetgegevens_DL&KGVparameters_18-01-2024.csv"
file_psd_data = "../data/TopInt_meetgegevens_DL&KGVparameters_29-04-2024.csv"
file_por_data = "../data/TopInt_porosity_sandy_samples_19-01-2023.csv"

file_data = "../data/data_PSD_Kf_por.csv"

### list of PSD filter sizes (predefined)
sieve_diam = [.00001,0.0001,0.0002,0.0005,.001,.002,.004,.008,.016,.025,.035,.05,.063,.075,.088,.105,.125,.150,.177,.21,.25,.3,.354,.42,.5,.6,.707,.85,1.,1.190,1.41,1.68,2]

### column names in excel file for relevant information:   
name_K = 'K (m/d 10C)'
name_litho = 'Hoofdlithologie_gemeten'
name_ID = 'LocDepth_ID'
name_por = 'porositeit'

### --- Load Data from excel file and perform filtering of samples 
##################################################################
#data = pd.read_csv(file_psd_data)   # read in data as panda data frame
data_psd = pd.read_csv(file_psd_data)   # read in data as panda data frame
data_por = pd.read_csv(file_por_data)   # read in data as panda data frame

### combines data frames, containing samples with also porosity --- ###
###########################################
#data = pd.merge(data_psd, data_por, how='inner', on=[name_ID])  # data set only containing samples with porosity 
data = pd.merge(data_psd, data_por, how='left', on=[name_ID]) # complete data set, where values of porosity are added (otherwise nan)
#print(test[name_por].count())

### --- extract PSD from data-frame --- ###
###########################################
sieve_classes = data.columns[[x.startswith("F") for x in data.columns]]
if len(sieve_diam)-1 != len(sieve_classes.values):
      print("WARNING: number of sieve classes does not match to pre-specified list of sieve diameters.")
data_app = pd.DataFrame(data, columns=sieve_classes)#.values

### --- extract Kf and lithoclass from data-frame --- ###
#########################################################
data_app['Kf'] = data[name_K]
data_app['logK'] = np.log10(data[name_K])
data_app['porosity'] = data[name_por]
data_app['litho_measured'] = data[name_litho]
# data_app['Grondsoort_gemeten'] = data['Grondsoort_gemeten']
print("Number of available samples:", len(data_app.index))

# ### --- drop samples with NAN values (in Kf and sieve samples)
# ##############################################################
data_app.dropna(subset = sieve_classes,inplace = True)
data_app.dropna(subset = ['Kf'],inplace = True)
data_app.reset_index(drop=True,inplace=True) # reset index in data frame
print("Total number of applied samples:", len(data_app.index))

# ### --- write filtered data to file --- ###
# ###########################################
data_app.to_csv(file_data,index = False)

