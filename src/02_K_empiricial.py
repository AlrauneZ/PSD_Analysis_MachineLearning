#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 20:29:44 2023

@author: zech0001
"""

#import matplotlib.pyplot as plt
#import numpy as np
# import pandas as pd
import PSD_K_empirical

###----------------------------------------------------------------------------
### Set file pathes and names
###----------------------------------------------------------------------------

# file_data = "../data/data_PSD_Kf_por.csv"
file_data = "../data/data_PSD_Kf_por_props.csv"
file_data_Kemp = "../data/data_PSD_Kf_por_props_Kemp.csv"
file_Kemp_all = "../results/Kemp_all.csv"

###----------------------------------------------------------------------------
### Initialize analysis creating class object and identify Kemp values
###----------------------------------------------------------------------------

Analysis_Kemp = PSD_K_empirical.PSD_to_K_Empirical() # instance of PSD class
data = Analysis_Kemp.read_data(file_data) # data from csv file
sieve_diam = Analysis_Kemp.sieve_diam
Analysis_Kemp.set_input_values()

###----------------------------------------------------------------------------
### Calculate empirical K for all implemented methods and save to file:
###----------------------------------------------------------------------------

K_empirical =  Analysis_Kemp.PSD2K_allMethods() # data frame with all values from empirical methods
Analysis_Kemp.write_to_csv(file_Kemp_all) # write results for all empirical methods to file

###----------------------------------------------------------------------------
### Select K_emp from methods with applicability to all samples:
###----------------------------------------------------------------------------

# app_cond = K_empirical.columns[[x.startswith("app") for x in K_empirical.columns]] # condition picking methods applicable to all samples
# app = pd.DataFrame(K_empirical, columns=app_cond)#.values
# print("Method with applicabilty to all samples:")
# print(app.columns[(app.sum() == len(app))])

K_empirical =  Analysis_Kemp.PSD2K_fullappMethods()
Analysis_Kemp.write_to_csv(file_data_Kemp,add_data = True)
#df = pd.concat([data,K_empirical],axis = 1)
