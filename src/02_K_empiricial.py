#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 20:29:44 2023

@author: zech0001
"""

#import matplotlib.pyplot as plt
#import numpy as np
import pandas as pd
import PSD_K_empirical

###----------------------------------------------------------------------------
### Set file pathes and names
###----------------------------------------------------------------------------
file_AI_data = "../data/AI_data.csv"
#file_psd_props = "../data/PSD_properties.csv"
file_Kemp_all = "../results/Kemp_all.csv"
file_Kemp_app = "../results/Kemp_app.csv"


###----------------------------------------------------------------------------
### Initialize analysis creating class object and identify Kemp values
###----------------------------------------------------------------------------

Analysis_Kemp = PSD_K_empirical.PSD_to_K_Empirical() 
data = Analysis_Kemp.read_data(file_AI_data)
Analysis_Kemp.set_input_values()

K_empirical =  Analysis_Kemp.PSD2K_allMethods()

Analysis_Kemp.write_to_csv(file_Kemp_all)

###----------------------------------------------------------------------------
### Some statistics on K_emp and applicability of methods:
###----------------------------------------------------------------------------

app_cond = K_empirical.columns[[x.startswith("app") for x in K_empirical.columns]]
app = pd.DataFrame(K_empirical, columns=app_cond)#.values
app_all = app.columns[(app.sum() == len(app))]
print("Method with applicabilty to all samples:")
print(app_all)

K_empirical =  Analysis_Kemp.PSD2K_fullappMethods()
Analysis_Kemp.write_to_csv(file_Kemp_app)
