"""
Script for loading in TopIntegraal dataset from Excel file and process it for
extracting the information needed for this study
    - PSD sieve data (including filtering on NAN values)
    - Kf + hydraulic conductivity measurments (including quality check) and 
            filtering out samples not matching quality check

Optional: data analysis of PSD including calculation of PSD properties such as d10 etc, 
          percentage of lutum, sand and silt as well as soil class specification
          soil classes will then also be written to condensed data file
--> soil classes are needed for data sub set specification in the AI analysis 
    and for plotting results

Writing condensed data to csv file (with standardised column naming) for further use in the study

Author: A. Zech
"""

import pandas as pd

### optional: preliminary PSD data analysis for samples statistics and soil class specification
data_analysis = True #False # 

# =============================================================================
# File names for input and output data
# =============================================================================

file_psd_data = "../data/20230822_tbl_Merge_DLprocedures_KGVparameters_paperValerie_filtered.xlsx"
#file_psd_data = "../data/TopIntegraal_PSD_K_SoilClasses.xlsx"

file_AI_data = "../data/AI_data.csv"
file_psd_props = "../data/PSD_properties_alt.csv"

# =============================================================================
# Load Data from excel file and perform filtering of samples applicable for our study
# =============================================================================
  
### read in data all data from excel-file as panda data frame
data = pd.read_excel(file_psd_data)
sieve_diam = [.00001,0.0001,0.0002,0.0005,.001,.002,.004,.008,.016,.025,.035,.05,.063,.075,.088,.105,.125,.150,.177,.21,.25,.3,.354,.42,.5,.6,.707,.85,1.,1.190,1.41,1.68,2]
### column names in excel file for relevant information:   
# name_K = 'tbl_Doorlatendheid_Procedures_M_D60/D10'  # hydraulic conductivity measurements
name_K = 'K (m/d 10C)'
name_Kquality = 'Kwaliteit_monster_upto2019'         # quality check of K measurement
name_Kquality_update = 'Eindoordeel_from2020onwards'

#filter_soil_type = self.data.soil_class.isin(soil_classes)

### specification of filter for K-data to be used (full-fulling quality check)
filter_q0 = data[name_Kquality].isin(['ok'])
filter_q1 = data[name_Kquality_update].isin(['Niet beoordeeld'])
filter_q2 = data[name_Kquality_update].isin(['OK','OK(G)','OK(M)','OK(Z)'])
filter_q = filter_q0*filter_q1 + filter_q2

### filtering all data in agreement with K measurement quality check
data = data[filter_q]

### extract PSD from data-frame
sieve_classes = data.columns[[x.startswith("F") for x in data.columns]]
if len(sieve_diam)-1 != len(sieve_classes.values):
    print("WARNING: number of sieve classes does not match to pre-specified list of sieve diameters.")
data_AI = pd.DataFrame(data, columns=sieve_classes)#.values
data_AI['Kf'] = data[name_K]

### drop samples with NAN values in sieve samples and reset index in data frame
data_AI.dropna(inplace = True)
data_AI.reset_index(drop=True,inplace=True)
print(len(data_AI.index))
data_AI.to_csv(file_AI_data)

# =============================================================================
###Perform preliminary data analysis of PSD for derived parameters and their stats
# =============================================================================

if data_analysis:
    import PSD_Analysis
  
    ### initiate analysis
    Analysis = PSD_Analysis.PSD_Analysis(data_AI)
    
    ### perform data analysis on psd
    # Analysis.filter_psd_data()
    Analysis.calc_psd_diameters()
    Analysis.calc_psd_soil_class()
    Analysis.calc_NEN5104_classification()
    #Analysis.calc_psd_folk()
    
    psd_props = Analysis.psd_properties
    psd_props.to_csv(file_psd_props)   
    psd_props.describe()
