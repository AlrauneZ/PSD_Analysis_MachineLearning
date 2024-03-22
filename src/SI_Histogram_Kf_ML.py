#!/usr/bin/env python3
"""

"""

import PSD_2K_ML
import matplotlib.pyplot as plt
plt.close('all')

algorithm = 'RF' #'ANN' #'DT' #'SVR' #'LR' #'LR' #
soil_type = 'topall'
feature = 'PSD' #'dX_por' #'dX' #
target = 'Kf' # 'por' #
verbose = True #False #

# =============================================================================
# Load Data and perform Algorithm fitting to produce predictions
# =============================================================================
      
filename = "../data/TopIntegraal_PSD_K_SoilClasses.xlsx"

print("Training and Prediction of {}".format(algorithm))
print("###############################")

### ===========================================================================
### Set file pathes and names
### ===========================================================================
# file_data = "../data/AI_data.csv"
file_data = "../data/data_PSD_Kf_por_props.csv"

# Instance of PSD analysis with AI
Analysis = PSD_2K_ML.PSD_2K_ML(
                        algorithm = algorithm,
                        feature = feature,
                        target = target,                            
                        )

data_PSD = Analysis.prepare_data(filename=file_data,
                      soil_type = soil_type, 
                      remove_outlier = False,
                      verbose = verbose,      
                      )

### ===========================================================================
### Speficy Algorithm and set target and feature variables, run training
### ===========================================================================

### specify AI algorithm
Analysis.set_algorithm(algorithm = algorithm,
                       verbose = verbose)

### specifying feature (input) and target (output) variables
Analysis.set_feature_variables()
Analysis.set_target_variables()

### split data for training and train 
Analysis.data_split(verbose = verbose)
Analysis.training()

### ===========================================================================
###   Algorithm Performance with optimal Parameters
### ===========================================================================

### determine prediction data on trained algorithm for specified data set
# Analysis.prediction(x_pred = 'testing_set', verbose = verbose)
# Analysis.prediction(x_pred = 'training_set',verbose = verbose)
Analysis.prediction(x_pred = 'full_set', verbose = verbose)

### calculate percentiles for plot
#bc5,pc5 = Analysis.quantiles_4_plot(bins=10,nth=5)
#bc95,pc95 = Analysis.quantiles_4_plot(bins=10,nth=95)

# =============================================================================
### plot specifications
fig, ax = plt.subplots(figsize=(3.5, 2.5))
# fig, ax = plt.subplots(figsize=(3.75, 3.))
# fig, ax = plt.subplots(figsize=(7.5, 6))
textsize = 12 #8
#soil_class_name, soil_class_sample = Analysis.soil_class_specification()

# =============================================================================

hist = plt.hist(Analysis.y_pred,density = True, bins = 30,rwidth=0.9,color='goldenrod')#,zorder = 3)
#hist = plt.hist(Analysis.y_obs,density = True, bins = 30,rwidth=0.9,color='goldenrod')#,zorder = 3)

ax.set_xlabel("$\log_{10}(K_{obs}$)",fontsize=textsize)
#ax.set_ylabel("density",fontsize=textsize)
ax.set_title(r'$K$ - distribution from {}'.format(algorithm),fontsize=textsize)
ax.grid(True, zorder = 1)

ax.set_xlim([-6.8,2.2])
ax.tick_params(axis="both",which="major",labelsize=textsize)

plt.tight_layout()
# plt.savefig('../results/Fig_Histogram_{}.png'.format(algorithm),dpi=300)
#plt.savefig('../results/Fig_Histogram_{}.pdf'.format(algorithm))
