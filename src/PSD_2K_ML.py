#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 21:16:51 2023

@author: zech0001
"""
import numpy as np
import scipy
import copy
import pandas as pd
from PSD_Analysis import PSD_Analysis

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV #RepeatedKFold, cross_val_score,
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR

import skopt

DEF_settings = dict(
        sieve_diam = [.00001,0.0001,0.0002,0.0005,.001,.002,.004,.008,.016,.025,.035,.05,.063,.075,.088,.105,.125,.150,.177,.21,.25,.3,.354,.42,.5,.6,.707,.85,1.,1.190,1.41,1.68,2], # in mm
        )    

class PSD_2K_ML(PSD_Analysis):
    
    def __init__(
          self,
          algorithm = 'LR',
          **settings_new,
          ):

        self.algorithm = algorithm
        self.settings = copy.copy(DEF_settings)
        self.settings.update(**settings_new)

    # def read_data(self,
    #               filename,
    #               **settings,               
    #               ): 

    #     """
    #     Function to read in data from condensed data file containing PSD, 
    #     K-values and soil_classes
    #     """    
    #     self.settings.update(**settings)

    #     self.data = pd.read_csv(filename)

    #     ### Identify PSD data and check number of sieve classes
    #     sieve_classes = self.data.columns[[x.startswith("F") for x in self.data.columns]]
    #     self.sieve_diam = np.array(self.settings['sieve_diam'])
    #     if len(self.sieve_diam)-1 != len(sieve_classes.values):
    #         print("WARNING: number of sieve classes does not match to pre-specified list of sieve diameters.")
    #     self.psd = pd.DataFrame(self.data, columns=sieve_classes)#.values

    #     return self.data

    # def set_psd(self,
    #             psd,
    #             sieve_diam,
    #             ):
        
    #     self.psd = psd
    #     self.sieve_diam = sieve_diam

    def prepare_data(self,
                     filename = False,
                     soil_type = 'all', 
                     remove_outlier = False,
                     verbose = False,                  
                  ): 


        if filename:
            self.read_data(filename)
            #self.data = pd.read_excel(filename)
            # self.data.rename(columns = {'K (m/d 10C)':'Kf'}, inplace = True)
       
        self.data = self.data.assign(logK=np.log10(self.data["Kf"].values))       
        
        if remove_outlier:
            self.remove_outliers(verbose = verbose)
            
        self.soil_type = soil_type            
        if soil_type != 'all':
            self.sub_sample_soil_type(soil_type = soil_type,
                                      inplace = True,
                                      filter_props = False,
                                      verbose = False,
                                      )

        if verbose:        
            print('---------------------------------\n   Data Preparation \n---------------------------------')
            print("Input data of soil types: {}".format(soil_type))
            print("Number of samples: {}\n".format(len(self.data["Kf"].values)))

        return self.data

    def remove_outliers(self,
                        z_score_limit = 3,
                        verbose = False,
                        ):
        
        z_scores = scipy.stats.zscore(self.data["Kf"])
        filtered_entries = (np.abs(z_scores) < z_score_limit)
        self.data = self.data[filtered_entries]
        if verbose:        
            print("Number of outlier removed: {}".format(len(filtered_entries)-np.sum(filtered_entries)))
            print("Number of samples: {}".format(len(self.data["Kf"].values)))

        return self.data

    # def sub_sample_soil_type(self,
    #                          soil_type = 'sand',
    #                          verbose = True,
    #                          ):

    #     self.soil_type = soil_type
    #     if self.soil_type == 'sand':
    #         # soil_classes = ['Zs1', 'Zs2', 'Zs3', 'Zs4', 'Zk','Lz3']
    #         soil_classes = ['Zs1', 'Zs2', 'Zs3', 'Zs4', 'Zk']
    #     elif self.soil_type == 'clay':
    #         soil_classes = ['Ks1', 'Ks2', 'Ks3', 'Ks4']
    #     elif self.soil_type == 'silt':
    #         # soil_classes = ['Lz1', 'Lz2', 'Kz1', 'Kz2', 'Kz3']
    #         soil_classes = ['Lz1', 'Lz2','Lz3', 'Kz1', 'Kz2', 'Kz3']
    #     else:
    #         print("WARNING: soil_type not in the list. \nSelect from: 'sand', 'clay', 'silt'.")

    #     filter_soil_type = self.data.soil_class.isin(soil_classes)
    #     self.data = self.data.loc[filter_soil_type]
    #     if verbose:        
    #         print("Input data filtered to soil type: {}".format(self.soil_type))
    #         print("Number of samples in sub-set: {}".format(len(self.data["Kf"].values)))
        
    #     return self.data

    def soil_class_specification(self,
                                 verbose = True):
     
        ### List of soil classes in data file 
        self.soil_class_names = np.unique(self.data.soil_class)
        ### corresponds to:
        # soil_class_names = ['Ks1', 'Ks2', 'Ks3', 'Ks4', 'Kz1', 'Kz2', 'Kz3', 'Lz1', 'Lz3', 'Zk', 'Zs1', 'Zs2', 'Zs3', 'Zs4']

        ### samples with number coded soil class name for color specification in plots
        self.soil_class_sample = self.data.soil_class.astype('category').cat.codes

        if verbose:        
            print("Soil classes of data (sub-)set: \n {}".format(self.soil_class_names))

        return self.soil_class_names,self.soil_class_sample

    def stats_data(self, 
                   verbose = True):

        stats = self.data["logK"].describe()

        if verbose:        
            print("Statistics of log-conductivity:")
            print(stats)
        return stats

    def set_algorithm(self,
                      algorithm = None,
                      verbose = False,
                      ):

        print('----------------------------------\n  Set ML algorithm --> {} \n----------------------------------'.format(algorithm))
        self.scale_feature=False

        max_depth = [2, 4 , 6, 10, 20, 25, 30]
        min_samples_split = [2, 3, 5, 10, 15, 20, 30] # [2, 5, 10, 20]
        n_estimators = [20, 60, 100, 150, 200, 300]
        
        if algorithm is not None:
            self.algorithm = algorithm

        if self.algorithm == 'DT':
            self.AI = DecisionTreeRegressor(random_state = 42)
            self.best_params = dict(
                            max_depth = 5, 
                            min_samples_split= 30
                            )
            self.search_space_GS = {
                    "max_depth" : max_depth,
                    "min_samples_split" : min_samples_split
                    }
#           self.search_space_GS = {
#                    "max_depth" : [2,3,5,7,10,15,20,25,30],
#                    "min_samples_split" : [2,3,5,7,10,15,20,30],
#                   }
            ### maybe adapt!

            self.search_space_scopt = [
                      skopt.space.Integer(max_depth[0], max_depth[-1], name='max_depth'),
                      skopt.space.Integer(min_samples_split[0],min_samples_split[-1], name='min_samples_split'),
                      ]
  
            
        elif self.algorithm == 'RF':
            self.AI = RandomForestRegressor(random_state = 42, bootstrap = True)
            self.best_params = dict(
                            max_depth = 25, 
                            min_samples_split= 2, 
                            n_estimators = 200)
            self.search_space_GS = {
                "max_depth" : max_depth,
                "min_samples_split" : min_samples_split, 
                "n_estimators" : n_estimators,
                # "bootstrap" : [True]
                }

            self.search_space_scopt = [
                      skopt.space.Integer(max_depth[0], max_depth[-1], name='max_depth'),
                      skopt.space.Integer(min_samples_split[0],min_samples_split[-1], name='min_samples_split'),
                      skopt.space.Integer(n_estimators[0],n_estimators[-1], name='n_estimators'),
                      ]

        elif self.algorithm == 'XG':
            self.AI = XGBRegressor()
            self.best_params = dict(n_estimators = 60, 
                                    max_depth = 5, 
                                    learning_rate = 0.1)

            self.search_space_GS = {
                            'max_depth': max_depth,
                            'n_estimators': n_estimators, 
                            'learning_rate' : [0.01, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0],
                            }

            self.search_space_scopt = [
                      skopt.space.Integer(max_depth[0], max_depth[-1], name='max_depth'),
                      skopt.space.Real(0.01, 1.0,prior = 'log-uniform', name='learning_rate'),
                      skopt.space.Integer(n_estimators[0],n_estimators[-1],name = 'n_estimators')
                      ]

        elif self.algorithm == 'LR':
            self.AI = Ridge(random_state = 42,solver = 'svd')

            if self.soil_type == 'silt':
                self.best_params = dict(alpha = 0.1) 
            else: #"all"
                self.best_params = dict(alpha =  1)

            self.search_space_GS = {
                "alpha" : [0.0001, 0.001, 0.01, 0.1, 1, 10, 100], 
                }

            self.search_space_scopt = [
                      skopt.space.Real(0.0001, 100 ,prior = 'log-uniform', name='alpha'),
                      # skopt.space.Real(0.0, 1,name='alpha'),
                     ]
      
        elif self.algorithm == 'SVR':
            self.AI = SVR(kernel = 'rbf')
            self.scale_feature=True
            self.best_params = dict(C = 10, 
                          gamma = 0.1, 
                          )

            self.search_space_GS = {'C': [0.1,1, 10, 100], 
                             'gamma': [0.001,0.01,0.1,1.],
                             }

            self.search_space_scopt = [
                      skopt.space.Real(0.1, 100,prior = 'log-uniform', name='C'),
                      skopt.space.Real(0.001, 1,prior = 'log-uniform', name='gamma'),
                      ]

        elif self.algorithm == 'ANN':
            self.AI = MLPRegressor(random_state = 42, 
                                   solver = 'adam',
                                   max_iter=500,
                                   alpha = 0.0001
                                   ) #solver='lbfgs')
            self.scale_feature=True
            
            if self.soil_type == 'clay':
                self.best_params = dict(activation = 'relu', 
                                    # alpha = 0.001, 
                                    hidden_layer_sizes = (100, 50, 30),
                                    learning_rate = 'adaptive', 
                                    # learning_rate = 'constant', 
                                    )
            else:
                self.best_params = dict(activation = 'relu',  
                                  # alpha = 0.0001, 
                                  hidden_layer_sizes = (120, 80, 40),
                                  learning_rate = 'constant', 
                                  )                      
            self.search_space_GS = {
                'hidden_layer_sizes': [(150,100,50), (120,80,40), (100,50,30)],
                'activation': ['tanh', 'relu','logistic'],
                # 'alpha': [0.0001,0.001,0.05, 0.1,1.],
                'learning_rate': ['constant','adaptive'],
            }

            self.search_space_scopt =  [
                      skopt.space.Integer(50, 150, name='hidden_layer_sizes'),
                      skopt.space.Categorical(categories = ['constant','adaptive'], name = 'learning_rate'),
                      skopt.space.Categorical(categories = ['tanh', 'relu','logistic'],name="activation")
                      ]
    
            # self.AI = MLPRegressor(random_state = 42, solver = 'sgd') #solver='lbfgs')
            # self.scale_feature=True
            
            # if self.soil_type == 'clay':
            #     self.best_params = dict(activation = 'relu', 
            #                         alpha = 0.001, 
            #                         hidden_layer_sizes = (100, 50, 30),
            #                         learning_rate = 'adaptive', 
            #                         # learning_rate = 'constant', 
            #                         max_iter = 3000, 
            #                         )
            # else:
            #     self.best_params = dict(activation = 'relu',  
            #                       alpha = 0.0001, 
            #                       hidden_layer_sizes = (120, 80, 40),
            #                       learning_rate = 'constant', 
            #                       max_iter = 1000, 
            #                       )                      
            
            # self.search_space_GS = {
            #     'hidden_layer_sizes': [(150,100,50), (120,80,40), (100,50,30)],
            #     'max_iter': [500, 1000, 1500, 3000],
            #     'activation': ['tanh', 'relu','logistic'],
            #     'alpha': [0.0001,0.001,0.05, 0.1,1.],
            #     'learning_rate': ['constant','adaptive'],
            # }


             # self.search_space_scopt =  [
             #           skopt.space.Integer(500, 3000, name='max_iter'),
             #           skopt.space.Integer(50, 150, name='hidden_layer_sizes'),
             #           skopt.space.Real(0.0001, 1, prior = 'log-uniform', name='alpha'),
             #           # skopt.space.Categorical(categories = ['constant','adaptive'], name = 'learning_rate'),
             #           # skopt.space.Categorical(categories = ['tanh', 'relu','logistic'],name="activation")
             #           ]
     
           
        else:
            print("Warning: specified algorithm '{}' not implemented.".format(self.algorithm))
            print("Select from : ANN, DT, RF, XG, LR or SVR")

        self.AI.set_params(**self.best_params)
        if verbose:
            print("Selected ML algorithm/regressor and parameter settings:")
            print(self.AI)
            # print(self.best_params)

        
#       self.settings=copy.copy(DEF_settings)
    def set_feature_variables(self,
                              # scale_feature = False,
                              ):

        print('----------------------------------\n  Set Feature Variables --> PSD \n----------------------------------')

        # sieve_classes = self.data.columns[[x.startswith("F") for x in self.data.columns]]
        # feature_var = pd.DataFrame(self.data, columns=sieve_classes).values
        self.feature_var = self.psd
        if self.scale_feature:
            self.feature_var = StandardScaler().fit_transform(self.feature_var)
        
        return self.feature_var

    def set_target_variables(self,
                             log_transform=True,
                             **kwargs,
                             ):
        
        print('----------------------------------\n  Set Target Variable --> (log) Kf \n----------------------------------')
        # target_var = pd.DataFrame(self.data, columns=['K']).values

        if log_transform:
            self.target_var = self.data.logK.values
            # self.target_var = np.log10(target_var)
        else:
            self.target_var = self.data.Kf.values
            # self.target_var = target_var

        return self.target_var

    def data_split(self,
                   test_size = 0.2,
                   verbose = False,
                   **kwargs):
    
        print('----------------------------------\n  Perform data split  \n----------------------------------')
        # self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.feature_var, self.target_var , test_size = test_size, random_state = 42)
        self.x_train, self.x_test, y_train, y_test = train_test_split(self.feature_var, self.target_var , test_size = test_size, random_state = 42)

        self.y_train = y_train.squeeze()
        self.y_test = y_test.squeeze()

        if verbose:
            print("Training data: {}% ".format(100*(1-test_size)))
            print("Number of samples in training data set: {}".format(len(y_train)))
            print("Test data: {}%".format(100*(test_size)))
            print("Number of samples in test data set: {}".format(len(y_test)))
        return self.x_train, self.x_test, self.y_train, self.y_test


    def hyperparameter_GS(self,
                          file_results = "../results/Hyper_GS_{}_{}.csv",
                          verbose = False,
                          ):

        print('----------------------------------\n  Hyperparameter testing with GridSearch  \n----------------------------------')
        
        # =============================================================================
        # #Hyperparameter tuning
        # =============================================================================

        self.GS = GridSearchCV(estimator = self.AI, 
                          param_grid = self.search_space_GS, 
                          scoring = "r2", #
                          # scoring = ["r2", "neg_root_mean_squared_error"],
                          # refit = "r2",
                          refit = True,
                          cv = 10, 
                          verbose = False)
 
        # grid search on training data set
        self.GS.fit(self.x_train, self.y_train)


        # performance measures for optimal parameters on test data 
        y_pred = self.GS.predict(self.x_test)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        r2 = r2_score(self.y_test, y_pred)

        self.best_params = self.GS.best_params_
        self.AI.set_params(**self.best_params)

        if verbose:
            print("#########################")
            print(" GridSearchCV - Results:")
            print(" Best Estimator: ", self.GS.best_estimator_)
            print(" Best hyperparameters: ", self.GS.best_params_)
            print(" Best R2-score, ",self.GS.best_score_)
            print("#########################")
            print(" Performance of test data set with optimal parameters:")
            print(" RMSE is" ,rmse)
            print(" R2 is:" ,r2)


        if file_results:
            df = pd.DataFrame(self.GS.cv_results_)
            df = df.round(decimals=3)
            # df = df.sort_values("rank_test_r2")
            df.to_csv(file_results.format(self.soil_type,self.algorithm))
      
        return df

    def hyperparameter_skopt(self,
                             # save_results = True,
                             file_results = "../results/Hyper_Skopt_{}_{}.csv",
                             verbose = False,
                              **kwargs,
                             ):        

        print('----------------------------------\n  Hyperparameter testing with scikit-optimize  \n----------------------------------')
    

        HPO_params = {
                      'n_calls':100,
                      'n_random_starts':20,
                      'base_estimator':'ET',
                      'acq_func':'EI',
                      }

        HPO_params.update(**kwargs)

        @skopt.utils.use_named_args(self.search_space_scopt)
        def objective(**params):
            self.AI.set_params(**params)  
            self.AI.fit(self.x_train, self.y_train)
            y_hat = self.AI.predict(self.x_train)
            # r2 = r2_score(self.y_train, y_hat)
            mse = mean_squared_error(self.y_train, y_hat)
            # y_hat = self.AI.predict(self.x_test)
            # r2 = r2_score(self.y_test, y_hat)

            return mse                      
#            return(1-r2)                      
            # rmse_val = np.sqrt(((self.y_test - y_hat)**2).mean())            
            # return (rmse_val)

        results = skopt.forest_minimize(objective,self.search_space_scopt,**HPO_params)

        names = []
        for i in range(len(results['x'])):
            names.append(self.search_space_scopt[i].name)

        df_results = pd.DataFrame(data = results['x'],index=names)
        

        if file_results:
            # np.savetxt(file_results.format(self.algorithm,self.soil_type),results['x'])
            # df_results = pd.DataFrame(data = np.array(results['x']))
            df_results.to_csv(file_results.format(self.soil_type,self.algorithm),header = False)

        if verbose:
            print("#########################")
            print(" Scikit Optimize - Results:")
            # print(" Best Estimator: ", self.GS.best_estimator_)
            print(" Best hyperparameters: ")
            print(df_results)

            # print(" Best hyperparameters: ", results['x'])
            # for i in range(len(results['x'])):
            #     name = self.search_space_scopt[i].name
            #     print(" {} = {}".format(name,results['x'][i]))
            
        return results


    def training(self,
                 algorithm = None,
                 **kwargs,
                 ):

        if algorithm is not None:
            self.algorithm = algorithm
            self.set_algorithm(**kwargs)
            self.set_feature_variables()
            self.set_target_variables(**kwargs)

        self.AI.fit(self.x_train, self.y_train)

    def prediction(self,
                   x_pred = 'full_set',
                   y_obs = 'full_set',
                   verbose = False,
                   ):
       
        if x_pred == 'full_set':
            self.x_pred = self.feature_var
            self.y_obs = self.target_var
        elif x_pred == 'training_set':
            self.x_pred = self.x_train
            self.y_obs = self.y_train
        elif x_pred == 'testing_set':
            self.x_pred = self.x_test
            self.y_obs = self.y_test
        elif isinstance(x_pred,np.array()) and isinstance(y_obs,np.array()):
            if len(x_pred)!=len(y_obs):
                raise ValueError('Input and target (observation) variable have to have same length!')
            else:
                self.x_pred = x_pred
                self.y_obs = y_obs
        else:
            print("Warning: variables x_pred and y_obs not specified correctly")

        # self.AI.set_params(**self.best_params)
        self.y_pred = self.AI.predict(self.x_pred)

        self.mse = mean_squared_error(self.y_obs, self.y_pred)
        self.rmse = np.sqrt(self.mse)
        self.r2 = r2_score(self.y_obs, self.y_pred)

        if verbose:
            print('\nPerformance measures for {}'.format(x_pred))
            print("--------------------------------------")
            print("MSE is {:.3f}".format(self.mse))
            print("RMSE is {:.3f}".format(self.rmse))
            print("R2 is  {:.3f}".format(self.r2))

        return self.y_pred,self.mse, self.rmse, self.r2
        
    # def performance_measures(self,
    #                          y_obs = 'full_set',
    #                          verbose = False):

    #     if y_obs == 'full_set':
    #         self.y_obs = self.target_var
    #     elif y_obs == 'training_set':
    #         self.y_obs = self.y_train
    #     elif y_obs == 'testing_set':
    #         self.y_obs = self.y_test
    #     else:
    #         if len(self.x_pred)!=len(y_obs):
    #             raise ValueError('Input and target (observation) variable have to have same length!')
           
    #     self.mse = mean_squared_error(self.y_obs, self.y_pred)
    #     self.rmse = np.sqrt(self.mse)
    #     self.r2 = r2_score(self.y_obs, self.y_pred)

    #     if verbose:
    #         print('\nPerformance measures for {}'.format(y_obs))
    #         print("--------------------------------------")
    #         print("MSE is {:.3f}".format(self.mse))
    #         print("RMSE is {:.3f}".format(self.rmse))
    #         print("R2 is  {:.3f}".format(self.r2))

    #     return self.mse, self.rmse, self.r2

    def quantiles_4_plot(self,
                  bins=10,
                  nth=5,
                  ):

        bc,pc = quantiles(np.squeeze(self.y_obs),np.squeeze(self.y_pred),bins,nth)
        # t,v = quantiles(np.squeeze(self.y_obs),np.squeeze(self.y_pred),Nbins,nth)

        return bc,pc


def quantiles(y_obs, y_pred, bins, nth):  

    """
    calculating the nth percentile of a series of data split of into Nbins 
    
    Parameters:
        y_obs : 1D array
            A sequence of values to be binned.
        y_pred : 1D array
            The data on which the statistic will be computed. 
            This must be the same shape as y_obs.
        bins : int
            Number of equal-width bins to split data into
        nth : int
            percentile value to be calculated 
    
    Returns:                        
        bin_centers : 1D array of length Nbins
            List of bin center values for each of the Nbins where y_obs has been
            distributed to.
        percentiles : 1D array of length Nbins
            List of values of nth percentile of the data in each of the Nbins.
    """    

    def myperc(y):
        return np.percentile(y,nth)

    t=scipy.stats.binned_statistic(y_obs,y_pred,statistic=myperc,bins=bins)
    bin_centers = t[1][:-1]+0.5*np.diff(t[1])
    percentiles = t[0]

    return bin_centers, percentiles

