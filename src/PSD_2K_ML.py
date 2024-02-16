#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Maschine Learning application to estimate hydraulic conductivity K from 
Particle size distribution data 
Training against K from permeameter observations using TopIntegraal data base
provided by TNO

@author: A. Zech
"""
import numpy as np
import scipy
import copy
import pandas as pd
from PSD_Analysis import PSD_Analysis
from data_dictionaries import DIC_best_params

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
          feature = 'PSD',
          target = 'Kf',
          **settings_new,
          ):

        """
            algorithm - type of ML algorithm, options:
                - LR = linear regression with Ridge
                - DT = decision tree
                - RF = random forest
                - XG = XGBoost
                - SVR = support vector regression
                - ANN = Artifical neural networks
                
            feature - type of feature variable, options:
                - PSD (default)
                - dX 
                - dX_por
        
            target - type of target variable, options:
                - Kf (default)
                - por
        """

        self.algorithm = algorithm
        self.feature = feature
        self.target = target
        self.settings = copy.copy(DEF_settings)
        self.settings.update(**settings_new)

    def prepare_data(self,
                     filename = False,
                     soil_type = 'full', 
                     remove_outlier = False,
                     verbose = False,                  
                  ): 

        if filename:
            # function inheritated from PDS_Analysis
            self.read_data(filename) # read in data function given in superior class PSD_Analysis
       
        self.data = self.data.assign(logK=np.log10(self.data["Kf"].values))       
        
        if remove_outlier:
            self.remove_outliers(verbose = verbose)
            
        self.soil_type = soil_type            
        if soil_type not in ['full','all','por']:
            # function inheritated from PDS_Analysis
            self.sub_sample_soil_type(soil_type = soil_type,
                                      inplace = True,
                                      filter_props = False,
                                      verbose = verbose,
                                      )

        if self.feature in ['dX','dX_por']:
            # self.prep_dx(verbose = verbose)
            self.dX = self.calc_psd_diameters(diams = [10,50,60])
            
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
        self.psd = self.psd[filtered_entries]

        if verbose:        
            print("Number of outlier removed: {}".format(len(filtered_entries)-np.sum(filtered_entries)))
            print("Number of samples: {}".format(len(self.data["Kf"].values)))

        return self.data

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


    def set_algorithm(self,
                      algorithm = None,
                      verbose = False,
                      **kwargs,
                      ):

        print('----------------------------------\n  Set ML algorithm --> {} \n----------------------------------'.format(algorithm))
        self.scale_feature=False

        max_depth = [2, 3, 4 , 5, 6, 8, 10, 20, 25, 30]
        min_samples_split = [2, 3, 5, 10, 15, 20, 30] # [2, 5, 10, 20]
#        n_estimators = [20, 60, 100, 150, 200, 300]
        
        if algorithm is not None:
            self.algorithm = algorithm

        if self.algorithm == 'DT':
            self.AI = DecisionTreeRegressor(random_state = 42)
                
            self.search_space_GS = {
                    "max_depth" : max_depth,
                    "min_samples_split" : min_samples_split
                    }

            self.search_space_scopt = [
                      skopt.space.Integer(max_depth[0], max_depth[-1], name='max_depth'),
                      skopt.space.Integer(min_samples_split[0],min_samples_split[-1], name='min_samples_split'),
                      ]
             
        elif self.algorithm == 'RF':
            self.AI = RandomForestRegressor(random_state = 42, 
                                            bootstrap = True,
                                            n_estimators = 300,
                                            )

            self.search_space_GS = {
                "max_depth" : max_depth,
                "min_samples_split" : min_samples_split, 
#                "n_estimators" : n_estimators,
                }

            self.search_space_scopt = [
                      skopt.space.Integer(max_depth[0], max_depth[-1], name='max_depth'),
                      skopt.space.Integer(min_samples_split[0],min_samples_split[-1], name='min_samples_split'),
 #                     skopt.space.Integer(n_estimators[0],n_estimators[-1], name='n_estimators'),
                      ]

        elif self.algorithm == 'XG':
            self.AI = XGBRegressor()#n_estimators = 200)

            self.search_space_GS = {
                            'max_depth': max_depth,
                            # 'n_estimators': n_estimators, 
                            'learning_rate' : [0.01, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0],
                            }

            self.search_space_scopt = [
                      skopt.space.Integer(max_depth[0], max_depth[-1], name='max_depth'),
                      skopt.space.Real(0.01, 1.0,prior = 'log-uniform', name='learning_rate'),
                      # skopt.space.Integer(n_estimators[0],n_estimators[-1],name = 'n_estimators')
                      ]

        elif self.algorithm == 'LR':
            self.AI = Ridge(random_state = 42,solver = 'svd')

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

            self.search_space_GS = {'C': [1, 10, 100,1000], 
                             'gamma': [0.001,0.01,0.1,1.,10],
                             }

            self.search_space_scopt = [
                      skopt.space.Real(1, 1000,prior = 'log-uniform', name='C'),
                      skopt.space.Real(0.001, 10,prior = 'log-uniform', name='gamma'),
                      ]

        elif self.algorithm == 'ANN':
            self.AI = MLPRegressor(random_state = 42, 
                                   solver = 'adam',
                                   max_iter=500,
                                   alpha = 0.0001,
                                   learning_rate = 'adaptive',
                                   ) #solver='lbfgs')
            self.scale_feature=True
            
            self.search_space_GS = {
                # 'hidden_layer_sizes': [(50),(60),(80),(100),(120),(150)],
                'hidden_layer_sizes': [(150,100,50), (120,80,40), (100,50,30)],
                'activation': ['tanh', 'relu','logistic'],
                # 'learning_rate': ['constant','adaptive'],
            }

            self.search_space_scopt =  [
                      skopt.space.Integer(50, 150, name='hidden_layer_sizes'),
                      # skopt.space.Categorical(categories = ['constant','adaptive'], name = 'learning_rate'),
                      skopt.space.Categorical(categories = ['tanh', 'relu','logistic'],name="activation")
                      ]     
           
        else:
            print("Warning: specified algorithm '{}' not implemented.".format(self.algorithm))
            print("Select from : ANN, DT, RF, XG, LR or SVR")

        # self.best_params = DIC_best_params[self.algorithm][self.soil_type]
        if self.feature == 'PSD' and self.target =='por':
            self.best_params = DIC_best_params['PSD_por'][self.algorithm][self.soil_type]   
        else:
            self.best_params = DIC_best_params[self.feature][self.algorithm][self.soil_type]   
        
        self.AI.set_params(**self.best_params)
        if verbose:
            print("Selected ML algorithm/regressor and parameter settings:")
            print(self.AI)
            # print(self.best_params)
        
    def set_feature_variables(self,
                              feature = None,
                              scale_feature = None,
                              **kwargs,
                              ):

        """
            feature - type of feature variable, options:
                - PSD (default)
                - dX 
                - dX_por
        
        """
        if feature is not None: #update feature variable setting
            self.feature=feature

        print('----------------------------------\n  Set Feature Variables --> {} \n----------------------------------'.format(self.feature))

        if scale_feature is not None:
            self.scale_feature=scale_feature

        # sieve_classes = self.data.columns[[x.startswith("F") for x in self.data.columns]]
        # feature_var = pd.DataFrame(self.data, columns=sieve_classes).values

        if self.feature == 'PSD':
            self.feature_var = self.psd
        elif self.feature == 'dX':
            self.feature_var = self.dX
        elif self.feature == 'dX_por':
            self.feature_var = self.dX
            self.feature_var['por'] = self.data['por']
        else: 
            raise ValueError('Choice of feature variable not implemented.')

        if self.scale_feature:
            self.feature_var = StandardScaler().fit_transform(self.feature_var)
        
        return self.feature_var

    def set_target_variables(self,
                             target = None,
                             log_transform=True,
                             **kwargs,
                             ):

        if target is not None: #update feature variable setting
            self.target=target
        
        print('----------------------------------\n  Set Target Variable --> {} \n----------------------------------'.format(self.target))
        # target_var = pd.DataFrame(self.data, columns=['K']).values

        if self.target == 'Kf':
            if log_transform:
                self.target_var = self.data.logK.values
            else:
                self.target_var = self.data.Kf.values
        elif self.target == 'por':
            self.target_var = self.data['por']
        else: 
            raise ValueError('Choice of feature variable not implemented.')

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
                          file_results = "../results/Hyper_{}_{}_GS.csv",
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
                             file_results = "../results/Hyper_{}_{}_Skopt.csv",
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
            self.set_feature_variables(**kwargs)
            self.set_target_variables(**kwargs)

        self.AI.fit(self.x_train, self.y_train)

    def prediction(self,
                   x_pred = 'full_set',
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
        else:
            print("Warning: variables x_pred and y_obs not specified correctly")

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

    def application(self, 
                    x_app, 
                    y_app = False, 
                    verbose = False,
                   ):
       
        # if not isinstance(x_pred,np.array()) and not isinstance(y_obs,np.array()):
        #     raise ValueError('Input and target (observation) not given as arrays!')
        # if len(x_pred)!=len(y_obs):
        #     raise ValueError('Input and target (observation) variable do not have the same length!')
        # else:
        #     self.y_app = y_app

        y_app_pred = self.AI.predict(x_app)
        if isinstance(y_app,np.ndarray):
            if len(x_app)!=len(y_app):
                raise ValueError('Input and target (observation) variable do not have the same length!')
            mse = mean_squared_error(y_app, self.y_app_pred)
            rmse = np.sqrt(self.mse)
            r2 = r2_score(y_app, y_app_pred)

            if verbose:
                print('\nPerformance measures')
                print("--------------------------------------")
                print("MSE is {:.3f}".format(self.mse))
                print("RMSE is {:.3f}".format(self.rmse))
                print('\nPerformance measures:')
                print("--------------------------------------")
                print("MSE is {:.3f}".format(mse))
                print("RMSE is {:.3f}".format(rmse))
                print("R2 is  {:.3f}".format(r2))

                return (y_app_pred, mse, rmse, r2)

            else:
                return y_app_pred

        else:
            return y_app_pred

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

