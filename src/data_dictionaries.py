#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script containing dictionaries with hyperparameters 
    - for the 6 ML algorithms
    - determined by hyperparameter tuning
    - for the different data(sub)sets:
            - topall (entire data set, all soil types) = Top-All
            - sand (only sandy samples) = Top-Sand
            - silt (only sandy samples) = Top-Silt
            - clay (only sandy samples) = Top-Clay
            - por (only sandy samples) = Top-Por            
    - for different combinations of feature and target variables:
            - PSD: PSD --> Kf (Top-All + subsets)
            - dX: d10,d50,d60 --> Kf (Top-topall)
            - dX_por: d10,d50,d60,por --> Kf (Top-Por)
            - PSD_por: PSD  --> por (Top-Por)
@author: A. Zech
"""

DIC_best_params = dict()

### best hyperparameter values for PSD --> Kf (Top-All + subsets)
DIC_best_params['PSD']=dict(
    topall = dict(
        DT = dict(#GS 6&30; skopt: 30&2
            max_depth = 6, 
            min_samples_split= 5,
            ), 
        RF = dict( #GS: 30/5/300; Skopt: 22/2/285
            max_depth = 22, 
            min_samples_split= 2, 
            n_estimators = 300,
            ),
        XG = dict(#GS: 5/0.05; Skopt:  20/1
            max_depth = 14, 
            learning_rate = 0.05,
            ),
        LR = dict(#
            alpha =  1,
            ),
        SVR = dict( #GS 100/0.01; Skopt: 100/1.4
            C = 100, 
            gamma = 0.1,
            ),
        ANN = dict(#
            activation = 'relu',  
            hidden_layer_sizes = (150,150,150), # more then one hidden layer --> improvements
            learning_rate = 'adaptive', # does not matter of constant or adaptiv!
            ),                     
        ),
    sand = dict(
        DT = dict( #GS 5&30; skopt: 28&2
            max_depth = 6, 
            min_samples_split= 5,
            ),
        RF = dict(
            max_depth = 25, 
            min_samples_split= 2, 
            n_estimators = 300,
            ),
        XG = dict(#GS: 3; Skopt:  20/0.355
            max_depth = 4, 
            learning_rate = 0.5, #0.35,
            ),
        LR = dict(#
            alpha =  1,
            ),
        SVR = dict(
            C = 10, 
            gamma = 0.1,
            ),
        ANN = dict(
            activation = 'relu',  
            hidden_layer_sizes = (130,130,130), # more then one hidden layer --> improvements
            learning_rate = 'adaptive',  # does not matter of constant or adaptiv!
            ),
        ),
    silt = dict(
        DT = dict(#GS 5&30; skopt: 22&2
            max_depth = 3, 
            min_samples_split= 5,
            ),
        RF = dict(
            max_depth = 21, 
            min_samples_split= 2, 
            n_estimators = 300,
            ),
        XG = dict(#GS: 2/0.05; Skopt:  12/1
            max_depth = 20, 
            learning_rate = 0.05, #0.05,
            ),
        LR = dict(
            alpha =  0.01,
            ),
        SVR = dict(
            C = 10, 
            gamma = 0.1,
            ),
        ANN = dict(
            activation = 'relu',  #'logistic',  
            hidden_layer_sizes = (150,150,150), # more then one hidden layer --> improvements
            learning_rate = 'adaptive', #'constant',  # # does not matter of constant or adaptiv!
            ),

        ),
    clay = dict(
        DT = dict(#GS 2&2; skopt: 25&2
                    max_depth = 3, 
                    min_samples_split= 5,
                    ),
        RF = dict(
            max_depth = 25, 
            min_samples_split= 2, 
            n_estimators = 300,
            ),
        XG = dict(#GS: 6/0.05; Skopt:  10/0.91
            max_depth = 12, 
            learning_rate = 0.05,
            ),
        LR = dict(#
            alpha =  1,
            ),
        SVR = dict(
            C = 10, 
            gamma = 0.1,
            ),
        ANN = dict(
            activation = 'logistic',   #'relu',  #
            hidden_layer_sizes = (100), #(120,80,40),
            learning_rate = 'adaptive',  # does not matter of constant or adaptiv!
            ),
        ),
    por = dict(
        DT = dict( 
                max_depth = 14, 
                min_samples_split= 3,
                ),
        RF = dict(
                max_depth = 18, 
                min_samples_split= 2, 
                ),
        XG = dict(
                max_depth = 12, 
                learning_rate =  0.05,
                ),
        LR = dict(
                alpha =  1,
                ),
        SVR = dict(
                C = 10, 
                gamma = 0.1,
                ),
        ANN = dict(
                activation = 'relu',  #'logistic',   #
                hidden_layer_sizes =  (120,80,40), #(100,50,30), #
            ),       
        ),
)

### best hyperparameter values for d10,d50,d60 --> Kf (Top-topall)
DIC_best_params['dX'] = dict(
    topall = dict(
        DT = dict(
                max_depth = 10, 
                min_samples_split= 2,
            ),
        RF = dict(
                max_depth =17, 
                min_samples_split= 2, 
            ),
     
        XG = dict(
                max_depth = 18, 
                learning_rate = 0.05,
            ),
        LR = dict(#
                alpha =  0.001,
           ),
        SVR = dict(
                C = 10, 
                gamma = 1.0,
                ),
        ANN = dict(#
                activation = 'relu',  
                hidden_layer_sizes = (150,100,50), 
            ),                     
    ),
    por = dict(
        DT = dict(
                max_depth = 10, 
                min_samples_split= 3,
            ),
        RF = dict(
                max_depth =18, 
                min_samples_split= 2, 
            ),
     
        XG = dict(
                max_depth = 12, 
                learning_rate = 0.05,
            ),
        LR = dict(#
                alpha =  0.1,
           ),
        SVR = dict(
                C = 10, 
                gamma = 0.1,
                ),
        ANN = dict(#
                activation = 'relu',  
                hidden_layer_sizes = (120,80,40), 
            ),                     
    ),

)

### best hyperparameter values for d10,d50,d60 & por --> Kf (Top-Por)
DIC_best_params['dX_por'] = dict(
    por = dict(
        DT = dict(
                max_depth = 9, 
                min_samples_split= 3,
            ),
        RF = dict(
                max_depth = 18, 
                min_samples_split= 2, 
            ),
     
        XG = dict(
                max_depth = 12, 
                learning_rate = 0.1,
            ),
        LR = dict(#
                alpha =  0.01,
            ),
        SVR = dict(
                C = 10, 
                gamma = 0.1,
            ),
        ANN = dict(#
                activation = 'tanh',  
                hidden_layer_sizes = (120,80,40), 
            ),                     
    )
)
### best hyperparameter values for PSD  --> por (Top-Por)
DIC_best_params['PSD_por'] = dict(
    por = dict(
        DT = dict(
                max_depth = 4, 
                min_samples_split= 2,
                ),
        RF = dict(
                max_depth = 20, 
                min_samples_split= 2, 
                ),
        XG = dict(
                max_depth =10, 
                learning_rate = 0.05,
                ),
        LR = dict(#
                alpha =  0.01,
                ),
        SVR = dict(
                C = 1, 
                gamma = 0.001,
            ),
        ANN = dict(#
                activation = 'logistic',  
                hidden_layer_sizes = (100,50,30), 
                ),                     
        ),
)