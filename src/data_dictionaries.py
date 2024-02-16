#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 14:59:12 2024

@author: alraune
"""

DIC_best_params = dict()

### best hyperparameter values for PSD --> Kf (Top-All + subsets)
DIC_best_params['PSD']=dict(
    DT = dict(
        full = dict(#GS 6&30; skopt: 30&2
            max_depth = 6, 
            min_samples_split= 5,
            ),
        sand = dict( #GS 5&30; skopt: 28&2
            max_depth = 6, 
            min_samples_split= 5,
            ),
        silt = dict(#GS 5&30; skopt: 22&2
            max_depth = 3, 
            min_samples_split= 5,
            ),
        clay = dict(#GS 2&2; skopt: 25&2
            max_depth = 3, 
            min_samples_split= 5,
            ),
        por = dict( 
            # max_depth = 6, 
            # min_samples_split= 5,
            ),
        ),
    RF = dict(
        full = dict( #GS: 30/5/300; Skopt: 22/2/285
            max_depth = 22, 
            min_samples_split= 2, 
            n_estimators = 300,
            ),
        sand = dict(
            max_depth = 25, 
            min_samples_split= 2, 
            n_estimators = 300,
            ),
        silt = dict(
            max_depth = 21, 
            min_samples_split= 2, 
            n_estimators = 300,
            ),
        clay = dict(
            max_depth = 25, 
            min_samples_split= 2, 
            n_estimators = 300,
            ),
        por = dict(
            # max_depth = 25, 
            # min_samples_split= 2, 
            # n_estimators = 300,
            ),
        ),
 
    XG = dict(
        full = dict(#GS: 5/0.05; Skopt:  20/1
            max_depth = 14, 
            learning_rate = 0.05,
            ),
        sand = dict(#GS: 3; Skopt:  20/0.355
            max_depth = 4, 
            learning_rate = 0.5, #0.35,
            ),
        silt = dict(#GS: 2/0.05; Skopt:  12/1
            max_depth = 20, 
            learning_rate = 0.05, #0.05,
            ),
        clay = dict(#GS: 6/0.05; Skopt:  10/0.91
            max_depth = 12, 
            learning_rate = 0.05,
            ),
        por = dict(
            # max_depth = 14, 
            # learning_rate = 0.05,
            ),
        ),
    LR = dict(
        full = dict(#
            alpha =  1,
            ),
        sand = dict(
            alpha =  1,
            ),
        silt = dict(
            alpha =  0.01,
            ),
        clay = dict(
            alpha =  1,
            ),
        por = dict(
            # alpha =  1,
            ),
        ),
    SVR = dict(
        full = dict( #GS 100/0.01; Skopt: 100/1.4
            C = 100, 
            gamma = 0.1,
            ),
        sand = dict(
            C = 10, 
            gamma = 0.1,
            ),
        silt = dict(
            C = 10, 
            gamma = 0.1,
            ),
        clay = dict(
            C = 10, 
            gamma = 0.1,
            ),
        por = dict(
            # C = 10, 
            # gamma = 0.1,
            ),
        ),
    ANN = dict(
        full = dict(#
            activation = 'relu',  
            hidden_layer_sizes = (150,150,150), # more then one hidden layer --> improvements
            learning_rate = 'adaptive', # does not matter of constant or adaptiv!
            ),                     
        sand = dict(
            activation = 'relu',  
            hidden_layer_sizes = (130,130,130), # more then one hidden layer --> improvements
            learning_rate = 'adaptive',  # does not matter of constant or adaptiv!
            ),
        silt = dict(
            activation = 'relu',  #'logistic',  
            hidden_layer_sizes = (150,150,150), # more then one hidden layer --> improvements
            learning_rate = 'adaptive', #'constant',  # # does not matter of constant or adaptiv!
            ),
        clay = dict(
            activation = 'logistic',   #'relu',  #
            hidden_layer_sizes = (100), #(120,80,40),
            learning_rate = 'adaptive',  # does not matter of constant or adaptiv!
            ),
        por = dict(
            # activation = 'logistic',   #'relu',  #
            # hidden_layer_sizes = (100), #(120,80,40),
            # learning_rate = 'adaptive',  # does not matter of constant or adaptiv!
            ),
        ),                     
    )

### best hyperparameter values for d10,d50,d60 --> Kf (Top-All)
DIC_best_params['dX'] = dict(
    DT = dict(
        full = dict(#GS 6&30; skopt: 30&2
            # max_depth = 6, 
            # min_samples_split= 5,
            ),
        ),
    RF = dict(
        full = dict( #GS: 30/5/300; Skopt: 22/2/285
            # max_depth = 22, 
            # min_samples_split= 2, 
            ),
        ),
 
    XG = dict(
        full = dict(#GS: 5/0.05; Skopt:  20/1
            # max_depth = 14, 
            # learning_rate = 0.05,
            ),
        ),
    LR = dict(
        full = dict(#
            alpha =  1,
            ),
        ),
    SVR = dict(
        full = dict( #GS 100/0.01; Skopt: 100/1.4
            # C = 100, 
            # gamma = 0.1,
            ),
        ),
    ANN = dict(
        full = dict(#
            # activation = 'relu',  
            # hidden_layer_sizes = (150,150,150), # more then one hidden layer --> improvements
            ),                     
        ),                     
    )

### best hyperparameter values for d10,d50,d60,por --> Kf (Top-Por)
DIC_best_params['dX_por'] = dict(
    DT = dict(
        por = dict(#GS 6&30; skopt: 30&2
            # max_depth = 6, 
            # min_samples_split= 5,
            ),
        ),
    RF = dict(
        por = dict( #GS: 30/5/300; Skopt: 22/2/285
            # max_depth = 22, 
            # min_samples_split= 2, 
            ),
        ),
 
    XG = dict(
        por = dict(#GS: 5/0.05; Skopt:  20/1
            # max_depth = 14, 
            # learning_rate = 0.05,
            ),
        ),
    LR = dict(
        por = dict(#
            alpha =  1,
            ),
        ),
    SVR = dict(
        por = dict( #GS 100/0.01; Skopt: 100/1.4
            # C = 100, 
            # gamma = 0.1,
            ),
        ),
    ANN = dict(
        por = dict(#
            # activation = 'relu',  
            # hidden_layer_sizes = (150,150,150), # more then one hidden layer --> improvements
            ),                     
        ),                     
    )
### best hyperparameter values for d10,d50,d60,por --> Kf (Top-Por)
DIC_best_params['PSD_por'] = dict(
    DT = dict(
        por = dict(#GS 6&30; skopt: 30&2
            # max_depth = 6, 
            # min_samples_split= 5,
            ),
        ),
    RF = dict(
        por = dict( #GS: 30/5/300; Skopt: 22/2/285
            # max_depth = 22, 
            # min_samples_split= 2, 
            ),
        ),
 
    XG = dict(
        por = dict(#GS: 5/0.05; Skopt:  20/1
            # max_depth = 14, 
            # learning_rate = 0.05,
            ),
        ),
    LR = dict(
        por = dict(#
            alpha =  1,
            ),
        ),
    SVR = dict(
        por = dict( #GS 100/0.01; Skopt: 100/1.4
            # C = 100, 
            # gamma = 0.1,
            ),
        ),
    ANN = dict(
        por = dict(#
            # activation = 'relu',  
            # hidden_layer_sizes = (150,150,150), # more then one hidden layer --> improvements
            ),                     
        ),                     
    )
