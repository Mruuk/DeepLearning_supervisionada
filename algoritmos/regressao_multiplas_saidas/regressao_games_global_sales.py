#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 10:02:48 2022

@author: lisboa
"""

import pandas as pd
from keras.layers import Dense, Dropout, Activation, Input
from keras.models import Model
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor

base = pd.read_csv('recursos/games.csv')
base = base.drop('Other_Sales', axis = 1)
base = base.drop('Developer', axis = 1)
base = base.drop('NA_Sales', axis = 1)
base = base.drop('EU_Sales', axis = 1)
base = base.drop('JP_Sales', axis = 1)
base = base.dropna(axis = 0)
base = base.loc[base['Global_Sales'] > 1]
nome_jogos = base.Name
base = base.drop('Name', axis = 1)

previsores = base.iloc[:, [0,1,2,3,5,6,7,8,9]]
vendas_Global = base.iloc[:, 4].values

onehotencoder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(),
                                                 [0,2,3,8])],
                                  remainder='passthrough')
previsores = onehotencoder.fit_transform(previsores).toarray()

def cria_rede(neurons, activation, dropout, loss, kernel_initializer):
    camada_entrada = Input(shape=(99,))
    camada_oculta1 = Dense(units = neurons, kernel_initializer=kernel_initializer)(camada_entrada)
    camada_activate1 = Activation(activation)(camada_oculta1)
    camada_drop1 = Dropout(dropout)(camada_activate1)
    camada_oculta2 = Dense(units = neurons, kernel_initializer=kernel_initializer)(camada_drop1)
    camada_activate2 = Activation(activation)(camada_oculta2)
    camada_drop2 = Dropout(dropout)(camada_activate2)
    camada_saida = Dense(units = 1, activation = 'linear')(camada_drop2)
    
    regressor = Model(inputs = camada_entrada,
                      outputs = [camada_saida])
    regressor.compile(optimizer= 'adam',
                      loss= loss)
    return regressor 

regressor = KerasRegressor(build_fn = cria_rede)

parametros = {'batch_size': [100],
              'epochs': [5000, 8000],
              'dropout': [0.2],
              'activation': ['relu', 'sigmoid'],
              'kernel_initializer': ['normal', 'random_uniform'],
              'neurons': [50, 62],
              'loss': ['mse', 'mean_absolute_error',
                       'squared_hinge']}
grid_search = GridSearchCV(estimator= regressor,
                           param_grid= parametros,
                           cv = 2)
grid_search = grid_search.fit(previsores, vendas_Global)
melhores_paramentros = grid_search.best_params_
melhor_precisao = grid_search.best_score_


#regressor.fit(previsores, [vendas_Global],
#              epochs = 5000, batch_size = 100)
#previsao_Global = regressor.predict(previsores)
