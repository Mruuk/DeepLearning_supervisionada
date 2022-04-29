#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 10:46:50 2022

@author: lisboa
"""

import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from keras.wrappers.scikit_learn import KerasRegressor

base = pd.read_csv('recursos/autos.csv', encoding = 'ISO-8859-1')
base = base.drop('dateCrawled', axis = 1)
base = base.drop('dateCreated', axis = 1)
base = base.drop('nrOfPictures', axis = 1)
base = base.drop('postalCode', axis = 1)
base = base.drop('lastSeen', axis = 1)

base['name'].value_counts()
base = base.drop('name', axis = 1)
base['seller'].value_counts()
base = base.drop('seller', axis = 1)
base['offerType'].value_counts()
base = base.drop('offerType', axis = 1)

base = base[base.price > 10]
base = base.loc[base.price < 350000]

valores = {'vehicleType': 'limousine',
           'gearbox': 'manuell',
           'model': 'golf',
           'fuelType': 'benzin',
           'notRepairedDamage': 'nein'}

base = base.fillna(value = valores)

# separa os atributos
previsores = base.iloc[:, 1:13].values
preco_real = base.iloc[:, 0].values

labelencoder_previsores = LabelEncoder()
onehotencorder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(),
                                                  [0,1,3,5,8,9,10])],
                                   remainder='passthrough')
previsores = onehotencorder.fit_transform(previsores).toarray()

def criar_rede(optimizer, loss, activation, kernel_initializer, neurons, dropout):
    regressor = Sequential()
    regressor.add(Dense(units= neurons, activation= activation,
                        kernel_initializer= kernel_initializer, input_dim= 316))
    regressor.add(Dropout(dropout))
    regressor.add(Dense(units= neurons, activation= activation,
                        kernel_initializer= kernel_initializer))
    regressor.add(Dropout(dropout))
    regressor.add(Dense(units= 1, activation= 'linear'))
    regressor.compile(loss= loss, optimizer= optimizer,
                      metrics=['mean_absolute_error'])
    return regressor
regressor = KerasRegressor(build_fn= criar_rede)

parametros = {'batch_size': [300],
              'epochs': [100],
              'dropout': [0.2],
              'optimizer': ['adam'],
              'kernel_initializer': ['random_uniform'],
              'activation': ['relu'],
              'neurons': [158],
              'loss': ['mean_absolute_error', 'mean_squared_error',
                       'mean_absolute_percentage_error',
                       'mean_squared_logarithmic_error', 'squared_hinge']}

grid_search = GridSearchCV(estimator= regressor, 
                         param_grid= parametros,
                         cv= 2)
grid_search = grid_search.fit(previsores, preco_real)
melhores_parametros = grid_search.best_params_
melhor_precisao = grid_search.best_score_