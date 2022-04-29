#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 18:49:00 2022

@author: lisboa
"""

import pandas as pd
from keras.layers import Dense, Dropout, Activation, Input
from keras.models import Model

base = pd.read_csv('recursos/games.csv')

base = base.drop('Other_Sales', axis=1)
base = base.drop('Global_Sales', axis=1)
base = base.drop('Developer', axis=1)

base = base.dropna(axis = 0)
base = base.loc[base['NA_Sales'] > 1]
base = base.loc[base['EU_Sales'] > 1]

base['Name'].value_counts()
nome_jogos = base.Name
base = base.drop('Name', axis = 1)

previsores = base.iloc[:, [0,1,2,3,7,8,9,10,11]].values
venda_na = base.iloc[:, 4].values
venda_eu = base.iloc[:, 5].values
venda_jp = base.iloc[:, 6].values

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

onehotencorder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(),
                                                  [0,2,3,8])],
                                   remainder='passthrough')
previsores = onehotencorder.fit_transform(previsores).toarray()

camada_entrada = Input(shape=(61,))
camada_oculta1 = Dense(units = 32, kernel_initializer= 'normal')(camada_entrada)
camada_activate1 = Activation('sigmoid')(camada_oculta1)
camada_dropout1 = Dropout(0.2)(camada_activate1)
camada_oculta2 = Dense(units = 32, kernel_initializer= 'normal')(camada_dropout1)
camada_activate2 = Activation('sigmoid')(camada_oculta2)
camada_dropout2 = Dropout(0.2)(camada_activate2)
camada_saida1 = Dense(units = 1, activation = 'linear')(camada_dropout2)
camada_saida2 = Dense(units = 1, activation = 'linear')(camada_saida1)
camada_saida3 = Dense(units = 1, activation = 'linear')(camada_saida2)

regressor = Model(inputs = camada_entrada,
                  outputs = [camada_saida1, camada_saida2, camada_saida3])
regressor.compile(optimizer= 'adam',
                  loss = 'mse')
regressor.fit(previsores, [venda_na, venda_eu, venda_jp],
              epochs= 5000, batch_size = 100)

previsao_na, previsao_eu, previsao_jp = regressor.predict(previsores)
