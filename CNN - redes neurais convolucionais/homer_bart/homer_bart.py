#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  2 15:48:06 2022

@author: lisboa
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

base = pd.read_csv('recursos/personagens.csv')
previsores = base.iloc[:, 0:6].values
classe = base.iloc[:, 6].values

labelencoder = LabelEncoder()
classe = labelencoder.fit_transform(classe)
classe_dummy = np_utils.to_categorical(classe)

previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe_dummy, test_size= 0.25)

classificador = Sequential()
classificador.add(Dense(units= 6, activation= 'relu', input_dim= 6))
classificador.add(Dropout(0.2))
classificador.add(Dense(units= 6, activation= 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units=2, activation='softmax'))

classificador.compile(optimizer='adam', loss='binary_crossentropy',
                      metrics= ['accuracy'])
classificador.fit(previsores_treinamento, classe_treinamento,
                  batch_size= 6, epochs= 100,
                  validation_data=(previsores_teste, classe_teste))

previsoes = classificador.predict(previsores_teste)
previsoes = (previsoes > 0.5)

