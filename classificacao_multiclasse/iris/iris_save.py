#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 08:32:11 2022

@author: lisboa
"""

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
import numpy as np

base = pd.read_csv('recursos/iris.csv')
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values

labelencoder = LabelEncoder()
classe = labelencoder.fit_transform(classe)
classe_dummy = np_utils.to_categorical(classe)

classificador = Sequential()
classificador.add(Dense(units = 8, activation= 'relu',
                        kernel_initializer= 'normal', input_dim= 4))
classificador.add(Dropout(0.1))
classificador.add(Dense(units = 8, activation = 'relu',
                        kernel_initializer= 'normal'))
classificador.add(Dropout(0.1))
classificador.add(Dense(units = 3, activation= 'softmax'))
classificador.compile(optimizer = 'adam', loss = 'categorical_crossentropy',
                         metrics = ['accuracy'])

classificador.fit(previsores, classe_dummy, batch_size= 10,
                  epochs= 2000)


classificador_json = classificador.to_json()
with open('recursos/classificador_iris.json', 'w') as json_file:
    json_file.write(classificador_json)
classificador.save_weights('recursos/classificador_iris.h5')
