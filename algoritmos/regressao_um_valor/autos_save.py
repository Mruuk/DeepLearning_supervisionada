#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 17:28:39 2022

@author: lisboa
"""

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

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

previsores = base.iloc[:, 1:13].values
preco_real = base.iloc[:, 0].values

labelencoder_previsores = LabelEncoder()
onehotencorder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(),
                                                  [0,1,3,5,8,9,10])],
                                   remainder='passthrough')
previsores = onehotencorder.fit_transform(previsores).toarray()

regressor = Sequential()
regressor.add(Dense(units= 158, activation= 'relu', input_dim= 316))
regressor.add(Dropout(0.3))
regressor.add(Dense(units= 158, activation= 'relu'))
regressor.add(Dropout(0.3))
regressor.add(Dense(units= 1, activation= 'linear'))
regressor.compile(loss= 'squared_hinge', optimizer= 'adam',
                  metrics=['mean_absolute_error'])
 
regressor.fit(previsores, preco_real, batch_size= 300, epochs= 100)

regressor_json = regressor.to_json()
with open('recursos/regressor_autos.json', 'w') as json_file:
    json_file.write(regressor_json)
regressor.save_weights('recursos/regressor_autos.h5')
