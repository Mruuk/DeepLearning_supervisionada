#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 17:30:33 2022

@author: lisboa
"""

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout

# retira tudo que não tem relação com o dado observado(price),
# para que possa encontrar um padrão mais facilmente
base = pd.read_csv('recursos/autos.csv', encoding = 'ISO-8859-1')
base = base.drop('dateCrawled', axis = 1)
base = base.drop('dateCreated', axis = 1)
base = base.drop('nrOfPictures', axis = 1)
base = base.drop('postalCode', axis = 1)
base = base.drop('lastSeen', axis = 1)

# remove a variabilidade que não atrapalha a busca pelo padrão
base['name'].value_counts()
base = base.drop('name', axis = 1)
base['seller'].value_counts()
base = base.drop('seller', axis = 1)
base['offerType'].value_counts()
base = base.drop('offerType', axis = 1)
base['price'].value_counts()
base['abtest'].value_counts()
base['vehicleType'].value_counts()
base['yearOfRegistration'].value_counts()
base['gearbox'].value_counts()
base['powerPS'].value_counts()
base['model'].value_counts()
base['kilometer'].value_counts()
base['monthOfRegistration'].value_counts()
base['fuelType'].value_counts()
base['brand'].value_counts()
base['notRepairedDamage'].value_counts()

# remoção de valores inconsistentes
i1 = base.loc[base.price <= 10]
base.price.mean()
base = base[base.price > 10]
i2 = base.loc[base.price > 350000]
base = base.loc[base.price < 350000]

# editar valores faltantes, todo null recebe o valor com maior moda
base.loc[pd.isnull(base['vehicleType'])]
base['vehicleType'].value_counts() # limousine
base.loc[pd.isnull(base['gearbox'])]
base['gearbox'].value_counts() # manuell
base.loc[pd.isnull(base['model'])]
base['model'].value_counts() # golf
base.loc[pd.isnull(base['fuelType'])]
base['fuelType'].value_counts() # benzin = gasolina
base.loc[pd.isnull(base['notRepairedDamage'])]
base['notRepairedDamage'].value_counts() # nein

valores = {'vehicleType': 'limousine',
           'gearbox': 'manuell',
           'model': 'golf',
           'fuelType': 'benzin',
           'notRepairedDamage': 'nein'}

base = base.fillna(value = valores)

# separa os atributos
previsores = base.iloc[:, 1:13].values
preco_real = base.iloc[:, 0].values


# preprocesso desatualizado
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# labelencoder_previsores = LabelEncoder()

# transforma valores categóricos em numéricos
# previsores[:, 0] = labelencoder_previsores.fit_transform(previsores[:, 0])
# previsores[:, 1] = labelencoder_previsores.fit_transform(previsores[:, 1])
# previsores[:, 3] = labelencoder_previsores.fit_transform(previsores[:, 3])
# previsores[:, 5] = labelencoder_previsores.fit_transform(previsores[:, 5])
# previsores[:, 8] = labelencoder_previsores.fit_transform(previsores[:, 8])
# previsores[:, 9] = labelencoder_previsores.fit_transform(previsores[:, 9])
# previsores[:, 10] = labelencoder_previsores.fit_transform(previsores[:, 10])

# 0 -: 0 0 0
# 2 -: 0 1 0
# 3 -: 0 0 1
# onehotencoder = OneHotEncoder(categorical_features = [0,1,3,5,8,9,10])
# previsores = onehotencoder.fit_transform(previsores).toarray()

# faz o mesmo que o cod desatualizado
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_previsores = LabelEncoder()
 
onehotencorder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(),
                                                  [0,1,3,5,8,9,10])],
                                   remainder='passthrough')
previsores = onehotencorder.fit_transform(previsores).toarray()


regressor = Sequential()
regressor.add(Dense(units= 158, activation= 'relu', input_dim= 316))
regressor.add(Dense(units= 158, activation= 'relu'))
regressor.add(Dense(units= 1, activation= 'linear'))
regressor.compile(loss= 'mean_absolute_error', optimizer= 'adam',
                  metrics=['mean_absolute_error'])

regressor.fit(previsores, preco_real, batch_size= 300, epochs= 100)

previsoes = regressor.predict(previsores)
preco_real.mean()
previsoes.mean()


