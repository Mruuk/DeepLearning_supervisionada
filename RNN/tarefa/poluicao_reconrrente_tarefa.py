#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 11 13:28:49 2022

@author: lisboa
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

base = pd.read_csv('recursos/poluicao.csv')

# prepocesso
base = base.dropna(axis = 0)
base = base.drop('No', axis = 1)
base = base.drop('year', axis = 1)
base = base.drop('month', axis = 1)
base = base.drop('day', axis = 1)
base = base.drop('hour', axis = 1)
base = base.drop('cbwd', axis = 1)

base_treinamento = base.iloc[:, 1:7].values

poluicao = base.iloc[:, 0].values

normalizador = MinMaxScaler(feature_range=(0,1))
base_treinamento_normalizada = normalizador.fit_transform(base_treinamento)

poluicao = poluicao.reshape(-1, 1)
poluicao_nomalizada = normalizador.fit_transform(poluicao)

previsores = []
poluicao_real = []

for i in range(10, 41757):
    previsores.append(base_treinamento_normalizada[i-10:i, 0:6])
    poluicao_real.append(poluicao_nomalizada[i, 0])
previsores, poluicao_real = np.array(previsores), np.array(poluicao_real)

regressor = Sequential()
regressor.add(LSTM(units= 100, return_sequences = True,
                   input_shape= (previsores.shape[1], 6)))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units= 50, return_sequences= True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units= 50, return_sequences= True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units= 50))
regressor.add(Dropout(0.3))

regressor.add(Dense(units= 1, activation= 'linear'))

regressor.compile(optimizer = 'rmsprop', loss= 'mean_squared_error',
                  metrics= ['mean_absolute_error'])

es = EarlyStopping(monitor = 'loss', min_delta= 1e-10, patience = 10, verbose = 1)
rlr = ReduceLROnPlateau(monitor = 'loss', factor = 0.1, patience = 5, verbose = 1)
mcp = ModelCheckpoint(filepath= 'pesos_tarefa.h5', monitor= 'loss',
                      save_best_only = True, verbose = 1)

regressor.fit(previsores, poluicao_real, epochs = 100, batch_size = 64,
              callbacks = [es, rlr, mcp])

previsoes = regressor.predict(previsores)
previsoes = normalizador.inverse_transform(previsoes)

media_previsao = previsoes.mean()
media_poluicao = poluicao.mean()
abs(previsoes.mean() - poluicao.mean())

plt.plot(poluicao, color = 'red', label = 'Poluição real')
plt.plot(previsoes, color = 'blue', label = 'Previsões')
plt.title('Previsão poluição')
plt.xlabel('Horas')
plt.ylabel('Valor poluição')
plt.legend()
plt.show()


