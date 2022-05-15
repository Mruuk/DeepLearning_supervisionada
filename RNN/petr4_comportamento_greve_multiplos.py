#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 11 10:09:19 2022

@author: lisboa
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  9 18:44:16 2022

@author: lisboa
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

base = pd.read_csv('recursos/petr4_treinamento_ex.csv')
base = base.dropna()
base_treinamento = base.iloc[:, 1:7].values

normalizador = MinMaxScaler(feature_range=(0,1))
base_treinamento_normalizada = normalizador.fit_transform(base_treinamento)

normalizador_previsao = MinMaxScaler(feature_range=(0,1))
normalizador_previsao.fit_transform(base_treinamento[:, 0:1])

previsores = []
preco_real = []

for i in range(90, 1342):
    previsores.append(base_treinamento_normalizada[i-90:i, 0:6])
    preco_real.append(base_treinamento_normalizada[i, 0])
previsores, preco_real = np.array(previsores), np.array(preco_real)

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

regressor.add(Dense(units= 1, activation= 'sigmoid'))

regressor.compile(optimizer = 'adam', loss= 'mean_squared_error',
                  metrics= ['mean_absolute_error'])

es = EarlyStopping(monitor= 'loss', min_delta= 1e-10, patience= 10, verbose= 1)
rlr = ReduceLROnPlateau(monitor= 'loss', factor= 0.2, patience= 5, verbose = 1)
mcp = ModelCheckpoint(filepath= 'pesos.h5', monitor= 'loss',
                      save_best_only=True, verbose = 1)

regressor.fit(previsores, preco_real, epochs = 100, batch_size = 32,
              callbacks = [es, rlr, mcp])

base_teste =pd.read_csv('recursos/petr4_teste_ex.csv')
preco_real_teste = base_teste.iloc[:, 1:2].values
# base_completa = pd.concat((base['Open'], base_teste['Open']), axis = 0)
frames = [base, base_teste] # equivalente
base_completa = pd.concat(frames) # concat
base_completa = base_completa.drop('Date', axis = 1) # drop da coluna date

entradas = base_completa[len(base_completa) - len(base_teste) - 90:].values
entradas = normalizador.transform(entradas)

X_teste = []

for i in range(90,109):
    X_teste.append(entradas[i-90:i, 0:6])
X_teste = np.array(X_teste)

previsoes = regressor.predict(X_teste)
previsoes = normalizador_previsao.inverse_transform(previsoes)

plt.plot(preco_real_teste, color = 'red', label = 'Preço real')
plt.plot(previsoes, color = 'blue', label = 'Previsões')
plt.title('Previsão preço das ações')
plt.xlabel('Tempo')
plt.ylabel('Valor Yahoo')
plt.legend()
plt.show

plt.savefig('petr4_multiplos_greve.png')

previsoes.mean()
preco_real_teste.mean()
diferenca_media = abs(previsoes.mean() - preco_real_teste.mean())
