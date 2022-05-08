#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 20:05:21 2022

@author: lisboa
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importa base de dados
base = pd.read_csv('recursos/petr4_treinamento.csv')
# retira atributos sem valor(nan)
base = base.dropna()
# seleciona o que será alvo de previsão
base_treinamento = base.iloc[:, 1:2].values

# normalização dos valores utilizando o método(fuction) minmaxscaler
# irá transformar os valores nessa escala defenida na função
normalizador = MinMaxScaler(feature_range=(0,1))
base_treinamento_normalizada = normalizador.fit_transform(base_treinamento)

# faz uma codificação para efetivamente contruir a base de dados
# será usado 90 dias anteriores em nosso intervalo temporal
# começamos do registro 90 e vamos até o final
previsores = []
preco_real = []
for i in range(90, 1242):
    previsores.append(base_treinamento_normalizada[i-90:i, 0])
    preco_real.append(base_treinamento_normalizada[i, 0])

# transformando em vetor    
previsores, preco_real = np.array(previsores), np.array(preco_real)

# formato do input shape que o keras necessita para redes recorentes:
    # batch_size-: quantidade de registros
    # timesteps-: intervalo de tempo
    # input_dim-: quantos atributos previsores vamos utilizar,
        ## também chamados de indicador, quando trabalhamos com série temporal
        ## chamamos esses atributos previsores de indicadores
previsores = np.reshape(previsores, (previsores.shape[0], previsores.shape[1],1))

# estrutura da rede neural
regressor = Sequential()
regressor.add(LSTM(units= 100, return_sequences = True,
                   input_shape= (previsores.shape[1], 1)))
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
regressor.fit(previsores, preco_real, epochs = 100, batch_size= 32)

# importa base 
base_teste = pd.read_csv('recursos/petr4_teste.csv')
# seleciona apenas a coluna Open, que é o alvo
preco_real_teste = base_teste.iloc[:, 1:2].values

# concatena a base original com a base teste, por colunas
base_completa = pd.concat((base['Open'], base_teste['Open']), axis = 0)
# lower bound
entradas = base_completa[len(base_completa) - len(base_teste) - 90:].values
entradas = entradas.reshape(-1, 1)
entradas = normalizador.transform(entradas)

X_teste = []

for i in range(90, 112):
    X_teste.append(entradas[i-90:i, 0])
X_teste = np.array(X_teste)
X_teste = np.reshape(X_teste, (X_teste.shape[0], X_teste.shape[1], 1))
previsoes = regressor.predict(X_teste)
previsoes = normalizador.inverse_transform(previsoes)

previsoes.mean()
preco_real_teste.mean()
abs(previsoes.mean() - preco_real_teste.mean())

plt.plot(preco_real_teste, color = 'red', label = 'Preço real')
plt.plot(previsoes, color= 'blue', label = 'Previsões')
plt.title('Previsão preço das ações')
plt.xlabel('Tempo')
plt.ylabel('Valor Yahoo')
plt.legend()
plt.show
