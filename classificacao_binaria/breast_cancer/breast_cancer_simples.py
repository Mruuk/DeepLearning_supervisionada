#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 12:25:47 2022

@author: lisboa
"""

import pandas as pd

previsores = pd.read_csv('recursos/entradas_breast.csv')
classe = pd.read_csv('recursos/saidas_breast.csv')

from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25)

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
classificador = Sequential()
classificador.add(Dense(units = 16, activation= 'relu', 
                        kernel_initializer= 'random_uniform', input_dim = 30))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 16, activation= 'relu', 
                        kernel_initializer= 'random_uniform'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units= 1, activation='sigmoid'))

import tensorflow as tf
otimizador = tf.keras.optimizers.Adam(learning_rate = 0.001, decay = 0.0001, clipvalue = 0.5)
classificador.compile(optimizer= otimizador, loss = 'binary_crossentropy',
                      metrics = ['binary_accuracy'])

#classificador.compile(optimizer= 'adam', loss = 'binary_crossentropy',
#                      metrics = ['binary_accuracy'])
classificador.fit(previsores_treinamento, classe_treinamento,
                  batch_size = 10, epochs = 100)

#visializaãço de pesos

pesos0 = classificador.layers[0].get_weights()
print(pesos0)
print(len(pesos0))
pesos1 = classificador.layers[1].get_weights()
pesos2 = classificador.layers[2].get_weights()

#verificação em matriz 

previsoes = classificador.predict(previsores_teste)
previsoes = (previsoes > 0.5)
from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)

#ou usando o proprio keras

resultado = classificador.evaluate(previsores_teste, classe_teste)
