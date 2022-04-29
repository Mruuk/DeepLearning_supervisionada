#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 15:52:52 2022

@author: lisboa
"""

from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.utils import np_utils
from keras.layers.normalization.batch_normalization import BatchNormalization
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
previsores_treinamento = x_train.reshape(x_train.shape[0],
                                         32, 32, 3)
previsores_teste = x_test.reshape(x_test.shape[0],
                                  32, 32, 3)
previsores_treinamento = previsores_treinamento.astype('float32')
previsores_teste = previsores_teste.astype('float32')
previsores_treinamento /= 255
previsores_teste /= 255
classe_treinamento = np_utils.to_categorical(y_train, 10)
classe_test = np_utils.to_categorical(y_test, 10)

def criar_rede(activation, neurons, filters):
    classificador = Sequential()
    classificador.add(Conv2D(filters, (3, 3), input_shape = (32, 32, 3),
                            activation = activation))
    classificador.add(BatchNormalization())
    classificador.add(MaxPooling2D(pool_size= (2, 2)))
    classificador.add(Conv2D(filters, (3, 3), activation = activation))
    classificador.add(BatchNormalization())
    classificador.add(MaxPooling2D(pool_size= (2, 2)))
    classificador.add(Flatten())
    
    classificador.add(Dense(units = neurons, activation = activation))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units = 10, activation = 'softmax'))
    classificador.compile(loss = 'categorical_crossentropy',
                         optimizer = 'adam', metrics = ['accuracy'])
    return classificador

classificador = KerasClassifier(build_fn = criar_rede)
parametros = {'batch_size': [128],
              'activation': ['relu'],
              'neurons': [512,1024],
              'filters': [32]}
grid_search = GridSearchCV(estimator= classificador,
                           param_grid= parametros,
                           cv= 2)
grid_search = grid_search.fit(previsores_treinamento, classe_treinamento)
melhores_parametros = grid_search.best_params_
melhor_precisao = grid_search.best_score_
    
