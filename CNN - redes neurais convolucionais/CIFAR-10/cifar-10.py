#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 10:22:52 2022

@author: lisboa
"""

import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Activation
from keras.layers.normalization.batch_normalization import BatchNormalization
from keras.utils import np_utils

(X_treinamento, y_treinamento), (X_teste, y_teste) = cifar10.load_data()

i = 2
plt.imshow(X_treinamento[i])
plt.title('Classe '+ str(y_treinamento[i]))

previsores_treinamento = X_treinamento.reshape(X_treinamento.shape[0],
                                               32, 32, 3)
previsores_teste = X_teste.reshape(X_teste.shape[0],
                                  32, 32, 3)
previsores_treinamento = previsores_treinamento.astype('float32')
previsores_teste = previsores_teste.astype('float32')
previsores_teste /= 255
previsores_treinamento /= 255
classe_treinamento = np_utils.to_categorical(y_treinamento, 10)
classe_teste = np_utils.to_categorical(y_teste, 10)

classificador = Sequential()
classificador.add(Conv2D(32, (3, 3), input_shape=(32,32,3), activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size=(2,2)))

classificador.add(Conv2D(32, (3, 3), activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size=(2,2)))

classificador.add(Flatten())

classificador.add(Dense(units = 128))
classificador.add(Activation('relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 128))
classificador.add(Activation('relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(10, activation='softmax'))
classificador.compile(loss='categorical_crossentropy', 
                      optimizer="adamax", metrics=['accuracy'])
classificador.fit(previsores_treinamento, classe_treinamento, 
                  batch_size=128, epochs=5, 
                  validation_data=(previsores_teste, classe_teste))
precisao = classificador.predict(previsores_teste)
loss, accuracy = classificador.evaluate(previsores_teste, classe_teste)

