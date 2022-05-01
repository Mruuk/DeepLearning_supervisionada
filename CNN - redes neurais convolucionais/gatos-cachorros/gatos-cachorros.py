#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 14:40:59 2022

@author: lisboa
"""

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
from keras.layers.normalization.batch_normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator 
import numpy as np
from keras.preprocessing import image

classificador = Sequential()
classificador.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3)))
classificador.add(Activation('relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2, 2)))

classificador.add(Conv2D(32, (3, 3)))
classificador.add(Activation('relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2, 2)))

classificador.add(Flatten())

classificador.add(Dense(units = 128))
classificador.add(Activation('relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 128))
classificador.add(Activation('relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 1))
classificador.add(Activation('sigmoid'))

classificador.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                      metrics = ['accuracy'])

gerador_treinamento = ImageDataGenerator(rescale = 1./255,
                                         rotation_range = 7,
                                         horizontal_flip = True,
                                         shear_range = 0.2,
                                         height_shift_range = 0.07,
                                         zoom_range = 0.2)

gerador_teste = ImageDataGenerator(rescale = 1./255)

base_treinamento = gerador_treinamento.flow_from_directory('dataset/training_set',
                                                           target_size = (64, 64),
                                                           batch_size = 32,
                                                           class_mode = 'binary')
base_teste = gerador_teste.flow_from_directory('dataset/test_set',
                                               target_size = (64, 64),
                                               batch_size = 32,
                                               class_mode = 'binary')

classificador.fit_generator(base_treinamento, steps_per_epoch= 4000/32,
                            epochs = 10, validation_data= base_teste,
                            validation_steps= 1000/32)

imagem_test = image.load_img('/home/lisboa/Downloads/IMG-20180629-WA0001.jpg',
                             target_size= (64, 64))
imagem_test = image.img_to_array(imagem_test)
imagem_test /= 255
imagem_test = np.expand_dims(imagem_test, axis = 0)
previsao = classificador.predict(imagem_test)
previsao = (previsao > 0.5)
base_treinamento.class_indices

if previsao == True:
    print('Gato')
else:
    print('Cachorro')
