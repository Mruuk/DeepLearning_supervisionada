#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  1 10:28:24 2022

@author: lisboa
"""

import numpy as np
from keras_preprocessing import image
from keras.models import model_from_json
import matplotlib.pyplot as plt

arquivo = open('save_rede/classificador_GC.json', 'r')
estrutura_rede = arquivo.read()
arquivo.close()

classificador = model_from_json(estrutura_rede)
classificador.load_weights('save_rede/classificador_GC.h5')

def consulta(imagem):
    imagem_teste = image.load_img(imagem,
                                  target_size=(64, 64))
    imagem_teste = image.img_to_array(imagem_teste)
    imagem_teste /= 255
    imagem_teste = np.expand_dims(imagem_teste, axis = 0)
    previsao = classificador.predict(imagem_teste)
    previsao = (previsao > 0.5)
    
    if previsao == True:
        print('gato')
    else:
        print('cachorro')
    return 0

consulta('dataset/test_set/gato/cat.3522.jpg')

consulta('/home/lisboa/Downloads/IMG_20220211_170727.jpg')
consulta('/home/lisboa/Downloads/IMG-20180603-WA0000.jpg')



