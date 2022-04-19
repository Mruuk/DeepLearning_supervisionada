#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 11:05:04 2022

@author: lisboa
"""

import numpy as np
from keras.models import model_from_json

arquivo = open('recursos/classificador_iris.json', 'r')
estrutura_rede = arquivo.read()
arquivo.close()

classificador = model_from_json(estrutura_rede)
classificador.load_weights('recursos/classificador_iris.h5')

novo = np.array([[7.3,2.9,6.3,1.8]])

previsao = classificador.predict(novo) 
previsao = (previsao > 0.5)

if previsao[0][0] == True and previsao[0][1] == False and previsao[0][2] == False:
    print('Iris setosa')
elif previsao[0][0] == False and previsao[0][1] == True and previsao[0][2] == False:
    print('Iris versicolor')
elif previsao[0][0] == False and previsao[0][1] == False and previsao[0][2] == True:
    print('Iris virginica')