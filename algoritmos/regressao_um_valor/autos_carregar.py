#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 18:04:34 2022

@author: lisboa
"""

from keras.models import models_from_json
import numpy as np

arquivo = open('recursos/regressor_autos.json', 'r')
estrutura_rede = arquivo.read()
arquivo.close()

regressor = models_from_json(estrutura_rede)
regressor.load_weights('recursos/regressor_autos.h5')

novo =  np.array
previsoes = regressor.predict(novo)