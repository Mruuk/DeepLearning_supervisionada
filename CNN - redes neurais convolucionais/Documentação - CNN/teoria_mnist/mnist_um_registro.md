```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  2 13:06:06 2022

@author: lisboa
"""
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.datasets import mnist
from keras.utils import np_utils
from keras.layers.normalization.batch_normalization import BatchNormalization
import numpy as np
from keras.preprocessing.image import ImageDataGenerator 

(X_treinamento, y_treinamento), (X_teste, y_teste) = mnist.load_data()

previsores_treinamento = X_treinamento.reshape(X_treinamento.shape[0],
                                               28, 28, 1)
previsores_teste = X_teste.reshape(X_teste.shape[0], 28, 28, 1)
previsores_teste = previsores_teste.astype('float32')
previsores_treinamento = previsores_treinamento.astype('float32')


classe_treinamento = np_utils.to_categorical(y_treinamento, 10)
classe_teste = np_utils.to_categorical(y_teste, 10)

classificador = Sequential()
classificador.add(Conv2D(32, (3, 3), activation = "relu"))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size=(2, 2)))

classificador.add(Conv2D(32, (3, 3), activation = "relu"))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size=(2, 2)))
classificador.add(Flatten())

classificador.add(Dense(units= 128, activation= "relu"))
classificador.add(Dropout(0.2))
classificador.add(Dense(units= 128, activation = "relu"))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 10, activation = "softmax"))

classificador.compile(loss = "categorical_crossentropy", optimizer= 'adam',
                      metrics= ["accuracy"])

gerador_treinamento = ImageDataGenerator(rescale = 1./255)

gerador_teste = ImageDataGenerator(rescale = 1./255)

base_treinamento = gerador_treinamento.flow(previsores_treinamento,
                                            classe_treinamento, batch_size= 128)
base_teste = gerador_teste.flow(previsores_teste,
                                classe_teste, batch_size= 128)

classificador.fit_generator(base_treinamento, steps_per_epoch= 60000/128,
                            epochs = 5, validation_data= base_teste,
                            validation_steps= 10000 / 128)

```


### todo o codigo foi feito com o ImageDataGenerator, apenas para automatizar a normalização, mas poderias ser feito manualmente sem problemas, como foi feita a normalização da imagem(um_registro)

### instanciamos uma unica imagem dos previsores_teste necessário acrescentar mais uma coluna, devido a interpretação do tensorflow, essa coluna indica quantas imagens possui a variavel normalizamos os valores realizamos a classificação da imagem retornará um vetor com duas colunas, sendo ele: (1, 10). Nos 10 atributos estão o percentual que a rede neural indicou para cada número da previsão o valor mais alto é o que a rede acredita ser o valor correto basta entao pegar apenas o valor mais alto, utilizando o numpy, como o método argmax, que retorna o valor mais alto dentre os passados depois foi verificado se a rede neural realmente teve acertividade na classificação

```python
imagem = previsores_teste[0]
imagem = np.expand_dims(imagem, axis = 0)
imagem /= 255
precisao = classificador.predict(imagem)

resultado = np.argmax(precisao)

plt.imshow(X_teste[0], cmap = 'gray')
plt.title('Classe ' + str(y_teste[0]))
```