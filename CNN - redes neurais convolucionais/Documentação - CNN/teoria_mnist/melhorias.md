# melhorando a acurácia da rede neural, aumentando as camandas, tanto do pré-processamento, camanda convolucional e pooling, como das camadas ocultas.

### importação de normalização de camadas

### o batchnomalization cuida de normalizar os parametros da camada de convolucional

### seu objetivo é melhoras a eficiencia e reduzir o tempo de processamento

```python
    from keras.layers.normalization.batch_normalization import BatchNormalization
```

### como feito neste caso, a normalização dos valores, tornando variaveis entre 0 e 1

```python
    previsores_treinamento /= 255
    previsores_teste /= 255
```

### a adição das camadas do batch nomalization, após o conv2D e importante, como fizemos aqui, dois processos do pré-processamento o flatten, que é responsável por transformar a matriz gerada em um vetor, com o objetivo de adaptar para a camada de entrada, ele só será incluido no ultimo processamento, devido a transformação foram incluidos também camadas de dropout para evitar underfintting e mais uma camada oculta

```python
    classificador = Sequential()
    classificador.add(Conv2D(32, (3,3), input_shape= (28, 28, 1),
                            activation='relu'))
    classificador.add(BatchNormalization())
    classificador.add(MaxPooling2D(pool_size = (2, 2)))
    #classificador.add(Flatten())

    classificador.add(Conv2D(32, (3,3), activation= 'relu'))
    classificador.add(BatchNormalization())  
    classificador.add(MaxPooling2D(pool_size= (2,2)))
    classificador.add(Flatten())

    classificador.add(Dense(units= 128, activation= 'relu'))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units= 128, activation= 'relu'))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units = 10, activation= 'softmax'))
    classificador.compile(loss= 'categorical_crossentropy', optimizer= 'adam',
                        metrics= ['accuracy'])
    classificador.fit(previsores_treinamento, classe_treinamento,
                    batch_size= 128, epochs = 5,
                    validation_data= (previsores_teste, classe_teste))
```