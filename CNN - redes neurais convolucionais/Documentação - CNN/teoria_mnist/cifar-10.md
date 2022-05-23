```python
classificador = Sequential()
classificador.add(Conv2D(32, (3, 3), input_shape = (32, 32, 3),
                        activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size= (2, 2)))
classificador.add(Flatten())

classificador.add(Dense(units = 128, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 1, activation = 'sigmoid'))
classificador.compile(loss = 'categorical_crossentropy',
                        optimizer = 'adam', metrics = ['accuracy'])
classificador.fit(previsores_treinamento, classe_treinamento,
                    batch_size = 128, epochs = 5,
                    validation_data = (previsores_teste, classe_test))
```