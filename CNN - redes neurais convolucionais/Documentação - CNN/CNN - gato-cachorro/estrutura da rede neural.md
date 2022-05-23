## Será trabalhada com imagens fisicas no disco mesmo, importante manter a estrutura da pasta dataset,para que não tenha necessidade de transformação nos codigos.
## Esse modelo, permitirá fazer a mesma classificação com o que quiser.

## usaremos o ImageDataGenerator, para gerar algumas imagens adicionais, devido o dataset ser pequeno
- na camada de convolução usamos um shape de 64 x 64, um shape um pouco maior que as imagens,
- isso porque temos imagens no dataset com tamanhos variados, e dessa forma será feita uma conversão
- das dimensões que temos originalmente para esses valores.
- é recomendado colocar valores maiores no input_shape, de acordo com as imagens do dataset
- e no input_shape teremos 3 canais-: RGB
```python
        from keras.models import Sequential
        from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
        from keras.layers.normalization.batch_normalization import BatchNormalization
        from keras.preprocessing.image import ImageDataGenerator 

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

        classificador.compile(optimizer = 'adam', loss = 'binary_crossenntropy',
                            metrics = ['accuracy'])
```