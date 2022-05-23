## o metodo augumentation do keras, serve para aumentar a quantidade de imagens, bastante útil quando se tem base de dados com poucas imagens.

## ele vai aumentar zoom da imagem, rotacionar, alterar a direção dos pixels, gerando varias outras imagens se adapta melhor e evita overfitting


### foi feita uma rede neural simples com divisão dos dados em treinamento e teste, sem usar a validação cruzada, pensando no processamento, uma vez que será gerada mais imagens

```python
        from keras.datasets import mnist
        from keras.models import Sequential
        from keras.layers import Dense, Flatten, MaxPooling2D, Conv2D
        from keras.utils import np_utils
# importação do metodo augumentation        
        from keras.preprocessing.image import ImageDataGenerator


        (X_treinamento, y_treinamento), (X_teste, y_teste) = mnist.load_data()
        previsores_treinamento = X_treinamento.reshape(X_treinamento.shape[0],
                                                    28, 28, 1)
        previsores_teste = X_teste.reshape(X_teste.shape[0],
                                                    28, 28, 1)
        previsores_treinamento = previsores_treinamento.astype('float32')
        previsores_teste = previsores_teste.astype('float32')
        previsores_teste /= 255
        previsores_treinamento /= 255
        classe_treinamento = np_utils.to_categorical(y_treinamento, 10)
        classe_teste = np_utils.to_categorical(y_teste, 10)

        classificador = Sequential()
        classificador.add(Conv2D(32, (3, 3), input_shape = (28,28,1), activation = 'relu'))
        classificador.add(MaxPooling2D(pool_size= (2,2)))
        classificador.add(Flatten())

        classificador.add(Dense(units = 128, activation = 'relu'))
        classificador.add(Dense(units = 10, activation = 'softmax'))
        classificador.compile(loss = 'categorical_crossentropy',
                            optimizer = 'adam', metrics = ['accuracy'])

# processo do augumentation
# paramentros-:
    # rotation_range -: o grau que será feita a rotação 
    # horizontal_flip -: giros horizontais na imagem
    # shear_range -: mudar os pixels para outra direção, será feita alterações nos valores dos pixels
    # height_shift_range -: responsável pela modificação na faixa de altura da imagem
    # zoom_range -: vai mudar o zoom da imagem
        gerador_treinamento = ImageDataGenerator(rotation_range = 7,
                                                horizontal_flip= True,
                                                shear_range= 0.2,
                                                height_shift_range= 0.07,
                                                zoom_range= 0.2)
# mesmo processo pra base teste 
# sem paramentros, não será feita a transformação das imagens  
# pois vamos aumentas a quantidade das imagens de treinamentos apenas,
# não há necessidade da base teste                                            
        gerador_teste = ImageDataGenerator()

# criando base treinamentos/teste que recebe os gerador
# .flow -: passa os paramentros
# o batch_size vem aqui, no lugar do metodo fit
        base_treinamento = gerador_treinamento.flow(previsores_treinamento,
                                                    classe_treinamento, batch_size= 128)
        base_teste = gerador_teste.flow(previsores_teste,
                                        classe_teste, batch_size = 128)

# o fit_generator no lugar de apenas o fit, pois estamos usando o augumentation
# steps_per_epoch-:  indica o numero total de etapas, lotes de amostras
# a ser gerados pelo gerados, antes de concluir uma época -: 
    ## se coluca a nº de imagem que se tem, dividido pelo batch_size
    ## se não for feita a divisão e passar o valor total de registros,
    ## ele irá passar de um por um e demorará muito pra ser feito o teste
    ## é recomendado pela documentação do keras, para executar mais rapido
# validation_steps-: segue amesma ideia do steps_per_epochs
        classificador.fit_generator(base_treinamento, steps_per_epoch= 60000 / 128,
                                    epochs = 5, validation_data = base_teste,
                                    validation_steps= 10000 / 128)

```

### nessa base de dados, nnão é necessário usar o método augumentation,]

### pois já possui dados satisfatórios para testes e treinamentos  

### tendo assim resultados expressivos   

### mais é um recurso bom, caso tenha uma base de dados com resultados não tão bons,

### ele irá melhorar os resultados