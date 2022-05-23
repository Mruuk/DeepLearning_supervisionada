# criamos a rede neural, sendo ela sequencial

## seguindo as etapas de pré-processamento

- primeiro a camada de convolução -: conv2d

  - filters: 32 detequetore ou  kernels, será kernels personalizados e não os kernels já existentes,
        - até por tem menos que 32
        - uma pratica recomendável é começar com 64, foi iniciado com 32 para ser mais leve,
        - e podendo seguir com os múltiplos, 64, 128, 256, 512, 1024...
  - (3, 3) : kernel_size, será o tamanho do detector de características,
        -  vai depender muito do tamanho da imagem, se tiver imagem maiores,
        -  deve-se aumentar o tamanho desse detector de características
    - strides : parametros que indica como que quer que a janela se mova (1 ,1)
    - nesse exemplo, 1 linha e 1 coluna
    - input_shape : são as dimensões da imagem, 28 x 28 e o canal rgb, que nesse caso é apenas 1
    - função de ativação: relu, por padrão
- segundo o pooling, passando apenas o tamanho da janela de 2 por 2
  - por default o parametro ja vem (2, 2)
- terceiro o flattening, transformando matriz do pooling em vetor
- quarto criação da rede neural
  - units = 128: quando se trabalha com CNN,
    - não é muito comum se utilizar da fórmula de somar entradas mais saidas e dividir por dois,
    - geralmente começa com 128, 256, 512... por se tratar de imagens
- no classificador.fit: fará o treinamento, parametros novo é o validation_data,
  - ele já vai mostrando os resultados nos teste
  - similar ao evaluate e predict. ele faz o treinamento e ja vai testanto

```python
        classificador = Sequential()
        classificador.add(Conv2D(32, (3,3), input_shape= (28, 28, 1),
                                activation='relu'))
        classificador.add(MaxPooling2D(pool_size = (2, 2)))
        classificador.add(Flatten())

        classificador.add(Dense(units= 128, activation= 'relu'))
        classificador.add(Dense(units = 10, activation= 'softmax'))
        classificador.compile(loss= 'categorical_crossentropy', optimizer= 'adam',
                            metrics= ['accuracy'])
        classificador.fit(previsores_treinamento, classe_treinamento,
                        batch_size= 128, epochs = 5,
                        validation_data= (previsores_teste, classe_test))

        resultado = classificador.evaluate(previsores_teste, classe_test)
```