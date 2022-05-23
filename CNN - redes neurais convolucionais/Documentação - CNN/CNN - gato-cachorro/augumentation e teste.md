## importação 
```python
        from keras.preprocessing.image import ImageDataGenerator 
```

## variavel que irá gerar as imagens de treinamento

## rescale:
- é a automatização da normalização que foi feita no mnist manualmente

- previsore_treinamento /= 255
- previsores_teste /= 255
- no rescale passamos o valor 1./255 que fará a mesma normalização,
- colocando os valores de cada pixel entre 0 e 1
- como as imagens são salvas do disco usaremos o rescale
- OBS: automatiza todo um processo de transformação dos dados em um fmt do numpy
  - e carregar manualmente

- o mesmo deve ser feito com o gerador_teste, assim como foi feito manualmente nos casos anteriores
- o método flow_from_directory é semelhante ao flow, porém ele pegará os dados de um diretório
- necessário alguns parametros:
    - target_size -: tamanho das imagens,  precisa ser o mesmo valor colocado nan rede neural
    -  batch_size -: tamanho dos pacotes
    -  class_mode -: problema de classificação
    - ele identifica as classe com base na arvore do diretório
    - para melhores resultados o steps_per_epochs deve-se colocar o valor total de atributos,
    - assim ele vai percorrer imagem por imagem, sem saltos
    - o mesmo no validation_steps
    - e claro aumentas a quantidade de epochs
    
```python
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
                                    epochs = 5, validation_data= base_teste,
                                    validation_steps= 1000/32)

```

## OBS:

- no console o returno é esse
- as 5 epochs desejadas
- o 125 é o valor do steps_per_epoch -: 4000/32 = 125
- isso significar que será usada apenas 125 imagens por cada época e não as 4000

 ```python
        Epoch 5/5
        125/125 [==============================] - 72s 576ms/step - loss: 0.5866 
        - accuracy: 0.6820 - val_loss: 0.6510 - val_accuracy: 0.6120                                    
```