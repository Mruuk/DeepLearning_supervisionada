## Necessário a importação do numpy e da bibliotéca image do keras

```python
        import numpy as np
        from keras.preprocessing import image    
```

## captura a imagem que deseja identificar 
## colocamos o target_size no shape utilizado pela rede,
## importante manter as mesma dimenção para não gerar erros 

```python  
        imagem_test = image.load_img('/home/lisboa/Downloads/img010.jpg',
                                    target_size= (64, 64))
```

## transformamos em um vetor, para que o keras consiga interpretar  

```python                                   
        imagem_test = image.img_to_array(imagem_test)
```

## normalizamos        

```python
        imagem_test /= 255
```

## transformamos em uma formatação para o tensorflow, cria uma coluna com a quantidade de imagens ou tamanho do batch, esse formato é o formato que o tensorflow trabalha

```python
        imagem_test = np.expand_dims(imagem_test, axis = 0)

## previsão        
        previsao = classificador.predict(imagem_test)

## retorna os indices para verificar qual é cada classe        
        base_treinamento.class_indices

## normalizando para true ou false, parametro é o 0.5
        previsao = (previsao > 0.5)

## exibir um resultado para usuarios        
        if previsao == True:
            print('Gato')
        else:
            print('Cachorro')
```