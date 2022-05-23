 
## importação do matplotlib, é uma bibliotéca específica para visualização de dados

## vamos usa-la para visualizar as imagens que temos na base de dados

```python
        import matplotlib.pyplot as plt
```

## importamaos a base de dados que está no próprio keras

```python
        from keras.datasets import mnist
        from keras.models import Sequential
```

## o flatten é a 3ª etapa do pré-processo, que transforma uma matriz em um vetor

```python
        from keras.layers import Dense, Flatten
```

## importação do utils, utilizado para fazer mapeamentos das dummy variables

## necessário transforma os dados no tipo dummy

```python
        from keras.utils import np_utils
```

## conv2d é a camada de convolução, é o nosso operador de convolução

## o MaxPooling2d é a segunda etapa

## o 2d é por trabalharmos com imagem, (x, y)
 
 ```python
        from keras.layers import Conv2D, MaxPooling2D
```

## carregamento da base de dados, que ja vem com a divisão da base de treinamento e a de teste

```python
        (X_treinamento, y_treinamento), (X_teste, y_teste) = mnist.load_data()
```

## para visualizar o registro, plt.imshow, cmap, transforma a imagem em tons de cinza,

## já que foi setado 'gray'

## importante salientar que é importante saber em qual momento é necessário ter ou não ter a cor

## pois a cor trás 3 canais, tendo mais dados para processar, caso não use a cor,

## como foi o caso, será utilizado apenas 1 canal, facilitando o pré-processamento

## 3 canais pois são o RGB

## só estamos alterando a visualização neste momento

```python
        plt.imshow(X_treinamento[0], cmap = 'gray')
```

## exibir a classe da imagem visualizada

```python
        plt.title('Classe ' + str(y_treinamento[0]))
```

## transformação dos dados para que o tensorflow consiga fazer a leitura

## instancia a variavel, o reshape altera o formato a primeira entrada de parametro é a quantidade

## de imagem, ou seja a quantidade de regirstro que teremos.

## segunda entrada será a altura da imagem 

## terceira entrada, será a largura da imagem

## quarta entrada, será a quantidade de canais

```python
        previsores_treinamento = X_treinamento.reshape(X_treinamento.shape[0],
                                                    28, 28, 1)
        previsores_teste = X_teste.reshape(X_teste.shape[0], 28, 28, 1)
```

## convertemos para float32        

```python
        previsores_treinamento = previsores_treinamento.astype('float32')
        previsores_teste = previsores_teste.astype('float32')
```

## a conversão foi feita em prol dessa divisão.

## 255 refere-se ao canal, que vai de 0 a 255. do rgb

## foi modificada a escala do valores, para que o processamento seja mais rapido,

## já que os valores estao muito altos

## transformar em uma escala de 0 a 1,

## e para isso é aplicada uma tecnica samada de MinMaxNormalization

## normalização

```python
        previsores_treinamento /= 255
        previsores_teste /= 255
```

## criamos uma classe do tipo dummy com 10 atributos, pois são 10 classe
 
 ```python
        classe_treinamento = np_utils.to_categorical(y_treinamento, 10)
        classe_test = np_utils.to_categorical(y_teste, 10)
```