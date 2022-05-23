# neste caso a implementação da validação crusada será feito por um método diferente, do que já foi trabalhando, com aqueles wrappers do keras para o sklearn

## importações do dataset, do modelo Sequential, das layers... e o kfold

## o StratifiedKFold serve justamente pra trabalhar com a validação cruzada

```python
        from keras.datasets import mnist
        from keras.models import Sequential
        from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
        from keras.utils import np_utils
        import numpy as np
        from sklearn.model_selection import StratifiedKFold
```

### seed

- semente de geradora dos números aleatórios

### np.random.seed

- irar mudar a semente geradora dos números aleatórios

### pois será feita manualmente a divisão da base de dados

```python
        seed = 5
        np.random.seed(seed)
```

### importação da base

```python
        (X, y), (X_teste, y_teste) = mnist.load_data()
```

### transformações

```python
        previsores = X.reshape(X.shape[0], 28, 28, 1)
        previsores = previsores.astype('float32')
```

#### normalização

```python
        previsores /= 255
```

### criarção das variaveis do tipo dummy

```python
        classe = np_utils.to_categorical(y, 10)
```

### kfold vai controlar a validação cruzada

### n_splits, numero de folds que se quer usar, por padrão são 10,

### quer dizer que irar quebrar a base de dados em 10 parte

### shuffle, true irá pegar os dados aleatóriamente

### random_state, pega o valor do seed, se mantem o mesmo valor com a divisão da base de dados

```python
        kfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state = seed)
```

### cria-se a variavel resultados como uma lista vazia,

### pois será onde os ficará o resultado de cada execução
  
```python
        resultados = []
```

### cria um vetor float com 5 colunas, parametro passado, com o valor zero
  
```python
        a = np.zeros(5)
```

### recebe o parametro shape, formato que será a classe como os seus registros

### shape[0] inicia na posição 0, e o argumento 1, referi-se as colunas

```python
        b = np.zeros(shape = (classe.shape[0], 1))
```

### um for para percorrer os registros que precisamos indicar, qual serar da base de treinamentos e qual será da base de teste em cada execução da validação cruzada

### variaveis indices vão monitorar o que vai para test e o que vai para treinamento

### kfold.split

- o comando split fará efetivamente a divisão da base de dados,

### recebe os previsores e passamos o n.zeros como mostrado no exemplo a cima dentro do for será feito todo o processo de pré-processamento e criação da rede neural

```python
        for indice_treinamento, indice_teste in kfold.split(previsores,
                                                            np.zeros(shape = (classe.shape[0], 1))):

```

### print('Indices treinamentos: ', indice_treinamento, 'Indice teste', indice_teste)

```python
            classificador = Sequential()
            classificador.add(Conv2D(32, (3, 3), input_shape = (28,28,1), activation = 'relu'))
            classificador.add(MaxPooling2D(pool_size= (2,2)))
            classificador.add(Flatten())
            classificador.add(Dense(units = 128, activation = 'relu'))
            classificador.add(Dense(units = 10, activation = 'softmax'))
            classificador.compile(loss = 'categorical_crossentropy', optimizer = 'adam',
                                metrics = ['accuracy'])

# classificador.fit e evaluate, irá receber os mesmos parametro, porém com uma indicação do indice  
# pois fizemos as divisões e queremos pegar apenas os atributos dessa divisão                               
            classificador.fit(previsores[indice_treinamento], classe[indice_treinamento],
                            batch_size = 128, epochs = 5)
            precisao = classificador.evaluate(previsores[indice_teste], classe[indice_teste])
```

### resultados que foi criada uma lsita vazia, será adicionada em cada rodada do for o valor da precição, o método evaluate retorna duas posições 0 e 1, a posição 0 retorna a loss function e a posição 1 retorna a accuracy

```python
            resultados.append(precisao[1])
```

### como não temos um vetor, nós temos um obj do tipo lista, então faremos manualmente a média

### sum

- soma

### len

- tamanho

```python
        # media =  resultados.mean()
        media = sum(resultados) / len(resultados)
```