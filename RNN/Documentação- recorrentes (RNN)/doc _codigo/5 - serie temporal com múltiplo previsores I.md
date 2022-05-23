# Serie temporal com múltiplos previsores I

## nós utilizamos apenas um previsor nos outros casos, que foi o 'Open', neste caso usaremos todos os atributos da base, exceto a data, no caso, com base em todos os atributos vamos prever o  valor

```python
        from keras.models import Sequential
        from keras.layers import Dense, Dropout, LSTM
        from sklearn.preprocessing import MinMaxScaler
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

        base = pd.read_csv('recursos/petr4_treinamento.csv')
        base = base.dropna()

```

### definimos os atributos para a base de treinamento, como a coluna 0 são as data, ela não entra e vamos de 0:6, para pegarmos a coluna 6, colocamos no upper bound o número 7

```python
        base_treinamento = base.iloc[:, 1:7].values
```

### criamos o nosso normalizador, entre 0,1, e normalizamos aplicando o fit_transform na base de treinamento

```python
        normalizador = MinMaxScaler(feature_range=(0,1))
        base_treinamento_normalizada = normalizador.fit_transform(base_treinamento)

        previsores = []
        preco_real = []
```

### mesmo for dos casos anteriores, a unica alteração será nos previsores, que era apena o índice 0, porém vamos usar todos os atributos, então usaremos uma faixa pegando todos eles, 0:6, são 5 atributos, como queremos pegar o 5 também entao no upper bound colocamos 6

```python
        for i in range(90, 1242):
            previsores.append(base_treinamento_normalizada[i-90:i, 0:6])
            preco_real.append(base_treinamento_normalizada[i, 0])
        previsores, preco_real = np.array(previsores), np.array(preco_real)
```

### não foi necessário redefinir o shape(formato), do vetor previsores, pois como colocamos uma faixa, ele ja veio com o formato que o keras fará a leitura

### mesma estrutura neural, com a diferença de entradas, pois tiamos apenas 1 atributo previsor, agora temos 6 atributos previsores

```python
        regressor = Sequential()
        regressor.add(LSTM(units= 100, return_sequences = True,
                        input_shape= (previsores.shape[1], 6)))
        regressor.add(Dropout(0.3))

        regressor.add(LSTM(units= 50, return_sequences= True))
        regressor.add(Dropout(0.3))

        regressor.add(LSTM(units= 50, return_sequences= True))
        regressor.add(Dropout(0.3))

        regressor.add(LSTM(units= 50))
        regressor.add(Dropout(0.3))
```

### usamos na função de ativação o sigmoid como teste, pois como os valores foram normalizados entre 0 e 1, e a função retorna valores entre 0 e 1, não haverá mudança nos valores assim como o linear

```python
        regressor.add(Dense(units= 1, activation= 'sigmoid'))
```

### rmsprop é o optimizador recomendado para esse tipo de rede neural, e faremos o teste com o adam

```python
        regressor.compile(optimizer = 'adam', loss= 'mean_squared_error',
                        metrics= ['mean_absolute_error'])
```

### a mudaça consideravel

- callbbacks
  - conjunto de funções que podem ser aplicados em determinados estados do processo
    - de treinamento, as funções do callbacks pode ser usado para ter uma visão interna dos estados e estatisticas de um modelo durante o treinamento.

### o earlystopping, irá parar de fazer o treinamento antes, de acordo com algumas condições

- Ex: utilizando uma loss_function, mean_squared_error, e durante 20/30 repetições da atualização dos pesos, ele parou de melhorar os resultados. Então para o treinamneto, pois não vale apenas execultar todo o processamento, sendo que talvez não melhore os resultados
  - parametros
    - monitor
      - tipo de função à qual irá monitorar, geralmento é a loss
      - min_delta-: a mudançã mínima que deve ser monitorada, para considerar como melhoria
      - define 0.1, se não melhorar 0.1 na próxima rodada, entao para o treinamento
      - definimos um valor muito pequeno, em notação científica.
    - patience
      - vai dizer o número de épocas que vai seguir sem melhorias do resultado, como uma tolerância antes de parar o treinamento, no caso foi definido 10 epocas no patience, se passar as 10 epocas e não tiver uma melhoria na loss em um valor 1e-10, entao ele para o treinamento
    - verbose vai mostrar as msg na tela

### ReduceLROnPlateau, irá reduzir a taxa de aprendizagem quando uma métrica parar de melhorar, trabalhando junto com a outra função do callbacks, earlystopping

- Ex: supondo que a loss_function, não está conseguindo melhorar depois de 2 ou 3 rodas, isso indica que vai reduzir o valor da taxa de aprendizagem, para que ele consiga melhorar o valor da métrica
  - paramentros:
    - monitor-: irá monitorar uma função, no caso será a loss function
    - min_delta-: usado para melhorar um novo optimum
    - patience-: número de épocas que vai seguir sem malhoria
    - factor-: o valor que a learning rate(taxa de aprendizagem) vai ser reduzida
    - ele vai multiplicar a taxa de aprendizagem atual pelo factor. new_lr = lr * factor

### ModelCheckpoit, irá salvar os models dps de cada uma das épocas, salvar os pesos. Como a validação cruzada, porem o keras tem recusos para que possa gravar os pesos assim que consiga melhores resultados durante o processamento

- filepath-: passa o caminho do arquivo
- monitor-: vai verificar a função para ver se teve ou não melhoria
- save_best_only-: vai salvar sempre aquele que tiver melhor resultado

### no fit o callbacks é uma lista

```python
        es = EarlyStopping(monitor= 'loss', min_delta= 1e-10, patience= 10, verbose= 1)
        rlr = ReduceLROnPlateau(monitor= 'loss', factor= 0.2, patience= 5, verbose = 1)
        mcp = ModelCheckpoint(filepath= 'pesos.h5', monitor= 'loss',
                            save_best_only=True, verbose = 1)

        regressor.fit(previsores, preco_real, epochs = 100, batch_size = 32,
                    callbacks = [es, rlr, mcp])
```

### callbacks  auxilia no treinamento, para que não perca muito tempo treinamdo a rede neural sem necessidade, e já salva os pesos, para que possa implementar em outra maquina sem a necessidade de treinar a rede neural tudo de novo
