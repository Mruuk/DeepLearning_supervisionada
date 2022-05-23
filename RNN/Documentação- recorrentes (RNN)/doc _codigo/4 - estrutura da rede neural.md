# Estrutura da rede neural

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 20:05:21 2022

@author: lisboa
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

# importa base de dados
base = pd.read_csv('recursos/petr4_treinamento.csv')
# retira atributos sem valor(nan)
base = base.dropna()
# seleciona o que será alvo de previsão
base_treinamento = base.iloc[:, 1:2].values

# normalização dos valores utilizando o método(fuction) minmaxscaler
# irá transformar os valores nessa escala defenida na função
normalizador = MinMaxScaler(feature_range=(0,1))
base_treinamento_normalizada = normalizador.fit_transform(base_treinamento)
```

## faz uma codificação para efetivamente contruir a base de dados, será usado 90 dias anteriores em nosso intervalo temporal, começamos do registro 90 e vamos até o final

- criamos vetores vaziou
- um for que vai percorrer de 90 ao maximo da base de dados, nesse caso 1242
- adicionamos ao vetor previsor os atributos da base de dados ja normalizada, o intervalo temporal, dos 90 dias anteriores, por isso i-90, será igual a 0, neste caso, e vai até o valor de i, deste jeito sempre mantendo um intervalo de 90.
- a coluna 0, só teremos ela mesmo, pois estamos visualizando apenas o valores de open da tabela
- preco_real será adicionado o valor de i e a coluna 0, o preco_real é o valor que deve ser previsto

```python

previsores = []
preco_real = []
for i in range(90, 1242):
    previsores.append(base_treinamento_normalizada[i-90:i, 0])
    preco_real.append(base_treinamento_normalizada[i, 0])

# transformando em vetor    
previsores, preco_real = np.array(previsores), np.array(preco_real)
```

- formato do input shape que o keras necessita para redes recorentes:
  - batch_size-: quantidade de registros
  - timesteps-: intervalo de tempo
  - input_dim-: quantos atributos previsores vamos utilizar,
    - também chamados de indicador, quando trabalhamos com série temporal
    - chamamos esses atributos previsores de indicadores
- podemos fazer assim também:
  - previsores = np.reshape(previsores, (1242, 90,1))
  - pois sabemos os valores

### estrutura da rede neural

#### primeiramente a camada de LSTM, units, ele é que vale ao número de células de memória, deve ser um número grande, para adicionar mais dimencionalidade e capturar a tendência no decorrer do tempo se colucar um valor muito baixo, como 5, ou 6, ele não vai conseguir capturar a variação temporal

#### return_sequences, se utiliza somente quando tem mais uma camada LSTM, isso indica que ele vai passar a informação pra frente, para as outras camada subsequentes

#### input_shape, como que estaram os dados de entrada, um dropout de 30%, evitando o overfitting, quanto se trabalha com esse tipo de rede neural, é interessante adicionar mais camadas, se tiver poucas camadas ele tem a tendência de não dar resultados muito bons, nas camadas subsequentes, pode reduzir o numero de neoronios(células de memória), geralmente em alguns casos pode colocar valores mais altos para as primeiras e pode diminuir a camada de saida será uma dense, todos os neoronios da ultima camada oculta estaram ligadas ao neoronio com a resposta final, apenas uma saida, pois faremos a previsão apenas do valor open

#### activation function poder ser linear, pois ele apenas passarar o valor sem alteração, sem aplicar função de ativação, neste caso pode testar a função sigmoid, pois como os dados estao normalizados entre 0 e 1, e a função sigmoid também retorna valores entre 0 e 1. Compile, optimizer, o rmsprop é um bom optimizador para reder neurais recorrentes, porém se utilizar o adam também terá resultados semelhantes, loss será o mean_squared_error é um calculo de erro mais eficiente que o mean_absolute_error, esta sendo usado no calculo do erro de ajustes de pesos, e trabalhando com o parametro metrics, é apenas para visualizar os resultados, como ele tira o absoluto e a média, fica mais fácil de entender os resultados

#### fit o previsores já esta no formato adequando 3 dim, epocas, é importante rodar pelo menos umas 100 epocas, menos que isso, ele tem uma tendência de não se adaptar muito bem aos dados, batch_size 32 a ultima camada oculta do LSTM não pode ter o parametro de return_sequences, pois se coloca só quando tem mais camadas subsequentes, o mean absolute erro mostra a diferença entre as previsoes e o preço real

```python

previsores = np.reshape(previsores, (previsores.shape[0], previsores.shape[1],1))

regressor = Sequential()
regressor.add(LSTM(units= 100, return_sequences = True,
                   input_shape= (previsores.shape[1], 1)))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units= 50, return_sequences= True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units= 50, return_sequences= True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units= 50))
regressor.add(Dropout(0.3))

regressor.add(Dense(units= 1, activation= 'linear'))

regressor.compile(optimizer = 'rmsprop', loss= 'mean_squared_error',
                  metrics= ['mean_absolute_error'])
regressor.fit(previsores, preco_real, epochs = 100, batch_size= 32)

```

`output:
    Epoch 100/100
    36/36 [==============================] - 16s 452ms/step - loss: 0.0016 -
    mean_absolute_error: 0.0300`
