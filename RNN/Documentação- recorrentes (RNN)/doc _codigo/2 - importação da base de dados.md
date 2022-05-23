# importando base de dados

## Normalizando valores com o minmaxscaler

```python
from keras.models import Sequential
from keras.layers import Dense, Dropout
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
