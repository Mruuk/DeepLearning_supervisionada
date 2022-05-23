# Série temporal com múltiplas saídas I

## previsão de múltiplas saídas, ou seja, ao invés de prever somente o valor de abertura, vamos prever também o valor da alta da bolsa, qual foi o maior valor negociado em determinado dia, entao, isso pode ser bastante útil, caso queira saber o teto máximo de uma ação, quanto que uma ação vai custar no máximo em um determinado dia

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

### pegamos apenas o Open

```python
        base_treinamento = base.iloc[:, 1:2].values
```

### aqui pegamos o High

```python
        base_valor_maximo = base.iloc[:, 2:3].values
```

### normalizamos as bases

```python
        normalizador = MinMaxScaler(feature_range=(0,1))
        base_treinamento_normalizada = normalizador.fit_transform(base_treinamento)
        base_valor_maximo_normalizada = normalizador.fit_transform(base_valor_maximo)
```

### criamos os vetores, dois vetores para os preços, sendo eles o open e o high

```python
        previsores = []
        preco_real1 = []
        preco_real2 = []
```

### colocamos os dois preços e transformamos em array, precisamos no reshape pois temos apenas 1 atributo previsor

```python
        for i in range(90, 1242):
            previsores.append(base_treinamento_normalizada[i-90:i, 0])
            preco_real1.append(base_treinamento_normalizada[i, 0])
            preco_real2.append(base_valor_maximo_normalizada[i, 0])
        previsores, preco_real1, preco_real2 = np.array(previsores), np.array(preco_real1), np.array(preco_real2)
        previsores = np.reshape(previsores, (previsores.shape[0], previsores.shape[1], 1))
```

### precisamos unir os valores preços, pois temos 2 vetores e se mantivermos assim teremos erro, quando for passar essa variavel como parâmetro, temos 2 coluna, open e high

```python
        preco_real = np.column_stack((preco_real1, preco_real2))

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

# duas saidas, 2 neurônios
        regressor.add(Dense(units= 2, activation= 'linear'))

        regressor.compile(optimizer = 'rmsprop', loss= 'mean_squared_error',
                        metrics= ['mean_absolute_error'])

        es = EarlyStopping(monitor= 'loss', min_delta= 1e-10, patience= 10, verbose= 1)
        rlr = ReduceLROnPlateau(monitor= 'loss', factor= 0.2, patience= 5, verbose = 1)
        mcp = ModelCheckpoint(filepath= 'pesos.h5', monitor= 'loss',
                            save_best_only=True, verbose = 1)

        regressor.fit(previsores, preco_real, epochs = 100, batch_size = 32,
                    callbacks = [es, rlr, mcp])
```
