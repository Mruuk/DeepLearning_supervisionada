# Previsão de preço de ações

## para fazer as previsões vamos utilizar petr4_teste.csv, a base de dados de janeiro carregamos ela teremos 22 registros com 7 campos, porém queremos prever apenas o campo 'open'

```python
base_teste = pd.read_csv('recursos/petr4_teste.csv')
```

### para isso precisamos extrair apenas esse valor, vamos utilizar o iloc para pegar apenas a coluna open

```python

preco_real_teste = base_teste.iloc[:, 1:2].values
```

### porém não podemos submeter para a rede neural esses dados, dessa forma que eles estão, porque o modelo foi treinado utilizando 90 preços anteriores, e por isso vamos precisar dos 90 preços anteriores de cada um dos registros do preco_real_teste

### concatena a base original com a base teste, por colunas, o pd.concat serve para concatenar duas bases de dados, passamos as duas bases e depois determinamos se vai ser concatenado em linhas ou colunas, axis = 0  a concatenação é por coluna

```python
base_completa = pd.concat((base['Open'], base_teste['Open']), axis = 0)
```

- lower bound
- len(base_completa) = 1264
- len(base_teste) = 22

### com a concatenação das bases, os 22 registros que correspondem a janeiro, estão no final da base_completa, o primeiro len vai retornar a ultima ação de janeiro, quanto faz a subtração para len(base_teste), vai ta retornando a ultima ação de dezembro, e o -90, ele vai retornar as ultimas ações que elas devem ser buscadas

### isso quer dizer, que irá começar a buscar os registros da posição 1152 da base_completa, sendo elas as ultimas 90 que estão relacionadas ao mês de janeiro, values para já deixar no formato do numpy array

```python
entradas = base_completa[len(base_completa) - len(base_teste) - 90:].values
```

### um reshape, onde (-1, 1) indica que não vai trabalhar com as linha e que terá uma coluna, deixando assim no formato do numpy

```python
entradas = entradas.reshape(-1, 1)
```

### normalizamos, para deixar os dados na mesma escala

#### obs: .fit_transform = que dizer q ele vai se encaixar nos dados parametros passados

#### transform =  que dizer que ele vai normalizar de acordo com o que ja foi normalizado

#### aqui:>  normalizador = MinMaxScaler(feature_range=(0,1))

#### base_treinamento_normalizada = normalizador.fit_transform(base_treinamento)

### criamos um vetor

```python
entradas = normalizador.transform(entradas)

X_teste = []

```

#### um for com range de 90 ate 112, o motivo dos valorees é porque, 90 + 22 = 112, sendo 22 o número de registros que queremos fazer a previsão, a linha de entradas(variavel vetor) estão nessa dimensão faremos o mesmo append que foi feito para previsores, conversão para array, transformamos o formato como foi feito nos previsores, já podemos fazer um predict, para visualizar desnormalizamos com normalizador.inverse_transform

```python
for i in range(90, 112):
    X_teste.append(entradas[i-90:i, 0])
X_teste = np.array(X_teste)
X_teste = np.reshape(X_teste, (X_teste.shape[0], X_teste.shape[1], 1))
previsoes = regressor.predict(X_teste)
previsoes = normalizador.inverse_transform(previsoes)
```

#### para analizar os valores, fazemos uma média das previsões e dos preços reais

```python
previsoes.mean()
preco_real_teste.mean()
# a diferença
abs(previsoes.mean() - preco_real_teste.mean())

```
