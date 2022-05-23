# Serie temporal com múltiplos previsores II

## criamos nossa base de dados teste e o preço real também

```python
        base_teste =pd.read_csv('recursos/petr4_teste.csv')
        preco_real_teste = base_teste.iloc[:, 1:2].values # atributo Open apenas
```

### a concatenação era feita assim

- base_completa = pd.concat((base['Open'], base_teste['Open']), axis = 0)

### só que como temos mais atributos, não será possível rodar o código, faremos um código equivalente

- variavel frame recebe a base e a base teste, na variavem frame, teremos 2 dataframes usando o recurso do panda vamos agora concatenar

```python
        frames = [base, base_teste] # equivalente
        base_completa = pd.concat(frames) # concat
        base_completa = base_completa.drop('Date', axis = 1) # drop da coluna date
```

### não houve mudanças aqui

```python
        entradas = base_completa[len(base_completa) - len(base_teste) - 90:].values
```

### entradas = entradas.rechape(-1,1), não é necessário, pois estamos trabalhando com dimenções maiores

```python
        entradas = normalizador.transform(entradas)

        X_teste = []
```

### modificamos apenas o indice 0 para 0:6, pois estamos pegando todos os atributos

```python
        for i in range(90,112):
            X_teste.append(entradas[i-90:i, 0:6])
        X_teste = np.array(X_teste)
```

### X_teste = np.reshape(X_teste, (X_teste.shape[0], X_teste.shape[1], 1)), não é necessário redimencionar para esse formato pois ele já se encontra nele (22, 90, 6), quantidade de registros, quantidade do timesteps e quantidade de atributos previsores

```python
        normalizador_previsao = MinMaxScaler(feature_range=(0,1))
        normalizador_previsao.fit_transform(base_treinamento[:, 0:1])
```

### a mudança de escala não funcionará com o normalizador.inverse_transform(previsores), o normalizador, é necessário fazer um outro devido ao shape, ele foi feito a normalização na base_treinamento, que tem caracteristicas diferentes dos previsores, então necessário fazer outro normalizador

```python
        previsores = regressor.predict(X_teste)
        previsores = normalizador_previsao.inverse_transform(previsores)

        plt.plot(preco_real_teste, color = 'red', label = 'Preço real')
        plt.plot(previsores, color = 'blue', label = 'Previsões')
        plt.title('Previsão preço das ações')
        plt.xlabel('Tempo')
        plt.ylabel('Valor Yahoo')
        plt.legend()
        plt.show

        plt.savefig('petr4_multiplos.png')
```
