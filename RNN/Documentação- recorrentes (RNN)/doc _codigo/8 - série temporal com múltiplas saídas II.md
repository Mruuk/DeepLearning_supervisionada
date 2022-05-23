# série temporal com múltiplas saídas II

## importa base teste

### criamos e alocamos open e high

```python
        base_teste =pd.read_csv('recursos/petr4_teste.csv')
        preco_real_open = base_teste.iloc[:, 1:2].values
        preco_real_high = base_teste.iloc[:, 2:3].values

        base_completa = pd.concat((base['Open'], base_teste['Open']), axis = 0)
        entradas = base_completa[len(base_completa) - len(base_teste) - 90:].values
        entradas = entradas.reshape(-1, 1)
        entradas = normalizador.transform(entradas)

        X_teste = []

        for i in range(90, 112):
            X_teste.append(entradas[i-90:i, 0])
        X_teste = np.array(X_teste)
        X_teste = np.reshape(X_teste, (X_teste.shape[0], X_teste.shape[1], 1))
```

### o predict já cria a estrutura de matriz, colocando os respectivos valores de previsão, caso queira trabalha com mais valores de saídas, ele vai colocando uma colunas para cada valor que queira fazer a previsão

```python
        previsoes = regressor.predict(X_teste)
        previsoes = normalizador.inverse_transform(previsoes)
```

### criando plot com os parametros

```python
        plt.plot(preco_real_open, color = 'red', label = 'Preço abertura real')
        plt.plot(preco_real_high, color = 'black', label = 'Preço alta real')

        plt.plot(previsoes[:, 0], color= 'blue', label = 'Previsões abertura')
        plt.plot(previsoes[:, 1], color= 'orange', label = 'Previsões alta')

        plt.title('Previsão preço das ações')
        plt.xlabel('Tempo')
        plt.ylabel('Valor Yahoo')
        plt.legend()
        plt.show
```
