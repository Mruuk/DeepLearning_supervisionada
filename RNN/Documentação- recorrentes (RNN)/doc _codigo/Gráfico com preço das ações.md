# Gráfico com preço das ações

## vamos fazer um gráfico para comparar os resultados das previsoes da rede neural LSTM com os preços reais, para gerar esse gráfico, necessário a utilização da biblioteca matplotlib

- plt.plot, serão os parametros que geram as linhas de analise do gráfico
- plt.title seŕa o título
- plt.xlabel será a nomenclatura do x
- plt.ylabel será a nomenclatura do y
- plt.legend será a legenda
- plt.show irá exibir o gráfico

```python
        import matplotlib.pyplot as pl           
        plt.plot(preco_real_teste, color = 'red', label = 'Preço real')
        plt.plot(previsoes, color= 'blue', label = 'Previsões')
        plt.title('Previsão preço das ações')
        plt.xlabel('Tempo')
        plt.ylabel('Valor Yahoo')
        plt.legend()
        plt.show

```

### é bem robusto o modelo, possibilitando trabalhar com essa tarefa relativamente complexa, também pode observar que segue práticamente uma análise de tendência, não é apenas prever o preço da ação, se prever também a tendência de um preço de uma determinada empresa
