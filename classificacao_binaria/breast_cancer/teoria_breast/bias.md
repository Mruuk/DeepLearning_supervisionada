# neoronio adicional = bias

## valores diferentes mesmo se todas as entradas forem zero muda a saída com a unidade de bias

### já vem implementado nas bibliotecas

### ERRO

    algoritmo mais simples
        erro = respostaCorreta - resposta Calculada
        não é utilizada, falta robusteis

    
    Mean square error(MSE) e Root mean squaer error(RMSE)

    Descida do gradiente

        Batch gradient descent (BGD)
            Calcula o erro para todos os registros e atualiza os pesos

        Stochastic gradient descent (SGD)
            Calcula o erro para cada registro e atualiza os pesos 
            
            Ajuda a previnir mínimos locais(superfícies não convexas)
            Mais rápido (não precisa carregar todos os dados em memória)

        Mini bacth gradient descent
            Escolhe um número de registros para rodar e atualizar os pesos

        Parâmetros:
            Learning rate(tava de aprendizageem)
            Bacth size (tamanho de lote)
            Epochs(épocas)
