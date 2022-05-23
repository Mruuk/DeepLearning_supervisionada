# Estrutura da base para previsão temporal 1

## nós queremos prever valor de uma ação baseado nos dados anteriores, por tanto não iremos passar como parametro a base de dados original, é necessário criar uma estrutura de dados diferente, para conseguirmos fazer a aplicação das redes neurais recorrentes.

## para se trabalhar com uma série  temporal, a primeira coisa que precisa fazer, é definir o intervalo de tempo

### tabela

dia | dia nº | valor
------- | ------- | -------
quinta | 03 | 19,99
sexta | 04 | 19,80
segunda | 07 | 20,33
terça | 08 | 20,48
quarta | 09 | 20,11
quinta | 10 | 19,63
sexta | 11 | 19,77
segunda | 14 | 19,85

### supondo que queremos fazer uma previsão da ação de um determinado dia, com base não ações dos ultimos 4 dias

- para isso vai precisar criar uma estrutura para armazenar esses dados.

- previsores e preço_Real, são atributos previsores e a classe, como estamo trabalhando com uma regressão, queremos prever um valor.

- trabalhando num contexto de 4 dias:

  - seguindo a tabela acima como parametros, perceba que não podemos prever valores dos dias, 03, 04, 07 e 08, devido ao fato de não termos as 4 ações anteriores para serem analizadas, a previsão só pode ser feita a partir do dia 09.

  - poderia também fazer uma previsão com auma análise de apenas 1 ação anterior, porém não faz muito sentido, o algoritmo não iria se adaptar.

#### obs: ainda é um problema de aprendizagem supervisionada

- a utilização de 4 registros ainda é muito pouco, para o algoritmo se adaptar, foi utilizado apenas para exemplificação


previsores | preço_real
--------------------------
19,99 19,80 20,33 20,48 | 20,11
19,80 20,33 20,48 20,11 | 19,63
20,33 20,48 20,11 19,63 | 19,77
20,48 20,11 19,63 19,77 | 19,85