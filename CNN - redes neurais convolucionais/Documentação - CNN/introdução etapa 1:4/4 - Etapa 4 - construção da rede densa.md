# o vetor do flattening vai representar a camada de entrada 

## é uma construção de rede neural comum, como ja visto nas anteriores.

## com basa na ultima camada, a rede neural vai aprender com base nos valores altos de cada neuronio, vai se referir a um determinado resultado

 ### imagem -> kernal(detectores de caracteristicas) -> função de ativação(relu)-> pooling(principais caracteristicas) -> flattening(trnasforma matrz em vetor) -> rede neural densa

- Treinamento com a descida do gradiente
- Além do ajuste dos pesos, é feito também a mudança do detector de características

 ## ele vai fazer todo o processamento, vai coletar o valor do erro, depois que executa a rede neural densa, e volta para os pré-processamentos, escolhento outros mapas de caracteristicas(feature maps), e vai fazendo os testes


## As etapas de uma rede neural convolucional são respectivamente:

- operador de convolução, pooling, flattening e rede neural densa
- Uma rede neural convolucional procura identificar quais são as características mais importantes das imagens utilizando detectores de características

     1. O operador de convolução reduz a dimensionalidade da imagem com o intuito de capturar
          as características mais importantes

     2. Os detectores de características mais relevantes para tarefas de classificação de imagens
          são aqueles que realçam as bordas dos objetos

     3. A camada de convolução de uma rede neural convolucional é composta por vários mapas de 
     características

     4. A utilização da função max pooling é preferível do que usar funções de média ou mínimo
          porque ela realça as características mais importantes dos objetos

     5. O treinamento em uma rede neural convolucional é realizado pelo algoritmo de descida
          do gradiente e além de atualizar os pesos da rede neural densa, é também necessário 
          atualizar o mapa de características

