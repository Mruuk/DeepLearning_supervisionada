# Representação da rede neural recorente (RNN)

## Espremer as camadas e cada camada RN será somente um úninco nó, mas cada nó representa uma camada que pode conter cários neurônios. A camada escodida é representada por somente um único neurônio que representa n neurônios

### Camada escondida é conectada a ela mesma, mandando uma resposta para frente e também se realimentando

### Backpropagation para treinar, desenrolando a rede neural para termos uma rede padrão

### Os pesos são compartilhados, a memória é rápida, curta e lembra o que aconteceu apenas nas últimas interações

### A auto ligação é o temporal no loop

## Grandient vanish problem:

- milhares de camadas
- os pesos são multiplicados várias vezes e o resultado seŕa cada vez menor que não fará alteração

- topicos de problemas e suas soluções:

  - vanish:
    - Backpropagation through time (BPTT):
     - algoritmo semelhante que também fluirá para trás a partir do tempo futuro para os tempos atuais

    - Xavier initialization : inicialização dos pesos:
      - glorot_normal um initializer do keras
      - LSTM(long short term memory): um tipo especial de rede neural recorrente
    - Exploding gradient: o valor poderá ficar muito grande(ficando muito grande ele nunca atingirar o erro mínimo global)
      - Não visitar todas as camadas ocultas(não é recomendado, pode comprometer a aprendizagem)
      -  RMSProp: divide a taxa de aprendizagem por uma média exponencialmente decrescente do quadrado do gradiente(um optimizer assim como o SGD(Stochastic Gradient Descent), ou o adam)
            
      - Clipping gradient(grampo): impedi que ele suba do valor que foi clipado, forçando uma descida



