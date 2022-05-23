# em geral as principais aplicações envolvidas com esse tipo de rede neural, utilizam essa técnica ou essa arquitetura de rede neural

## frases pequenas como "as nuvens estão no (céu)", céu sendo a palavras que a RNN vai completar ou prever neste caso uma rede neural recorrente simple, é fácil fazer uma previsão.

### o espaço entre as palavras é pequeno, pode usar a informação passada

### mas num caso onde a frase é mais longa como "Eu sou do brasil... (eu falo português)" precisa ter o contexto Brasil é difícil para uma RNN 'simples' aprender esse padrão, por isso existem as LSTM

### Adiciona células de memória na rede neural e manipula asseas células

### Aprende dependências de longo prazo

- Forget gate: liberar da memória
- Input gate: adicinar na memória
- Output gate: ler da memória

### se utiliza a função de ativação a tangente hiperbólica(tanh) numa RNN 'simple', já numa LSTM, utiliza-se um conjunto de parametros, tendo o sigmoid e a tanh, fora as operações de apagar e adicionar, que evita os problemas no gradiente

### O que acontece dentro de um neurônio de uma LSTM:

1. primeiro processo é decidir o uqe será apagado
    o output do tempo anterior mais o valor atual, são submetidos à função sigmoid,
    cujo retorna valores entre 0 e 1, se o valor for 0 o dado não é importante e é apagado da memória
    - pensando  que esteja trabalhando com um pedaço de texte onde a RNN precisa completar, 
    qual a próxima palavra que a vai indicar
    - A memória pode incluir o gênero da pessoa para que os pronomes corretos possam ser usados.
        quando a rede encontra uma nova pessoa, pode apagar o gênero da pessoa anterior

2. segundo processo é decidir o que será armazenado
    vai verificar quais são os valores que serão alterados e usa a função tanh e cria um vetor dos 
    novos candidatos.
    se achou varias pessoas no texto, ele vai criar um vetor com as pessoas para 
    identificar qual o tipo de pronome vai utilizar.
    a função tanh retorna valores entre -1 e 1
    vai adicionar o novo gênero da pessoa no lugar daquele que foi apagado anteriormente.

3. terceiro processo é atualizar o estado antigo
    as etapas anteriores decidiram o que apagar e o que armagenar, e agora essas etapas são executadas

4. quarto processo é decidir qual será a saída
    aplicamaos a função sigmoid e também aplicamos a tanh que retorna valores entre -1 e 1,
    para retornar apenas as partes necessárias.(é como se fosse uma parte de filtro,
     onde efetivamente dirá o que é importande para ser adicionado e
      então propagar para o próximo estado, o output do neurônio)
      no exemplo do texto, se ele encontrou uma pessoa/objeto,
      pode retornar se éno plural ou singular, ou se é ele ou ela.


[fonte](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)