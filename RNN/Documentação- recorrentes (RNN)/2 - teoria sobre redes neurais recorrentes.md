# problemas com dados sequenciais(intervalos de tempo)
   ### ex:  eventos que ocorrem de hora em hora, e preciso fazer uma previsão de um evento da hora subsequente

- prever a próxima ação(frames anteriores podem prever o frame atual, trabalhando com video)
- essas redes são muito utilizada em processamento de liguagem natural(NLP)
  - previsão da próxima palavra em um texto
  - tradução automática(speak to text)
  - geração de poemas ou notícias
- geração de legendas em vídeos
- séries temporais (time series)
  - redes neurais recorrentes, são muito especificas para resolver estes problemas:
    - Preço de ação na bolsa de valores
    - Temperatura
    - Crescimento populacional
    - Nível de poluição
- outras áreas relevantes
  - Descrição de imagens        

- para entender o final de uma frase você precisa saber o que foi dito anntes
    - uma premissa base sobre a rede neural
    - essa frase expressa muito bem como que uma rede neural recorrente funciona
    - com base em eventos passados ela prever futuros

- Redes neurais tradicionais não armazenam informações no tempo(previsões independentes)    
    - as informações em uma rede neural tradicional como uma densa, ela apenas calcula o peso para aquele neurônio e passa a informação adiante, sem armazenar
- as redes recorrentes são redes neurais com loops que permitem que a informação persista ao decorrer do tempo       
- múltiplas cópias de si mesmas por sua definição será uma rede pesada, devido aos loops