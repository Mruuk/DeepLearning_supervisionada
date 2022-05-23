# rede neurais convolucionais(CNN) 

 - usando parar visão computacinal
 - carros autônomos, detecçõa de pedestres(umas das razões por deep learning funcionar bem)
 - Em geral, melhor do que SVM(support vector machines) : machine learning
    considerado algoritmos mais eficiente de apredizagem de maquina, por vários anos
    mas com a chegada do deep learning os CNN superaram 

Pixels - rgb

um rede neural convolucional não usa todas as entradas(pixels)

img = 32 x 32=> 10.24 pixels => 1024 x 3= 3072 pixels de entrada usando todos os pixels deixaria muito lenta
o processamento de dados da sua rede(o valor 3 refere-se ao rgb)
seleciona quais são as melhores caracteristica 

usa uma rede neural tradicional(densa), mas no começo transforma os dados na camada de entrada
um pré-processamento

as caracteriticas mais importantes é o que a rede CNN tem de descobrir.

caso use uma rede neural tradicional, terá que pasasr todos os pixels para serem processado,
porém usando um CNN, haverá um pré-processamento, para descobrir quais as caracteristicas
mais importantes para o reconhecimento da image

para faces: 
    localização do nariz
    distância entre os olhos
    localização da boca
*não precisa se precupar com esses parametros, qndo se trata de CNN 

diferença de uma face humana de um animal:
    a rede neural vai encontrar o melhor caminho
    ela vai descobrir efetivamente sobre isso

CNN - etapas:
    
    etapa 1 -: Operador de concolução
    etapa 2 -: Pooling
    etapa 3 -: Flattening
    etapa 4 -: Rede neural densa

    etapas 1:3 -: são pré-processamento das imagens
    etapa 4 -:construção da rede neural densa

    etapa 1 -: 
        Convolução é o processo de adicionar cada elemento da imagem para seus vizinhos,
        ponderado por um kernel
        ele faz calculos matemáticos, fazendo multiplicação, somas de matriz,
        para que consigamos fazer alteração de determinados pixels da imagem 

        a imagem é uma matriz e o kernel é outra matriz

        fórmula:

            (f*g)[n]= SOMATÓRIO indo de m= infinito negativo ao positivo( f[m]g[n-m])
            f[n-m]g[m]
### links:
- [explicação sobre os kernels](https://en.wikipedia.org/wiki/Kernel_(image_processing))
- [exemplo on-line](http://setosa.io/ev/image-kernels/)
            
        identify -: é a image original
        edge detection -: a mais importante em CNN, realça bastante as bordas,
                    elas são consideradas as principais características, quando trabalhamos com essa
                    parte de pré-processamento de uma imagem

        sharpen
        box blur
        gaussin blur 3x3
        gaussin blur 5x5
        unshrp masking

    a primeira etapa  é pegar a imagem original e aplicar desses tipos de kernel, para fazer,
    modificações na imagem

    com o mapa de características (filter map) a imagem fica menor para facilitar o processamento
    alguma informação sobre a imagem pode ser perdida, porém o propósito é detectar as partes 
    principais(quanto maior os números melhor)

    o processo de multiplicação é feito andando uma matriz menor que a matriz da imagem original,
    por toda matriz original indo de linha a linha e coluna a coluna, chamasse de janela, 
    por onde a matriz, feature detected

    o mapa de caracteristicas preserva as caracteristicas principais da imagem(olho, boca, nariz)

    aplicação da função relu -:
        
        depois que se tem a feature map(mapa de caracteristicas), aplica a função relu
        para cada célula da matriz.
        O objetivo disse é para retinar os valores negativos, a função relu
        vai de zero ao infinito positivo, portanto, todo valor negativo vai ser substituido por zero.
        ela facilita a detecção dos padrões, ajudando assim no processamento da imagem

    camada de Convolução -: 

        é a aplicação dos mapas de características(feature maps) ou detector de caracteristica