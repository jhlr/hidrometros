# Respostas do Mikael, conferidas

Base: texto do Mikael (2026-10-14), com a voz dele mantida. Mudei só o que estava errado, incompleto ou exagerado, conferido contra `results.csv`, `README.md` e `docs/`.

**1.1** O `data.csv` manda em tudo: liga a foto ao valor e à posição do mostrador (`location`, um polígono com x e y de 0 a 1). Pra treinar, usamos só a pasta `images/` e esse polígono. As máscaras (`masks/`) serviram só como tira-teima do recorte (IoU mediana de 0,983), e as colagens ficaram fora do treino.

**1.2** O rótulo é da foto inteira, não de cada número, então a gente tem que fatiar o mostrador em 8 e dar a cada fatia o dígito da posição dela. O perigo disso? Se o recorte sair torto ou cair no meio de duas rodas, o modelo vai tentar aprender um número que não está naquela fatia, sujando o treino com lixo.

**2.1** A fonte mais confiável é a coluna `value` do CSV. Padronizamos tudo para 5 dígitos inteiros e 3 decimais, completando com zeros (à esquerda nos inteiros e à direita nos decimais, porque o `value` perde os zeros do final: 21,86 é 21,860). Isso funciona porque bate com os nomes dos arquivos (1244 de 1244) e com a inspeção visual que fizemos.

**2.2** O dígito "0" domina geral, com 35,7% da base, porque a gente encheu a parte inteira de zeros à esquerda. No teste ele é 34,4% dos recortes, então um modelo inútil que chutasse "0" pra tudo já garantiria uns 34% de acerto. Essa é a nossa nota de corte.

**3.1** A divisão de treino e teste tem que ser feita pela foto original, nunca pelo recorte. Se fatiar antes, pedaços do mesmo hidrômetro vazam pro treino e pro teste, fazendo a rede "decorar" o fundo da foto em vez de aprender a ler o número. Todos os grupos usam o mesmo `splits.csv` pra que os resultados sejam comparáveis e ninguém ajuste a divisão a favor do próprio modelo.

**3.2** A gente endireitou 136 fotos tortas (84 verticais e 52 inclinadas) e desvirou 56 de cabeça para baixo (36 pela regra do vermelho e 20 pelo classificador de orientação). Só que o recorte automático continuou falhando de três jeitos: (a) fatiou errado os mostradores de 5 rodas, em que o rótulo tem 3 zeros decimais que não existem na imagem (36 imagens têm decimais `000`, e em 22 de 24 que olhamos o mostrador tem só 5 rodas); (b) cortou as bordas dos números: com a grade igual, 5450 dos 9944 recortes têm tinta na borda esquerda, 5289 na direita e 4238 nas duas (indicador aproximado); (c) se perdeu em engrenagens girando no meio da transição, principalmente na última casa: no teste, a posição 7 tem 17 dos 45 erros da ResNet18.

**3.3** Cortar a imagem numa grade de fatias iguais dá ruim se o medidor tiver menos rodas ou se a foto sair desalinhada. A ideia de corrigir foi achar a divisão de cada roda pela variação de intensidade dos pixels, em vez da grade fixa. Testamos isso no desafio extra, mas não funcionou: a acurácia por dígito caiu nos quatro modelos (por exemplo, 0,0133 na ResNet18). E nos mostradores de 5 rodas o recorte não resolve, porque o rótulo continua com 3 zeros que não existem; aí teria que detectar o número de rodas e ajustar o rótulo.

**4.1** Normalizamos as cores com o padrão da ImageNet pra os modelos pré-treinados não estranharem as fotos, já que foram treinados com entradas nessa escala. Aplicamos isso também na nossa rede menor (smallcnn) só pra manter a regra do jogo igual pra todo mundo, não porque ela dependesse disso.

**4.2** A diferença de tamanho é grande: a smallcnn tem 0,34 M de parâmetros, a ResNet18 11,18 M, a AlexNet 57,04 M e a VGG16 134,31 M. Na AlexNet, 95,7% dos parâmetros estão nas camadas densas do final (`classifier`), e na VGG16 são 89,0%. A ResNet18 evita isso usando uma média global: a cabeça dela tem só 5130 parâmetros, e o peso está nas convoluções.

**4.3** Treinar só a cabeça mexe apenas na última camada; o fine-tuning completo ajusta a rede inteira. Na primeira época a gente "congela" o corpo da rede pra que o classificador inicial (que começa aleatório) não destrua os filtros visuais que vieram da ImageNet. Depois destrava tudo e treina o modelo inteiro devagar (taxa 1e-4 no corpo, ou 1e-5 na VGG16).

**4.4** Espelhar a imagem (flip) aqui é tiro no pé, porque um 2, um 3 ou um 5 espelhado não existe num hidrômetro, e a rede aprenderia formas que nunca aparecem. O aumento de dados foi só girar um pouco (até 5°), mexer um pouco de posição e escala e mudar brilho e contraste, pra simular os defeitos de uma câmera de celular.

**5.1** A ResNet18 teve o melhor F1-macro (0,9654) e a maior acurácia por dígito (0,9699), seguida de VGG16 (0,9455 e 0,9532), AlexNet (0,9261 e 0,9412) e smallcnn (0,7711 e 0,8155). O ranking é o mesmo nas duas métricas, e nenhum dos três pré-treinados sacrifica uma classe pra inflar a nota nas fáceis. A diferença entre AlexNet e VGG16 é pequena e vem de uma execução só, então a ordem entre elas pede cautela.

**5.2** A ResNet18 quase não erra e tropeça em poucos pares: (0→1) 4 vezes, (1→7) 3, (2→1) 3 e (6→4) 3, traços parecidos (1 e 7) ou o 1 e o 0, que são os dígitos mais frequentes. Parte pode ser engrenagem rodando, mas não classificamos os erros um a um. A VGG16 erra mais (1→7) e (1→0), 6 vezes cada. A smallcnn erra mais e puxa pra classe mais frequente: prevê "0" em 36,8% dos recortes de teste (os zeros são 34,4%), e os pares que mais erra são (6→0), 13 vezes, e (1→0), 12.

**5.3** Acertar a fatura inteira é muito mais difícil que acertar um dígito solto. Como a leitura tem 8 números, errar um só já invalida a foto toda. É por isso que a ResNet tem 97% de acerto por dígito, mas cai pra 85,6% na leitura completa. Na smallcnn é pior: 81,6% por dígito vira 31,0% por leitura.

**5.4** Pra rodar num celular, a ResNet18 é a melhor opção: acerta 0,9699 por dígito e 0,8556 a leitura inteira, pesa só 44,8 MB e leva 6,42 ms por imagem. A VGG16 tem 537,3 MB e 20,80 ms, pesada demais pro aparelho. A smallcnn é leve (1,4 MB), mas erra uns 69% das leituras completas, o que seria inútil pro negócio. Esses tempos foram medidos no nosso Mac (MPS), não num celular, então valem pra comparar os modelos entre si.

**5.5** A ResNet18 ganhou da VGG16 em tudo: acurácia por dígito 0,9699 contra 0,9532, e leitura completa 0,8556 contra 0,7701. E ainda gasta 12 vezes menos memória (44,8 MB contra 537,3 MB), faz a inferência 3,2 vezes mais rápida (6,42 ms contra 20,80 ms) e treina uns 8 vezes mais rápido por época. Ela usa conexões residuais e não tem as camadas densas gigantes da VGG16.

**5.6** A smallcnn tinha um bug de arquitetura: o pooling global da cabeça esmagava os pixels e deixava a rede cega sobre onde estavam os traços do número (acurácia 0,2246). Corrigimos trocando por um Flatten e ela voltou a aprender (0,8155), mas ficou uns 15 pontos abaixo da ResNet18 e ainda melhorava na época 10, então 10 épocas limitam ela. Isso mostra que transferir inteligência pré-treinada ajuda bastante, mas é resultado de uma execução só.
