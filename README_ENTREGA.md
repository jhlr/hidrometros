# [Nome do grupo] — Classificação de Dígitos de Hidrômetros com AlexNet, VGG16 e ResNet

> **Modelo de README.** Copie este arquivo para o `README.md` do repositório do grupo e preencha
> todas as seções. Apague as linhas de instrução (as citações iniciadas por `>`) antes de entregar.
> A entrega é **obrigatoriamente o link deste repositório GitHub**, enviado no Google Classroom.

**Disciplina:** Visão Computacional (VC001) — CESAR School, 2026.2
**Integrantes:** Nome 1 (e-mail) · Nome 2 (e-mail) · Nome 3 (e-mail)

## Fonte dos dados e licença

Dataset: [Yandex.Toloka Water Meters](https://www.kaggle.com/datasets/tapakah68/yandextoloka-water-meters-dataset),
licença **CC BY-NC-ND 4.0**. Este repositório **não** contém imagens do dataset, recortes gerados nem checkpoints.
O uso é restrito a fins acadêmicos e não comerciais.

## Como reproduzir

> Descreva os passos exatos, do zero. Exemplo:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# colocar o kaggle.json em ~/.kaggle/ (chmod 600)
jupyter notebook pratica_watermeters.ipynb   # executar todas as células
```

- Ambiente usado (Colab/local, GPU/MPS/CPU, modelo da GPU):
- Tempo total de execução:
- Split utilizado: `splits.csv` **oficial** fornecido pelo professor (não alterado)

## Protocolo de treino

> Copie a tabela de protocolo do roteiro (README da prática) e registre qualquer alteração. Toda alteração deve valer para os quatro modelos.

| Modelo | Pesos | Entrada | Batch | LR cabeça | LR backbone | Épocas máx. | Melhor época |
|---|---|---|---|---|---|---|---|
| smallcnn | do zero | 64×48 | | | — | | |
| alexnet | ImageNet | | | | | | |
| vgg16_bn | ImageNet | | | | | | |
| resnet18 | ImageNet | | | | | | |

Alterações em relação ao protocolo oficial (e justificativa):

## Resultados (conjunto de teste)

> Os números abaixo devem ser **idênticos** aos do `results.csv` versionado. Divergência gera penalidade de −20%.

| model | acc_digit | f1_macro | acc_reading | params_M | s_per_epoch | ms_per_image | ckpt_MB | best_epoch |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

Figuras (pasta `figures/`):

- Matrizes de confusão normalizadas: `figures/confusion_matrices.png`
- Acurácia por posição: `figures/accuracy_by_position.png`
- Curvas de treino: `figures/training_curves.png`
- Desempenho × custo: `figures/performance_vs_cost.png`

## Respostas às questões

> Responda com base nos **seus** números. Respostas genéricas, sem referência aos resultados, são avaliadas como insuficientes.

**1.1** ...
**1.2** ...
**2.1** ...
**2.2** ...
**3.1** ...
**3.2** Quantidade de faixas corrigidas: ___ . Tipos de recorte errado e quantidades: ...
**3.3** ...
**4.1** ...
**4.2** ...
**4.3** ...
**4.4** ...
**5.1** ...
**5.2** ...
**5.3** ...
**5.4** ...
**5.5** ...
**5.6** ...

## Limitações

> O que o pipeline faz de errado e por quê? Quanto do erro restante vocês atribuem aos dados (rótulo, recorte)
> e quanto ao modelo? O que vocês mudariam?

## Declaração de uso de IA

> Seguir o template disponivel no Google Classroom da disciplina.