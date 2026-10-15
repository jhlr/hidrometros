# Declaração de Uso de IA

Template da disciplina (Visão Computacional · CESAR School · 2026.2), preenchido pelo grupo Luminus (Grupo 2). A cópia no formato original do template está em [`Declaracao_Uso_IA_Luminus_Grupo2.docx`](Declaracao_Uso_IA_Luminus_Grupo2.docx).

## Identificação da atividade

| | |
|---|---|
| Disciplina | Visão Computacional, VC001 |
| Atividade | Prática 2: Leitura de Hidrômetros com CNNs (repositório `github.com/jhlr/hidrometros`) |
| Integrantes | Angelita Dias de Santana, Carolline de Araújo Mariz, Fernando Cyreno Rangel da Silva, Guaraci de Barros Rios, Ivo Caetano Andrade Júnior, Mikael de Albuquerque Mulatinho, João Rietra |

## Seção A: uso de LLM / assistente de IA

**Utilizou IA?** Sim.

**Modelo(s) usado(s).** Claude Sonnet 5.5 (`claude-sonnet-5-5`), da Anthropic, pelo Claude Code (aplicativo desktop). Os agentes auxiliares (subagentes) rodaram dentro da mesma sessão e herdam o modelo da sessão por padrão; não registramos o modelo de cada subagente separadamente.

**Prompt(s) utilizado(s).** O prompt é o SDD (documento de especificação) completo, que está anexado a esta entrega: [`docs/SPEC.md`](../SPEC.md) (v1.3, 2026-10-13). Ele foi usado como prompt único de execução, entregue a um agente sem contexto prévio. Durante a sessão, o integrante João Rietra também deu instruções curtas de direção (por exemplo, "planeje antes! quero fazer em formato SDD" e "execute o spec num subagente") e respondeu perguntas de múltipla escolha sobre escopo.

**Resposta gerada (trecho incorporado à entrega).** O repositório: o código (`src/hidrometros/`, `tests/`), o notebook, os históricos e resultados, as figuras, o README e os documentos em `docs/`. A IA também rascunhou a seção Limitações do README e o texto das notas de auditoria e de desvios do plano.

**Adaptações realizadas (validação, correção, expansão).**
- O integrante João Rietra aprovou o SDD antes da execução (decisões humanas D1 a D7 na seção 7.1 do SDD) e conferiu as folhas de auditoria de orientação das faixas (todas estavam de pé).
- Os números do README são conferidos contra `results.csv` por um teste automático (`make check`, 13 critérios, e 40 testes); o SDD manda guardas de teste que foram vistas falhando antes de passar (`docs/2026-10-13-guard-red.md`).
- A `smallcnn` do código inicial tinha um defeito de arquitetura (acurácia abaixo da taxa da classe majoritária); a IA diagnosticou e corrigiu com experimentos registrados em `docs/2026-10-14-smallcnn-root-cause.md`, e a mudança está declarada no README como alteração de arquitetura, não de protocolo.
- O esqueleto inicial do código (`SmallCNN`, `build_model` etc.) foi escrito pelo integrante Mikael de Albuquerque Mulatinho, que informou não ter usado IA nele.
- As respostas às questões 1.1 a 5.6 partiram do texto escrito por Mikael; a IA conferiu cada número contra `results.csv` e os logs, corrigiu o que estava errado ou incompleto e ajustou a redação (registro em `docs/2026-10-14-respostas-mikael-revisadas.md`).
- Um segundo agente de IA, sem contexto do projeto, simulou a correção do professor com base nos requisitos publicados e apontou defeitos que foram corrigidos.
- A seção Limitações do README foi rascunhada pela IA. Revisão pelo grupo: pendente.

## Seção B: uso de skill (Claude Code / SKILL.md)

**Utilizou skill?** Sim.

**Nome da skill.** `spec` (do pacote gstack) e `minions`.

**Como foram usadas.** A `spec` guiou a elaboração do SDD (fases de entendimento, escopo e rascunho; as fases finais de arquivar issue no GitHub não foram executadas). A `minions` orientou como delegar a execução a subagentes (um agente por etapa, sem sub-agentes).

**Conteúdo do SKILL.md.** Arquivos `SKILL.md` das duas skills, anexados à entrega à parte (o template permite anexar como arquivo separado).

**Versão / hash.**

| Skill | Arquivo | SHA-256 | Última modificação |
|---|---|---|---|
| `spec` | `gstack/spec/SKILL.md` | `c37faa9c206ccb4ac5007165319436e5e057bfb81b3f0de0bb2878fafebfc0a5` | 2026-09-15 |
| `minions` | `minions/SKILL.md` | `9329ab6c5ab92b3c9d542d042888e448f6789d5feddb55ef776fd41f6eb9da9e` | 2026-09-28 |

## Seção C: uso de SDD (System Design Document)

**Utilizou SDD?** Sim.

**Arquivo SDD.** [`docs/SPEC.md`](../SPEC.md), anexado ao repositório (SHA-256 `2b6a2a676d700d2d8fadadb87e08ee2dd204774ade6fc6e8cd3e31121c1ade2c`, v1.3, 2026-10-13).

**Partes geradas por IA e como o SDD foi elaborado.** O SDD foi redigido pela IA (Claude), em conversa com o integrante João Rietra, a partir do README do professor, do PDF da prática e do esqueleto do Mikael. Todas as seções textuais foram geradas pela IA. As decisões de escopo são humanas (D1 a D7, seção 7.1 do SDD) e foram respondidas pelo integrante a perguntas objetivas (onde fica o repositório, nome do grupo, treino local, como tratar fotos do dataset, desafio extra). A seção 5.2 (fatos verificados do dataset) vem de uma inspeção executada pela IA sobre o dataset baixado, com código e contagens que o `make check` revalida.

## Declaração de responsabilidade

Cada integrante assina a sua cópia do template. O formulário original, com o texto de responsabilidade da disciplina, está na versão `.docx` desta pasta.
