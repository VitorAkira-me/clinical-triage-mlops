# Machine Learning Canvas - Triagem de Laudos Médicos

Framework de [Louis Dorard](https://www.louisdorard.com/machine-learning-canvas), adaptado para
foco em **Business Understanding**. Foram preenchidos apenas os blocos relevantes para este projeto -
blocos como *Data Collection*, *Features* ou *Building Models* não agregam nada aqui além do que
já está em [docs/specs/](specs/) e nos notebooks, então foram deixados de fora.

Contexto: Tech Challenge Fase 3, FIAP - sistema de triagem automática de laudos médicos.

---

## Value Proposition

Reduzir o tempo de triagem manual de laudos clínicos e apoiar a priorização da fila de
atendimento em pronto-socorro.

**O que este sistema não é**: uma ferramenta de diagnóstico autônomo. A predição é um sinal de
apoio à decisão - quem decide a prioridade real de atendimento continua sendo o julgamento
clínico humano. Isso não é uma ressalva genérica de compliance: é uma consequência direta da
limitação real do modelo, documentada em [MODEL_CARD.md](MODEL_CARD.md) - o classificador aprendeu
uma correspondência determinística do gerador sintético do dataset, não semântica clínica real.

Para quem usa: um sistema que reordena a fila de revisão por urgência prevista entrega valor
mesmo com essa limitação, desde que o output seja tratado como *sugestão de ordenação*, nunca como
decisão final automática.

## Prediction Task

Classificação multi-classe de texto: dado o texto livre de um laudo (`clinical_notes`), prever uma
entre três classes de urgência:

- `normal`
- `atencao`
- `urgente`

Supervisionado, uma predição por laudo, sem dependência de histórico do paciente ou de outros
laudos (cada requisição a `POST /predict` é independente - ver
[docs/specs/API-001-predict-endpoint.md](specs/API-001-predict-endpoint.md)).

## Decisions

A predição alimenta a **ordenação da fila de revisão humana**, não uma ação automática sobre o
paciente:

- `urgente` → priorizado para revisão imediata
- `atencao` → revisão em prazo intermediário
- `normal` → revisão em prazo padrão

O modelo não aciona nenhum fluxo assistencial sozinho (não solicita exame, não define protocolo,
não bloqueia atendimento). Essa fronteira é deliberada: dado o achado do vazamento determinístico
de rótulo ([ADR-002](decisions/ADR-002-text-leakage.md)), automatizar qualquer decisão clínica a
partir deste modelo seria irresponsável - o valor está em *ordenar mais rápido*, não em
*decidir sozinho*.

## Data Sources

- **[fedmml-ed-triage](https://huggingface.co/datasets/olaflaitinen/fedmml-ed-triage)**
  (Hugging Face, *gated* - exige aceite de termos antes do download): 87.234 encontros sintéticos
  de pronto-socorro, licença CC BY 4.0. Foi escolhido em [ADR-001](decisions/ADR-001-dataset.md) por já
  nascer rotulado por urgência (ESI 1-5, remapeado para as 3 classes do projeto) em vez de exigir
  um mapeamento inventado a partir de outro corpus.
- Campo usado: `clinical_notes` (texto), 85.679 registros não-nulos de 87.234 (1,78% nulos) -
  vitais e labs estruturados do mesmo dataset ficam fora de escopo do classificador de texto.
- Textos são **inteiramente sintéticos**, gerados por template (`"{idade}yo {sexo} {verbo}
  {chief_complaint}. {cláusula fixa}."`) - sem dado real de paciente, sem questão de privacidade,
  mas também sem a variação linguística de um laudo humano real (ver limitação central no
  [MODEL_CARD.md](MODEL_CARD.md)).

## Making Predictions

Serviço **real-time**, via API HTTP síncrona (`POST /predict`, FastAPI) - não é um job batch.
Essa não é uma escolha de infraestrutura isolada: é uma constatação do próprio problema, registrada
no [ADR-005](decisions/ADR-005-cloud-strategy.md) - "triagem" significa priorizar no momento em
que o laudo chega; processar em lote à noite anularia o propósito do sistema.

Modelo carregado uma única vez no startup do processo (não por requisição). Latência de inferência
pura medida e otimizada via ONNX (ver [MODEL_CARD.md](MODEL_CARD.md), seção *Model Details*);
latência ponta-a-ponta da API observada na prática fica na casa de ~6-9ms (monitorada via
Prometheus/Grafana, [OBS-001](specs/OBS-001.md)).

## Offline Evaluation

Baseline TF-IDF + LogisticRegression avaliado em split 80/20 estratificado (17.136 amostras de
teste): **F1 macro = 1.0**.

Esse número **não deve ser lido como desempenho real de NLP clínico** - ver a ressalva completa,
com os números da investigação, no [MODEL_CARD.md](MODEL_CARD.md#limitations). Em resumo: o F1
perfeito reflete um vazamento determinístico do gerador sintético do dataset, confirmado
comparando o classificador de texto contra um baseline ingênuo por `chief_complaint`, que produz a
mesma matriz de confusão.
