# KANBAN — Clinical Triage MLOps (Fase 3)

Colunas: BACKLOG · TODO · IN PROGRESS · REVIEW/VALIDATION · DONE

Cada tarefa: ID, Título, Objetivo, Motivação, O que vou aprender,
Pré-requisitos, Passos, Critério de aceite, Dependências, Complexidade,
Status.

---

## DONE

### ML-001 — Investigar e escolher dataset
- **Objetivo**: encontrar dataset de texto médico adequado para
  classificação de urgência (normal/atenção/urgente), ≥2.000 registros
- **Motivação**: todo o resto do projeto depende dessa decisão
- **Vou aprender**: avaliação de qualidade e licenciamento de dataset,
  checagem de balanceamento, riscos éticos em dados de saúde
- **Critério de aceite**: dataset escolhido e decisão documentada
- **Dependências**: nenhuma
- **Complexidade**: média
- **Resultado**: fedmml-ed-triage escolhido — ver
  [ADR-001](decisions/ADR-001-dataset.md)

### ML-002 — EDA do dataset escolhido
- **Objetivo**: entender completude, distribuição de classes (após
  remapeamento ESI→3 classes) e qualidade textual do campo `clinical_notes`
- **Motivação**: validar que o dataset sintético não trivializa o
  problema antes de investir em modelagem
- **Vou aprender**: análise exploratória de texto, detecção de
  desbalanceamento de classes, red flags de data leakage
- **Pré-requisitos**: ML-001 concluído, acesso ao dataset liberado no
  Hugging Face
- **Passos**: baixar dataset → aplicar remapeamento ESI→classe →
  checar completude/nulos → checar distribuição de classes →
  inspecionar amostras de texto manualmente
- **Critério de aceite**: notebook de EDA com achados registrados,
  decisão sobre necessidade de balanceamento (undersampling/
  class_weight/etc.) tomada e justificada
- **Dependências**: ML-001
- **Complexidade**: média
- **Resultado**: notebook `notebooks/01_eda_dataset.ipynb`. Distribuição de
  classes moderadamente desbalanceada (atenção 47,4% / normal 32,2% /
  urgente 20,3%) — decisão: `class_weight="balanced"`, sem
  under/oversampling, avaliação por F1 macro + recall de `urgente`.
  Achado principal: `clinical_notes` tem vazamento determinístico total
  (chief_complaint e cláusula final do template mapeiam 1:1 para a classe,
  0 exceções em 85.679 notas) — qualquer classificador de texto vai bater
  ~100%. Vitais/labs, fora de escopo, validados como gerados com ruído
  real (não determinísticos), confirmando que o vazamento é específico do
  texto. Decisão de manter o dataset e reportar com transparência
  documentada em [ADR-002](decisions/ADR-002-text-leakage.md).

### ML-003 — Baseline de classificação (TF-IDF + Logistic Regression)
- **Objetivo**: treinar e avaliar um baseline de classificação de urgência
  a partir de `clinical_notes`, estabelecendo o piso de referência para
  qualquer modelo futuro
- **Motivação**: validar o pipeline de treino ponta a ponta antes de
  investir em otimização; expor com transparência a limitação registrada
  no ADR-002
- **Vou aprender**: vetorização TF-IDF, avaliação multi-classe (F1 macro,
  matriz de confusão), como reportar um resultado "bom demais" com
  honestidade
- **Pré-requisitos**: ML-002 concluído
- **Passos**: carregar `data/raw/fedmml_ed_triage_raw.parquet` (precisa de
  `chief_complaint` além de `clinical_notes`, além do parquet processado
  da ML-002) → dropar nulos → split treino/teste estratificado
  (compartilhado pelos dois baselines) → TF-IDF + LogisticRegression
  (`class_weight="balanced"`) → baseline ingênuo por `chief_complaint`
  (ADR-002) → avaliar os dois (F1 macro, recall por classe, matriz de
  confusão) → salvar modelos em `models/` e métricas em
  `docs/experiments/`
- **Critério de aceite**: métricas do baseline TF-IDF e do baseline
  ingênuo reportadas lado a lado (esperado: equivalentes, conforme
  ADR-002); modelo serializado e reprodutível
- **Dependências**: ML-002
- **Complexidade**: média
- **Resultado**: notebook `notebooks/02_baseline.ipynb`, métricas em
  [docs/experiments/ML-003-baseline-metrics.json](../docs/experiments/ML-003-baseline-metrics.json).
  Confirmação empírica do ADR-002: os dois baselines (TF-IDF+LogReg e
  ingênuo por `chief_complaint`) batem exatamente F1 macro = recall macro
  = recall(`urgente`) = 1.00 sobre 17.136 exemplos de teste — matrizes de
  confusão idênticas, diagonais perfeitas. Não há distinção prática entre
  os dois modelos: o texto não carrega sinal além do que `chief_complaint`
  já entrega. README (tarefa separada) vai documentar isso na subseção
  6.1 "Limitações do dataset".

### API-001 — Especificar e implementar endpoint `/predict`
- **Objetivo**: expor o pipeline TF-IDF + LogisticRegression da ML-003 via API FastAPI
  (`POST /predict` + `GET /health`)
- **Dependências**: ML-003
- **Complexidade**: média
- **Resultado**: spec em
  [docs/specs/API-001-predict-endpoint.md](../docs/specs/API-001-predict-endpoint.md);
  implementação em `src/api/main.py` + `src/api/schemas.py`. `POST /predict` recebe só
  `clinical_notes`, devolve `urgencia` + `probabilidades` (dict nomeado por classe, montado via
  `pipeline.classes_` — não uma lista posicional, evita o gotcha de `predict_proba` ordenar
  alfabeticamente). Modelo carregado uma vez no startup (`lifespan`); ausência de
  `models/tfidf_logreg_baseline.joblib` falha o startup com mensagem citando
  `notebooks/02_baseline.ipynb`. `MODEL_PATH` configurável por variável de ambiente. 6 testes em
  `tests/test_api.py` (fixture sintética + 1 `@pytest.mark.slow` com o modelo real), todos
  passando; testado manualmente também via `uvicorn` real (não só `TestClient`). Fora de escopo:
  auth, CORS, versionamento de rota, métricas Prometheus (EPIC 09).

---

## TODO

*(Demais tarefas dos EPICS 05–14 serão detalhadas ao chegar em cada STEP
do roadmap — ver decisão de não planejar em excesso antecipadamente,
registrada na sessão de discovery.)*

---

## BACKLOG (nível de épico, não detalhado ainda)

- EPIC 05 — Docker
- EPIC 06 — Testes
- EPIC 07 — CI/CD (GitHub Actions)
- EPIC 08 — Airflow (DAG de treino)
- EPIC 09 — Observabilidade (Prometheus + Grafana)
- EPIC 10 — Otimização de inferência (ONNX/quantização/pruning)
- EPIC 11 — Benchmark (latência p50/p95, tamanho de modelo)
- EPIC 12 — Arquitetura de cloud (ADR-005)
- EPIC 13 — Documentação final
- EPIC 14 — Vídeo STAR
