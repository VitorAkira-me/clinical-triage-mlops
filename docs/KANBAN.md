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

### DOC-001 — README seção 6 (Modelo) + 6.1 (Limitações do dataset)
- **Objetivo**: documentar o baseline da ML-003 no README, incluindo a tabela comparativa
  TF-IDF vs. baseline ingênuo que evidencia o vazamento do ADR-002
- **Motivação**: números prontos desde a ML-003, pendência de redação registrada em duas
  sessões anteriores (ML-003 e API-001)
- **Dependências**: ML-003
- **Complexidade**: baixa (só redação — nenhuma decisão de código pendente)
- **Resultado**: README.md seções 6 e 6.1 preenchidas. Números conferidos contra
  [docs/experiments/ML-003-baseline-metrics.json](../docs/experiments/ML-003-baseline-metrics.json)
  (batem exatos). Tabela comparativa TF-IDF vs. ingênuo por `chief_complaint`, com nota sobre
  vitais/labs validados como não-determinísticos (ML-002) e link para
  [ADR-002](decisions/ADR-002-text-leakage.md). Docker/CI/demais seções TODO do README
  continuam fora de escopo — são decisões de arquitetura, não redação, e seguem o fluxo
  SPEC → discussão.

### DOCK-001 — Dockerizar a API de inferência (`src/api/`)
- **Objetivo**: empacotar a API FastAPI da API-001 numa imagem Docker autossuficiente que sobe
  com `docker run` e responde em `/health` e `/predict`, sem volume, rede ou secret
- **Motivação**: pré-requisito para CI/CD (EPIC 07) e arquitetura de cloud (EPIC 12) — a partir
  daqui a unidade deployável do projeto é a imagem, não o checkout do repo
- **Dependências**: API-001
- **Complexidade**: média
- **Resultado**: spec em
  [docs/specs/DOCK-001-dockerize-api.md](specs/DOCK-001-dockerize-api.md). `Dockerfile`
  single-stage (`python:3.12-slim-bookworm`), `uv sync --frozen --no-dev --no-install-project`
  a partir do `uv.lock`; código roda via `PYTHONPATH=/app` (não instala o pacote `src`), usuário
  não-root, `HEALTHCHECK` batendo em `GET /health` via `urllib` (sem curl na slim). Modelo entra
  por `COPY` explícito do `.joblib` no build (decisão: imagem autossuficiente; revisitar quando
  houver model registry — EPIC 08/12). Commit `chore:` separado moveu `pandas`/`pyarrow`/
  `huggingface-hub` para `[project.optional-dependencies] data` — fora do runtime da imagem,
  ainda instaláveis via `uv sync --extra data` para os notebooks. **Tamanho da imagem final:
  132 MB** (`docker image inspect` / `CONTENT SIZE`, compactado; ~568 MB descompactado). Chegou
  a 308 MB na 1ª versão e caiu com: remover `chown -R` (layer duplicada de ~296 MB), montar o
  binário do `uv` via `RUN --mount=from=` em vez de `COPY --from` (~58 MB), e mandar o cache de
  download do `uv` para `--mount=type=cache` em vez da layer. Validado de verdade: `docker build`
  limpo, container sobe `healthy` em ~7 s, `GET /health` → `200`, `POST /predict` → `200` com 3
  probabilidades somando 1.0, texto em branco → `422`; build sem o `.joblib` no contexto falha
  no `COPY` (exit 1), não em runtime. `docker-compose.yml` fora de escopo (só faz sentido no
  EPIC 09, com Prometheus + Grafana).

### CI-001 — Pipeline CI (GitHub Actions): lint → testes → build
- **Objetivo**: automatizar lint (ruff), testes (pytest) e validação de build da imagem Docker a
  cada PR/push para `main`, com o pipeline realmente bloqueando merge quando falha
- **Motivação**: nada garante hoje que um PR não quebrou lint/teste/build antes de chegar em
  `main`; requisito do Tech Challenge Fase 3 (pipeline CI/CD) — este card cobre a metade CI
- **Dependências**: DOCK-001
- **Complexidade**: média
- **Resultado**: spec em [docs/specs/CI-001-ci-pipeline.md](specs/CI-001-ci-pipeline.md).
  `.github/workflows/ci.yml` com 3 jobs: `lint` (`ruff check` + `ruff format --check`) e `test`
  (`pytest -m "not slow" --cov --cov-fail-under=70`) em paralelo, `build` com
  `needs: [lint, test]` — builda a imagem do DOCK-001, sobe o container e valida `GET /health` +
  `POST /predict` reais dentro do job. `scripts/gen_placeholder_model.py` gera um modelo
  sintético só para o `COPY` do Dockerfile (o `.joblib` real é gitignored, não existe no
  runner) — reutilizável localmente, não versionado. Cache de deps via `astral-sh/setup-uv`
  (`uv`/Python fixados em `0.11.7`/`3.12`, chaveado por `uv.lock`); `concurrency` cancela runs
  supersedidos.

  **Validado de verdade no PR #6**, não assumido: run inicial 100% verde (lint 9s, test 52s,
  build 37s, confirmando que `build` só inicia depois de `lint`+`test`). Commit proposital com
  `import sys` não usado → `lint` vermelho (`F401`), `build` **SKIPPED** — mas nesse momento,
  **sem branch protection ainda**, o PR seguia `mergeable: MERGEABLE` (prova de que até então o
  CI era só sinal, não bloqueio). Configurada branch protection em `main` via `gh api`
  (`required_status_checks` com `lint`/`test`/`build`, preservando as regras já existentes de
  exigir PR). Quebrado o lint de novo com a proteção ativa: `mergeStateStatus` virou `BLOCKED` e
  `gh pr merge` foi **recusado pelo próprio GitHub** (`"the base branch policy prohibits the
  merge"`). Revertido, run voltou a verde, `mergeable: MERGEABLE` de novo. Ciclo completo
  vermelho→bloqueado→verde→liberado, com saída real de `gh run watch`/`gh pr view`/`gh pr merge`
  em cada etapa (sem número ou resultado inventado).

### OBS-001 — Observabilidade da API (Prometheus + Grafana)
- **Objetivo**: instrumentar a API com métricas Prometheus (RED + negócio) e um dashboard
  Grafana, funcionando como proxy de drift já que não há rótulo verdadeiro em produção
- **Motivação**: EPIC 09 do roadmap; a API rodava sem nenhuma visibilidade operacional
- **Dependências**: DOCK-001
- **Complexidade**: alta
- **Resultado**: spec em [docs/specs/OBS-001.md](specs/OBS-001.md), 5 passos, todos implementados
  e validados de verdade (nenhum número inventado):

  **Passo 1** — `prometheus-fastapi-instrumentator` para RED automático. Achado real: `.instrument(app)`
  sozinho não grava métrica nenhuma, precisa de `.add(metrics.default(...))`. `/health`/`/metrics`
  excluídos (heartbeat do Docker não é tráfego de negócio). Buckets de latência recalibrados
  (0.5ms–10s) — o padrão da lib (mínimo 10ms) daria zero resolução pra um modelo linear leve.

  **Passo 2** — métricas de negócio via `prometheus_client` direto: `triage_predictions_total`
  (Counter) e `triage_prediction_confidence` (Histogram), ambas com label `classe_prevista` (não
  agregado global — mais diagnóstico, cardinalidade trivial).

  **Passo 3** — `docker-compose.yml`: API via `build: .` (sem registry), Prometheus/Grafana com
  `depends_on: condition: service_healthy` reaproveitando o `HEALTHCHECK` da DOCK-001, sem volume
  persistente (efêmero é suficiente pro objetivo de demonstração).

  **Passo 4** — `prometheus.yml` real: scrape em `api:8000` (nome do serviço, não `localhost`).
  Validado com número batendo exato: 3 chamadas reais → `triage_predictions_total` no Prometheus
  mostrou exatamente a mesma contagem.

  **Passo 5** — provisionamento do Grafana por arquivo (datasource + dashboard, 4 painéis: taxa
  de requisições, latência p95, distribuição de classes, confiança p50/p95 por classe). Antes de
  implementar, **verificação**: não existe nem nunca existiu `.joblib` versionado no repo
  (gitignored por design) — rodados de verdade `notebooks/01_eda_dataset.ipynb` (dataset real,
  87.234 linhas, mesmo vazamento do ADR-002 confirmado com dado real) e `02_baseline.ipynb`
  (resultado idêntico, byte a byte, ao `docs/experiments/ML-003-baseline-metrics.json` já
  commitado). **Achado principal do passo**: a hipótese de calibração de confiança do Passo 2
  era direcionalmente certa mas conservadora demais — a distribuição real do baseline satura
  numa faixa de ~0,005 de largura (`0.9949`–`0.9997`), não numa faixa genérica de "alta
  confiança"; buckets recalibrados contra 90 chamadas reais (dataset de verdade, 30 por classe),
  `histogram_quantile` foi de `NaN` para valores reais e coerentes (`p50≈0.9994`, `p95≈0.9997`).
  Validação em três camadas: Prometheus direto, `/api/dashboards/uid/...` do Grafana
  (`provisionado: true`), e proxy de query do próprio Grafana batendo com o número do Prometheus.
  `scripts/gen_placeholder_model.py` ganhou um guard (`--force`) pra nunca mais sobrescrever um
  `.joblib` real sem querer.

  Fora de escopo mantido: Alertmanager, detecção estatística formal de drift (o card entrega um
  proxy visual, não um teste estatístico), autenticação do `/metrics`/Grafana, tracing.

---

## TODO

*(Demais tarefas dos EPICS 05–14 serão detalhadas ao chegar em cada STEP
do roadmap — ver decisão de não planejar em excesso antecipadamente,
registrada na sessão de discovery.)*

---

## IN PROGRESS

*(nenhuma tarefa aberta)*

---

## BACKLOG (nível de épico, não detalhado ainda)

- EPIC 05 — Docker (DOCK-001 concluído; compose entregue na OBS-001)
- EPIC 06 — Testes
- EPIC 07 — CI/CD (GitHub Actions) (CI-001 concluído — cobre só o CI; o CD segue dependendo de
  registry/cloud, EPIC 12)
- EPIC 08 — Airflow (DAG de treino)
- EPIC 09 — Observabilidade (Prometheus + Grafana) (OBS-001 concluído)
- EPIC 10 — Otimização de inferência (ONNX/quantização/pruning)
- EPIC 11 — Benchmark (latência p50/p95, tamanho de modelo)
- EPIC 12 — Arquitetura de cloud (ADR-005)
- EPIC 13 — Documentação final
- EPIC 14 — Vídeo STAR
