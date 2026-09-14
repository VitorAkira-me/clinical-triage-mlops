# Clinical Triage MLOps

Sistema de triagem automática de laudos médicos por urgência
(normal / atenção / urgente), desenvolvido para o Tech Challenge — Fase 3
da Pós Tech em Machine Learning Engineering (FIAP).

> Projeto individual — continuação solo após o encerramento do grupo
> original (ver Fases 1 e 2 abaixo).

## 1. Visão geral

Laboratório pessoal de Machine Learning Engineering/MLOps, construído em público, STEP por STEP,
com decisões documentadas em ADRs — não só uma entrega acadêmica.

**Problema**: hospitais recebem laudos em texto livre e precisam priorizar atendimento. Este
projeto constrói um classificador de urgência a partir do texto do laudo.

**Objetivo**: entregar uma API de inferência em produção, com pipeline de treino automatizado
(Airflow), CI/CD (GitHub Actions), observabilidade (Prometheus + Grafana) e uma técnica de
otimização de inferência com benchmark real de latência.

### 1.1 Dataset

[fedmml-ed-triage](https://huggingface.co/datasets/olaflaitinen/fedmml-ed-triage)
(Hugging Face) — **sintético**, licença CC BY 4.0. Usamos apenas o campo de texto
`clinical_notes`; o alvo nativo ESI 1–5 foi remapeado para 3 classes (normal/atenção/urgente).
Decisão completa e trade-offs em [ADR-001](docs/decisions/ADR-001-dataset.md).

**Aviso importante**: o dataset é sintético e o mapeamento de urgência é uma adaptação de
engenharia deste projeto — não representa um padrão clínico oficial.

## 2. Decisão de arquitetura em nuvem

Análise exigida pelo Tech Challenge (`STEP 10` do [ROADMAP](docs/ROADMAP.md)) — **raciocínio
escrito, sem deploy real**. Decisão completa em [ADR-005](docs/decisions/ADR-005-cloud-strategy.md);
resumo:

- **Real-time, não batch, para a API.** Não é uma escolha nova — é uma constatação do que já foi
  construído: a API (API-001) é síncrona, request/response, com latência medida e importante em
  toda a stack (buckets de latência calibrados na OBS-001, benchmark de inferência no OPT-001).
  Triagem significa priorizar no momento em que o laudo chega; processar em lote anularia o
  propósito do sistema. O **treino**, em contraste, já é batch por natureza (Airflow, AIR-001,
  sob demanda) — o projeto já reflete essa distinção na prática, sem ter sido planejado assim.
- **Provedor recomendado: AWS** (`ECS Fargate` + `Amazon Managed Prometheus` + `Amazon Managed
  Grafana` + `MWAA`) — critério: não há lock-in hoje (Docker puro), então a escolha é pela menor
  distância entre o que já existe localmente e o equivalente gerenciado. Os dois produtos
  gerenciados de observabilidade da AWS são literalmente os mesmos dois já escolhidos no
  `docker-compose.yml` da OBS-001 — migração quase 1:1, não uma reescrita.
- **Alternativa honesta: GCP Cloud Run** — mais simples de operar pra uma API deste porte
  (serverless, escala a zero, tráfego baixo/esporádico); melhor critério se custo/simplicidade
  pesar mais que continuidade da stack de observabilidade já construída.

## 3. Como executar

### Pré-requisitos

- **Docker** (build/run da API, da stack de monitoramento e da DAG do Airflow)
- **`.env` na raiz** com `HF_TOKEN` (token do Hugging Face com acesso ao dataset — é gated,
  precisa aceitar os termos no site antes de gerar o token): necessário só para baixar o dataset
  do zero (notebooks de EDA/treino, ou a task `load_dataset` da DAG se não houver cache local).
  Se `data/raw/fedmml_ed_triage_raw.parquet` já existir, nada disso é consultado.
- **O artefato do modelo** (`models/tfidf_logreg_baseline.joblib`) — **não é versionado no Git**
  (gitignored de propósito). Gere-o rodando `notebooks/02_baseline.ipynb` **ou** a DAG do Airflow
  (seção 3.3 abaixo) antes de subir a API/imagem. Sem ele, a API falha no startup com mensagem
  acionável e o `docker build` falha no passo de `COPY` do modelo.

### 3.1 Build e run da API (container único)

```bash
docker build -t clinical-triage-api .
docker run --rm -p 8000:8000 clinical-triage-api
```

A imagem é autossuficiente: o modelo é embutido no build, não precisa de volume, rede externa ou
variável de ambiente. `docker ps` mostra `healthy` alguns segundos depois de subir.

```bash
curl http://localhost:8000/health
# {"status":"ok"}

curl -X POST http://localhost:8000/predict \
  -H 'content-type: application/json' \
  -d '{"clinical_notes": "67yo M c/o chest pain, diaphoretic, in moderate distress"}'
# {"urgencia":"urgente","probabilidades":{"atencao":0.089,"normal":0.171,"urgente":0.740}}
```

Sem Docker (desenvolvimento local): `uv sync && uv run uvicorn src.api.main:app --host 0.0.0.0
--port 8000` (`uv sync --extra data` adiciona pandas/pyarrow/huggingface-hub para os notebooks).
Contrato completo em [docs/specs/API-001-predict-endpoint.md](docs/specs/API-001-predict-endpoint.md);
decisões da imagem em [docs/specs/DOCK-001-dockerize-api.md](docs/specs/DOCK-001-dockerize-api.md).

### 3.2 Stack completa (API + Prometheus + Grafana)

```bash
docker compose up -d --build
```

Sobe os três serviços na ordem certa (API primeiro, `healthy`, só então Prometheus/Grafana —
`depends_on: condition: service_healthy`, reaproveitando o `HEALTHCHECK` da API). URLs:
`localhost:8000` (API), `localhost:9090` (Prometheus), `localhost:3000` (Grafana,
usuário/senha `admin`/`admin` — desenvolvimento local, não produtivo). Dashboard já vem
provisionado por arquivo, sem clique na UI — ver seção 5. Decisões em
[docs/specs/OBS-001.md](docs/specs/OBS-001.md).

### 3.3 Pipeline de treino (Airflow)

DAG com 2 tasks (`load_dataset → train_baseline`), container standalone — **Airflow nativo não
roda no Windows** (testado empiricamente: quebra na importação básica), por isso Docker em vez de
venv local, em qualquer sistema operacional.

```bash
docker build -t air001-airflow ./airflow

docker run --rm \
  -e AIRFLOW_HOME=/opt/airflow \
  -e AIRFLOW__CORE__DAGS_FOLDER=/opt/airflow/project/airflow/dags \
  -e AIRFLOW__CORE__LOAD_EXAMPLES=False \
  -v "$(pwd):/opt/airflow/project" \
  air001-airflow bash -c "airflow db migrate && airflow dags test air001_train_baseline $(date +%F)"
```

> **Windows + Git Bash**: prefixe o comando com `MSYS_NO_PATHCONV=1` — o Git Bash traduz
> `/opt/airflow` para um caminho Windows antes de passar pro Docker, e o Airflow quebra sem essa
> variável. Irrelevante em Linux/macOS ou PowerShell.

`load_dataset` reusa `data/raw/fedmml_ed_triage_raw.parquet` se já existir; senão tenta o cache
local do Hugging Face; só baixa de verdade (usando o `HF_TOKEN` do `.env`) como último recurso.
`train_baseline` salva o `.joblib` em `models/` no **host** (volume montado, não preso no
container) — o mesmo artefato que a API/Docker usam. Decisões em
[docs/specs/AIR-001.md](docs/specs/AIR-001.md).

### 3.4 CI/CD

`.github/workflows/ci.yml`: `lint` (ruff) e `test` (pytest) em paralelo, `build` (builda a imagem
e valida `/health`+`/predict` reais dentro dela) só depois dos dois passarem. Roda em todo PR e
push pra `main`; branch protection exige os 3 checks verdes — testado de propósito quebrando um
lint e confirmando que o GitHub recusa o merge, não só reporta vermelho. Decisões em
[docs/specs/CI-001-ci-pipeline.md](docs/specs/CI-001-ci-pipeline.md).

## 4. Resultados

### 4.1 Modelo

Baseline: TF-IDF + Logistic Regression (`class_weight="balanced"`), treinado sobre o campo
`clinical_notes`. Split treino/teste estratificado (80/20, `random_state=42`). Ver
[notebooks/02_baseline.ipynb](notebooks/02_baseline.ipynb) e métricas completas em
[docs/experiments/ML-003-baseline-metrics.json](docs/experiments/ML-003-baseline-metrics.json).

#### 4.1.1 Limitações do dataset — Vazamento de rótulo identificado

Durante a EDA (ML-002) e confirmado empiricamente na ML-003, identificamos que o campo
`clinical_notes` do `fedmml-ed-triage` é gerado por um template fixo: as 28 categorias de
`chief_complaint` e as 5 variantes de cláusula final do texto mapeiam para a classe de urgência
(normal/atenção/urgente) com 100% de precisão, sem exceção, nas 85.679 notas verificadas.

Para confirmar isso, comparamos o baseline de texto (TF-IDF + Logistic Regression) lado a lado com
um baseline ingênuo que classifica apenas pelo `chief_complaint` (dicionário complaint → classe
majoritária):

| Modelo | F1 macro | Recall macro | Recall (`urgente`) |
|---|---|---|---|
| TF-IDF + Logistic Regression | 1.00 | 1.00 | 1.00 |
| Ingênuo (só `chief_complaint`) | 1.00 | 1.00 | 1.00 |

*(17.136 exemplos de teste; matrizes de confusão idênticas, diagonais perfeitas — ver
[métricas completas](docs/experiments/ML-003-baseline-metrics.json))*

Os dois empatam exatamente. Isso significa que o classificador de texto não demonstra capacidade
real de NLP clínico — ele decorou uma tabela de busca embutida no template do gerador sintético,
não aprendeu linguagem clínica. Validamos que esse vazamento é específico do texto: os campos de
vitais/labs (fora do escopo do classificador), como `spo2`, `heart_rate` e troponina, seguem
distribuições com sobreposição real entre classes, sem determinismo perfeito.

Decidimos manter o dataset e reportar esse resultado com transparência, em vez de trocar de fonte
ou escondê-lo — decisão completa e alternativas descartadas em
[ADR-002](docs/decisions/ADR-002-text-leakage.md). Na prática, isso significa que as métricas de
~100% deste projeto não devem ser lidas como "o modelo é excelente", e sim como evidência de um
artefato de geração de dados sintéticos — uma lição de MLOps sobre validar dados antes de
comemorar métricas.

Reconfirmado de forma independente na AIR-001 (dados reais baixados de novo) e na OBS-001
(150 chamadas reais ao `/predict` com o baseline real): mesmo padrão, 100% das vezes.

### 4.2 Otimização (ONNX) e benchmark

Conversão do baseline (TF-IDF + Logistic Regression) pra ONNX via `skl2onnx` — **pipeline
inteiro**, não só o classificador (funcionou de primeira; o `TfidfVectorizer` foi treinado com
hiperparâmetros default, o caso mais simples de converter). Corretude validada **antes** de medir
qualquer velocidade: 150 textos reais do dataset, 150/150 classes batendo entre sklearn e ONNX,
diferença máxima de probabilidade `8.11e-08` (tolerância usada: `1e-4`).

Latência (300 iterações, mesma entrada, mesma máquina, após warmup):

| | sklearn original | ONNX (onnxruntime) | Speedup |
|---|---|---|---|
| Mediana | 0.40 ms | 0.036 ms | **~11x** |
| p95 | 0.55 ms | 0.050 ms | **~11x** |

Tamanho do artefato: 9.044 bytes (`.joblib`) vs. 7.076 bytes (`.onnx`), -22%.

**Ressalva importante**: esses 0.4ms do sklearn já são extremamente baixos em termos absolutos —
a latência ponta-a-ponta da API real (HTTP + serialização, medida na OBS-001) fica na casa de
6-9ms, dominada por overhead de rede, não pela inferência pura. O ganho de ~11x é real na camada
de inferência isolada; não foi medido (nem é o objetivo deste card) o quanto disso se traduziria
em latência percebida pelo cliente se a API passasse a servir via ONNX — troca que não foi feita
em produção, é comparação/benchmark. Detalhes completos, script reprodutível e a trajetória da
conversão (o que funcionou de primeira e o fallback que existia mas não foi preciso) em
[docs/specs/OPT-001.md](docs/specs/OPT-001.md) e
[docs/experiments/OPT-001-benchmark.json](docs/experiments/OPT-001-benchmark.json).

## 5. Monitoramento

Prometheus + Grafana, subindo junto com a API via `docker-compose up` (seção 3.2), dashboard
provisionado por arquivo — 5 painéis, todos validados com dados reais (não só a métrica existindo,
o número batendo com o tráfego gerado):

1. **Taxa de requisições** (por rota) — `http_requests_total` (RED, OBS-001 Passo 1)
2. **Latência p95** (por rota) — `histogram_quantile(0.95, ...)` sobre
   `http_request_duration_seconds_bucket`, buckets calibrados pra latência sub-segundo (o padrão
   da lib, pensado pra web genérica, daria zero resolução aqui)
3. **Distribuição de classes previstas ao longo do tempo** — `rate(triage_predictions_total)` por
   `classe_prevista` — proxy de drift: não há rótulo verdadeiro disponível em produção, então uma
   mudança anormal na proporção normal/atenção/urgente é o sinal indireto que temos
4. **Confiança por classe** (p50 e p95, linhas simples) — `histogram_quantile` sobre
   `triage_prediction_confidence_bucket`; buckets recalibrados contra o baseline real na OBS-001
   Passo 5 depois que a hipótese inicial (baseada só no padrão de separação perfeita da ML-003)
   se mostrou conservadora demais — a confiança real satura numa faixa de ~0,005 de largura perto
   de 1.0, não numa faixa genérica de "alta confiança"
5. **Taxa de erro** — `http_requests_total` filtrando status diferente de `2xx`, por rota e
   status agrupado

Nenhuma das métricas de negócio (painéis 3 e 4) é uma detecção estatística formal de drift — são
gatilhos pra investigação humana. Decisões completas em
[docs/specs/OBS-001.md](docs/specs/OBS-001.md).

## 6. Limitações e lições aprendidas

- **O dataset é sintético e vazado por template** (seção 4.1.1) — a métrica de ~100% não valida
  capacidade real de NLP clínico. A lição de MLOps mais importante do projeto: validar dados
  *antes* de comemorar métricas, e reportar isso com transparência em vez de esconder.
- **Métrica bem desenhada não é o mesmo que visualização útil.** Os buckets de confiança do
  Passo 2 da OBS-001 existiam, tinham o label certo, o teste passava — e ainda assim
  `histogram_quantile` devolvia `NaN` contra o modelo real, porque a distribuição real caía num
  intervalo que os buckets simplesmente não resolviam. Só apareceu rodando tráfego real contra o
  dashboard, não no design isolado da métrica.
- **`rate()`/`histogram_quantile()` precisam de dado espalhado no tempo, não uma rajada.** Uma
  rajada instantânea de tráfego prova que a métrica está correta, mas produz `rate() == 0` (ou
  resultado vazio) num painel — o Prometheus não viu a transição, só o valor final parado.
  Descoberto de verdade duas vezes (OBS-001 Passo 5 e na validação do painel de erro para este
  README), não hipoteticamente.
- **Drift de ambiente entre quem treina e quem serve é um risco real, não teórico**: o container
  do Airflow (AIR-001) instalou uma versão de `scikit-learn` diferente da API por não ter versão
  fixada no `requirements.txt` — o `.joblib` carregou mesmo assim (só warning), mas podia não ter
  carregado. Fixar versões exatas nas duas pontas deixou de ser opcional antes do ONNX.
- **Ferramentas de dado têm gotchas de versão binária**: `pyarrow` sem versão fixa quebrou a
  leitura de um parquet escrito por outra versão (`OSError: Repetition level histogram size
  mismatch`) — corrigido fixando a mesma versão nos dois ambientes.
- **Nem toda ferramenta do ecossistema roda em todo SO.** Airflow nativo não roda no Windows —
  descoberto testando de verdade (`ImportError` já na importação básica), não assumido. Rodar via
  Docker desde o início evitou reescrever a decisão depois.
- **Auditoria de documentação antes de consolidar**: specs antigos (API-001, AIR-001) tinham
  decisões que mudaram implicitamente ao longo do projeto sem o documento acompanhar — corrigido
  numa sessão dedicada antes deste README, exatamente para não propagar contradição documentada
  pro README final.

## 7. Decisões arquiteturais

Ver [docs/decisions/](docs/decisions/) — ADR-001 (dataset), ADR-002 (vazamento de rótulo),
ADR-005 (estratégia de cloud, seção 2 acima).

## 8. Estrutura do projeto

```
clinical-triage-mlops/
├── .github/workflows/          # CI (lint, test, build) — CI-001
├── airflow/
│   ├── dags/                   # DAG de treino — AIR-001
│   └── Dockerfile
├── benchmarks/                 # conversão ONNX + benchmark — OPT-001
├── data/{raw,processed}/
├── docs/{decisions,specs,experiments}/
├── monitoring/{prometheus,grafana}/  # scrape config + dashboard provisionado — OBS-001
├── notebooks/
├── src/api/                    # FastAPI — API-001
├── tests/
├── models/                     # .joblib (gitignored, gerado localmente)
├── .claude/CLAUDE.md
├── Dockerfile                  # DOCK-001
├── docker-compose.yml          # OBS-001
└── pyproject.toml
```

## 9. Roadmap

Ver [docs/ROADMAP.md](docs/ROADMAP.md).

## 10. Vídeo

TODO — próxima e última etapa.

## 11. Autor

Akira — Pós Tech em Machine Learning Engineering, FIAP.
Projetos anteriores em equipe: [ml-churn-prediction](https://github.com/fiap-postech-ml-engineering/ml-churn-prediction)
(Fase 1) e [ecommerce-recsys-mlops](https://github.com/fiap-postech-ml-engineering/ecommerce-recsys-mlops)
(Fase 2).
