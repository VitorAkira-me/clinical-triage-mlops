# Clinical Triage MLOps

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-F7931E?logo=scikitlearn&logoColor=white)
![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-005CED?logo=onnx&logoColor=white)
![Prometheus](https://img.shields.io/badge/Prometheus-E6522C?logo=prometheus&logoColor=white)
![Grafana](https://img.shields.io/badge/Grafana-F46800?logo=grafana&logoColor=white)
![Airflow](https://img.shields.io/badge/Apache%20Airflow-017CEE?logo=apacheairflow&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)
[![CI](https://github.com/VitorAkira-me/clinical-triage-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/VitorAkira-me/clinical-triage-mlops/actions/workflows/ci.yml)
![uv](https://img.shields.io/badge/uv-DE5FE9?logo=uv&logoColor=white)
![Ruff](https://img.shields.io/badge/lint-ruff-D7FF64?logo=ruff&logoColor=black)
![coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)
![license](https://img.shields.io/github/license/VitorAkira-me/clinical-triage-mlops)

Sistema de triagem automática de laudos médicos por urgência (normal / atenção / urgente),
desenvolvido pra o Tech Challenge - Fase 3 da Pós Tech em Machine Learning Engineering (FIAP).

> Projeto individual - continuação solo depois que o grupo original da Fase 1/2 se desfez.

## Autor

| Nome | RM | Contato |
|---|---|---|
| Vitor Akira Ucha Ito | RM371483 | [Github](https://github.com/VitorAkira-me) - [Linkedin](https://www.linkedin.com/in/vitor-akira/) |

🎥 Vídeo Explicativo em até 5 min: [ADICIONAR LINK]

## Fluxo do projeto

```mermaid
flowchart LR
    DS[("fedmml-ed-triage<br/>(Hugging Face)")] --> Load["Airflow: load_dataset"]
    Load --> Train["Airflow: train_baseline<br/>(TF-IDF + LogReg)"]
    Train --> Model["tfidf_logreg_baseline<br/>.joblib / .onnx"]
    Model --> API["FastAPI<br/>POST /predict · GET /health"]
    API -- "GET /metrics" --> Prom[("Prometheus")]
    Prom --> Grafana["Grafana<br/>dashboard (5 painéis)"]
```

```mermaid
flowchart LR
    Push["git push / PR"] --> Lint["lint: ruff"]
    Push --> Test["test: pytest + cobertura"]
    Lint --> Build["build: docker build<br/>+ smoke test real"]
    Test --> Build
    Build --> Gate{"branch protection<br/>nos 3 checks"}
    Gate -->|verde| Merge["merge em main"]
```

## Requisitos

- Docker + Docker Compose (build/run da API, da stack de monitoramento e da DAG do Airflow)
- Python 3.12+ e [uv](https://docs.astral.sh/uv/) - só se quiser rodar a API sem Docker
- `.env` na raiz com `HF_TOKEN` - só necessário pra baixar o dataset do zero (se
  `data/raw/fedmml_ed_triage_raw.parquet` já existir, nada disso é consultado; ver seção 3)

## 1. Visão geral

Comecei este projeto como laboratório pessoal de ML Engineering/MLOps, construído em público,
STEP por STEP, com cada decisão registrada num ADR - não é só a entrega da Fase 3.

**Problema**: hospitais recebem laudos em texto livre e precisam priorizar atendimento. Construí
um classificador de urgência a partir do texto do laudo.

**Objetivo**: uma API de inferência em produção, com pipeline de treino automatizado (Airflow),
CI/CD (GitHub Actions), observabilidade (Prometheus + Grafana) e uma técnica de otimização de
inferência com benchmark real de latência.

### 1.1 Dataset

[fedmml-ed-triage](https://huggingface.co/datasets/olaflaitinen/fedmml-ed-triage)
(Hugging Face) - **sintético**, licença CC BY 4.0. Uso só o campo de texto `clinical_notes`; o
alvo nativo ESI 1–5 foi remapeado por mim pra 3 classes (normal/atenção/urgente). Decisão
completa e trade-offs em [ADR-001](docs/decisions/ADR-001-dataset.md).

**Aviso importante**: o dataset é sintético e o mapeamento de urgência é uma adaptação de
engenharia deste projeto - não representa um padrão clínico oficial.

## 2. Decisão de arquitetura em nuvem

Fiz essa análise porque o Tech Challenge exige (`STEP 10` do [ROADMAP](docs/ROADMAP.md)), não
porque decidi implementar deploy de verdade - é raciocínio escrito, sem código, registrado em
[ADR-005](docs/decisions/ADR-005-cloud-strategy.md).

**Real-time, não batch, pra API.** Isso não é bem uma escolha nova, é uma constatação do que já
existe: desde a API-001 a API é síncrona - o cliente manda o texto, espera a resposta na mesma
requisição - e a latência importa o suficiente pra eu ter calibrado buckets de métrica na casa de
milissegundos (OBS-001) e medido a inferência isolada no benchmark ONNX (OPT-001). Processar em
lote mataria o próprio sentido de "triagem": o paciente já teria sido atendido, bem ou mal, antes
do lote rodar de madrugada. O treino é o oposto - já é batch por natureza (a DAG do Airflow roda
sob demanda, não fica ligada o tempo todo). Não planejei essa distinção de propósito, ela só
apareceu porque treino e inferência têm perfis de uso completamente diferentes.

**Provedor: recomendo AWS.** Não tenho compromisso com nenhum provedor hoje - é tudo Docker puro,
sem lock-in - então decidi pelo critério de menor distância entre o que já construí e o
equivalente gerenciado, não por "o que todo mundo usa". A AWS ganha nesse critério
especificamente: `ECS Fargate` roda o `Dockerfile` sem mudar uma linha, e `Amazon Managed
Prometheus` + `Amazon Managed Grafana` são literalmente os dois produtos que já escolhi rodando
local no `docker-compose.yml` da OBS-001 - migrar isso não é reescrever a stack de
observabilidade, é só apontar pra outro lugar. `MWAA` cobre o lado batch do treino sem eu
reescrever a DAG da AIR-001.

Isso não quer dizer que GCP estaria errado. `Cloud Run` é genuinamente mais simples de operar pra
uma API deste tamanho - serverless de verdade, escala a zero, `gcloud run deploy` direto do
Dockerfile. Se o critério fosse custo e simplicidade em vez de continuidade da stack que já
construí, eu escolheria diferente. Os dois caminhos, com os trade-offs completos, estão no
ADR-005.

## 3. Como executar

### Pré-requisito: o artefato do modelo

A API serve o baseline treinado na ML-003 (`models/tfidf_logreg_baseline.joblib`). Esse arquivo
**não é versionado no Git** (gitignored de propósito) - gere-o rodando
`notebooks/02_baseline.ipynb` **ou** a DAG do Airflow (seção 3.3) antes de subir a API/imagem.
Sem ele, a API falha no startup com mensagem acionável e o `docker build` falha no `COPY` do
modelo.

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

Sem Docker: `uv sync && uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000`
(`uv sync --extra data` adiciona pandas/pyarrow/huggingface-hub pros notebooks). Contrato
completo em [docs/specs/API-001-predict-endpoint.md](docs/specs/API-001-predict-endpoint.md);
decisões da imagem em [docs/specs/DOCK-001-dockerize-api.md](docs/specs/DOCK-001-dockerize-api.md).

### 3.2 Stack completa (API + Prometheus + Grafana)

```bash
docker compose up -d --build
```

Sobe os três serviços na ordem certa (API primeiro, `healthy`, só então Prometheus/Grafana -
`depends_on: condition: service_healthy`, reaproveitando o `HEALTHCHECK` da API). URLs:
`localhost:8000` (API), `localhost:9090` (Prometheus), `localhost:3000` (Grafana,
`admin`/`admin` - desenvolvimento local, não produtivo). Dashboard já vem provisionado por
arquivo, sem clique na UI - painéis descritos na seção 5. Decisões em
[docs/specs/OBS-001.md](docs/specs/OBS-001.md).

### 3.3 Pipeline de treino (Airflow)

DAG com 2 tasks (`load_dataset → train_baseline`), container standalone - testei e confirmei que
**Airflow nativo não roda no Windows** (quebra já na importação básica), por isso Docker em vez
de venv local, em qualquer sistema operacional.

```bash
# Linux, macOS, Git Bash (Windows)
./scripts/run_airflow_dag.sh
```

```powershell
# PowerShell nativo do Windows
.\scripts\run_airflow_dag.ps1
```

Os dois resolvem a raiz do repositório sozinhos (rodam de qualquer diretório) e confirmam que o
`.joblib` foi **atualizado** pela execução, não só que ele existe. `load_dataset` reusa
`data/raw/fedmml_ed_triage_raw.parquet` se já existir; senão tenta o cache local do Hugging Face;
só baixa de verdade (com o `HF_TOKEN` do `.env`) como último recurso. Comando manual equivalente
(bash e PowerShell) e o troubleshooting real por trás desses dois scripts -
inclusive um bug sutil que quase reintroduzi tentando simplificar o comando - estão em
[docs/specs/AIR-001.md](docs/specs/AIR-001.md).

**Dois scripts, dois propósitos diferentes - não confundir**: `run_airflow_dag.sh`/`.ps1` (acima)
**provam** que a DAG funciona (`airflow dags test`, é o que uso pra validar de verdade e é o que
o CI/desenvolvimento usariam). `run_airflow_ui.sh`/`.ps1` é só pra **demonstração visual** - sobe
`airflow standalone` (scheduler + webserver + triggerer de pé, UI completa em
`localhost:8080`), não é modo de produção:

```bash
./scripts/run_airflow_ui.sh       # ou .\scripts\run_airflow_ui.ps1 no PowerShell
```

A senha do usuário `admin` é gerada sozinha. Testei e confirmei que o jeito que funciona é ler o
arquivo (o banner de senha não apareceu no log nesta sessão, mesmo com o webserver já
respondendo) - em outro terminal, com o script ainda rodando:

```bash
docker exec air001-ui-standalone cat /opt/airflow/standalone_admin_password.txt
```

Pra disparar a DAG manualmente pela UI: acesse `localhost:8080`, entre com `admin`/a senha do
arquivo, ative o toggle ao lado de `air001_train_baseline` (vem pausada por padrão), clique no
nome da DAG e no botão de play (▶) pra disparar uma run - acompanha em tempo real na visão Grid.

### 3.4 CI/CD

`.github/workflows/ci.yml`: `lint` (ruff) e `test` (pytest) em paralelo, `build` (builda a imagem
e valida `/health`+`/predict` reais dentro dela) só depois dos dois passarem. Roda em todo PR e
push pra `main`; testei de propósito quebrando um lint e confirmando que o GitHub recusa o merge
com branch protection ativa, não só reporta vermelho. Decisões em
[docs/specs/CI-001-ci-pipeline.md](docs/specs/CI-001-ci-pipeline.md).

### 3.5 Como testar cada etapa

Comandos reais que rodei pra confirmar cada peça isoladamente, não só "subir tudo e torcer":

**A API está respondendo?**
```bash
curl -s http://localhost:8000/health
# {"status":"ok"}
curl -s -X POST http://localhost:8000/predict -H 'content-type: application/json' \
  -d '{"clinical_notes": "67yo M c/o chest pain"}'
# {"urgencia":"urgente","probabilidades":{...}}
```

**O Prometheus está coletando a API?**
```bash
curl -s http://localhost:9090/api/v1/targets | grep -o '"health":"[a-z]*"'
# "health":"up"
```
Se vier `"down"`, o alvo `api:8000` não é um erro de configuração - é o nome do serviço no
`docker-compose.yml`, resolvido pelo DNS interno do Compose (não `localhost`, que de dentro do
container do Prometheus apontaria pra ele mesmo). `down` normalmente significa que a API ainda
não terminou de subir - espere o `docker ps` mostrar `healthy` e tente de novo.

**O dashboard está populado?**
```bash
curl -s -u admin:admin http://localhost:3000/api/dashboards/uid/obs-001-clinical-triage \
  | python -c "import json,sys; d=json.load(sys.stdin); print(d['meta']['provisioned'], len(d['dashboard']['panels']))"
# True 5
```
Confirma que os 5 painéis foram provisionados por arquivo. Pra ver dado de verdade nos gráficos,
gere tráfego primeiro (os `curl` da seção "API está respondendo" já bastam) e abra
`localhost:3000` - o dashboard atualiza sozinho a cada 10s.

**A DAG do Airflow rodou com sucesso?**
Os scripts da seção 3.3 já fazem essa confirmação sozinhos (checam que `models/tfidf_logreg_
baseline.joblib` foi atualizado pela execução, não só que a DAG "reportou sucesso" - motivo
completo em `docs/specs/AIR-001.md`). Rodar de novo é a própria checagem:
```bash
./scripts/run_airflow_dag.sh   # termina com "Tudo certo." e exit code 0, ou explica o que falhou
```

## 4. Resultados

### 4.1 Modelo

Baseline: TF-IDF + Logistic Regression (`class_weight="balanced"`), treinado sobre
`clinical_notes`. Split treino/teste estratificado (80/20, `random_state=42`). Ver
[notebooks/02_baseline.ipynb](notebooks/02_baseline.ipynb) e métricas completas em
[docs/experiments/ML-003-baseline-metrics.json](docs/experiments/ML-003-baseline-metrics.json).

#### 4.1.1 Limitações do dataset - vazamento de rótulo

Durante a EDA (ML-002) e confirmado empiricamente na ML-003, achei que o campo `clinical_notes`
do `fedmml-ed-triage` é gerado por um template fixo: as 28 categorias de `chief_complaint` e as 5
variantes de cláusula final do texto mapeiam pra classe de urgência com 100% de precisão, sem
exceção, nas 85.679 notas verificadas.

Pra confirmar isso, comparei o baseline de texto (TF-IDF + Logistic Regression) lado a lado com
um baseline ingênuo que classifica só pelo `chief_complaint` (dicionário complaint → classe
majoritária):

| Modelo | F1 macro | Recall macro | Recall (`urgente`) |
|---|---|---|---|
| TF-IDF + Logistic Regression | 1.00 | 1.00 | 1.00 |
| Ingênuo (só `chief_complaint`) | 1.00 | 1.00 | 1.00 |

*(17.136 exemplos de teste; matrizes de confusão idênticas, diagonais perfeitas - ver
[métricas completas](docs/experiments/ML-003-baseline-metrics.json))*

Os dois empatam exatamente. O classificador de texto não demonstra nenhuma capacidade real de
NLP clínico - ele decorou uma tabela de busca embutida no template do gerador sintético, não
aprendeu linguagem clínica. Validei que esse vazamento é específico do texto: os campos de
vitais/labs (fora do escopo do classificador), como `spo2`, `heart_rate` e troponina, seguem
distribuições com sobreposição real entre classes, sem determinismo perfeito.

Decidi manter o dataset e reportar isso com transparência, em vez de trocar de fonte ou esconder
o resultado - decisão completa e alternativas descartadas em
[ADR-002](docs/decisions/ADR-002-text-leakage.md). As métricas de ~100% deste projeto não devem
ser lidas como "o modelo é excelente" - são evidência de um artefato de geração de dados
sintéticos. Reconfirmei isso duas vezes de forma independente, sem planejar: com dados baixados
de novo do zero na AIR-001, e com 150 chamadas reais ao `/predict` na OBS-001. Mesmo padrão, 100%
das vezes.

### 4.2 Otimização (ONNX) e benchmark

Converti o baseline pra ONNX via `skl2onnx` - o **pipeline inteiro**, não só o classificador.
Funcionou de primeira: o `TfidfVectorizer` foi treinado com hiperparâmetros default, o caso mais
simples de converter, então nem cheguei a precisar do fallback (classificador isolado, TF-IDF em
sklearn) que deixei implementado pra esse cenário.

Validei corretude **antes** de medir qualquer velocidade - não adianta ser mais rápido se estiver
errado: 150 textos reais do dataset, 150/150 classes batendo entre sklearn e ONNX, diferença
máxima de probabilidade de `8.11e-08` (tolerância usada: `1e-4`).

Latência (300 iterações, mesma entrada, mesma máquina, depois de warmup):

| | sklearn original | ONNX (onnxruntime) | Speedup |
|---|---|---|---|
| Mediana | 0.40 ms | 0.036 ms | **~11x** |
| p95 | 0.55 ms | 0.050 ms | **~11x** |

Tamanho do artefato: 9.044 bytes (`.joblib`) contra 7.076 bytes (`.onnx`), -22%.

Uma ressalva que preciso deixar clara: esses 0.4ms do sklearn já são extremamente baixos em
termos absolutos - a latência ponta-a-ponta da API real (HTTP + serialização, medida na OBS-001)
fica na casa de 6-9ms, dominada por overhead de rede, não pela inferência. O ganho de ~11x é real
na camada de inferência isolada; não medi (não era o objetivo deste card) o quanto isso se
traduziria em latência percebida pelo cliente se a API passasse a servir via ONNX - troca que não
fiz em produção, isto aqui é comparação/benchmark. Script reprodutível e a trajetória completa da
conversão em [docs/specs/OPT-001.md](docs/specs/OPT-001.md) e
[docs/experiments/OPT-001-benchmark.json](docs/experiments/OPT-001-benchmark.json).

## 5. Monitoramento

Prometheus + Grafana sobem junto com a API via `docker-compose up` (seção 3.2), dashboard
provisionado por arquivo. Os 5 painéis, todos validados com dado real - não só a métrica
existindo, o número batendo com o tráfego que gerei:

1. **Taxa de requisições** (por rota), a partir de `http_requests_total` (RED, OBS-001 Passo 1).
2. **Latência p95** (por rota), `histogram_quantile` sobre `http_request_duration_seconds_bucket`
   - os buckets padrão da lib são pensados pra web genérica e dariam zero resolução aqui, então
   calibrei pra latência sub-segundo.
3. **Distribuição de classes previstas ao longo do tempo**, `rate(triage_predictions_total)` por
   classe - proxy de drift: sem rótulo verdadeiro disponível em produção, uma mudança anormal na
   proporção normal/atenção/urgente é o sinal indireto que tenho.
4. **Confiança por classe** (p50 e p95, linhas simples), sobre
   `triage_prediction_confidence_bucket` - buckets recalibrados na prática depois que a hipótese
   inicial se mostrou conservadora demais (detalhes na seção 6).
5. **Taxa de erro**, `http_requests_total` filtrando status diferente de `2xx`. "No data" é o
   estado normal enquanto não houver erro real na janela do painel - confirmei gerando um erro de
   propósito e vendo o painel reagir na hora, não é bug nem configuração errada.

Os painéis 3 e 4 não são detecção estatística formal de drift - são gatilho pra eu (ou quem
estiver de plantão) ir investigar, não um alarme automático. Decisões completas em
[docs/specs/OBS-001.md](docs/specs/OBS-001.md).

## 6. Limitações e lições aprendidas

O achado mais importante do projeto inteiro é o vazamento de rótulo (seção 4.1.1): as métricas de
~100% não provam capacidade de NLP clínico, provam que o modelo decorou um template. É a lição de
MLOps que mais levo daqui - validar o dado antes de comemorar a métrica, e reportar isso com
transparência em vez de esconder.

Descobri outra coisa só rodando tráfego de verdade contra o dashboard: métrica bem desenhada não
é o mesmo que visualização útil. Os buckets de confiança que calibrei no Passo 2 da OBS-001 tinham
o label certo, o teste unitário passava - e mesmo assim `histogram_quantile` devolvia `NaN`
contra o modelo real, porque a distribuição real caía numa faixa de 0,005 de largura que os
buckets simplesmente não resolviam. Só apareceu quando gerei 90 chamadas reais e olhei o número,
não no design isolado da métrica.

Relacionado: `rate()`/`histogram_quantile()` precisam de tráfego espalhado no tempo, não uma
rajada. Uma rajada instantânea prova que a métrica está correta e ainda assim produz `rate() ==
0` (ou resultado vazio) num painel - o Prometheus não viu a transição, só o valor final parado.
Peguei isso duas vezes, sem querer: uma na validação do Passo 5 da OBS-001, outra de novo
validando o painel de erro pra este README.

Drift de ambiente entre quem treina e quem serve é risco real, não teórico. O container do
Airflow instalou uma versão de `scikit-learn` diferente da API porque eu não tinha fixado versão
no `requirements.txt` - o `.joblib` carregou mesmo assim (só um warning), mas podia não ter
carregado. O mesmo aconteceu com `pyarrow`: sem versão fixa, quebrou a leitura de um parquet
escrito por outra versão (`OSError: Repetition level histogram size mismatch`). Fixar versão
exata nos dois lados deixou de ser opcional antes de eu converter pra ONNX.

O Airflow nativo não roda no Windows - não assumi isso, testei: o próprio `import airflow`
imprime um aviso da lib avisando, e uma importação básica (`DagBag`) já quebra. Rodar em Docker
desde o início evitou eu ter que voltar atrás nessa decisão depois.

E a lição mais recente, literalmente da véspera do vídeo: documentação errada quebra ambiente de
verdade, não é só "falta de polimento". As instruções do Airflow que eu tinha escrito eram só
bash e diziam explicitamente que funcionariam em PowerShell - não funcionavam, quebravam por
completo. Pior: ao tentar simplificar o comando pra fugir desse bug de shell usando uma data fixa
arbitrária, criei um segundo bug, mais grave - o Airflow marca a run como sucesso, com exit code
0, sem rodar nenhuma task, se a data passada for anterior ao `start_date` da DAG. Silencioso, sem
erro nenhum. Só peguei porque o script de validação que escrevi checa se o arquivo do modelo foi
de fato atualizado, não só se a DAG "reportou sucesso".

## 7. Decisões arquiteturais

Ver [docs/decisions/](docs/decisions/) - ADR-001 (dataset), ADR-002 (vazamento de rótulo),
ADR-005 (estratégia de cloud, seção 2).

## 8. Estrutura do projeto

```
clinical-triage-mlops/
├── .github/workflows/          # CI (lint, test, build) - CI-001
├── airflow/
│   ├── dags/                   # DAG de treino - AIR-001
│   └── Dockerfile
├── benchmarks/                 # conversão ONNX + benchmark - OPT-001
├── data/{raw,processed}/
├── docs/{decisions,specs,experiments}/
├── monitoring/{prometheus,grafana}/  # scrape config + dashboard provisionado - OBS-001
├── notebooks/
├── scripts/                    # run_airflow_dag.sh/.ps1 - AIR-001
├── src/api/                    # FastAPI - API-001
├── tests/
├── models/                     # .joblib/.onnx (gitignored, gerado localmente)
├── .claude/CLAUDE.md
├── Dockerfile                  # DOCK-001
├── docker-compose.yml          # OBS-001
└── pyproject.toml
```

## 9. Roadmap

Ver [docs/ROADMAP.md](docs/ROADMAP.md) - concluído até o `STEP 11`; falta só o vídeo (`STEP 12`).
