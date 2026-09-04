# DOCK-001: Dockerizar a API de inferência

## Problema

A API da [API-001](API-001-predict-endpoint.md) só existe hoje como código que se roda com
`uv run uvicorn src.api.main:app` a partir de um checkout do repositório, com o `.venv` local e o
`.joblib` gerado por notebook. Não há uma unidade deployável: nada que se possa entregar a um
runtime de containers, a um registry, ou a um pipeline de CI/CD e esperar que suba igual em
qualquer lugar. Os STEPs seguintes do roadmap (CI/CD — EPIC 07, cloud — EPIC 12) pressupõem essa
unidade.

## Objetivo

Produzir uma imagem Docker **autossuficiente** da API de inferência: `docker run <imagem>` sobe o
serviço e responde em `/health` e `/predict` sem depender de volume, rede externa ou secret. A
imagem é a unidade deployável do projeto a partir deste ponto.

## Escopo

- `Dockerfile` single-stage para a API FastAPI (`src/api/`)
- Separar as dependências de runtime da API das dependências de notebook/treino no
  `pyproject.toml` (grupo opcional), para a imagem instalar só o que a inferência usa
- `.dockerignore` endurecido (contexto de build enxuto + nenhum secret na imagem)
- `HEALTHCHECK` no container apontando para o `GET /health` da API-001
- README seção 7 ("Como executar") com as instruções `docker build` / `docker run`
- Medição e registro do tamanho da imagem final (critério de aceite)

## Fora de escopo

- `docker-compose.yml` — um único container não justifica orquestração; o compose entra quando
  houver mais de um serviço para subir junto (Prometheus + Grafana, EPIC 09)
- Workflow de CI que constrói/publica a imagem (EPIC 07) — aqui a imagem é construída e validada
  localmente
- Publicação em registry (GHCR, ECR, etc.) — decisão de EPIC 07/12
- Multi-stage build — avaliado e descartado para este escopo (ver "Riscos" e a discussão no
  histórico do card): não há toolchain de compilação a descartar, todas as dependências instalam
  de wheel. Os dois custos que multi-stage tiraria — o binário do `uv` (~58 MB descompactado) e o
  cache de download do `uv` — foram resolvidos em single-stage com `RUN --mount=from=` e
  `RUN --mount=type=cache` (ver "Interface esperada"). O que sobra para multi-stage justificar
  (dois `FROM` + `COPY --from` + gestão de path de venv) não se paga aqui. Reavaliar se entrar
  uma dependência que compile do fonte.
- Download do modelo em runtime / model registry — o `.joblib` entra por `COPY` no build (ver
  "Fluxo de dados"); revisitar quando houver retraining automatizado (EPIC 08) ou arquitetura de
  cloud (EPIC 12), aí como ADR próprio
- Otimização de tamanho de imagem além da dieta de dependências (distroless, `--platform`
  específico, remoção cirúrgica de `site-packages`) — EPIC 10/11 se o benchmark pedir
- Tuning de workers/uvicorn-gunicorn para carga — a API-001 já definiu "um processo, modelo
  carregado uma vez"; ajuste de concorrência é assunto de benchmark (EPIC 11)

## Requisitos funcionais

- RF1: `docker build` a partir da raiz do repositório produz uma imagem sem erro, com o
  `.joblib` do modelo embutido
- RF2: `docker run -p 8000:8000 <imagem>` sobe o serviço; `GET /health` devolve `200`
  `{"status": "ok"}`
- RF3: `POST /predict` com `{"clinical_notes": "<texto>"}` devolve `200` com `urgencia` em
  `{normal, atencao, urgente}` e `probabilidades` com as três classes somando ~1.0 — mesmo
  contrato da API-001
- RF4: se o `.joblib` não estiver no contexto de build, o `docker build` **falha no passo de
  `COPY`** (não produz uma imagem quebrada que só falha em runtime)
- RF5: o container expõe um `HEALTHCHECK`; `docker ps` mostra `healthy` depois que o serviço
  sobe, `unhealthy` se o processo morrer ou parar de responder
- RF6: `pandas`, `pyarrow` e `huggingface-hub` continuam instaláveis para quem for rodar os
  notebooks, via um grupo opcional (`uv sync --extra data`) — `notebooks/01_eda_dataset.ipynb` e
  `notebooks/02_baseline.ipynb` não podem quebrar

## Requisitos não funcionais

- RNF1: imagem base `python:3.12-slim-bookworm` (tag fixa, não `-slim` flutuante). Se aparecer
  erro de biblioteca de sistema ausente (ex: `libgomp.so.1`), resolver com uma linha de
  `apt-get install --no-install-recommends`, não trocando para a imagem full
- RNF2: single-stage
- RNF3: instalação de dependências reprodutível — a partir do `uv.lock` (`uv sync --frozen`), não
  resolução solta
- RNF4: camadas ordenadas para cache — manifestos de dependência (`pyproject.toml` + `uv.lock`)
  copiados e instalados **antes** do código-fonte, para que mudança em `src/` não invalide a
  camada de dependências
- RNF5: processo roda como usuário **não-root**
- RNF6: `COPY` do modelo é **explícito** para o arquivo único
  (`COPY models/tfidf_logreg_baseline.joblib ...`), não `COPY models/` nem wildcard — não
  arrastar o `naive_chief_complaint_baseline.json` nem outros artefatos futuros
- RNF7: o `HEALTHCHECK` não pode depender de `curl`/`wget` (não existem na slim) — usar Python
  puro (`urllib`)
- RNF8: nenhum secret na imagem — `.env` (contém `HF_TOKEN`) fora do contexto de build via
  `.dockerignore`; `COPY` do código é seletivo (`src/`), não `COPY . .`
- RNF9: `MODEL_PATH` continua configurável por variável de ambiente (herdado da API-001); o
  default do código deve resolver corretamente dentro da imagem sem precisar setar a env var

## Interface esperada

### `pyproject.toml` — grupo opcional

```toml
[project]
dependencies = [
    "fastapi",
    "uvicorn",
    "scikit-learn",
    "prometheus-client",
    "python-dotenv>=1.2.3",
]

[project.optional-dependencies]
data = [
    "pandas",
    "pyarrow>=25.0.1",
    "huggingface-hub>=1.27.0",
]
```

- `joblib` não precisa ser dependência direta — vem transitivamente com `scikit-learn` e é o que
  `src/api/main.py` usa para desserializar. Fica como está (import transitivo já em uso na
  API-001).
- `prometheus-client` e `python-dotenv` permanecem em `dependencies`: o primeiro é preparação da
  EPIC 09 e é leve; o segundo é usado no boot. Nenhum dos dois pesa como `pandas`/`pyarrow`.

### `Dockerfile` — estrutura (não o arquivo final, que sai na implementação)

```
FROM python:3.12-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 PYTHONPATH=/app
WORKDIR /app

# camada de dependências (cacheável) — o binário do uv entra só via --mount
# durante o RUN, nunca vira layer da imagem final (uv não roda em runtime)
COPY pyproject.toml uv.lock ./
RUN --mount=from=ghcr.io/astral-sh/uv:0.11.7,source=/uv,target=/bin/uv \
    --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# código + modelo (COPY explícito do arquivo único)
COPY src/ ./src/
COPY models/tfidf_logreg_baseline.joblib ./models/tfidf_logreg_baseline.joblib

# usuário não-root, SEM chown -R (a API só lê seus arquivos; chown recursivo
# duplicaria o /app inteiro numa layer nova)
RUN useradd --create-home --uid 1000 appuser
USER appuser

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"]

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Pontos fixados na implementação:
- **`--no-install-project`**: instala só as dependências, não o pacote `src` — evita a build do
  hatchling precisar do `README.md` no contexto. O código roda via `src.api.main:app` com
  `PYTHONPATH=/app`, não como pacote instalado.
- **`MODEL_PATH` default**: `src/api/main.py` resolve
  `Path(__file__).resolve().parents[2] / "models" / "tfidf_logreg_baseline.joblib"`. Com o código
  em `/app/src/api/main.py`, `parents[2]` = `/app` → default = `/app/models/tfidf_logreg_baseline.joblib`.
  Bate com o destino do `COPY`. Não precisa setar `MODEL_PATH` na imagem.
- **`uv` fora da imagem final**: a primeira versão usava `COPY --from=ghcr.io/astral-sh/uv /uv
  /bin/uv`, que deixava o binário do uv (~58 MB descompactado / ~20 MB compactado) numa layer da
  imagem final, inútil em runtime. Trocado por `RUN --mount=from=...` (feature do BuildKit): o
  binário é montado só durante o `uv sync` e não vira layer. Continua single-stage — não há
  segundo `FROM` nem `COPY --from` para o estágio final.
- **`--mount=type=cache` no cache do uv**: sem ele, os wheels baixados ficavam dentro da própria
  layer do `RUN uv sync` (cache + venv na mesma layer, ~520 MB descompactado). Com o cache num
  mount, só o venv fica na layer (~295 MB descompactado). Não afeta reprodutibilidade (o
  `--frozen` garante isso), só tamanho e velocidade de rebuild.
- **Versão do `uv`**: tag fixa `ghcr.io/astral-sh/uv:0.11.7` (mesma linha do `uv` local), não
  `latest`.

### `.dockerignore` — alvo

Adicionar aos itens já existentes (`.git`, `.venv`, `__pycache__`, `*.pyc`, `notebooks/`,
`tests/`, `docs/`, `data/raw`, `.pytest_cache`, `.ruff_cache`):

```
.env
.env.*
data/
.github/
airflow/
benchmarks/
monitoring/
.pre-commit-config.yaml
.vscode/
.idea/
.DS_Store
Dockerfile
.dockerignore
```

Manter no contexto: `pyproject.toml`, `uv.lock`, `src/`, `models/tfidf_logreg_baseline.joblib`.

### README seção 7 — "Como executar"

Bloco novo com: pré-requisito (gerar o `.joblib` via `notebooks/02_baseline.ipynb`),
`docker build -t clinical-triage-api .`, `docker run -p 8000:8000 clinical-triage-api`, exemplo
de `curl` para `/health` e `/predict`, e a nota de que rodar sem Docker é
`uv sync && uv run uvicorn src.api.main:app` (com `uv sync --extra data` para os notebooks).

## Fluxo de dados

### Build

1. `docker build` recebe o contexto (raiz do repo menos o que o `.dockerignore` exclui)
2. `COPY --from` traz o binário do `uv`
3. `COPY pyproject.toml uv.lock` → `uv sync --frozen --no-dev --no-install-project` cria
   `/app/.venv` só com as deps de runtime (sem grupo `dev`, sem grupo `data`)
4. `COPY src/` → código da API
5. `COPY models/tfidf_logreg_baseline.joblib ./models/` — **se o arquivo não estiver no contexto,
   o build falha aqui** (RF4)
6. cria `appuser`, ajusta dono, `USER appuser`
7. imagem final: base slim + `.venv` + `src/` + o `.joblib`

### Runtime

1. `docker run` → `CMD` inicia `uvicorn src.api.main:app`
2. `lifespan` da API-001 carrega `/app/models/tfidf_logreg_baseline.joblib` via `joblib.load`
   (modelo ausente → processo não sobe, herdado da API-001)
3. `HEALTHCHECK` começa a rodar após `--start-period`; `GET /health` interno → `docker ps` marca
   `healthy`
4. requisições externas em `POST /predict` seguem o fluxo da API-001

## Critérios de aceite

- `docker build -t clinical-triage-api .` termina sem erro (RF1)
- **Tamanho da imagem medido e registrado** — `docker images clinical-triage-api` colado no
  Resultado do card DOCK-001 no KANBAN e na seção "Métricas" desta spec (com o número real, não
  estimativa)
- `docker run -p 8000:8000 clinical-triage-api` sobe; `curl localhost:8000/health` → `200`
  `{"status":"ok"}` (RF2)
- `curl -X POST localhost:8000/predict -H 'content-type: application/json' -d '{"clinical_notes":"..."}'`
  → `200`, `urgencia` válida, três probabilidades somando ~1.0 — output real colado na validação
  (RF3)
- `docker build` sem o `.joblib` no contexto falha no `COPY` do modelo, não em runtime (RF4) —
  verificado
- `docker ps` mostra `healthy` para o container depois do `start-period` (RF5)
- `uv sync --extra data` instala `pandas`/`pyarrow`/`huggingface-hub`; `uv sync` (sem extra) não
  os instala (RF6)
- `uv run ruff check .` limpo; `uv run pytest` passando (a mudança no `pyproject.toml` não pode
  quebrar os testes da API-001)
- Dois commits na branch: `chore:` (dieta de deps) e `feat:` (Dockerfile + `.dockerignore` +
  README) — separados, histórico legível

## Estratégia de testes

Não há teste automatizado novo em `tests/` — o que DOCK-001 entrega (Dockerfile, `.dockerignore`,
grupos de dependência) não é código Python exercitável por `pytest`. A validação é manual e
reprodutível, registrada na sessão:

1. `uv lock` após editar o `pyproject.toml`; `uv sync` e `uv run pytest` para confirmar que a
   suíte da API-001 continua verde sem os pacotes movidos
2. `uv sync --extra data` e um smoke-check de import (`python -c "import pandas, pyarrow,
   huggingface_hub"`) para confirmar RF6
3. `docker build` → `docker images` (tamanho)
4. `docker run` → `curl /health` e `curl /predict` com um laudo de exemplo → outputs colados
5. `docker ps` para o status `healthy`
6. Teste negativo do RF4: renomear temporariamente o `.joblib` (ou usar um contexto sem ele) e
   confirmar que o `docker build` falha no `COPY` do modelo com mensagem clara

A automação disso (build da imagem + `curl` de fumaça) é responsabilidade do workflow de CI na
EPIC 07, fora do escopo aqui.

## Métricas

**Tamanho da imagem final: 132 MB** (`docker image inspect clinical-triage-api --format
'{{.Size}}'` = 132 040 532 bytes ≈ 126 MiB; mesma coluna `CONTENT SIZE` do `docker images` no
Docker 29 com containerd store — é o tamanho compactado, o que se paga num `docker pull`).
`DISK USAGE` descompactado em disco: ~568 MB.

Base `python:3.12-slim-bookworm` sozinha ≈ 141 MB descompactada. O peso da imagem é dominado pelo
venv de runtime (numpy + scipy + scikit-learn); `fastapi`/`uvicorn`/`pydantic` somam pouco.

Trajetória da medição durante a implementação (todos single-stage, mesma base, só o Dockerfile
mudando) — registra o efeito de cada decisão:

| Versão do Dockerfile | `CONTENT SIZE` |
|---|---|
| `chown -R appuser /app` + `COPY --from` do uv + cache do uv na layer | 308 MB |
| sem `chown -R` (só `USER appuser` no fim) | 221 MB |
| `uv` via `--mount=from=` + cache do uv via `--mount=type=cache` | **132 MB** |

As deps de notebook/treino (`pandas`, `pyarrow`, `huggingface-hub`) foram tiradas do runtime da
imagem por decisão desta tarefa (commit `chore:` separado, grupo `[project.optional-dependencies]
data`) — a imagem nunca chegou a ser construída com elas; a economia (estimada em ~250 MB
descompactado) está embutida em todos os números da tabela.

Tempo de `docker build`: ~71 s do zero (download da base + wheels), ~3–26 s com cache quente —
informativo, não é critério de aceite.

## Riscos

- **`.joblib` gitignored + `COPY` no build**: um `docker build` num ambiente que não tem o
  artefato (clone limpo, CI antes de existir passo de geração do modelo) falha no `COPY`. É o
  comportamento desejado (RF4), mas registra a dependência: o workflow de CI da EPIC 07 vai
  precisar de um passo que gera ou restaura o `.joblib` antes do `docker build`.
- **`libgomp` / bibliotecas de sistema na slim**: os wheels recentes de `scikit-learn` vendoram a
  runtime de OpenMP, então normalmente a slim basta. Se o `import sklearn` dentro do container
  falhar por `.so` ausente, a correção é uma linha de `apt-get install --no-install-recommends
  libgomp1` — a validação (subir o container e fazer `/predict`) cobre esse risco.
- **Mudança no `pyproject.toml` quebrar os testes**: `tests/test_api.py` tem um teste `slow` que
  treina um `Pipeline` scikit-learn em memória e um que carrega o modelo real — nenhum usa
  `pandas`/`pyarrow`. O risco é baixo, mas a validação roda `uv run pytest` completo antes do
  commit para confirmar.
- **`--no-install-project` e resolução de import**: se o `uv sync` não instalar o pacote `src`, o
  `uvicorn src.api.main:app` depende do `WORKDIR`/`PYTHONPATH` apontar para `/app`. Se der
  `ModuleNotFoundError: src`, a alternativa é `ENV PYTHONPATH=/app` explícito ou instalar o
  projeto (`--no-install-project` removido, exige `README.md` no contexto). Decidir na
  implementação conforme o comportamento real.
- **Cache de camada do modelo**: `COPY` do `.joblib` depois do `COPY src/` significa que retreinar
  o modelo invalida menos camadas. Como o modelo é gitignored e muda por fora do git, e é 9 KB,
  o impacto de cache é irrelevante — a ordem escolhida é por clareza (deps → código → artefato),
  não por otimização.

## Perguntas em aberto

Nenhuma — decisões fechadas na discussão prévia (registrada no histórico do card DOCK-001):
`COPY` no build, single-stage, `python:3.12-slim-bookworm`, dieta de deps dentro do DOCK-001 como
commit `chore:` separado, `HEALTHCHECK` via `urllib`, `docker-compose.yml` fora de escopo.

## Experimentos

Não aplicável — não há hipótese de ML/latência sendo testada. A única medição (tamanho da imagem)
é um critério de aceite objetivo, não um experimento com hipótese.
