# Imagem de inferência da Clinical Triage API (DOCK-001).
# Single-stage: todas as dependências instalam de wheel, não há toolchain de
# compilação a descartar, então multi-stage não se pagaria aqui.
# Spec: docs/specs/DOCK-001-dockerize-api.md

FROM python:3.12-slim-bookworm

# PYTHONDONTWRITEBYTECODE: não gerar .pyc em disco no runtime (o UV_COMPILE_BYTECODE
#   abaixo já pré-compila no build).
# PYTHONUNBUFFERED: stdout/stderr sem buffer, para o log aparecer na hora em `docker logs`.
# UV_LINK_MODE=copy: evita warning de hardlink entre o cache do uv e o /app/.venv.
# UV_COMPILE_BYTECODE=1: pré-compila os .pyc no build → startup do container mais rápido.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PYTHONPATH=/app

WORKDIR /app

# --- Camada de dependências (cacheável) ------------------------------------- -
# Copiada e instalada ANTES do código: mudar src/ não invalida esta layer.
# O binário do uv entra via --mount=from=... só durante este RUN — nunca vira
# layer da imagem final (uv não é usado em runtime). --mount=type=cache guarda
# o cache de download do uv entre builds (não afeta o tamanho da imagem).
#   --frozen            : usar o uv.lock como está, sem re-resolver
#   --no-dev            : sem o grupo [dependency-groups] dev (pytest, ruff, ...)
#   --no-install-project: instalar só as dependências, não o pacote `src` — assim
#                         o build não precisa do README.md no contexto; o código
#                         roda via PYTHONPATH=/app, não como pacote instalado
COPY pyproject.toml uv.lock ./
RUN --mount=from=ghcr.io/astral-sh/uv:0.11.7,source=/uv,target=/bin/uv \
    --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# --- Código + modelo ------------------------------------------------------- --
COPY src/ ./src/
# COPY explícito do artefato único (não `models/` inteiro): se o .joblib não
# estiver no contexto de build, o build FALHA AQUI — não gera imagem quebrada.
COPY models/tfidf_logreg_baseline.joblib ./models/tfidf_logreg_baseline.joblib

# Usuário não-root. Sem `chown -R`: a API só LÊ seus arquivos em runtime, e os
# arquivos copiados já são world-readable — um chown recursivo só criaria uma
# layer duplicada do /app inteiro (incl. o venv de ~500 MB).
RUN useradd --create-home --uid 1000 appuser
USER appuser

# Coloca o /app/.venv/bin no PATH → `uvicorn` e `python` resolvem para o venv.
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

# Healthcheck bate no GET /health da API-001. Python puro (urllib) porque a slim
# não traz curl/wget. --start-period cobre o boot do uvicorn + load do modelo.
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD ["python", "-c", "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status == 200 else 1)"]

# Um processo, modelo carregado uma vez no startup (decisão da API-001).
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
