# CI-001: Pipeline CI (GitHub Actions) — lint → testes → build

## Problema

Hoje, lint (`ruff`), testes (`pytest`) e a validação de que a imagem Docker builda (DOCK-001) só
rodam se alguém lembrar de rodar localmente antes de abrir/mergear um PR. Nada impede uma
regressão de chegar em `main`. O Tech Challenge Fase 3 pede explicitamente um pipeline CI/CD.

## Objetivo

Automatizar lint, testes e validação de build via GitHub Actions, disparado em todo PR para
`main` (e em push direto na `main`, como rede de segurança), com o pipeline **de fato bloqueando
merge** quando falha — não só reportando vermelho.

## Escopo

- Workflow `.github/workflows/ci.yml` com 3 jobs: `lint`, `test`, `build`
- Script `scripts/gen_placeholder_model.py` — gera um modelo sintético só para o job de `build`
  conseguir validar o `COPY` do Dockerfile sem depender do `.joblib` real (gitignored)
- Cache de dependências (`uv`) entre runs
- Branch protection em `main` marcando os 3 jobs como *required status checks*
- Demonstração real do critério de aceite: um commit que quebra lint de propósito, dentro do
  próprio PR do CI-001, seguido do commit que conserta

## Fora de escopo

- **CD** (deploy automatizado, push de imagem para registry) — depende de registry/cloud, decisão
  do EPIC 12. Este card cobre só a metade CI do "pipeline CI/CD" do enunciado.
- Testar o modelo real (`@pytest.mark.slow`) em CI — exigiria trazer o `.joblib` real (secret do
  Hugging Face + download, ou artefato versionado), desproporcional ao objetivo deste card. O
  teste `slow` já se auto-skipa em CI hoje (não há `.joblib` no runner) — comportamento herdado da
  API-001, não uma decisão nova.
- Cache de layers Docker via buildx (`type=gha`) — o build frio (~70s) é aceitável; fica como
  otimização futura se incomodar.
- Publicar/gerar relatório de cobertura (Codecov, etc.) — só *enforçar* o `fail_under` já
  configurado em `pyproject.toml`, sem publicar em serviço externo.
- Notificações (Slack, e-mail) de falha de CI — o próprio PR/commit status já é o canal.

## Requisitos funcionais

- RF1: todo `pull_request` para `main` dispara os 3 jobs (`lint`, `test`, `build`)
- RF2: push direto em `main` também dispara os 3 jobs (rede de segurança pós-merge)
- RF3: `lint` roda `ruff check .` e `ruff format --check .`
- RF4: `test` roda `pytest -m "not slow" --cov --cov-fail-under=70` — os testes da API-001 sem o
  teste de integração com modelo real, com cobertura enforçada
- RF5: `build` roda **depois** de `lint` e `test` passarem (`needs: [lint, test]`), gera um modelo
  placeholder, faz `docker build`, sobe o container e valida `GET /health` + `POST /predict` de
  verdade (não só `docker build` sem rodar nada)
- RF6: os 3 jobs aparecem como *required status checks* em `main` — um PR com qualquer um deles
  vermelho **não pode ser mergeado** pela UI do GitHub (branch protection), não é decoração
- RF7: falhas devem apontar a causa direto no log do job (sem precisar reproduzir localmente pra
  entender o que quebrou)

## Requisitos não funcionais

- RNF1: cache de dependências via `astral-sh/setup-uv@v6` (`enable-cache: true`), chaveado por
  `uv.lock` — evita rebaixar todas as wheels a cada run
- RNF2: `concurrency` cancela runs supersedidos do mesmo PR/branch (evita gastar minutos em runs
  obsoletos quando há push seguido de push)
- RNF3: `timeout-minutes` em cada job — um job travado não deve consumir minutos indefinidamente
- RNF4: `permissions: contents: read` no nível do workflow — o pipeline não escreve no repo, só lê
- RNF5: versão do `uv` no workflow fixada na mesma linha da usada localmente e no `Dockerfile`
  (`0.11.7`) e `python-version: "3.12"` explícito (não há `.python-version` no repo; sem pin, o uv
  poderia resolver para uma versão mais nova de Python que o `requires-python = ">=3.12"` permite)
- RNF6: o modelo placeholder gerado pelo `build` **não é versionado** — nasce no CI e também pode
  ser gerado localmente (`uv run python scripts/gen_placeholder_model.py`) para testar a imagem
  Docker sem rodar o notebook completo

## Interface esperada

### `scripts/gen_placeholder_model.py`

```python
"""Gera um modelo placeholder para validar o build da imagem Docker (CI-001).

NÃO é o baseline real — esse vem de notebooks/02_baseline.ipynb (ML-003). Este script treina um
Pipeline TF-IDF + LogisticRegression minúsculo sobre poucas linhas sintéticas (mesmas classes,
mesmo formato de tests/test_api.py), só para o Dockerfile ter algo válido para o COPY e o job de
build poder subir o container e bater em /health e /predict de verdade.
"""

from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

TEXTS = [...]   # mesmas 6 linhas sintéticas de tests/test_api.py, reaproveitadas
LABELS = [...]  # normal/normal/atencao/atencao/urgente/urgente

OUTPUT_PATH = Path("models/tfidf_logreg_baseline.joblib")


def main() -> None:
    pipeline = Pipeline([("tfidf", TfidfVectorizer()), ("clf", LogisticRegression(max_iter=1000))])
    pipeline.fit(TEXTS, LABELS)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, OUTPUT_PATH)
    print(f"Modelo placeholder salvo em {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
```

### `.github/workflows/ci.yml`

```yaml
name: CI

on:
  pull_request:
    branches: [main]
  push:
    branches: [main]
  workflow_dispatch:

concurrency:
  group: ci-${{ github.ref }}
  cancel-in-progress: true

permissions:
  contents: read

jobs:
  lint:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v6
        with:
          version: "0.11.7"
          enable-cache: true
          python-version: "3.12"
      - run: uv sync --frozen
      - run: uv run ruff check .
      - run: uv run ruff format --check .

  test:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v6
        with:
          version: "0.11.7"
          enable-cache: true
          python-version: "3.12"
      - run: uv sync --frozen
      - run: uv run pytest -m "not slow" --cov --cov-fail-under=70

  build:
    needs: [lint, test]
    runs-on: ubuntu-latest
    timeout-minutes: 15
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v6
        with:
          version: "0.11.7"
          enable-cache: true
          python-version: "3.12"
      - name: Gera modelo placeholder para o COPY do Dockerfile
        run: |
          uv sync --frozen --no-dev
          uv run python scripts/gen_placeholder_model.py
      - name: docker build
        run: docker build -t clinical-triage-api:ci .
      - name: Smoke test da imagem
        run: |
          docker run -d --name ci-smoke -p 8000:8000 clinical-triage-api:ci
          for i in $(seq 1 30); do
            s=$(docker inspect -f '{{.State.Health.Status}}' ci-smoke)
            echo "health=$s"; [ "$s" = healthy ] && break; sleep 1
          done
          curl -fsS http://localhost:8000/health
          curl -fsS -X POST http://localhost:8000/predict \
            -H 'content-type: application/json' \
            -d '{"clinical_notes":"67yo M c/o chest pain, moderate distress"}'
          docker rm -f ci-smoke
```

## Fluxo de dados

1. PR aberto/atualizado contra `main` (ou push direto em `main`) → GitHub dispara os 3 jobs
2. `lint` e `test` rodam em paralelo, cada um faz checkout + `setup-uv` (cache) + `uv sync`
3. `lint`: `ruff check` + `ruff format --check`; qualquer um falhando → job vermelho
4. `test`: `pytest -m "not slow" --cov --cov-fail-under=70`; falha de teste ou cobertura abaixo de
   70% → job vermelho
5. `build` só inicia se `lint` **e** `test` tiverem passado (`needs`); gera o placeholder, builda
   a imagem, sobe o container, espera `healthy`, bate em `/health` e `/predict` reais; qualquer
   passo falhando → job vermelho
6. Com branch protection (RF6): PR com qualquer job vermelho não pode ser mergeado pela UI —
   botão de merge fica bloqueado até os 3 ficarem verdes

## Critérios de aceite

- Os 3 jobs disparam automaticamente no PR do CI-001 e em pushes para `main`
- `lint` pega erro de formatação/lint real — **demonstrado**: commit proposital quebrando uma
  regra de `ruff`, run fica vermelho (saída real de `gh run watch` colada no card), commit
  seguinte conserta, run fica verde
- `test` roda sem o teste `slow` (auto-skip preexistente, sem mudança de comportamento) e enforça
  `fail_under = 70` do `pyproject.toml`
- `build` só roda depois de `lint`/`test` verdes, gera o placeholder, builda a imagem e valida
  `/health` + `/predict` de verdade dentro do job (não só `docker build`)
- Branch protection em `main` marca os 3 jobs como required status checks — PR com check vermelho
  fica com merge bloqueado na UI, verificado de verdade (não só assumido)
- `uv run ruff check .` e `uv run pytest` continuam limpos localmente

## Estratégia de testes

Não há testes automatizados novos em `tests/` — o que este card entrega é infraestrutura de CI,
não código de aplicação. A validação é a execução real do pipeline, documentada com evidência:

1. Push da branch → PR aberto → `gh run watch` no run inicial (deve ficar todo verde, já que o
   código em si está correto)
2. Commit proposital quebrando `ruff check` (ex: import não usado) → push → `gh run watch` no novo
   run → job `lint` vermelho, saída real colada
3. Commit de correção → push → `gh run watch` → todos os jobs verdes, saída real colada
4. Checar `mergeStateStatus`/`statusCheckRollup` do PR via `gh pr view --json` antes e depois de
   configurar branch protection, para confirmar que o merge fica de fato bloqueado com check
   vermelho e liberado com todos verdes

## Métricas

- Tempo total do pipeline (do trigger ao último job verde) — informativo, não é critério de
  aceite. Registrar o valor observado nos runs reais da validação.

## Riscos

- **Token do `gh` sem escopo `workflow`**: criar/editar arquivos em `.github/workflows/` via
  `git push` usando um token OAuth exige o escopo `workflow`. Se o push for rejeitado por esse
  motivo, a correção é `gh auth refresh -h github.com -s workflow` (ou re-login) antes de
  commitar o workflow.
- **Branch protection travar o próprio PR do CI-001**: os required status checks só existem
  *depois* que o workflow rodar ao menos uma vez no repo (o GitHub precisa "ver" o nome do job).
  Ordem correta: abrir o PR primeiro (jobs rodam, ainda sem ser required), demonstrar
  vermelho/verde, **depois** configurar branch protection — não o contrário.
- **`--cov-fail-under=70` mudar de comportamento no futuro**: hoje a cobertura é 100%, folga
  grande. Se `src/` crescer sem teste correspondente, o job `test` passa a falhar por cobertura,
  não por bug — comportamento esperado e desejado, mas vale deixar registrado para não causar
  surpresa numa sessão futura.
- **`docker build` no runner do GitHub ser mais lento/rápido que local**: o tempo medido no
  DOCK-001 (~70s frio) foi numa máquina local; o runner `ubuntu-latest` tem características
  diferentes (rede, CPU). Não é um critério de aceite, só uma expectativa a calibrar com o número
  real do primeiro run.

## Perguntas em aberto

Nenhuma — decisões fechadas na discussão prévia (registrada no histórico do card CI-001):
placeholder via script (não fixture binária commitada), `build` com `needs: [lint, test]`,
branch protection via `gh api`, `push: [main]` além de `pull_request`, demo de falha/correção no
próprio PR do CI-001.

## Experimentos

Não aplicável — não há hipótese de ML/latência sendo testada.
