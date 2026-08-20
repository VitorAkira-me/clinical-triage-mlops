# API-001: Endpoint `POST /predict`

## Problema

O modelo treinado na ML-003 (`models/tfidf_logreg_baseline.joblib`) existe hoje só como artefato
de notebook — não há forma de consultá-lo fora de um script Python local. Para o projeto avançar
para os STEPs seguintes (Docker, CI/CD, observabilidade), o modelo precisa estar servido atrás de
uma API HTTP.

## Objetivo

Expor o pipeline TF-IDF + LogisticRegression (`class_weight="balanced"`) da ML-003 via uma API
FastAPI mínima, com um endpoint de predição e um de health-check.

## Escopo

- `POST /predict`: recebe texto de laudo clínico, devolve classe de urgência prevista +
  probabilidades por classe
- `GET /health`: confirma que o serviço está no ar
- Carregamento do modelo uma vez na inicialização do processo (não por request)
- Falha rápida e acionável se o artefato do modelo não existir
- Validação de entrada (texto não-vazio)

## Fora de escopo

- Autenticação/autorização
- CORS
- Versionamento de rota (`/predict` puro, sem prefixo `/api/v1`)
- Métricas Prometheus / instrumentação de observabilidade (EPIC 09)
- Servir o baseline ingênuo por `chief_complaint` (existe só para comparação/documentação do
  ADR-002, nunca foi pensado como algo a servir em produção)
- Docker/deploy (EPIC 05) — este spec cobre só a aplicação FastAPI, não como ela é empacotada
- Batch prediction (múltiplos textos por request)

## Requisitos funcionais

- RF1: `POST /predict` aceita `{"clinical_notes": str}` e devolve a classe prevista
  (`normal`/`atencao`/`urgente`) e as probabilidades das três classes
- RF2: `GET /health` devolve `200` com um corpo simples confirmando que o processo está de pé
- RF3: entrada com `clinical_notes` vazio ou só espaços é rejeitada com `422`, sem chegar ao
  modelo
- RF4: se `models/tfidf_logreg_baseline.joblib` não existir no caminho configurado, a aplicação
  falha ao iniciar (não sobe em estado "quebrado"), com mensagem de erro que diz exatamente o que
  fazer para resolver

## Requisitos não funcionais

- Modelo carregado uma única vez no startup do processo (não recarregar por request)
- Caminho do modelo configurável por variável de ambiente (`MODEL_PATH`), com default relativo à
  raiz do projeto — não precisa mudar código para apontar para outro artefato (testes, ambientes
  diferentes)
- Código em `src/api/`: `main.py` (app + rotas) e `schemas.py` (modelos Pydantic de
  request/response); carregamento do modelo é uma função simples dentro de `main.py`, sem módulo
  dedicado — não se justifica para uma função de poucas linhas
- Type hints em todas as funções novas; sem abstrações (camada de serviço, repositório, etc.) que
  este escopo de 2 endpoints não pede

## Interface esperada

### `POST /predict`

Request:
```json
{"clinical_notes": "67yo M c/o Chest pain. Patient in moderate distress..."}
```

Response `200`:
```json
{
  "urgencia": "urgente",
  "probabilidades": {
    "normal": 0.001,
    "atencao": 0.004,
    "urgente": 0.995
  }
}
```

Response `422` (texto vazio/só espaços):
```json
{"detail": [{"type": "value_error", "loc": ["body", "clinical_notes"], "msg": "..."}]}
```
(formato padrão do FastAPI/Pydantic para erro de validação — não precisamos customizar)

### `GET /health`

Response `200`:
```json
{"status": "ok"}
```

### Detalhes de implementação a fixar

- **Validação de `clinical_notes`**: `min_length=1` do Pydantic sozinho não barra string só de
  espaços. Usar um `field_validator` que faz `strip()` e rejeita se o resultado ficar vazio.
- **Alinhamento de probabilidades**: `pipeline.predict_proba(...)` devolve as colunas na ordem de
  `pipeline.classes_` (ordem alfabética do scikit-learn: `atencao`, `normal`, `urgente`) — **não**
  assumir a ordem `["normal", "atencao", "urgente"]` usada nos notebooks da ML-002/ML-003. Montar
  o dicionário de probabilidades fazendo `zip(pipeline.classes_, proba[0])`, nunca com uma lista
  de classes hardcoded separada do pipeline.
- **Resolução do caminho do modelo**:
  ```python
  DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "tfidf_logreg_baseline.joblib"
  MODEL_PATH = Path(os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH))
  ```
- **Erro de modelo ausente** (falha no startup, via `lifespan` do FastAPI), mensagem acionável:
  ```
  Modelo não encontrado em {MODEL_PATH}. Rode notebooks/02_baseline.ipynb (ML-003) para gerar o
  artefato, ou defina a variável de ambiente MODEL_PATH apontando para um .joblib já treinado.
  ```

## Fluxo de dados

1. Startup do processo: `lifespan` carrega `MODEL_PATH` via `joblib.load`; se o arquivo não
   existir, levanta erro acionável e o processo não sobe
2. Cliente envia `POST /predict` com `clinical_notes`
3. Pydantic valida (não vazio após `strip()`) — falha aqui vira `422` automático, sem tocar o
   modelo
4. Handler chama `pipeline.predict([texto])` e `pipeline.predict_proba([texto])`
5. Resposta monta `urgencia` (classe prevista) + `probabilidades` (dict alinhado via
   `pipeline.classes_`)
6. `200` com `PredictResponse`

## Critérios de aceite

- `POST /predict` com texto válido devolve `200`, classe em `{normal, atencao, urgente}` e as três
  probabilidades somando ~1.0
- `POST /predict` com `clinical_notes` vazio ou só espaços devolve `422`
- `GET /health` devolve `200` quando a aplicação está no ar
- Subir a aplicação sem `models/tfidf_logreg_baseline.joblib` presente falha com mensagem citando
  o comando/notebook para gerar o artefato — não um erro genérico de arquivo não encontrado
- `MODEL_PATH` sobrescrito por variável de ambiente é respeitado (testável apontando para um
  modelo-fixture)
- `uv run ruff check .` limpo, testes descritos abaixo passando

## Estratégia de testes

`tests/test_api.py`, marcadores `api` (todos) e `slow` (o teste de integração):

1. **Fixture sintética** (`unit`/`api`): treina um `Pipeline(TfidfVectorizer, LogisticRegression)`
   minúsculo em memória, sobre poucas linhas sintéticas cobrindo as 3 classes, salva num
   `tmp_path` via `joblib.dump`, e usa `monkeypatch`/override de `MODEL_PATH` para apontar a
   aplicação para esse artefato. Não depende do dataset real nem do Hugging Face.
   - Testa `POST /predict` feliz (200, classe válida, probabilidades somam ~1)
   - Testa `POST /predict` com texto vazio/whitespace (422)
   - Testa `GET /health` (200)
   - Testa startup com `MODEL_PATH` apontando para um arquivo inexistente (erro acionável,
     mensagem cita o notebook)
2. **Teste de integração** (`@pytest.mark.slow`, `api`): carrega o
   `models/tfidf_logreg_baseline.joblib` real e faz uma predição de verdade, checando só que a
   resposta tem o formato esperado (não trava em números exatos — isso já está coberto pela
   ML-003). Objetivo é pegar problemas reais de compatibilidade do artefato (versão do
   scikit-learn, schema de features) que a fixture sintética não pode detectar. Usa
   `pytest.skip(...)` automático se o arquivo não existir — não quebra CI antes de existir um
   passo que gere o artefato.

## Métricas

Não aplicável — este spec não introduz uma métrica de negócio nova, só serve o modelo cujas
métricas já foram medidas e documentadas na ML-003
(`docs/experiments/ML-003-baseline-metrics.json`).

## Riscos

- `models/` é gitignored: um clone limpo do repositório não tem o `.joblib` até alguém rodar
  `notebooks/02_baseline.ipynb`. É o comportamento esperado (RF4 cobre isso com erro claro), mas
  vale deixar registrado que "clonar e rodar" não funciona sem esse passo manual — Docker/deploy
  (fora de escopo aqui) vai precisar decidir como o artefato chega à imagem
- Ordem de `pipeline.classes_` divergente da ordem usada nos notebooks é uma fonte plausível de
  bug silencioso (probabilidades trocadas entre classes sem erro nenhum) — mitigado pela regra
  explícita na seção "Interface esperada", mas vale um teste específico que não deixe essa
  ambiguidade passar (checar que a probabilidade mais alta bate com a classe retornada em
  `urgencia`)

## Perguntas em aberto

Nenhuma — decisões fechadas na discussão prévia a este documento.

## Experimentos

Não aplicável — não há hipótese nova sendo testada aqui, é a implementação do que a ML-003 já
validou.
