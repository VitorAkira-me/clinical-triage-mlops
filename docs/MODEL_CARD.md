# Model Card — Baseline de Triagem (TF-IDF + LogisticRegression)

Estrutura inspirada em [Mitchell et al., 2019](https://arxiv.org/abs/1810.03993), focada em
**Data Understanding**. A seção mais importante deste documento é *Limitations* — o achado que
mais moldou as decisões do projeto a partir da EDA.

---

## Model Details

- **Algoritmo**: `TfidfVectorizer` (hiperparâmetros default: `ngram_range=(1,1)`, sem
  `stop_words` customizado) + `LogisticRegression(class_weight="balanced", max_iter=1000,
  random_state=42)`, em um único `sklearn.Pipeline`.
- **Treino**: `notebooks/02_baseline.ipynb` (ML-003) e, de forma equivalente,
  `airflow/dags/air001_train_baseline.py` (AIR-001) — mesma lógica, mesmo seed.
- **Formato de serving**: convertido para **ONNX** via `skl2onnx` (`benchmarks/opt001_onnx_benchmark.py`,
  [OPT-001](specs/OPT-001.md)). O pipeline inteiro converteu de primeira — não foi necessário o
  fallback de converter só o classificador e manter o TF-IDF em sklearn.
- **Ganho de latência medido** (300 iterações, mesma entrada, mesma máquina, após warmup):

  | | sklearn original | ONNX (onnxruntime) | Speedup |
  |---|---|---|---|
  | Mediana | 0,40 ms | 0,036 ms | ~11x |
  | p95 | 0,55 ms | 0,050 ms | ~11x |

  Corretude validada **antes** de reportar essa velocidade: 150 textos reais do dataset, 100% das
  classes batendo entre sklearn e ONNX, diferença máxima de probabilidade de `8,11e-08`
  (tolerância usada: `1e-4`). Sem essa checagem, um número de velocidade sobre um resultado errado
  não teria valor nenhum.
  Artefato também ficou menor: 9.044 bytes (`.joblib`) vs. 7.076 bytes (`.onnx`), -22%.
  Números completos em [docs/experiments/OPT-001-benchmark.json](experiments/OPT-001-benchmark.json).

  Essa latência é só a camada de inferência pura — a API real tem overhead de rede/HTTP/serialização
  bem maior (latência ponta-a-ponta observada: ~6–9ms). O ganho de ~11x é real, mas não aparece como
  11x na latência percebida pelo cliente (a API atual serve via sklearn, não via ONNX — a troca de
  serving é uma decisão em aberto, fora do escopo do OPT-001).

## Intended Use

**Uso pretendido**: apoiar a priorização da fila de revisão humana de laudos de pronto-socorro,
ordenando por urgência prevista.

**Explicitamente NÃO é**:
- uma ferramenta de diagnóstico autônomo;
- um substituto do julgamento clínico;
- validado contra qualquer padrão-ouro clínico real (o dataset é sintético — ver *Training Data*).

Essa fronteira não é uma formalidade — é a consequência direta da limitação descrita abaixo.

## Training Data

- **Fonte**: [fedmml-ed-triage](https://huggingface.co/datasets/olaflaitinen/fedmml-ed-triage)
  (Hugging Face, *gated*), 87.234 encontros sintéticos de pronto-socorro, licença CC BY 4.0
  ([ADR-001](decisions/ADR-001-dataset.md)).
- **Campo usado**: `clinical_notes` — 85.679 registros não-nulos de 87.234 (1.555 nulos, 1,78%),
  confirmado em `notebooks/01_eda_dataset.ipynb` e `notebooks/02_baseline.ipynb`.
- **Rótulo**: ESI nativo (1–5) remapeado para 3 classes (`1–2→urgente`, `3→atencao`, `4–5→normal`).
- **Geração**: texto **inteiramente sintético**, produzido por template fixo:
  `"{idade}yo {sexo} {verbo} {chief_complaint}. {cláusula fixa}."` — não é texto clínico real
  escrito por humanos.
- **Split**: 80/20 estratificado por `urgencia`, `random_state=42` (17.136 amostras no split de
  teste, de um total de 85.679).

## Limitations

**Achado principal da EDA (ML-002), documentado em
[ADR-002](decisions/ADR-002-text-leakage.md): vazamento determinístico total de rótulo no texto.**

O `clinical_notes` é montado por template a partir de duas variáveis discretas: 28 categorias de
`chief_complaint` e 5 variantes de cláusula final. A checagem quantitativa completa (todas as
85.679 notas não-nulas, sem amostragem) encontrou **0 exceções**: cada uma das 28 categorias de
`chief_complaint` mapeia para exatamente uma classe de urgência, e o mesmo vale para as 5 variantes
de cláusula final. Ou seja, **100% de precisão determinística**, não uma correlação forte — uma
correspondência exata, tabela-verdade completa.

Consequência direta: **qualquer classificador de texto reporta ~100% de acurácia/F1 neste
dataset**, e isso é exatamente o que a ML-003 mediu (ver *Evaluation* abaixo) — não porque o
modelo aprendeu linguagem clínica, mas porque decorou (via TF-IDF, sem precisar de nada mais
sofisticado) uma tabela de busca embutida no próprio gerador sintético do dataset.

**Por que isso importa**: F1 = 1.0 aqui **não implica robustez clínica real**. Um laudo real,
escrito por um profissional de saúde, não segue um template fechado de 28×5 combinações — o
modelo nunca foi exposto a variação linguística real, sinônimos, negação, ambiguidade ou
gravidade descrita de forma implícita. Não há evidência, neste projeto, de que o classificador
generalizaria para texto clínico real. O número alto é um artefato do gerador de dados, não uma
medida de capacidade de NLP clínico.

**Validação cruzada com dados estruturados**: vitais e labs (`spo2`, `heart_rate`, `troponin`
etc., fora do escopo deste classificador de texto) seguem distribuições condicionais ao ESI *com
sobreposição real entre classes* — sem o determinismo perfeito visto no texto. Isso indica que o
vazamento é específico da lógica de geração do campo `clinical_notes`, não uma característica do
dataset como um todo.

**Decisão tomada diante disso** ([ADR-002](decisions/ADR-002-text-leakage.md)): manter o dataset
(a alternativa — trocar de fonte — reintroduziria o problema original que o
[ADR-001](decisions/ADR-001-dataset.md) evitou, de rotular urgência sem base clínica) e reportar o
resultado com transparência radical, incluindo o baseline ingênuo de comparação abaixo.

## Evaluation

Split de teste: 17.136 amostras (20% de 85.679, estratificado, `random_state=42`).

| Modelo | F1 macro | Recall macro | Recall (`urgente`) |
|---|---|---|---|
| TF-IDF + LogisticRegression | 1.0 | 1.0 | 1.0 |
| Baseline ingênuo (só `chief_complaint`) | 1.0 | 1.0 | 1.0 |

As matrizes de confusão dos dois modelos são **idênticas** (`normal`: 5.522, `atencao`: 8.128,
`urgente`: 3.486 — zero erro nos dois, nas mesmas células). Isso não é coincidência: é a prova
direta de que o classificador de texto não faz nada além do que o baseline ingênuo já faz — dado o
vazamento determinístico descrito acima, os dois são matematicamente equivalentes sobre este
dataset.

Números completos: [docs/experiments/ML-003-baseline-metrics.json](experiments/ML-003-baseline-metrics.json).

**Leitura correta destes números**: eles confirmam que a implementação do pipeline está correta
(sem bug de vazamento de dados entre treino/teste, sem erro de alinhamento de classes) — não que o
modelo resolveu triagem clínica por NLP. O valor deste baseline para o restante do projeto (API,
Docker, CI/CD, observabilidade, otimização) é ter um modelo funcional para servir e monitorar, não
uma reivindicação de acurácia clínica.
