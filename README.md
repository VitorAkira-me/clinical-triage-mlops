# Clinical Triage MLOps

Sistema de triagem automática de laudos médicos por urgência
(normal / atenção / urgente), desenvolvido para o Tech Challenge — Fase 3
da Pós Tech em Machine Learning Engineering (FIAP).

> Projeto individual — continuação solo após o encerramento do grupo
> original (ver Fases 1 e 2 abaixo).

## 1. Projeto

Laboratório pessoal de Machine Learning Engineering/MLOps, construído em
público, STEP por STEP, com decisões documentadas em ADRs.

## 2. Problema

Hospitais recebem laudos em texto livre e precisam priorizar atendimento.
Este projeto constrói um classificador de urgência a partir do texto do
laudo.

## 3. Objetivo

Entregar uma API de inferência em produção, com pipeline de treino
automatizado (Airflow), CI/CD (GitHub Actions), observabilidade
(Prometheus + Grafana) e uma técnica de otimização de inferência com
benchmark real de latência.

## 4. Arquitetura

TODO — será documentada após o STEP 2 (API) e o STEP 10 (ADR de cloud).

## 5. Dataset

[fedmml-ed-triage](https://huggingface.co/datasets/olaflaitinen/fedmml-ed-triage)
(Hugging Face) — **sintético**, licença CC BY 4.0. Usamos apenas o campo
de texto `clinical_notes`; o alvo nativo ESI 1–5 foi remapeado para
3 classes (normal/atenção/urgente). Decisão completa e trade-offs em
[ADR-001](docs/decisions/ADR-001-dataset.md).

**Aviso importante**: o dataset é sintético e o mapeamento de urgência é
uma adaptação de engenharia deste projeto — não representa um padrão
clínico oficial.

## 6. Modelo

Baseline: TF-IDF + Logistic Regression (`class_weight="balanced"`), treinado sobre o campo
`clinical_notes`. Split treino/teste estratificado (80/20, `random_state=42`). Ver
[notebooks/02_baseline.ipynb](notebooks/02_baseline.ipynb) e métricas completas em
[docs/experiments/ML-003-baseline-metrics.json](docs/experiments/ML-003-baseline-metrics.json).

### 6.1 Limitações do dataset — Vazamento de rótulo identificado

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

## 7. Como executar

TODO — instruções chegam junto com o STEP 2/3 (API + Docker).

## 8. API

TODO — especificação em `docs/specs/` antes da implementação.

## 9. Pipeline de treinamento

TODO — DAG Airflow no STEP 6.

## 10. CI/CD

TODO — workflow GitHub Actions no STEP 5.

## 11. Monitoramento

TODO — Prometheus + Grafana no STEP 7.

## 12. Otimização

TODO — investigação de ONNX/quantização/pruning no STEP 8.

## 13. Benchmark

TODO — resultados reais no STEP 9. Nenhum número aqui até a execução
efetiva do experimento.

## 14. Resultados

TODO.

## 15. Decisões arquiteturais

Ver [docs/decisions/](docs/decisions/).

## 16. Estrutura do projeto

```
clinical-triage-mlops/
├── .github/workflows/
├── airflow/dags/
├── data/{raw,processed}/
├── docs/{decisions,specs,experiments,architecture}/
├── monitoring/{prometheus,grafana}/
├── notebooks/
├── src/{api,data,models,training,monitoring}/
├── tests/
├── benchmarks/
├── models/
├── CLAUDE.md
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

## 17. Roadmap

Ver [docs/ROADMAP.md](docs/ROADMAP.md).

## 18. Aprendizados

TODO — registrado ao final de cada STEP.

## 19. Vídeo

TODO.

## 20. Autor

Akira — Pós Tech em Machine Learning Engineering, FIAP.
Projetos anteriores em equipe: [ml-churn-prediction](https://github.com/fiap-postech-ml-engineering/ml-churn-prediction)
(Fase 1) e [ecommerce-recsys-mlops](https://github.com/fiap-postech-ml-engineering/ecommerce-recsys-mlops)
(Fase 2).
