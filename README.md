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

TODO — baseline a definir no STEP 1 (candidatos: TF-IDF + Logistic
Regression, TF-IDF + Random Forest).

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
