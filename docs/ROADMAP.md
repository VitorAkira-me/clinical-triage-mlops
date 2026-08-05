# ROADMAP — Clinical Triage MLOps (Fase 3)

Ordem pensada para que cada camada só seja construída sobre uma anterior
que já funciona — não pela ordem em que os requisitos aparecem no
enunciado.

- **STEP 0 — Entender o problema + escolher dataset** ✅
  Decisão registrada em [ADR-001](decisions/ADR-001-dataset.md)

- **STEP 1 — Baseline de modelo (notebook)**
  Validar que o problema é solucionável antes de engenheirar em torno dele

- **STEP 2 — Transformar modelo em API (FastAPI)**
  Primeiro artefato "servível"

- **STEP 3 — Containerizar (Docker)**
  Empacotar antes de automatizar

- **STEP 4 — Testes (pytest)**
  Precisa existir antes do CI — senão o CI não testa nada de verdade

- **STEP 5 — CI (lint + testes + build)**
  Automatiza o que já existe manualmente

- **STEP 6 — Airflow (pipeline de treino)**
  Trata o *treino* como pipeline, separado da API de inferência

- **STEP 7 — Observabilidade (Prometheus + Grafana)**
  Só faz sentido observar uma API que já está no ar

- **STEP 8 — Otimização de inferência (ONNX/quantização/pruning)**
  Só otimiza o que já está medido

- **STEP 9 — Benchmark comparativo**
  Mede o efeito real do STEP 8 (latência p50/p95, tamanho de modelo)

- **STEP 10 — Arquitetura de cloud (ADR)**
  Discussão teórica (batch vs real-time, AWS vs Azure vs GCP) — sem deploy
  obrigatório

- **STEP 11 — Documentação final + vídeo STAR**
  Fecha o ciclo
