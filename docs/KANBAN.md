# KANBAN — Clinical Triage MLOps (Fase 3)

Colunas: BACKLOG · TODO · IN PROGRESS · REVIEW/VALIDATION · DONE

Cada tarefa: ID, Título, Objetivo, Motivação, O que vou aprender,
Pré-requisitos, Passos, Critério de aceite, Dependências, Complexidade,
Status.

---

## DONE

### ML-001 — Investigar e escolher dataset
- **Objetivo**: encontrar dataset de texto médico adequado para
  classificação de urgência (normal/atenção/urgente), ≥2.000 registros
- **Motivação**: todo o resto do projeto depende dessa decisão
- **Vou aprender**: avaliação de qualidade e licenciamento de dataset,
  checagem de balanceamento, riscos éticos em dados de saúde
- **Critério de aceite**: dataset escolhido e decisão documentada
- **Dependências**: nenhuma
- **Complexidade**: média
- **Resultado**: fedmml-ed-triage escolhido — ver
  [ADR-001](decisions/ADR-001-dataset.md)

---

## TODO

### ML-002 — EDA do dataset escolhido
- **Objetivo**: entender completude, distribuição de classes (após
  remapeamento ESI→3 classes) e qualidade textual do campo `clinical_notes`
- **Motivação**: validar que o dataset sintético não trivializa o
  problema antes de investir em modelagem
- **Vou aprender**: análise exploratória de texto, detecção de
  desbalanceamento de classes, red flags de data leakage
- **Pré-requisitos**: ML-001 concluído, acesso ao dataset liberado no
  Hugging Face
- **Passos**: baixar dataset → aplicar remapeamento ESI→classe →
  checar completude/nulos → checar distribuição de classes →
  inspecionar amostras de texto manualmente
- **Critério de aceite**: notebook de EDA com achados registrados,
  decisão sobre necessidade de balanceamento (undersampling/
  class_weight/etc.) tomada e justificada
- **Dependências**: ML-001
- **Complexidade**: média
- **Status**: TODO — próxima tarefa recomendada

### ML-003 — Baseline de classificação (TF-IDF + Logistic Regression)
- **Dependências**: ML-002
- **Status**: TODO (detalhamento completo quando ML-002 fechar)

### API-001 — Especificar e implementar endpoint `/predict`
- **Dependências**: ML-003
- **Status**: TODO

*(Demais tarefas dos EPICS 05–14 serão detalhadas ao chegar em cada STEP
do roadmap — ver decisão de não planejar em excesso antecipadamente,
registrada na sessão de discovery.)*

---

## BACKLOG (nível de épico, não detalhado ainda)

- EPIC 05 — Docker
- EPIC 06 — Testes
- EPIC 07 — CI/CD (GitHub Actions)
- EPIC 08 — Airflow (DAG de treino)
- EPIC 09 — Observabilidade (Prometheus + Grafana)
- EPIC 10 — Otimização de inferência (ONNX/quantização/pruning)
- EPIC 11 — Benchmark (latência p50/p95, tamanho de modelo)
- EPIC 12 — Arquitetura de cloud (ADR-005)
- EPIC 13 — Documentação final
- EPIC 14 — Vídeo STAR
