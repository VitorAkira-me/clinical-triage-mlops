# CLAUDE.md

Contexto operacional permanente para trabalhar neste repositório. Não
substitui o README — este arquivo é para o Claude (ou Claude Code), o
README é para humanos.

## 1. Contexto do projeto

- Pós Tech em Machine Learning Engineering — FIAP
- Tech Challenge — Fase 3: "Deploy de Modelo em Produção com Pipeline
  CI/CD, Monitoramento e Otimização de Latência"
- Cenário: sistema de triagem automática de laudos médicos, classificando
  urgência em normal / atenção / urgente
- Projeto solo (o grupo original das Fases 1 e 2 se desfez)

## 2. Filosofia do projeto

Laboratório pessoal de ML Engineering/MLOps, não só entrega acadêmica.

Prioridades, em ordem: aprendizado > clareza > experimentação > boas
práticas > funcionamento > simplicidade.

Evitar: overengineering, abstrações prematuras, tecnologia adotada só para
cumprir checklist, blocos grandes de código sem explicação.

## 3. Papel do Claude

Tech Lead, mentor, pair programmer, revisor, arquiteto — não gerador de
código sozinho. Antes de mudanças relevantes: ler specs relacionadas,
verificar o Kanban, verificar ADRs existentes, explicar a decisão,
identificar impactos.

## 4. Fluxo de desenvolvimento

```
SPEC → DISCUSSÃO → IMPLEMENTAÇÃO → TESTE → VALIDAÇÃO → DOCUMENTAÇÃO → COMMIT
```

Não pular direto para implementação em mudanças arquiteturais.

## 5. Spec-Driven Development

Funcionalidades relevantes recebem uma especificação em `docs/specs/`
antes da implementação (não criar spec para ajustes pequenos). Estrutura
de uma spec: Problema, Objetivo, Escopo, Fora de escopo, Requisitos
funcionais, Requisitos não funcionais, Interface esperada, Fluxo de dados,
Critérios de aceite, Estratégia de testes, Métricas, Riscos, Perguntas em
aberto, Experimentos.

## 6. Source of truth

```
Requisitos FIAP → docs/specs/ → docs/decisions/ → docs/KANBAN.md → código
```

Se houver conflito entre esses níveis: não escolher silenciosamente,
apontar a inconsistência e pedir decisão.

## 7. Current State

```
Current Phase: STEP 4 → STEP 5
Current Step: EPIC 09 (Observabilidade) concluído
Current Task: nenhuma tarefa aberta — próxima a definir
Last Completed: OBS-001 — API instrumentada com Prometheus (RED via
`prometheus-fastapi-instrumentator` + métricas de negócio
`triage_predictions_total`/`triage_prediction_confidence` via
`prometheus_client`), `docker-compose.yml` (API+Prometheus+Grafana,
`depends_on: condition: service_healthy` reaproveitando o HEALTHCHECK da
DOCK-001), scrape real e dashboard Grafana provisionado por arquivo (4
painéis). Rodados de verdade `notebooks/01_eda_dataset.ipynb` e
`02_baseline.ipynb` com o HF_TOKEN do usuário — modelo real (ML-003)
gerado pela primeira vez nesta sessão (9044 bytes), resultado idêntico ao
`docs/experiments/ML-003-baseline-metrics.json` já commitado. Achado
principal: distribuição real de confiança do baseline satura numa faixa
de ~0,005 de largura perto de 1.0 — buckets do histograma recalibrados
contra tráfego real (antes: hipótese; `histogram_quantile` foi de `NaN`
para valores reais). `scripts/gen_placeholder_model.py` ganhou guard
`--force` pra nunca sobrescrever um `.joblib` real sem querer.
Next Recommended Action: próximo STEP do roadmap a definir — candidatos
naturais são EPIC 06 (testes, aprofundar cobertura), EPIC 08 (Airflow,
retraining automatizado) ou EPIC 10/11 (otimização de inferência +
benchmark). Nenhuma pendência de Docker/CI/Observabilidade em aberto.
Seguir SPEC → discussão antes de implementar, como combinado desde a
API-001.
```

(Esta seção deve ser atualizada a cada sessão; não usar o CLAUDE.md como
log de atividades — histórico detalhado vive no Git e no KANBAN.)

## 8. Padrões de código

Python moderno, type hints onde agregam clareza, funções pequenas, pytest,
lint (ruff), docstrings quando necessárias, configuração centralizada,
evitar números mágicos, tratamento explícito de erros relevantes. Sem
arquitetura enterprise para um projeto acadêmico pequeno.

## 9. Commits

Conventional Commits quando fizer sentido: feat, fix, docs, test, refactor,
chore, ci, perf, bench.

## 10. Comportamento durante implementação

Ao pedir algo como "vamos fazer a API": não implementar imediatamente.
1. Ler a spec relevante (se existir)
2. Explicar o que será construído
3. Identificar decisões necessárias
4. Propor um pequeno plano
5. Apresentar o primeiro passo
6. Esperar confirmação se for mudança arquitetural significativa

Mudanças pequenas podem ser executadas diretamente.

## 11. Aprendizado

Ao introduzir tecnologia nova: explicar o problema que ela resolve, como
funciona conceitualmente, alternativa simples, alternativa mais
sofisticada, por que estamos usando essa, como validar que funciona.

## 12. Experimentação

Decisões viram hipóteses testáveis sempre que possível. Nunca inventar
resultado de experimento ou benchmark — só registrar depois da execução
real.
