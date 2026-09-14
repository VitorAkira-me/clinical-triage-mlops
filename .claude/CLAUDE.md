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
Current Phase: STEP 6 → STEP 7
Current Step: EPIC 10/11 (ONNX + benchmark) concluídos; EPIC 08/09
concluídos em sessões anteriores
Current Task: nenhuma tarefa aberta — próxima a definir
Last Completed: OPT-001/BENCH-001 — pipeline inteiro (TfidfVectorizer +
LogisticRegression) convertido pra ONNX via `skl2onnx`, funcionou de
primeira (fallback de classificador isolado implementado mas não
acionado). Corretude validada antes de medir latência: 150 textos reais,
150/150 classes batendo, diff máxima de probabilidade 8.11e-08. Latência
real (300 iterações): sklearn mediana 0.40ms vs. ONNX 0.036ms — ~11x de
speedup, reproduzido em 3 execuções. Tamanho: 9044 bytes (.joblib) vs.
7076 bytes (.onnx). Script: `benchmarks/opt001_onnx_benchmark.py`.
Antes disso, nesta mesma sessão: fixado `scikit-learn==1.9.0` na API e
no container do Airflow (eliminava o drift de versão documentado no
AIR-001) e retreinado o baseline sem o warning de versão.
Next Recommended Action: EPIC 12 (arquitetura de cloud, ADR-005) é o
próximo passo OBRIGATÓRIO, não opcional — é o STEP 10 do
docs/ROADMAP.md, única etapa antes do STEP 11 (documentação final).
Achado de auditoria (pré-README, nesta sessão): docs/KANBAN.md listava
isso como item plano de backlog, sem sinalizar a obrigatoriedade que o
ROADMAP já definia — corrigido. EPIC 06 (testes) continua opcional/nice-
to-have, sem STEP associado no ROADMAP. Seguir SPEC → discussão antes de
implementar, como combinado desde a API-001.
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
