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
Current Phase: STEP 3 → STEP 4
Current Step: STEP 4 (CI/CD) concluído na metade CI; CD segue no EPIC 12
Current Task: nenhuma tarefa aberta — próxima a definir
Last Completed: CI-001 — pipeline `.github/workflows/ci.yml` (lint → testes
→ build) via GitHub Actions, branch protection em `main` com os 3 jobs
como required status checks. Validado com run vermelho e verde reais
(PR #6): commit proposital quebrando lint → job `lint` falha, `build`
skipado (`needs: [lint, test]`); com branch protection ativa,
`mergeStateStatus` vira `BLOCKED` e `gh pr merge` é recusado pelo
GitHub — não é só sinal, bloqueia de fato. DOCK-001 (Dockerfile
single-stage, imagem final 132 MB) mergeado em `main` nesta mesma
sessão, antes do CI-001 (o job de build depende do Dockerfile existir).
Next Recommended Action: próximo STEP do roadmap a definir — candidatos
naturais são EPIC 06 (testes, se quiser aprofundar cobertura além dos
testes da API-001) ou EPIC 08 (Airflow). Nenhuma pendência de Docker/CI
em aberto. Seguir SPEC → discussão antes de implementar, como combinado
desde a API-001.
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
