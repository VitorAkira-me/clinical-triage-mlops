---
name: kanban
description: Consulta e atualiza o Kanban do projeto (docs/KANBAN.md) — pegar tarefa, mover status, checklist pré-PR. Use ao iniciar uma tarefa, ao conferir o status de um card, ou antes de abrir qualquer PR.
---

# Kanban do projeto (docs/KANBAN.md)

O Kanban de desenvolvimento vive em [`docs/KANBAN.md`](../../../docs/KANBAN.md), versionado no
próprio repositório (é o source of truth definido no CLAUDE.md, seção 6 — não há board externo
para este projeto). Colunas: BACKLOG · TODO · IN PROGRESS · REVIEW/VALIDATION · DONE. IDs seguem
o padrão `<PREFIXO>-<NNN>` (ex: `ML-002`, `API-001`) ou `EPIC-NN` para itens de backlog ainda não
detalhados.

Projeto solo (ver CLAUDE.md seção 1) — não há campo de atribuição de pessoa nem necessidade de
perguntar quem vai trabalhar em um card.

## Ações cobertas

- **Pegar uma tarefa**: ler `docs/KANBAN.md`, listar os cards em TODO para o usuário escolher e,
  ao escolher um, mover o bloco do card da seção TODO para IN PROGRESS (editando o arquivo
  diretamente) e atualizar o campo **Status** dentro do card.
- **Consultar detalhes de uma tarefa pelo ID** (ex: `ML-002`): localizar o bloco correspondente em
  `docs/KANBAN.md` e trazer objetivo/critério de aceite/dependências.
- **Apoio à skill `pr-description`**: quando a branch não tiver ID de tarefa no nome, ou quando o
  diff parecer tocar mais de um card, consultar `docs/KANBAN.md` para identificar candidatos a
  card relacionado em vez de assumir.
- **Fechar uma tarefa**: mover o bloco do card para DONE e preencher o campo **Resultado** com o
  que foi entregue (ver exemplo de `ML-001` no arquivo) — só depois que o checklist pré-PR abaixo
  confirmar que o trabalho está 100% completo.

## Checklist obrigatório antes de qualquer PR

Antes de abrir um PR (via skill `pr-description`), revisar todos os cards em IN PROGRESS
relacionados ao trabalho feito na branch atual:

1. Ler em `docs/KANBAN.md` os cards IN PROGRESS que tenham relação com o diff/commits da branch.
2. Para cada card candidato, avaliar se o trabalho da branch o resolve **por completo**.
3. **Regra dura**: um card só pode ser movido para DONE e referenciado no PR se estiver 100%
   finalizado. Se um card está apenas parcialmente resolvido pelo diff atual, **não** movê-lo nem
   incluí-lo como resolvido no PR — avisar o usuário explicitamente e instruir a concluir o card
   antes de abrir o PR (ou manter em IN PROGRESS e referenciar como "Part of" no PR).
4. Só depois desse checklist, prosseguir para a skill `pr-description` com a lista de IDs
   confirmados como concluídos, e atualizar `docs/KANBAN.md` (mover para DONE, preencher
   Resultado) como parte do mesmo commit/PR.

Isso evita dois problemas simétricos: cards esquecidos (nunca fechados em nenhum PR) e cards
marcados como concluídos antes do trabalho estar de fato completo.
