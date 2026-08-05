---
name: pr-description
description: Gera título e descrição em Markdown para Pull Requests deste projeto, no formato Conventional PR (Summary, Changes, Test plan, Related). Use ao abrir um PR ou quando o usuário pedir a descrição/mensagem do PR.
---

# Descrição de Pull Request

Gera o título e o corpo (Markdown) de um PR para o `clinical-triage-mlops`, baseado nos commits
da branch atual em relação a `main`.

## Passo 1 — Coletar contexto

```bash
git branch --show-current
git log main..HEAD --oneline
git diff main...HEAD --stat
```

## Passo 2 — Identificar o(s) ID(s) de tarefa e checar o checklist do Kanban

Antes de montar qualquer coisa, use a skill `kanban` para confirmar os cards que este PR
resolve:

1. Se o nome da branch seguir o padrão `<tipo>/<ID>-<slug>` (ex: `feat/ML-002-eda-dataset`),
   extraia o `<ID>` (ex: `ML-002`) como candidato.
2. Se a branch **não** tiver esse padrão, ou se o diff parecer tocar mais de um card, pergunte ao
   usuário (ou consulte `docs/KANBAN.md` via `kanban`) se há outros cards relacionados — não
   assuma que só existe um.
3. Para cada card candidato, rode o **checklist obrigatório pré-PR** da skill `kanban`: só inclua
   no PR os cards que o trabalho da branch resolve **por completo**. Se algum card estiver apenas
   parcialmente resolvido, não o referencie como concluído no PR — avise o usuário e instrua a
   concluir o card primeiro (ou referencie como "Part of").
4. Se nenhum card sobreviver ao checklist (branch não relacionada a nenhum card do Kanban), siga
   sem a seção `## Related`.

## Passo 3 — Montar o título

- Com ID confirmado: `[<ID>] <descrição curta no imperativo>` (se houver mais de um ID,
  `[<ID1>][<ID2>] <descrição>`)
- Sem ID: `<descrição curta no imperativo>`

A descrição curta resume o objetivo geral da branch (não é só o último commit). O ID no título é
só para legibilidade humana — não há integração automática que leia o título ou a descrição deste
projeto (ver Passo 4).

## Passo 4 — Montar a descrição (template)

```markdown
## Summary
- <1-3 bullets descrevendo o que foi feito e por quê>

## Changes
- <bullets das mudanças principais — módulos/arquivos afetados, decisões relevantes>

## Test plan
- [ ] <como validar — ex: `uv run ruff check .`, `uv run pytest`, execução manual>

## Related
- <ID> (docs/KANBAN.md)
```

Regras:
- `## Related` lista os IDs confirmados no checklist do Passo 2, com uma nota curta se o card
  foi fechado por completo (`<ID> — concluído`) ou apenas parcialmente contemplado
  (`<ID> — parcial, segue em IN PROGRESS`). Não há automação GitHub↔Kanban neste projeto: mover o
  card em `docs/KANBAN.md` para DONE é um passo manual, feito como parte do mesmo commit/PR (ver
  skill `kanban`).
- `## Related` só aparece se houver ao menos um ID confirmado no Passo 2. Se não houver, omita a
  seção inteira.
- `Summary` foca no "porquê"/objetivo; `Changes` foca no "o quê" (arquivos, módulos, decisões
  técnicas). Não repita a lista de commits literalmente — sintetize.
- `Test plan` deve ser um checklist verificável (comandos reais deste projeto via `uv run`: ruff,
  pytest, pre-commit, ou passos manuais quando aplicável — não há Makefile neste repo).
- Linguagem: português, seguindo a convenção de commits do projeto.

## Passo 5 — Aprovação obrigatória antes de abrir o PR

**Em toda e qualquer hipótese**, mostrar o título e a descrição completos (já com a seção
`## Related` resolvida pelo checklist do Passo 2) ao usuário e esperar aprovação explícita antes
de rodar `gh pr create` ou qualquer comando que efetivamente abra o PR. Pedir para abrir o PR
autoriza o fluxo, não dispensa a revisão do conteúdo gerado — só prosseguir depois que o usuário
confirmar ou pedir ajustes no texto.
