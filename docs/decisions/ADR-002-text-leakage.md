# ADR-002: Vazamento determinístico de rótulo no texto do fedmml-ed-triage

## Contexto
A ML-002 (EDA) identificou que o campo `clinical_notes` do dataset `fedmml-ed-triage`
([ADR-001](ADR-001-dataset.md)) é gerado por um template fixo, sem ruído entre a queixa e o
nível de gravidade: as 28 categorias de `chief_complaint` e as 5 variantes de cláusula final de
texto mapeiam para a classe de urgência (normal/atenção/urgente) com 100% de precisão, sem
exceção, nas 85.679 notas não-nulas verificadas. Um classificador de texto (mesmo um
TF-IDF + regressão logística simples) vai reportar ~100% de acurácia/F1 nesse dataset — não
porque aprendeu linguagem clínica, mas porque decorou uma tabela de busca embutida no template do
gerador sintético.

O ADR-001 já havia registrado esse risco de forma genérica ("texto gerado por template... tende a
facilitar artificialmente a classificação"), mas sem prever que o vazamento seria total e
determinístico, e não apenas parcial.

Validação cruzada com vitais e labs (fora do escopo do classificador de texto) mostrou que esse
vazamento é específico da lógica de geração do texto: `spo2`, `heart_rate`, `troponin` etc.
seguem distribuições condicionais ao ESI com sobreposição real entre classes, sem determinismo
perfeito — o restante do dataset parece gerado de forma estatisticamente coerente.

## Opções consideradas
1. **Reabrir o ADR-001 e trocar de dataset** (ex: Medical Abstracts TC Corpus). Descartada: essa
   fonte rotula tipo de condição clínica, não urgência — usá-la exigiria inventar um mapeamento
   condição→urgência sem base clínica, reintroduzindo exatamente o problema que o ADR-001 evitou
   ao escolher o fedmml-ed-triage.
2. **Remover a cláusula final do texto antes de vetorizar.** Descartada: a investigação mostrou
   que o vazamento não está isolado na cláusula final — o próprio `chief_complaint` (parte
   "legítima" do texto) já é 100% determinístico da classe. Remover a cláusula não resolveria o
   problema, só trocaria uma fonte de vazamento por outra.
3. **Manter o dataset e o campo de texto como estão, reportando o resultado com transparência
   radical.** Escolhida.

## Decisão
Manter o `fedmml-ed-triage` e o campo `clinical_notes` sem alteração. Na ML-003, reportar o
desempenho do baseline de texto (TF-IDF + LogisticRegression) lado a lado com um "baseline
ingênuo" que classifica apenas por `chief_complaint` (ex: dicionário complaint→classe
majoritária), evidenciando que os dois são equivalentes. Documentar essa limitação explicitamente
no README do projeto.

## Prós
- Preserva a validade do rótulo de urgência (motivo original do ADR-001 para escolher esse
  dataset) — não trocamos um problema de dificuldade por um problema de validade de rótulo.
- Sem retrabalho de ML-001/EDA; o restante do roadmap (API, Docker, CI/CD, Airflow,
  monitoramento, benchmark) depende de ter um modelo/pipeline funcionando, não de o problema de
  texto ser difícil.
- Transparência sobre a limitação tem valor pedagógico: documentar por que a métrica é enganosa é
  mais honesto do que escondê-la ou fingir uma dificuldade que não existe.

## Contras / riscos assumidos
- O classificador de texto não demonstra capacidade real de NLP clínico — é, na prática,
  equivalente a uma tabela de busca por `chief_complaint`.
- Métricas de monitoramento de drift/degradação (EPIC 09) em produção terão pouco significado
  real sobre um classificador que já parte de ~100% — vale registrar essa limitação também quando
  chegarmos nesse EPIC.
- Se a avaliação do Tech Challenge exigir demonstração de dificuldade real de NLP, esta decisão
  pode precisar ser revisitada.

## Consequências
- ML-003 deve incluir o baseline ingênuo por `chief_complaint` como comparação, não só o
  TF-IDF + LogisticRegression.
- README deve citar este ADR e deixar claro, desde a primeira versão, que a métrica de
  classificação de texto reflete um artefato do gerador sintético, não desempenho real de NLP
  clínico.
- ML-002 fechada com este achado como resultado principal, além da distribuição de classes e da
  decisão de balanceamento (`class_weight="balanced"`, sem under/oversampling).

## Status
Aceito em 16/08/2026.
