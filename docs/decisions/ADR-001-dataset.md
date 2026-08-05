# ADR-001: Escolha do dataset para triagem de laudos médicos

## Contexto
A Fase 3 do Tech Challenge exige um classificador NLP de urgência clínica
(normal / atenção / urgente), treinado sobre um dataset de texto médico com
pelo menos ~2.000 registros. O enunciado cita a Medical Abstracts TC Corpus
como exemplo, mas essa fonte rotula *tipo de condição clínica*, não
*urgência* — usá-la exigiria inventar um mapeamento sem base clínica.

## Opções consideradas

1. **Medical Abstracts TC Corpus** (sebischair) — 14.438 registros, 5 classes
   de tipo de doença, licença CC BY-SA, maduro e citado em papers.
   Descartada: classes não correspondem a urgência.

2. **fedmml-ed-triage** (Hugging Face, olaflaitinen) — 87.234 encontros
   sintéticos de pronto-socorro, campo de texto `clinical_notes` + dados
   estruturados (vitais, labs), alvo nativo ESI 1–5, licença CC BY 4.0.

3. **Turkish Medical Emergency Triage Dataset** (Kaggle) — rótulos de
   urgência reais, mas em turco. Descartada por idioma.

4. **Synthetic Medical Triage Priority Dataset** (Kaggle) — formato
   (texto livre vs tabular) não confirmado. Descartada por incerteza antes
   de investir tempo em validação.

## Decisão
Usar o **fedmml-ed-triage**, restrito ao campo de texto `clinical_notes`
(ignorando vitais e labs — fora de escopo do classificador NLP leve pedido
pelo desafio). O alvo ESI 1–5 será remapeado para 3 classes:

| ESI nativo | Classe do projeto |
|---|---|
| 1–2 | urgente |
| 3   | atenção |
| 4–5 | normal |

## Prós
- Já nasce como problema de triagem/urgência — sem reinterpretação forçada
  de rótulos
- Volume muito acima do mínimo exigido (87k >> 2k)
- Sintético e sem dados reais de paciente — sem questões de privacidade/PHI
- Licença permissiva (CC BY 4.0), só exige atribuição

## Contras / riscos assumidos
- Dataset comunitário recente, artigo ainda em preprint (não peer-reviewed)
  — credibilidade da fonte é menor que a de um corpus estabelecido
- Texto gerado por template, não escrito por humanos reais — pode ter
  padrões linguísticos mais repetitivos que notas clínicas reais, o que
  tende a *facilitar* artificialmente a classificação
- Requer aceite de termos de acesso no Hugging Face antes do download
- Mapeamento ESI→3 classes é uma decisão de engenharia, não uma verdade
  clínica — deve ser citado explicitamente no README como simplificação

## Consequências
- EDA (ML-002) deve confirmar: completude do campo `clinical_notes`,
  distribuição das 3 classes após remapeamento, e se o texto sintético tem
  variação suficiente para não trivializar o classificador
- README do projeto deve deixar claro, desde a primeira versão, que o
  dataset é sintético e o mapeamento de urgência é uma adaptação do autor,
  não um padrão clínico oficial

## Status
Aceito em 04/08/2026.
