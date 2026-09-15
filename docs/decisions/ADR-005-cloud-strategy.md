# ADR-005: Estratégia de deploy em nuvem (análise, sem implementação)

Análise textual exigida pelo Tech Challenge - `STEP 10` do [ROADMAP](../ROADMAP.md), a única
etapa obrigatória antes da documentação final. Escopo: **raciocínio e decisão, não deploy real**.

## Contexto

O projeto hoje é: uma API FastAPI síncrona (`POST /predict`, request/response, sem fila) rodando
num único container Docker autossuficiente (DOCK-001), um pipeline de treino separado rodando
como DAG Airflow sob demanda (AIR-001, não agendado, não contínuo), e uma stack de observabilidade
local (Prometheus + Grafana, OBS-001). Nada disso está publicado num registry de nenhum cloud
ainda - CI-001 valida o build da imagem, mas não faz push. Não há lock-in de nenhum provedor.

## Batch ou real-time?

**Real-time - não é uma escolha nova, é uma constatação do que já foi construído.** A API-001 já
definiu o contrato como síncrono: cliente manda `clinical_notes`, espera a resposta na mesma
requisição, com latência importando (medida em toda a stack: RED metrics no Passo 1 da OBS-001,
buckets calibrados pra sub-10ms; benchmark ONNX no OPT-001 comparando latência de inferência pura).
Isso é inerente ao próprio problema: **triagem** significa priorizar atendimento no momento em que
o laudo chega - processar em lote à noite anularia o propósito do sistema (o paciente já teria
sido atendido, bem ou mal, antes do lote rodar).

O projeto já reflete essa distinção na prática, sem ter sido planejado assim explicitamente: o
**treino** (Airflow, AIR-001) é batch por natureza - não precisa rodar continuamente, só quando
alguém decide retreinar - enquanto a **inferência** (API) precisa estar sempre no ar, respondendo
em tempo real. Mapeado pra cloud, isso vira dois primitivos diferentes, não um só (ver "Decisão").

## Qual provedor?

**Critério de decisão**: não há lock-in hoje (Docker puro), então o critério não é "o que já
usamos", é "menor distância conceitual entre o que já construímos localmente e o equivalente
gerenciado do provedor" - menos re-arquitetura, não só menos código.

- **AWS**: `ECS Fargate` roda o `Dockerfile` do DOCK-001 sem alteração (só apontar pra uma imagem
  no ECR). **Amazon Managed Service for Prometheus + Amazon Managed Grafana** são,
  literalmente, os dois mesmos produtos já escolhidos localmente no `docker-compose.yml` da
  OBS-001 - a migração do `prometheus.yml`/dashboard provisionado é quase 1:1, não uma reescrita
  pra CloudWatch ou outra ferramenta. `MWAA` (Managed Workflows for Apache Airflow) cobre o lado
  batch do treino sem reescrever a DAG do AIR-001. GitHub Actions integra com AWS via OIDC
  (sem chave estática) pro CD que o CI-001 ainda não tem.
- **GCP**: `Cloud Run` é o mais simples dos três pra este tamanho de projeto - serverless de
  verdade, escala a zero, `gcloud run deploy` direto do Dockerfile, paga por requisição. Pra uma
  API pequena com tráfego esporádico (o caso deste projeto), é o melhor custo/simplicidade dos
  três. `Cloud Composer` cobre o Airflow gerenciado.
- **Azure**: `Container Apps`/AKS rodam o container igual aos outros dois, mas `Azure Monitor`
  não é um drop-in do que já foi construído (Prometheus/Grafana) - exigiria mais re-trabalho de
  observabilidade que os outros dois.

## Decisão

**Real-time, não batch, para a API - batch, via Airflow gerenciado, só para o retraining.**

**AWS (ECS Fargate + Amazon Managed Prometheus + Amazon Managed Grafana + MWAA)** como
recomendação primária: é o provedor com menor distância entre o que já existe localmente (Docker
+ Prometheus + Grafana + Airflow) e o equivalente gerenciado - a stack de observabilidade
construída na OBS-001 não precisaria ser re-pensada, só re-apontada.

**GCP Cloud Run** fica registrado como alternativa honesta se o critério dominante for custo e
simplicidade operacional em vez de continuidade da stack de observabilidade - é genuinamente mais
simples de operar pra uma API deste porte (tráfego baixo, sem necessidade de ficar sempre quente).
Não seria uma escolha errada, é uma escolha com um critério diferente.

## Fora de escopo

Deploy real (nenhum dos dois foi implementado - a análise é o entregável, conforme o próprio
ROADMAP define para o `STEP 10`), dimensionamento de custo real (número de requisições/mês não
medido em produção), IaC (Terraform/CDK), estratégia de secrets em produção (`HF_TOKEN`, credenciais
de cloud).
