#!/usr/bin/env bash
# AIR-001 - builda a imagem do Airflow e roda a DAG de treino (load_dataset -> train_baseline)
# uma vez, de ponta a ponta. Funciona em Linux, macOS e Git Bash no Windows.
#
# Uso: ./scripts/run_airflow_dag.sh   (de qualquer diretório - resolve a raiz do repo sozinho)
#
# MSYS_NO_PATHCONV=1 é exportado sempre: no Git Bash (Windows) evita que /opt/airflow seja
# traduzido pra um caminho Windows antes de chegar no Docker; em Linux/macOS não tem efeito
# nenhum (a variável simplesmente não é lida por nada) - sem risco de setar sempre.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "==> Repo root: $REPO_ROOT"
echo "==> Build da imagem air001-airflow (primeira vez baixa a imagem base + instala deps - pode levar alguns minutos)"
docker build -t air001-airflow ./airflow
if [ $? -ne 0 ]; then
  echo "==> docker build falhou - veja o erro acima." >&2
  exit 1
fi

MODEL_PATH="$REPO_ROOT/models/tfidf_logreg_baseline.joblib"
MARKER_FILE="$(mktemp)"
sleep 1  # garante que o marker fique estritamente anterior ao mtime do .joblib desta run,
         # mesmo em filesystems com granularidade de timestamp de 1s

EXEC_DATE="$(date +%F)"
echo
echo "==> Rodando a DAG air001_train_baseline (data de execução: $EXEC_DATE - precisa ser igual"
echo "    ou posterior ao start_date da DAG, 2026-01-01; qualquer data anterior faz o Airflow"
echo "    marcar a run como sucesso SEM rodar nenhuma task, silenciosamente - achado real,"
echo "    não hipotético, pego testando com uma data fixa anterior por engano)"
echo

export MSYS_NO_PATHCONV=1
docker run --rm \
  -e AIRFLOW_HOME=/opt/airflow \
  -e AIRFLOW__CORE__DAGS_FOLDER=/opt/airflow/project/airflow/dags \
  -e AIRFLOW__CORE__LOAD_EXAMPLES=False \
  -v "$REPO_ROOT:/opt/airflow/project" \
  air001-airflow bash -c "airflow db migrate && airflow dags test air001_train_baseline $EXEC_DATE"
DOCKER_EXIT=$?

# Exit code é o sinal primário (confirmado empiricamente: airflow dags test propaga exit 1 de
# verdade quando uma task falha, não só quando o DAG não existe) - não texto de log, que muda
# de formato entre versões do Airflow.
echo
if [ "$DOCKER_EXIT" -ne 0 ]; then
  echo "==> A DAG falhou (exit $DOCKER_EXIT) - procure 'Traceback' ou 'DagRun failed' no log acima." >&2
  exit 1
fi
echo "==> DAG rodou com sucesso (as 2 tasks: load_dataset, train_baseline)."

# Não basta o arquivo existir - pode ser de uma run anterior enquanto ESTA run silenciosamente
# não fez nada (foi exatamente o bug real encontrado testando este script: data de execução
# antes do start_date da DAG faz o Airflow reportar sucesso sem rodar task nenhuma). Confirma
# que o arquivo foi de fato (re)escrito por esta execução.
if [ -f "$MODEL_PATH" ] && [ "$MODEL_PATH" -nt "$MARKER_FILE" ]; then
  echo "==> $MODEL_PATH existe e foi atualizado agora:"
  ls -la "$MODEL_PATH"
else
  echo "==> $MODEL_PATH não foi atualizado por esta execução (existe mas está desatualizado, ou" >&2
  echo "    não existe) - a DAG reportou sucesso mas não rodou a task de treino de verdade." >&2
  rm -f "$MARKER_FILE"
  exit 1
fi

rm -f "$MARKER_FILE"
echo
echo "==> Tudo certo."
