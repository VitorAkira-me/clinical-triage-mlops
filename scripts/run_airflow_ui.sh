#!/usr/bin/env bash
# AIR-001 - sobe o Airflow em modo standalone (UI completa), SÓ PRA DEMONSTRAÇÃO VISUAL
# (vídeo/gravação). NÃO é o script que prova que a DAG funciona - isso é
# scripts/run_airflow_dag.sh (airflow dags test, usado no CI/desenvolvimento) e continua
# intocado. Este aqui sobe scheduler + webserver + triggerer de pé, em foreground, pra
# navegar na UI e disparar a DAG manualmente. Não é modo de produção (é literalmente o que o
# próprio Airflow avisa no log: "Standalone mode is for development purposes only").
#
# Uso: ./scripts/run_airflow_ui.sh   (de qualquer diretório - resolve a raiz do repo sozinho)
# Ctrl+C encerra (container --rm, limpa sozinho ao sair).
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if ! docker image inspect air001-airflow > /dev/null 2>&1; then
  echo "==> Imagem air001-airflow não existe ainda - buildando (mesma do script de teste)"
  docker build -t air001-airflow ./airflow
fi

echo "==> Subindo Airflow standalone em http://localhost:8080 (Ctrl+C encerra)"
echo "==> A senha do usuário admin é gerada automaticamente. Testado de verdade: o banner de"
echo "    senha NÃO apareceu neste log mesmo com o webserver já respondendo - o método"
echo "    confirmado é ler o arquivo, em OUTRO terminal, com este container já rodando:"
echo
echo "      docker exec air001-ui-standalone cat /opt/airflow/standalone_admin_password.txt"
echo
echo "==> DAG 'air001_train_baseline' já aparece na lista, mas paused por padrão - dá pra"
echo "    disparar/pausar pela UI (toggle ao lado do nome, depois o botão de play)."
echo

export MSYS_NO_PATHCONV=1
docker run --rm --name air001-ui-standalone -p 8080:8080 \
  -e AIRFLOW_HOME=/opt/airflow \
  -e AIRFLOW__CORE__DAGS_FOLDER=/opt/airflow/project/airflow/dags \
  -e AIRFLOW__CORE__LOAD_EXAMPLES=False \
  -v "$REPO_ROOT:/opt/airflow/project" \
  air001-airflow airflow standalone
