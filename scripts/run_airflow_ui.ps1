# AIR-001 - sobe o Airflow em modo standalone (UI completa), SO PRA DEMONSTRACAO VISUAL
# (video/gravacao). NAO e o script que prova que a DAG funciona - isso e
# scripts\run_airflow_dag.ps1 (airflow dags test, usado no CI/desenvolvimento) e continua
# intocado. Este aqui sobe scheduler + webserver + triggerer de pe, em foreground, pra
# navegar na UI e disparar a DAG manualmente. Nao e modo de producao (e literalmente o que o
# proprio Airflow avisa no log: "Standalone mode is for development purposes only").
#
# Uso: .\scripts\run_airflow_ui.ps1   (de qualquer diretorio - resolve a raiz do repo sozinho)
# Ctrl+C encerra (container --rm, limpa sozinho ao sair).

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
Set-Location $RepoRoot

docker image inspect air001-airflow *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "==> Imagem air001-airflow nao existe ainda - buildando (mesma do script de teste)"
    docker build -t air001-airflow ./airflow
    if ($LASTEXITCODE -ne 0) {
        Write-Error "docker build falhou."
        exit 1
    }
}

Write-Host "==> Subindo Airflow standalone em http://localhost:8080 (Ctrl+C encerra)"
Write-Host "==> A senha do usuario admin e gerada automaticamente. Testado de verdade: o banner"
Write-Host "    de senha NAO apareceu neste log mesmo com o webserver ja respondendo - o metodo"
Write-Host "    confirmado e ler o arquivo, em OUTRO terminal, com este container ja rodando:"
Write-Host ""
Write-Host "      docker exec air001-ui-standalone cat /opt/airflow/standalone_admin_password.txt"
Write-Host ""
Write-Host "==> DAG 'air001_train_baseline' ja aparece na lista, mas paused por padrao - da pra"
Write-Host "    disparar/pausar pela UI (toggle ao lado do nome, depois o botao de play)."
Write-Host ""

# Sem "2>&1" de proposito (mesmo motivo do run_airflow_dag.ps1): no PowerShell 5.1 isso
# embrulha cada linha de stderr num NativeCommandError e aborta o script.
docker run --rm --name air001-ui-standalone -p 8080:8080 `
    -e AIRFLOW_HOME=/opt/airflow `
    -e AIRFLOW__CORE__DAGS_FOLDER=/opt/airflow/project/airflow/dags `
    -e AIRFLOW__CORE__LOAD_EXAMPLES=False `
    -v "${RepoRoot}:/opt/airflow/project" `
    air001-airflow airflow standalone
