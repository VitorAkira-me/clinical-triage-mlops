# AIR-001 - builda a imagem do Airflow e roda a DAG de treino (load_dataset -> train_baseline)
# uma vez, de ponta a ponta. PowerShell nativo do Windows (nao Git Bash).
#
# Uso: .\scripts\run_airflow_dag.ps1   (de qualquer diretorio - resolve a raiz do repo sozinho)
#
# Existe porque o bloco de comando bash do README/AIR-001.md (continuacao com \, $(date +%F))
# nao funciona colado direto no PowerShell - cada linha vira um comando separado e "-e"/"-v"
# sao interpretados como cmdlets inexistentes. Este script usa sintaxe PowerShell nativa
# (crase pra continuacao) e NAO usa "2>&1" no docker run: no PowerShell 5.1, redirecionar
# stderr de um comando nativo embrulha cada linha num NativeCommandError e aborta o script
# com $ErrorActionPreference = "Stop", mesmo quando o comando termina com exit code 0.

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
Set-Location $RepoRoot

Write-Host "==> Repo root: $RepoRoot"
Write-Host "==> Build da imagem air001-airflow (primeira vez baixa a imagem base + instala deps - pode levar alguns minutos)"

docker build -t air001-airflow ./airflow
if ($LASTEXITCODE -ne 0) {
    Write-Error "docker build falhou (exit $LASTEXITCODE)."
    exit 1
}

$ModelPath = Join-Path $RepoRoot "models\tfidf_logreg_baseline.joblib"
$MarkerTime = Get-Date

$ExecDate = Get-Date -Format "yyyy-MM-dd"
Write-Host ""
Write-Host "==> Rodando a DAG air001_train_baseline (data de execucao: $ExecDate - precisa ser"
Write-Host "    igual ou posterior ao start_date da DAG, 2026-01-01; qualquer data anterior faz"
Write-Host "    o Airflow marcar a run como sucesso SEM rodar nenhuma task, silenciosamente -"
Write-Host "    achado real, nao hipotetico, pego testando com uma data fixa anterior por engano)"
Write-Host ""

# Sem "2>&1" de proposito (ver nota no topo) - a saida do Airflow imprime direto no console.
docker run --rm `
    -e AIRFLOW_HOME=/opt/airflow `
    -e AIRFLOW__CORE__DAGS_FOLDER=/opt/airflow/project/airflow/dags `
    -e AIRFLOW__CORE__LOAD_EXAMPLES=False `
    -v "${RepoRoot}:/opt/airflow/project" `
    air001-airflow bash -c "airflow db migrate && airflow dags test air001_train_baseline $ExecDate"

$DagExitCode = $LASTEXITCODE

Write-Host ""
if ($DagExitCode -ne 0) {
    Write-Error "==> A DAG falhou (exit $DagExitCode) - procure 'Traceback' ou 'DagRun failed' no log acima."
    exit 1
}
Write-Host "==> DAG rodou com sucesso (as 2 tasks: load_dataset, train_baseline)."

# Nao basta o arquivo existir - pode ser de uma run anterior enquanto ESTA run silenciosamente
# nao fez nada (foi exatamente o bug real encontrado testando este script: data de execucao
# antes do start_date da DAG faz o Airflow reportar sucesso sem rodar task nenhuma). Confirma
# que o arquivo foi de fato (re)escrito por esta execucao.
if ((Test-Path $ModelPath) -and ((Get-Item $ModelPath).LastWriteTime -gt $MarkerTime)) {
    $info = Get-Item $ModelPath
    Write-Host "==> $ModelPath existe e foi atualizado agora: $($info.Length) bytes, $($info.LastWriteTime)"
} else {
    Write-Error "==> $ModelPath nao foi atualizado por esta execucao (existe mas esta desatualizado, ou nao existe) - a DAG reportou sucesso mas nao rodou a task de treino de verdade."
    exit 1
}

Write-Host ""
Write-Host "==> Tudo certo."
