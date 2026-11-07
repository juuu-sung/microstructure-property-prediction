param([switch]$CheckOnly)
$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Push-Location -LiteralPath $taskRoot
try {
    $taskPython = Join-Path $taskRoot '.venv-microstructure\Scripts\python.exe'
    $taskMode = if ($CheckOnly) { 'check' } else { 'run' }
    & $taskPython -X utf8 scripts/resume_mitra_v2.py $taskMode
    if ($LASTEXITCODE -ne 0) { throw 'Mitra resume stopped. Completed outer cells are preserved; inspect the printed log.' }
} finally { Pop-Location }
