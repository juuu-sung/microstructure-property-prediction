param()
$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $taskRoot
$taskPython = Join-Path $taskRoot '.venv-microstructure\Scripts\python.exe'
function Invoke-Experiment([string[]]$Arguments) {
    Write-Host ('Running: python ' + ($Arguments -join ' '))
    & $taskPython @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Stopped after failure (exit $LASTEXITCODE). Rerun to reuse completed cells." }
}
Invoke-Experiment -Arguments @('scripts/mitra_v2_experiment.py','prepare')
Invoke-Experiment -Arguments @('scripts/mitra_v2_experiment.py','evaluate','--seed','42')
Invoke-Experiment -Arguments @('scripts/mitra_v2_experiment.py','evaluate','--seed','2026')
Invoke-Experiment -Arguments @('scripts/mitra_v2_experiment.py','compare')
Write-Host 'Complete. See reports/mitra_v2/comparison.json and seed42/seed2026/report.txt.'
