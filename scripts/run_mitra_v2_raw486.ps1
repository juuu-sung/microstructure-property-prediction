param([switch]$CheckOnly)
$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $taskRoot
$taskPython = Join-Path $taskRoot '.venv-microstructure\Scripts\python.exe'
function Invoke-Raw486([string[]]$Arguments) {
    Write-Host ('Running: python ' + ($Arguments -join ' '))
    & $taskPython @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Stopped (exit $LASTEXITCODE). Rerun to reuse completed outer cells. See worker_result.log." }
}
Invoke-Raw486 -Arguments @('scripts/mitra_v2_raw486_experiment.py','check')
Invoke-Raw486 -Arguments @('scripts/mitra_v2_raw486_experiment.py','prepare')
if ($CheckOnly) { Write-Host 'Check/preparation complete. No competition training or prediction.'; exit 0 }
Invoke-Raw486 -Arguments @('scripts/mitra_v2_raw486_experiment.py','evaluate','--seed','42')
Invoke-Raw486 -Arguments @('scripts/mitra_v2_raw486_experiment.py','evaluate','--seed','2026')
Invoke-Raw486 -Arguments @('scripts/mitra_v2_raw486_experiment.py','compare')
Write-Host 'Complete. See reports/mitra_v2_raw486/comparison.json and seed42/seed2026/report.txt.'
