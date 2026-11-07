param([ValidatePattern('^cuda(:[0-9]+)?$')][string]$Device = 'cuda')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv-microstructure\Scripts\python.exe'
$experiment = 'scripts/micronet_mim_experiment.py'
function Invoke-ExperimentPython {
    param([string[]]$Arguments)
    Write-Host ('Running: python ' + ($Arguments -join ' '))
    & $pythonPath -X utf8 @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Stopped after failure (exit $LASTEXITCODE). Rerun after fixing; completed cells and 5-epoch checkpoints are reused." }
}
Push-Location -LiteralPath $projectRoot
try {
    Invoke-ExperimentPython -Arguments @('scripts/setup_micronet.py', '--validate-only')
    Invoke-ExperimentPython -Arguments @($experiment, 'prepare')
    Invoke-ExperimentPython -Arguments @($experiment, 'check', '--device', $Device)
    Invoke-ExperimentPython -Arguments @($experiment, 'train', '--seed', '42', '--device', $Device)
    Invoke-ExperimentPython -Arguments @($experiment, 'train', '--seed', '2026', '--device', $Device)
    Invoke-ExperimentPython -Arguments @($experiment, 'compare')
    Write-Host 'Complete. See reports/micronet_mim/comparison.json and seed42/seed2026/report.txt.'
}
finally { Pop-Location }
