param([ValidatePattern('^cuda(:[0-9]+)?$')][string]$Device = 'cuda')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv-microstructure\Scripts\python.exe'
$experiment = 'scripts/dinov2_factorial_experiment.py'
function Invoke-ExperimentPython {
    param([string[]]$Arguments)
    Write-Host ('Running: python ' + ($Arguments -join ' '))
    & $pythonPath -X utf8 @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Stopped after failure (exit $LASTEXITCODE). Rerun the same command to reuse completed chunks/cells." }
}
Push-Location -LiteralPath $projectRoot
try {
    Invoke-ExperimentPython -Arguments @($experiment, 'prepare')
    Invoke-ExperimentPython -Arguments @('scripts/setup_dinov2_factorial.py')
    Invoke-ExperimentPython -Arguments @($experiment, 'check', '--device', $Device)
    Invoke-ExperimentPython -Arguments @($experiment, 'extract', '--device', $Device, '--batch-size', '1')
    Invoke-ExperimentPython -Arguments @($experiment, 'evaluate', '--seed', '42')
    Invoke-ExperimentPython -Arguments @($experiment, 'evaluate', '--seed', '2026')
    Invoke-ExperimentPython -Arguments @($experiment, 'compare')
    Write-Host 'Complete. See reports/dinov2_factorial/comparison.json and seed42/seed2026/report.txt.'
}
finally { Pop-Location }
