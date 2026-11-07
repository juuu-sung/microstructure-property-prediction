param([ValidatePattern('^cuda(:[0-9]+)?$')][string]$Device = 'cuda')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv-microstructure\Scripts\python.exe'
$experiment = 'scripts/residual_correction_experiment.py'
function Invoke-ExperimentPython {
    param([string[]]$Arguments)
    Write-Host ('Running: python ' + ($Arguments -join ' '))
    & $pythonPath -X utf8 @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Stopped after failure (exit $LASTEXITCODE). Completed context/correction cells are reused on rerun." }
}
Push-Location -LiteralPath $projectRoot
try {
    Invoke-ExperimentPython -Arguments @($experiment, 'prepare')
    Invoke-ExperimentPython -Arguments @($experiment, 'check', '--device', $Device)
    Invoke-ExperimentPython -Arguments @($experiment, 'evaluate', '--seed', '42', '--device', $Device)
    Invoke-ExperimentPython -Arguments @($experiment, 'evaluate', '--seed', '2026', '--device', $Device)
    Invoke-ExperimentPython -Arguments @($experiment, 'compare')
    Write-Host 'Complete. See reports/residual_correction/comparison.json and seed42/seed2026/report.txt.'
}
finally { Pop-Location }
