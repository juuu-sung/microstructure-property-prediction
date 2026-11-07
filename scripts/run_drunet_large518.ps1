param([string]$Device = 'cuda')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv-microstructure\Scripts\python.exe'
$experiment = 'scripts/drunet_large518_experiment.py'
function Invoke-ExperimentPython {
    param([string[]]$Arguments)
    Write-Host ('Running: python ' + ($Arguments -join ' '))
    & $pythonPath -X utf8 @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Stopped after failure (exit $LASTEXITCODE). Rerun to reuse verified caches."
    }
}
Push-Location -LiteralPath $projectRoot
try {
    Invoke-ExperimentPython -Arguments @($experiment, 'prepare')
    Invoke-ExperimentPython -Arguments @($experiment, 'extract', '--device', $Device)
    Invoke-ExperimentPython -Arguments @($experiment, 'evaluate', '--seed', '42')
    Invoke-ExperimentPython -Arguments @($experiment, 'evaluate', '--seed', '2026')
    Invoke-ExperimentPython -Arguments @($experiment, 'compare')
    Write-Host 'Complete. See reports/drunet_large518/comparison.json and seed42/seed2026/report.txt.'
}
finally { Pop-Location }
