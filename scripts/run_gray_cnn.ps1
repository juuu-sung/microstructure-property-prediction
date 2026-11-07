param([string]$Device = 'cuda')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv-microstructure\Scripts\python.exe'
$experiment = 'scripts/gray_cnn_experiment.py'
function Invoke-GrayPython {
    param([string[]]$Arguments)
    Write-Host ('Running: python ' + ($Arguments -join ' '))
    & $pythonPath -X utf8 @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Stopped after failure (exit $LASTEXITCODE). Rerun to reuse verified cells and checkpointed progress."
    }
}
Push-Location -LiteralPath $projectRoot
try {
    Invoke-GrayPython -Arguments @($experiment, 'prepare')
    Invoke-GrayPython -Arguments @($experiment, 'check', '--device', $Device)
    Invoke-GrayPython -Arguments @($experiment, 'train', '--device', $Device, '--seed', '42')
    Invoke-GrayPython -Arguments @($experiment, 'train', '--device', $Device, '--seed', '2026')
    Invoke-GrayPython -Arguments @($experiment, 'compare')
    Write-Host 'Complete. See reports/gray_cnn/comparison.json and seed42/seed2026/report.txt.'
}
finally { Pop-Location }
