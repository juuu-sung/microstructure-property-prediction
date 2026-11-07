param(
    [ValidatePattern('^(cpu|cuda(:[0-9]+)?)$')]
    [string]$Device = 'cuda',
    [ValidateRange(1, 128)]
    [int]$BatchSize = 8
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv-microstructure\Scripts\python.exe'
$experiment = 'scripts/micronet_multiview_experiment.py'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw "Python not found: $pythonPath" }

function Invoke-ExperimentPython {
    param([string[]]$Arguments)
    Write-Host ("Running: python " + ($Arguments -join ' '))
    & $pythonPath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Stopped after failure (exit $LASTEXITCODE). Fix this step and rerun; verified completed cells are reused."
    }
}

Push-Location -LiteralPath $projectRoot
try {
    Invoke-ExperimentPython -Arguments @($experiment, 'prepare')
    Invoke-ExperimentPython -Arguments @($experiment, 'check', '--device', $Device, '--batch-size', "$BatchSize")
    Invoke-ExperimentPython -Arguments @($experiment, 'extract', '--device', $Device, '--batch-size', "$BatchSize")
    Invoke-ExperimentPython -Arguments @($experiment, 'train', '--seed', '42', '--device', $Device)
    Invoke-ExperimentPython -Arguments @($experiment, 'train', '--seed', '2026', '--device', $Device)
    Invoke-ExperimentPython -Arguments @($experiment, 'compare')
    Write-Host 'Complete. See reports/micronet_multiview/comparison.json and seed42/seed2026/report.txt.'
}
finally { Pop-Location }
