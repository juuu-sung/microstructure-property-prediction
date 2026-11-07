param(
    [ValidatePattern('^(cpu|cuda(:[0-9]+)?)$')]
    [string]$Device = 'cuda',
    [ValidateRange(1, 128)]
    [int]$BatchSize = 8
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv-microstructure\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "Project Python environment not found: $pythonPath"
}

function Invoke-ExperimentPython {
    param([string[]]$Arguments)
    Write-Host ("Running: python " + ($Arguments -join ' '))
    & $pythonPath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Stopped after failure (exit $LASTEXITCODE). Fix this step and rerun; completed artifacts are verified and reused."
    }
}

Push-Location -LiteralPath $projectRoot
try {
    Invoke-ExperimentPython -Arguments @('scripts/setup_micronet.py')
    Invoke-ExperimentPython -Arguments @('scripts/efficientnet_micronet_experiment.py', 'prepare')
    Invoke-ExperimentPython -Arguments @('scripts/efficientnet_micronet_experiment.py', 'check', '--device', $Device, '--batch-size', "$BatchSize")
    Invoke-ExperimentPython -Arguments @('scripts/efficientnet_micronet_experiment.py', 'extract', '--device', $Device, '--batch-size', "$BatchSize")
    Invoke-ExperimentPython -Arguments @('scripts/efficientnet_micronet_experiment.py', 'evaluate', '--seed', '42')
    Invoke-ExperimentPython -Arguments @('scripts/efficientnet_micronet_experiment.py', 'evaluate', '--seed', '2026')
    Write-Host 'Complete. See reports/efficientnet_micronet/comparison.json and seed42/seed2026/report.md.'
}
finally {
    Pop-Location
}
