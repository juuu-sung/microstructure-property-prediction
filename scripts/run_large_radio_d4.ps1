param(
    [ValidatePattern('^cuda(:[0-9]+)?$')][string]$Device = 'cuda',
    [switch]$CheckOnly
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv-microstructure\Scripts\python.exe'
$experiment = 'scripts/large_radio_d4_experiment.py'
function Invoke-D4Python {
    param([string[]]$Arguments)
    Write-Host ('Running: python ' + ($Arguments -join ' '))
    & $pythonPath -X utf8 @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Stopped after failure (exit $LASTEXITCODE). Completed chunks/cells are preserved; rerun the same command."
    }
}
Push-Location -LiteralPath $projectRoot
try {
    Invoke-D4Python -Arguments @($experiment, 'prepare')
    Invoke-D4Python -Arguments @($experiment, 'check', '--device', $Device)
    if (-not $CheckOnly) {
        Invoke-D4Python -Arguments @($experiment, 'extract', '--device', $Device)
        Invoke-D4Python -Arguments @($experiment, 'evaluate', '--seed', '42')
        Invoke-D4Python -Arguments @($experiment, 'evaluate', '--seed', '2026')
        Invoke-D4Python -Arguments @($experiment, 'compare')
        Write-Host 'Complete. See reports/large_radio_d4/comparison.json and seed42/seed2026/report.txt.'
    }
}
finally { Pop-Location }
