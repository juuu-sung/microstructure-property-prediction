param(
    [ValidatePattern('^cuda(:[0-9]+)?$')][string]$Device = 'cuda',
    [switch]$CheckOnly
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv-microstructure\Scripts\python.exe'
$experiment = 'scripts/cradio_spatial_experiment.py'
function Invoke-SpatialPython {
    param([string[]]$Arguments)
    Write-Host ('Running: python ' + ($Arguments -join ' '))
    & $pythonPath -X utf8 @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Stopped after failure (exit $LASTEXITCODE). Completed token chunks/training cells are preserved; rerun the same command." }
}
Push-Location -LiteralPath $projectRoot
try {
    Invoke-SpatialPython -Arguments @($experiment, 'check', '--device', $Device)
    Invoke-SpatialPython -Arguments @($experiment, 'prepare')
    if (-not $CheckOnly) {
        Invoke-SpatialPython -Arguments @($experiment, 'extract', '--device', $Device)
        Invoke-SpatialPython -Arguments @($experiment, 'evaluate', '--seed', '42', '--device', $Device)
        Invoke-SpatialPython -Arguments @($experiment, 'evaluate', '--seed', '2026', '--device', $Device)
        Invoke-SpatialPython -Arguments @($experiment, 'compare')
        Write-Host 'Complete. See reports/cradio_spatial/comparison.json and seed42/seed2026/report.txt.'
    }
}
finally { Pop-Location }
