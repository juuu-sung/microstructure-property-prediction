param(
    [ValidatePattern('^cuda(:[0-9]+)?$')][string]$Device = 'cuda',
    [switch]$CheckOnly
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv-microstructure\Scripts\python.exe'
$experiment = 'scripts/cradiov4_h_d4_experiment.py'
function Invoke-HRadioPython {
    param([string[]]$Arguments)
    Write-Host ('Running: python ' + ($Arguments -join ' '))
    & $pythonPath -X utf8 @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Stopped after failure (exit $LASTEXITCODE). Completed H chunks/cells are preserved; rerun this command."
    }
}
Push-Location -LiteralPath $projectRoot
try {
    Invoke-HRadioPython -Arguments @('scripts/setup_cradiov4_h.py')
    Invoke-HRadioPython -Arguments @($experiment, 'prepare')
    Invoke-HRadioPython -Arguments @($experiment, 'check', '--device', $Device)
    if (-not $CheckOnly) {
        Invoke-HRadioPython -Arguments @($experiment, 'extract', '--device', $Device)
        Invoke-HRadioPython -Arguments @($experiment, 'evaluate')
        Write-Host 'Complete. See reports/cradiov4_h_d4/comparison.json and seed42/seed2026/report.txt.'
    }
}
finally { Pop-Location }
