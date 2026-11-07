param(
    [ValidatePattern('^cuda(:[0-9]+)?$')][string]$Device = 'cuda',
    [switch]$CheckOnly
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv-microstructure\Scripts\python.exe'
$experiment = 'scripts/dinov2_large_register_d4_experiment.py'
function Invoke-RegisterPython {
    param([string[]]$Arguments)
    Write-Host ('Running: python ' + ($Arguments -join ' '))
    & $pythonPath -X utf8 @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Stopped after failure (exit $LASTEXITCODE). Completed register-Large chunks/cells are preserved; rerun this command."
    }
}
Push-Location -LiteralPath $projectRoot
try {
    Invoke-RegisterPython -Arguments @('scripts/setup_dinov2_large_register.py')
    Invoke-RegisterPython -Arguments @($experiment, 'prepare')
    Invoke-RegisterPython -Arguments @($experiment, 'check', '--device', $Device)
    if (-not $CheckOnly) {
        Invoke-RegisterPython -Arguments @($experiment, 'extract', '--device', $Device)
        Invoke-RegisterPython -Arguments @($experiment, 'evaluate')
        Write-Host 'Complete. See reports/dinov2_large_register_d4/comparison.json and seed42/seed2026/report.txt.'
    }
}
finally { Pop-Location }
