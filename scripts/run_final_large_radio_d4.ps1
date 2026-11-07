param(
    [ValidatePattern('^cuda(:[0-9]+)?$')][string]$Device = 'cuda',
    [switch]$PrepareOnly
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv-microstructure\Scripts\python.exe'
$experiment = 'scripts/final_large_radio_d4_submission.py'
function Invoke-FinalD4Python {
    param([string[]]$Arguments)
    Write-Host ('Running: python ' + ($Arguments -join ' '))
    & $pythonPath -X utf8 @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Stopped after failure (exit $LASTEXITCODE). Rerun this command to reuse verified heads, CV cells and test chunks."
    }
}
Push-Location -LiteralPath $projectRoot
try {
    Invoke-FinalD4Python -Arguments @($experiment, 'prepare')
    if (-not $PrepareOnly) {
        Invoke-FinalD4Python -Arguments @($experiment, 'train')
        Invoke-FinalD4Python -Arguments @($experiment, 'extract', '--device', $Device)
        Invoke-FinalD4Python -Arguments @($experiment, 'predict')
        Write-Host 'Complete. Separate CSV: reports/final_large_radio_d4_blend/submission/submission.csv'
    }
}
finally { Pop-Location }
