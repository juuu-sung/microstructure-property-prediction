$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv-microstructure\Scripts\python.exe'
$experiment = 'scripts/evaluate_dinov2_large518_confirmation.py'
function Invoke-ConfirmationPython {
    param([string[]]$Arguments)
    Write-Host ('Running: python ' + ($Arguments -join ' '))
    & $pythonPath -X utf8 @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Stopped after failure (exit $LASTEXITCODE). Rerun the same command to reuse verified completed cells."
    }
}
Push-Location -LiteralPath $projectRoot
try {
    Invoke-ConfirmationPython -Arguments @($experiment, 'prepare')
    Invoke-ConfirmationPython -Arguments @($experiment, 'evaluate')
    Invoke-ConfirmationPython -Arguments @($experiment, 'compare')
    Write-Host 'Complete. See reports/dinov2_large518_confirmation/confirmation_summary.json and report.txt.'
}
finally { Pop-Location }
