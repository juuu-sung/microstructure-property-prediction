param([ValidatePattern('^cuda(:[0-9]+)?$')][string]$Device = 'cuda')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv-microstructure\Scripts\python.exe'
$experiment = 'scripts/final_cradiov4_submission.py'
function Invoke-FinalPython {
    param([string[]]$Arguments)
    Write-Host ('Running: python ' + ($Arguments -join ' '))
    & $pythonPath -X utf8 @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Stopped after failure (exit $LASTEXITCODE). Rerun this command to reuse verified completed contexts/stages/chunks."
    }
}
Push-Location -LiteralPath $projectRoot
try {
    Invoke-FinalPython -Arguments @($experiment, 'prepare')
    Invoke-FinalPython -Arguments @($experiment, 'train', '--device', $Device)
    Invoke-FinalPython -Arguments @($experiment, 'extract', '--device', $Device)
    Invoke-FinalPython -Arguments @($experiment, 'predict')
    Write-Host 'Complete. Separate CSV: reports/final_cradiov4_blend/submission/submission.csv'
}
finally { Pop-Location }
