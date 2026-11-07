param(
    [ValidatePattern('^cuda(:[0-9]+)?$')][string]$Device = 'cuda',
    [string]$OutputDir = 'reports/cradiov4_confirmation'
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv-microstructure\Scripts\python.exe'
$experiment = 'scripts/cradiov4_confirmation.py'
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
    Invoke-ConfirmationPython -Arguments @($experiment, 'prepare', '--output-dir', $OutputDir)
    Invoke-ConfirmationPython -Arguments @($experiment, 'check', '--device', $Device, '--output-dir', $OutputDir)
    foreach ($seed in @(31415, 27182, 16180)) {
        Invoke-ConfirmationPython -Arguments @($experiment, 'evaluate', '--seed', "$seed", '--device', $Device, '--output-dir', $OutputDir)
    }
    Invoke-ConfirmationPython -Arguments @($experiment, 'compare', '--output-dir', $OutputDir)
    Write-Host ('Complete. See ' + $OutputDir + '/confirmation_summary.json and report.txt.')
}
finally { Pop-Location }
