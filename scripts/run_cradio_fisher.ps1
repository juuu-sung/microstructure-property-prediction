param(
    [ValidatePattern('^cuda(:[0-9]+)?$')][string]$Device = 'cuda',
    [switch]$CheckOnly
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv-microstructure\Scripts\python.exe'
$experiment = 'scripts/cradio_fisher_experiment.py'
function Invoke-FisherPython {
    param([string[]]$Arguments)
    Write-Host ('Running: python ' + ($Arguments -join ' '))
    & $pythonPath -X utf8 @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Stopped after failure (exit $LASTEXITCODE). Verified token chunks, dictionaries and Ridge cells are preserved; rerun this command."
    }
}
Push-Location -LiteralPath $projectRoot
try {
    Invoke-FisherPython -Arguments @($experiment, 'prepare')
    Invoke-FisherPython -Arguments @($experiment, 'check', '--device', $Device)
    if (-not $CheckOnly) {
        Invoke-FisherPython -Arguments @($experiment, 'extract', '--device', $Device)
        Invoke-FisherPython -Arguments @($experiment, 'evaluate')
        Write-Host 'Complete. See reports/cradio_fisher_d4/comparison.json and seed42/seed2026/report.txt.'
    }
}
finally { Pop-Location }
