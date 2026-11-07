param([ValidatePattern('^cuda(:[0-9]+)?$')][string]$Device='cuda',
      [string]$OutputDir='reports/dinov3_large')
$ErrorActionPreference='Stop'
$projectRoot=Split-Path -Parent $PSScriptRoot
$pythonPath=Join-Path $projectRoot '.venv-microstructure\Scripts\python.exe'
$experiment='scripts/dinov3_large_experiment.py'
function Invoke-ExperimentPython {
    param([string[]]$Arguments)
    Write-Host ('Running: python '+($Arguments -join ' '))
    & $pythonPath -X utf8 @Arguments
    if ($LASTEXITCODE -ne 0) { throw 'Stopped. Rerun the same command to reuse verified chunks/cells.' }
}
Push-Location -LiteralPath $projectRoot
try {
    Invoke-ExperimentPython -Arguments @('scripts/setup_dinov3_large.py')
    Invoke-ExperimentPython -Arguments @($experiment,'prepare','--output-dir',$OutputDir)
    Invoke-ExperimentPython -Arguments @($experiment,'check','--device',$Device,'--output-dir',$OutputDir)
    Invoke-ExperimentPython -Arguments @($experiment,'extract','--device',$Device,'--output-dir',$OutputDir)
    Invoke-ExperimentPython -Arguments @($experiment,'evaluate','--seed','42','--output-dir',$OutputDir)
    Invoke-ExperimentPython -Arguments @($experiment,'evaluate','--seed','2026','--output-dir',$OutputDir)
    Invoke-ExperimentPython -Arguments @($experiment,'compare','--output-dir',$OutputDir)
    Write-Host ('Complete. See '+$OutputDir+'/comparison.json and seed42/seed2026/report.txt.')
} finally { Pop-Location }
