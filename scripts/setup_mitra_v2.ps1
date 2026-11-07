$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Push-Location -LiteralPath $taskRoot
try {
    $taskTmp = Join-Path $taskRoot '.task-tmp\mitra'
    New-Item -ItemType Directory -Force -Path $taskTmp | Out-Null
    $env:TEMP = $taskTmp
    $env:TMP = $taskTmp
    $taskPython = Join-Path $taskRoot '.venv-microstructure\Scripts\python.exe'
    $mitraPython = Join-Path $taskRoot '.venv-mitra-v2\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $mitraPython)) {
        & $taskPython -m venv .venv-mitra-v2
        if ($LASTEXITCODE -ne 0) { throw 'Could not create the isolated Mitra environment.' }
    }
    & $taskPython -m pip --python $mitraPython install -r scripts/mitra_v2_requirements.lock.txt
    if ($LASTEXITCODE -ne 0) { throw 'Mitra dependency installation failed.' }
    & $mitraPython -m pip check
    if ($LASTEXITCODE -ne 0) { throw 'Mitra dependency conflict.' }
    & $taskPython scripts/mitra_v2_assets.py --download
    if ($LASTEXITCODE -ne 0) { throw 'Pinned Mitra assets failed verification.' }
    Write-Host 'Isolated environment ready. Main .venv-microstructure is unchanged.'
} finally { Pop-Location }
