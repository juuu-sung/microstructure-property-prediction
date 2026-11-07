[CmdletBinding()]
param([string]$Python = '')

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$environmentPath = Join-Path $projectRoot '.venv-microstructure'
$environmentPython = Join-Path $environmentPath 'Scripts\python.exe'

function Invoke-Checked {
    param([string]$Program, [string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE : $Program $Arguments"
    }
}

function Test-Python312 {
    param([string]$Program)
    if (-not (Test-Path -LiteralPath $Program -PathType Leaf)) { return $false }
    $versionResult = & $Program -c 'import sys; print(str(sys.version_info.major)+chr(46)+str(sys.version_info.minor))' 2>$null
    return ($LASTEXITCODE -eq 0 -and "$versionResult".Trim() -eq '3.12')
}

if (-not (Test-Path -LiteralPath $environmentPython -PathType Leaf)) {
    if (Test-Path -LiteralPath $environmentPath) {
        throw "$environmentPath already exists but has no Scripts/python.exe. Use another environment or inspect it manually."
    }
    $pythonCandidates = @()
    if ($Python) {
        $pythonCandidates += $Python
    } else {
        $pythonCandidates += Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
        $registeredPython = Get-Command python -ErrorAction SilentlyContinue
        if ($registeredPython) { $pythonCandidates += $registeredPython.Source }
        $launcher = Get-Command py -ErrorAction SilentlyContinue
        if ($launcher) {
            try {
                $fromLauncher = & $launcher.Source -3.12 -c 'import sys; print(sys.executable)' 2>$null
                if ($LASTEXITCODE -eq 0) { $pythonCandidates += "$fromLauncher".Trim() }
            } catch {
                # A Python launcher can exist without any registered interpreters.
            }
        }
        # Existing desktop runtime: used only as the base for a separate project venv.
    }
    $basePython = $pythonCandidates | Where-Object { Test-Python312 $_ } | Select-Object -First 1
    if (-not $basePython) {
        throw 'Python 3.12 was not found. Install Python 3.12, or supply -Python C:\path\to\python.exe.'
    }
    Write-Host "Creating project environment using $basePython"
    Invoke-Checked -Program $basePython -Arguments @('-m', 'venv', $environmentPath)
}

if (-not (Test-Python312 $environmentPython)) {
    throw 'The existing .venv-microstructure must use Python 3.12. It has not been modified.'
}

# Official paired CUDA 12.8 wheels; no global Python or existing ML environment changes.
Invoke-Checked -Program $environmentPython -Arguments @('-m', 'pip', 'install', '--upgrade', 'pip', '--index-url', 'https://pypi.org/simple')
Invoke-Checked -Program $environmentPython -Arguments @('-m', 'pip', 'install', 'torch==2.10.0', 'torchvision==0.25.0', '--index-url', 'https://download.pytorch.org/whl/cu128')
Invoke-Checked -Program $environmentPython -Arguments @('-m', 'pip', 'install', '-r', (Join-Path $projectRoot 'requirements-frozen-resnet.txt'), '--index-url', 'https://pypi.org/simple')
Invoke-Checked -Program $environmentPython -Arguments @('-m', 'pip', 'check')

$frozenPackages = & $environmentPython -m pip freeze
if ($LASTEXITCODE -ne 0) { throw 'pip freeze failed.' }
$reportPath = Join-Path $projectRoot 'reports'
New-Item -ItemType Directory -Path $reportPath -Force | Out-Null
$frozenPackages | Set-Content -LiteralPath (Join-Path $reportPath 'frozen_resnet_environment.txt') -Encoding utf8
Write-Host "Environment ready: $environmentPython"
Write-Host 'VS Code: Python: Select Interpreter -> .venv-microstructure\Scripts\python.exe'
Write-Host 'Next: .\.venv-microstructure\Scripts\python.exe scripts/extract_frozen_resnet.py --check-only'
