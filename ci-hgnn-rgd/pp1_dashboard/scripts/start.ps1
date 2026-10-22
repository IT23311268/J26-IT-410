param(
    [Parameter(Mandatory=$true)]
    [ValidateSet('backend', 'frontend')]
    [string]$Service
)
$ErrorActionPreference = 'Stop'
$dashboardRoot = Split-Path -Parent $PSScriptRoot
if ($Service -eq 'backend') {
    $pythonPath = Join-Path $dashboardRoot 'backend\.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $pythonPath)) {
        throw 'Create the isolated backend environment using README.md first.'
    }
    Push-Location (Join-Path $dashboardRoot 'backend')
    try { & $pythonPath -m uvicorn app.main:app --host 127.0.0.1 --port 8000 }
    finally { Pop-Location }
} else {
    Push-Location (Join-Path $dashboardRoot 'frontend')
    try { & npm.cmd run dev }
    finally { Pop-Location }
}
