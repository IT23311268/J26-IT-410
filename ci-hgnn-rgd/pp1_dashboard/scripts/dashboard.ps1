param(
    [ValidateSet('Start', 'Stop', 'Status')]
    [string]$Action = 'Start',
    [switch]$NoBrowser
)

$ErrorActionPreference = 'Stop'
$dashboardRoot = Split-Path -Parent $PSScriptRoot
$runtimeRoot = Join-Path $dashboardRoot '.run'
$statePath = Join-Path $runtimeRoot 'services.json'
$dashboardUrl = 'http://127.0.0.1:5173/'
$services = @{}
if (Test-Path -LiteralPath $statePath) {
    $saved = Get-Content -Raw -LiteralPath $statePath | ConvertFrom-Json
    foreach ($entry in $saved.PSObject.Properties) { $services[$entry.Name] = $entry.Value }
}

function Save-Services {
    New-Item -ItemType Directory -Path $runtimeRoot -Force | Out-Null
    ConvertTo-Json -InputObject $services -Depth 4 | Set-Content -LiteralPath $statePath -Encoding UTF8
}

function Test-Port([int]$Port) {
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $connection = $client.BeginConnect('127.0.0.1', $Port, $null, $null)
        if (-not $connection.AsyncWaitHandle.WaitOne(500)) { return $false }
        $client.EndConnect($connection)
        return $true
    } catch { return $false }
    finally { $client.Dispose() }
}

function Test-Backend([string]$Url) {
    try {
        $health = Invoke-RestMethod -Uri $Url -TimeoutSec 3
        return ($health.status -eq 'ok' -and $health.read_only -eq $true -and $health.default_mode -eq 'real')
    } catch { return $false }
}

function Test-Frontend {
    try {
        $page = Invoke-WebRequest -UseBasicParsing -Uri $dashboardUrl -TimeoutSec 3
        return ($page.StatusCode -eq 200 -and $page.Content -match 'CI-HGNN-RGD')
    } catch { return $false }
}

function Get-OwnedProcess([string]$Name) {
    if (-not $services.ContainsKey($Name)) { return $null }
    $record = $services[$Name]
    $process = Get-Process -Id $record.pid -ErrorAction SilentlyContinue
    if ($null -ne $process -and $process.Path -eq $record.executable -and
        $process.StartTime.ToUniversalTime().Ticks.ToString() -eq $record.started_ticks) {
        return $process
    }
    return $null
}

function Stop-OwnedProcess([string]$Name) {
    $process = Get-OwnedProcess $Name
    if ($null -ne $process) {
        # Python's Windows venv redirector and Vite can each have child processes.
        # /T stops only this verified launcher-owned process tree, never a port owner.
        & taskkill.exe /PID $process.Id /T /F | Out-Null
        if ($LASTEXITCODE -ne 0 -and $null -ne (Get-OwnedProcess $Name)) {
            throw "Could not stop the owned $Name process."
        }
        Write-Host "Stopped $Name."
    }
    $services.Remove($Name)
    Save-Services
}

function Start-ServiceProcess([string]$Name, [string]$Executable, [string[]]$Arguments, [string]$Directory) {
    $process = Start-Process -FilePath $Executable -ArgumentList $Arguments -WorkingDirectory $Directory `
        -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $runtimeRoot "$Name.stdout.log") `
        -RedirectStandardError (Join-Path $runtimeRoot "$Name.stderr.log")
    $services[$Name] = @{
        pid = $process.Id
        executable = $process.Path
        started_ticks = $process.StartTime.ToUniversalTime().Ticks.ToString()
    }
    Save-Services
}

if ($Action -eq 'Status') {
    $backendReady = Test-Backend 'http://127.0.0.1:8000/api/health'
    $frontendReady = Test-Frontend
    $connectionReady = Test-Backend ($dashboardUrl + 'api/health')
    Write-Host "Backend ready: $backendReady"
    Write-Host "Website ready: $frontendReady"
    Write-Host "Website/API connected: $connectionReady"
    Write-Host "URL: $dashboardUrl"
    if (-not ($backendReady -and $frontendReady -and $connectionReady)) { exit 1 }
    exit 0
}

if ($Action -eq 'Stop') {
    Stop-OwnedProcess 'frontend'
    Stop-OwnedProcess 'backend'
    Write-Host 'Stopped services started by this launcher. Manually started services are left running.'
    exit 0
}

$backendDirectory = Join-Path $dashboardRoot 'backend'
$frontendDirectory = Join-Path $dashboardRoot 'frontend'
$pythonPath = Join-Path $backendDirectory '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Install the isolated backend environment using README.md first.' }
if (-not (Test-Path -LiteralPath (Join-Path $frontendDirectory 'node_modules\vite\bin\vite.js'))) {
    throw 'Run npm.cmd ci inside frontend first, as described in README.md.'
}
$nodePath = (Get-Command node.exe -ErrorAction Stop).Source
New-Item -ItemType Directory -Path $runtimeRoot -Force | Out-Null

# Refuse port conflicts before starting anything. Never terminate an unknown server.
$backendReady = Test-Backend 'http://127.0.0.1:8000/api/health'
$frontendReady = Test-Frontend
if ((Test-Port 8000) -and -not $backendReady) { throw 'Port 8000 is occupied by an unrecognized or unready server. No process was stopped.' }
if ((Test-Port 5173) -and -not $frontendReady) { throw 'Port 5173 is occupied by an unrecognized or unready server. No process was stopped.' }
$startedNow = @()
try {
    if (-not $backendReady) {
        if ($null -ne (Get-OwnedProcess 'backend')) { throw 'The backend is still starting. Retry in a few seconds, or run Stop Dashboard.cmd first.' }
        Start-ServiceProcess 'backend' $pythonPath @('-m', 'uvicorn', 'app.main:app', '--host', '127.0.0.1', '--port', '8000') $backendDirectory
        $startedNow += 'backend'
    }
    if (-not $frontendReady) {
        if ($null -ne (Get-OwnedProcess 'frontend')) { throw 'The frontend is still starting. Retry in a few seconds, or run Stop Dashboard.cmd first.' }
        Start-ServiceProcess 'frontend' $nodePath @('node_modules/vite/bin/vite.js', '--host', '127.0.0.1') $frontendDirectory
        $startedNow += 'frontend'
    }
    $deadline = (Get-Date).AddSeconds(45)
    do {
        foreach ($name in $startedNow) {
            if ($null -eq (Get-OwnedProcess $name)) { throw "$name exited during startup. Read .run\$name.stderr.log for the error." }
        }
        $ready = (Test-Backend 'http://127.0.0.1:8000/api/health') -and (Test-Frontend) -and (Test-Backend ($dashboardUrl + 'api/health'))
        if ($ready) { break }
        Start-Sleep -Milliseconds 500
    } while ((Get-Date) -lt $deadline)
    if (-not $ready) { throw 'Startup timed out. Read the backend and frontend logs inside .run.' }
} catch {
    foreach ($name in $startedNow) { Stop-OwnedProcess $name }
    throw
}

Write-Host "Dashboard ready: $dashboardUrl"
Write-Host 'Both services are running in the background. Stop them using Stop Dashboard.cmd.'
Write-Host 'After restarting Windows, run Start Dashboard.cmd again.'
if (-not $NoBrowser) {
    # This is the requested dashboard page, not a visible helper console.
    Start-Process $dashboardUrl
}
