# Run simulations and independent verification for issue2, issue3, or both.
# Resolve paths from this script so callers do not need to change directory.
[CmdletBinding()]
param(
    [ValidateSet('issue2', 'issue3', 'all')]
    [string]$Scope = 'all',
    [switch]$Details,
    [switch]$RequireComplete
)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).ProviderPath
$pythonExe = Join-Path $projectRoot '.venv\Scripts\python.exe'

# Prefer the project environment; otherwise use Python from PATH.
if (-not (Test-Path -LiteralPath $pythonExe -PathType Leaf)) {
    $pythonCommand = Get-Command python -CommandType Application -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($null -eq $pythonCommand) {
        throw 'Python not found. Install Python 3.12+ or create the project .venv first.'
    }
    $pythonExe = $pythonCommand.Source
}

# Child failures are collected by the runner so all selected checks can finish.
# PowerShell 7 may otherwise turn an expected needs-review exit into an exception.
$PSNativeCommandUseErrorActionPreference = $false
$runnerArguments = @('--scope', $Scope.ToLowerInvariant())
if ($Details) { $runnerArguments += '--details' }
if ($RequireComplete) { $runnerArguments += '--require-complete' }
& $pythonExe -B -X utf8 (Join-Path $PSScriptRoot 'run_all.py') @runnerArguments
exit $LASTEXITCODE
