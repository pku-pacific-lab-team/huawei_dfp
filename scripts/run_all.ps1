# Run all four default INT8 timing simulations and write reports to results.
# Resolve paths from this script so callers do not need to change directory.
[CmdletBinding()]
param()

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

$entries = @('ascend.py', 'tpu.py', 'tsp.py', 'tsp_cim.py')
for ($index = 0; $index -lt $entries.Count; $index++) {
    $entry = $entries[$index]
    $entryPath = Join-Path $projectRoot $entry
    Write-Output ("[{0}/{1}] Running {2}" -f ($index + 1), $entries.Count, $entry)
    & $pythonExe -B -X utf8 $entryPath
    # Native process failures must stop the batch instead of reporting success.
    if ($LASTEXITCODE -ne 0) {
        throw "Stopped: $entry failed with exit code $LASTEXITCODE. Later simulators were not run."
    }
}

Write-Output ("All four simulations completed. Reports: {0}" -f (Join-Path $projectRoot 'results'))
