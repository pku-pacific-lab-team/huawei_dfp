# Delete generated JSON reports directly under this project's results directory.
# Run with -WhatIf to preview. Subdirectories and non-JSON files are preserved.
[CmdletBinding(SupportsShouldProcess = $true, ConfirmImpact = 'Medium')]
param()

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).ProviderPath
$resultsPath = Join-Path $projectRoot 'results'

if (-not (Test-Path -LiteralPath $resultsPath)) {
    Write-Output "No reports to clear: $resultsPath"
    return
}

$resultsDirectory = Get-Item -LiteralPath $resultsPath -Force
if (-not $resultsDirectory.PSIsContainer -or
    ($resultsDirectory.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
    throw "Expected a regular results directory, not a file or link: $resultsPath"
}

$reports = @(Get-ChildItem -LiteralPath $resultsDirectory.FullName -File -Filter '*.json' -Force |
    Where-Object { -not ($_.Attributes -band [IO.FileAttributes]::ReparsePoint) })
$removedCount = 0
foreach ($report in $reports) {
    if ($PSCmdlet.ShouldProcess($report.FullName, 'Delete generated JSON report')) {
        Remove-Item -LiteralPath $report.FullName
        $removedCount++
    }
}

if (-not $WhatIfPreference) {
    Write-Output "Removed $removedCount report(s) from $resultsPath"
}
