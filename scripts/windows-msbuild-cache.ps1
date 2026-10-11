# Intel/MSBuild tracker logs provide the actual source-to-object dependency graph.
# Keep the logs intact: removing one stale output lets MSBuild rebuild only its
# producer instead of discarding every object in the target.
function Assert-MomBuildArtifactPath([string]$Path, [string]$BuildRoot) {
  if ($Path -ne [IO.Path]::GetFullPath($Path) -or
      -not $Path.StartsWith($BuildRoot + '\', [StringComparison]::OrdinalIgnoreCase) -or
      [IO.Path]::GetExtension($Path) -notin @('.obj','.pch','.res','.lib','.exp','.dll','.node')) {
    throw 'Invalid MSBuild artifact receipt.'
  }
  # A lexical prefix is insufficient: junctions can redirect a build-owned
  # path outside the build tree. Check the leaf and every ancestor through root.
  $ancestor = $Path
  while ($ancestor -and $ancestor.Length -ge $BuildRoot.Length) {
    if (Test-Path -LiteralPath $ancestor) {
      $entry = Get-Item -LiteralPath $ancestor -Force
      if (($entry.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw 'Build artifact crosses a reparse point.'
      }
    }
    if ($ancestor -eq $BuildRoot) { break }
    $ancestor = Split-Path -Parent $ancestor
  }
}

function Assert-MomBuildOutputTree([string]$Directory, [string]$BuildRoot) {
  Assert-MomBuildArtifactPath (Join-Path $Directory '.mom-clean.obj') $BuildRoot
  if (-not (Test-Path -LiteralPath $Directory -PathType Container)) { return }
  # Inspect each directory before descending so cleanup never traverses junctions.
  $directories = New-Object 'Collections.Generic.Stack[string]'
  $directories.Push($Directory)
  while ($directories.Count -gt 0) {
    $directory = $directories.Pop()
    foreach ($entry in (Get-ChildItem -LiteralPath $directory -Force)) {
      if (($entry.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) { throw 'Reparse-point build output.' }
      if ($entry.PSIsContainer) { $directories.Push($entry.FullName) }
    }
  }
}

function Clear-MomMsbuildRelease([string]$BuildRoot) {
  if ($BuildRoot -ne [IO.Path]::GetFullPath((Join-Path (Get-Location).ProviderPath 'build'))) {
    throw 'MSBuild cleanup requires the local build directory.'
  }
  $release = Join-Path $BuildRoot 'Release'
  Assert-MomBuildOutputTree $release $BuildRoot
  if (-not (Test-Path -LiteralPath $release -PathType Container)) { return }
  Remove-Item -LiteralPath $release -Recurse -Force
}

function Read-MomTrackerLog([string]$Path, [bool]$IncludeRoots = $false) {
  $groups = @{}
  if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $groups }
  $key = ''
  # PS5 Get-Content decorates strings with provider metadata, which JSON expands
  # recursively in receipts. ReadAllLines preserves BOM decoding without wrappers.
  foreach ($line in [IO.File]::ReadAllLines($Path)) {
    if ($line.StartsWith('^')) {
      $key = $line.Substring(1)
      if (-not $groups.ContainsKey($key)) { $groups[$key] = @() }
      if ($IncludeRoots) {
        $groups[$key] += @($key -split '\|')
      }
    } elseif ($key -and $line) {
      # Retain missing paths too. Signing must fail rather than silently shrink
      # the consumed dependency graph after a header or output disappears.
      $groups[$key] += $line
    }
  }
  return $groups
}

function Get-MomTrackedOutputs([string]$BuildRoot) {
  $rows = @()
  $release = Join-Path $BuildRoot 'Release'
  if (-not (Test-Path -LiteralPath $release)) { return $rows }
  Assert-MomBuildOutputTree $release $BuildRoot
  foreach ($readLog in (Get-ChildItem -LiteralPath $release -Filter '*.read.*.tlog' -File -Recurse)) {
    $reads = Read-MomTrackerLog $readLog.FullName $true
    $writes = Read-MomTrackerLog ($readLog.FullName -replace '\.read\.', '.write.')
    foreach ($key in $reads.Keys) {
      if (-not $writes.ContainsKey($key)) { continue }
      $inputs = @($reads[$key] | Sort-Object -Unique | ForEach-Object {
        @{ Path = $_; Hash = Get-MomBuildHash $_ }
      })
      foreach ($output in ($writes[$key] | Sort-Object -Unique)) {
        $output = [IO.Path]::GetFullPath($output)
        # Tracker roots may list input files too. Only generated Release artifacts can
        # ever be removed; refuse unrelated or source paths from malformed logs.
        if ($output.StartsWith($release + '\', [StringComparison]::OrdinalIgnoreCase) -and
            [IO.Path]::GetExtension($output) -in @('.obj','.pch','.res','.lib','.exp','.dll','.node')) {
          Assert-MomBuildArtifactPath $output $BuildRoot
          $rows += @{ Path = $output; Hash = Get-MomBuildHash $output; Inputs = $inputs }
        }
      }
    }
  }
  return $rows
}

function Assert-MomTrackedOutputCoverage($Outputs, [string]$BuildRoot) {
  $release = Join-Path $BuildRoot 'Release'
  Assert-MomBuildOutputTree $release $BuildRoot
  $covered = @{}
  foreach ($row in @($Outputs)) {
    $inputs = @($row.Inputs)
    if ($inputs.Count -gt 0 -and @($inputs | Where-Object {
      -not $_ -or -not $_.Path -or $_.Hash -notmatch '^[A-Fa-f0-9]{64}$'
    }).Count -eq 0) { $covered[$row.Path] = $true }
  }
  $intermediates = @(Get-ChildItem -LiteralPath $release -File -Recurse | Where-Object {
    $_.Extension -in @('.obj','.pch','.res')
  })
  # A partial/unsupported tracker graph must not become a successful cache receipt.
  if ($intermediates.Count -eq 0) { throw 'MSBuild produced no tracked Release intermediates.' }
  foreach ($file in $intermediates) {
    if (-not $covered.ContainsKey($file.FullName)) { throw 'MSBuild tracker graph omitted a Release intermediate.' }
  }
}

function Repair-MomTrackedOutputs($State, [string]$BuildRoot) {
  if (-not $State) { return }
  $release = Join-Path $BuildRoot 'Release'
  # Validate the complete receipt before deleting anything; SDKs also live under build/.
  foreach ($row in @($State.Outputs)) {
    Assert-MomBuildArtifactPath $row.Path $BuildRoot
    if (-not $row.Path.StartsWith($release + '\', [StringComparison]::OrdinalIgnoreCase)) {
      throw 'Tracked output is outside generated Release.'
    }
  }
  foreach ($row in @($State.Outputs)) {
    $valid = $true
    try {
      $valid = @($row.Inputs).Count -gt 0 -and $row.Hash -eq (Get-MomBuildHash $row.Path) -and
        @($row.Inputs | Where-Object { $_.Hash -ne (Get-MomBuildHash $_.Path) }).Count -eq 0
    } catch { $valid = $false }
    if (-not $valid) { Remove-Item -LiteralPath $row.Path -Force -ErrorAction SilentlyContinue }
  }
}
