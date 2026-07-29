function Import-MomVcVars64 {
  $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
  $vsRoot = if (Test-Path $vswhere) {
    & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
      -property installationPath 2>$null | Select-Object -First 1
  } else { $null }
  $vcvars = @(
    $(if ($vsRoot) { Join-Path $vsRoot 'VC\Auxiliary\Build\vcvars64.bat' }),
    'C:\BuildTools\VC\Auxiliary\Build\vcvars64.bat'
  ) | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1
  if (-not $vcvars) { throw 'vcvars64.bat not found (need VS 2022 C++ build tools)' }

  # vcvars appends to inherited PATH/INCLUDE/LIB values. GitHub's package job has already loaded
  # oneAPI, CUDA, HIP, Node and Git by this point; feeding that environment through cmd.exe can
  # exceed its 8191-character expanded-line limit before vcvars gets to `set`. Seed vcvars from a
  # small Windows environment, retain its authoritative x64 include/library values, then append
  # the original executable search path for the explicitly selected compiler SDKs.
  $originalPath = $env:Path
  $saved = @{}
  foreach ($name in @('INCLUDE', 'LIB', 'LIBPATH')) {
    $saved[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
    [Environment]::SetEnvironmentVariable($name, $null, 'Process')
  }
  $env:Path = @(
    (Join-Path $env:SystemRoot 'System32'),
    $env:SystemRoot,
    (Join-Path $env:SystemRoot 'System32\Wbem')
  ) -join ';'

  try {
    $lines = & cmd.exe /d /s /c "call `"$vcvars`" >nul && set"
    if ($LASTEXITCODE -ne 0) { throw "vcvars64.bat failed with exit code $LASTEXITCODE" }
    foreach ($line in $lines) {
      if ($line -match '^([^=]+)=(.*)$') {
        [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], 'Process')
      }
    }
  } catch {
    $env:Path = $originalPath
    foreach ($name in $saved.Keys) {
      [Environment]::SetEnvironmentVariable($name, $saved[$name], 'Process')
    }
    throw
  }

  $seen = [Collections.Generic.HashSet[string]]::new(
    [StringComparer]::OrdinalIgnoreCase)
  $mergedPath = foreach ($entry in "$env:Path;$originalPath" -split ';') {
    if ($entry -and $seen.Add($entry)) { $entry }
  }
  $env:Path = $mergedPath -join ';'
}
