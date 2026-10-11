# Shared by the clang-based Windows workers. State is a receipt, never a substitute
# for verifying the current inputs and the successful output.
. (Join-Path $PSScriptRoot 'windows-msbuild-cache.ps1')
$script:MomBuildSnapshot = @()

function Initialize-MomBuildEnvironment([string]$RepoRoot) {
  if (-not $env:npm_package_json) { return }
  # Normalize npm's proven launcher scaffold before resolving compiler tools,
  # not by filtering PATH out of their fingerprints. JSON is ASCII-safe on PS5.
  $global:LASTEXITCODE = $null
  $encodedPath = & node (Join-Path $RepoRoot 'scripts\build-environment.js') --json
  $normalizeExit = $global:LASTEXITCODE
  if ($normalizeExit -eq 3) { return }
  if ($null -eq $normalizeExit -or $normalizeExit -ne 0) { throw "Build environment normalization failed ($normalizeExit)." }
  try {
    $normalizedPath = ConvertFrom-Json -InputObject ($encodedPath -join "`n") -ErrorAction Stop
    if ($normalizedPath -isnot [string]) { throw 'Expected PATH string.' }
  } catch { throw 'Invalid normalized build environment.' }
  $env:PATH = $normalizedPath
  Remove-Item Env:npm_package_json -ErrorAction SilentlyContinue
}

function Remove-MomAbandonedBuildReceipts([string]$CacheDir) {
  $CacheDir = Resolve-MomBuildPath $CacheDir
  $buildRoot = Resolve-MomBuildPath 'build'
  Assert-MomBuildArtifactPath (Join-Path $CacheDir '.mom-cache.obj') $buildRoot
  if (-not (Test-Path -LiteralPath $CacheDir -PathType Container)) { return }
  foreach ($file in (Get-ChildItem -LiteralPath $CacheDir -File -Force)) {
    if ($file.Name -notmatch '^(?:.+\.obj|link|configure|outputs)\.json\.[a-fA-F0-9]{32}\.tmp$') { continue }
    if (($file.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) { throw 'Reparse-point build receipt.' }
    Remove-Item -LiteralPath $file.FullName -Force
  }
}

function Remove-MomObsoleteCompileOutputs([string]$CacheDir, [string[]]$Objects) {
  $CacheDir = Resolve-MomBuildPath $CacheDir
  $buildRoot = Resolve-MomBuildPath 'build'
  Assert-MomBuildArtifactPath (Join-Path $CacheDir '.mom-cache.obj') $buildRoot
  $current = @{}
  foreach ($object in $Objects) { $current[(Resolve-MomBuildPath $object)] = $true }
  foreach ($file in (Get-ChildItem -LiteralPath $CacheDir -File -Force)) {
    if ($file.Name -notmatch '^(.+\.obj)(?:\.(?:(?:spir64|nvptx64-nvidia-cuda)\.)?d|\.json)?$') { continue }
    $object = Join-Path $CacheDir $Matches[1]
    if ($current.ContainsKey((Resolve-MomBuildPath $object))) { continue }
    Assert-MomBuildArtifactPath $object $buildRoot
    if (($file.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) { throw 'Reparse-point compile output.' }
    Remove-Item -LiteralPath $file.FullName -Force
  }
  Remove-MomAbandonedBuildReceipts $CacheDir
}

function Resolve-MomBuildPath([string]$Path) {
  if (-not [IO.Path]::IsPathRooted($Path)) { $Path = Join-Path (Get-Location).ProviderPath $Path }
  return [IO.Path]::GetFullPath($Path)
}

function Get-MomBuildHash([string]$Path) {
  if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "Build input missing: $Path" }
  return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash
}

function ConvertTo-MomCanonical($Value) {
  if ($Value -is [Collections.IDictionary]) {
    $ordered = [ordered]@{}
    foreach ($key in @($Value.Keys | Sort-Object)) { $ordered[$key] = ConvertTo-MomCanonical $Value[$key] }
    return $ordered
  }
  if ($Value -is [Array]) { return ,@($Value | ForEach-Object { ConvertTo-MomCanonical $_ }) }
  return $Value
}

function Get-MomBuildKey($Value) {
  $bytes = [Text.Encoding]::UTF8.GetBytes((ConvertTo-Json -InputObject (ConvertTo-MomCanonical $Value) -Compress -Depth 12))
  $sha = [Security.Cryptography.SHA256]::Create()
  try { return ([BitConverter]::ToString($sha.ComputeHash($bytes))).Replace('-', '') } finally { $sha.Dispose() }
}

function Initialize-MomIncrementalBuild([string]$CacheDir, [string[]]$Tools, [string[]]$ToolchainRoots) {
  Remove-MomAbandonedBuildReceipts $CacheDir
  New-Item -ItemType Directory -Force $CacheDir | Out-Null
  $toolRows = @($Tools | Sort-Object -Unique | ForEach-Object {
    @{ Path = Resolve-MomBuildPath $_; Hash = Get-MomBuildHash $_ }
  })
  $rootRows = @($ToolchainRoots | Where-Object { $_ } | Sort-Object -Unique | ForEach-Object {
    $root = $_
    # Bound discovery to installed tools/libraries, not SDK source/build trees.
    # Actual included headers are content-hashed through compiler dependencies.
    $files = @(Get-ChildItem -LiteralPath $root -File)
    foreach ($directory in @('bin', 'lib', 'lib64', 'Lib')) {
      $leaf = Join-Path $root $directory
      if (Test-Path -LiteralPath $leaf -PathType Container) {
        $files += @(Get-ChildItem -LiteralPath $leaf -File -Recurse -Depth 3)
      }
    }
    $files | Sort-Object FullName -Unique | ForEach-Object {
      @{ Path = $_.FullName; Size = $_.Length; Modified = $_.LastWriteTimeUtc.Ticks }
    }
  })
  $cohortRows = @($ToolchainRoots | Where-Object { $_ } | ForEach-Object {
    Get-ChildItem -LiteralPath $_ -File | Where-Object { $_.Extension -in @('.json', '.txt', '.manifest') } | Sort-Object FullName | ForEach-Object {
      @{ Path = $_.FullName; Hash = Get-MomBuildHash $_.FullName }
    }
  })
  # Bitcode, wrapper configuration and selected compiler helpers are consumed
  # without appearing in -MD; timestamps alone do not identify their contents.
  $sdkInputs = @($ToolchainRoots | Where-Object { $_ } | ForEach-Object {
    foreach ($relative in @('nvvm\libdevice', 'amdgcn\bitcode', 'lib\clang', 'lib\hipSYCL', 'etc\hipSYCL', 'etc\AdaptiveCpp')) {
      $leaf = Join-Path $_ $relative
      if (Test-Path -LiteralPath $leaf -PathType Container) {
        Get-ChildItem -LiteralPath $leaf -File -Recurse -Depth 4 | Where-Object { $_.Extension -in @('.bc', '.json', '.conf') }
      }
    }
    foreach ($relative in @('bin\clang++.exe', 'bin\lld-link.exe', 'bin\ptxas.exe', 'bin\nvlink.exe',
        'bin\sycl-post-link.exe', 'bin\llvm-spirv.exe', 'bin\clang-offload-wrapper.exe', 'bin\clang-offload-bundler.exe',
        'lib\sycl.lib', 'lib\amdhip64.lib', 'lib\x64\nvrtc.lib', 'lib\x64\cuda.lib')) {
      $leaf = Join-Path $_ $relative
      if (Test-Path -LiteralPath $leaf -PathType Leaf) { Get-Item -LiteralPath $leaf }
    }
    $lib = Join-Path $_ 'lib'
    if (Test-Path -LiteralPath $lib -PathType Container) {
      Get-ChildItem -LiteralPath $lib -File | Where-Object { $_.Extension -eq '.lib' -and $_.Name -match '(sycl|acpp|adaptivecpp)' }
    }
  } | Sort-Object FullName -Unique | ForEach-Object { @{ Path = $_.FullName; Hash = Get-MomBuildHash $_.FullName } })
  # Linker search order matters, but duplicate directories do not. Hash only
  # immediate library candidates on the selected architecture's LIB search path.
  $seenLibraries = @{}
  $script:MomBuildLibraryDirectories = @($env:LIB -split ';' | Where-Object { -not [string]::IsNullOrWhiteSpace($_) } | ForEach-Object {
    $directory = Resolve-MomBuildPath ($_.Trim().Trim('"'))
    if (-not $seenLibraries.ContainsKey($directory)) { $seenLibraries[$directory] = $true; $directory }
  })
  $libraryInputs = @($script:MomBuildLibraryDirectories | ForEach-Object {
    if (Test-Path -LiteralPath $_ -PathType Container) { Get-ChildItem -LiteralPath $_ -File -Filter '*.lib' }
  } | Sort-Object FullName -Unique | ForEach-Object { @{ Path = $_.FullName; Hash = Get-MomBuildHash $_.FullName } })
  $sdkInputs = @($sdkInputs + $libraryInputs | Sort-Object { $_.Path } -Unique)
  $repoInputs = @(foreach ($directory in @('native', 'sycl', 'xmrig')) {
    if (Test-Path -LiteralPath $directory -PathType Container) {
      Get-ChildItem -LiteralPath $directory -File -Recurse | ForEach-Object { @{ Path = $_.FullName; Hash = Get-MomBuildHash $_.FullName } }
    }
  })
  $includeMembership = @($ToolchainRoots | Where-Object { $_ } | ForEach-Object {
    $leaf = Join-Path $_ 'include'
    if (Test-Path -LiteralPath $leaf -PathType Container) { Get-ChildItem -LiteralPath $leaf -File -Recurse | Select-Object -ExpandProperty FullName }
  } | Sort-Object -Unique)
  $script:MomBuildDirectories = @(@('native', 'sycl', 'xmrig') + @($ToolchainRoots | Where-Object { $_ } | ForEach-Object { Join-Path $_ 'include' }) |
    Where-Object { Test-Path -LiteralPath $_ -PathType Container } | ForEach-Object { Resolve-MomBuildPath $_ } | Sort-Object -Unique)
  $sourceMembership = @($repoInputs | ForEach-Object { $_.Path })
  $script:MomBuildMembership = Get-MomBuildKey @($sourceMembership + $includeMembership + @($libraryInputs | ForEach-Object { $_.Path }) | Sort-Object -Unique)
  $script:MomBuildSnapshot = @($toolRows + $cohortRows + $sdkInputs + $repoInputs)
  # Repeated PATH prepending does not change tool lookup. Keep first occurrence/order.
  $seen = @{}
  $searchPath = @($env:PATH -split ';' | Where-Object { if ($seen.ContainsKey($_.ToLowerInvariant())) { return $false }; $seen[$_.ToLowerInvariant()] = $true; return $true })
  return Get-MomBuildKey @{ Version = 2; Tools = $toolRows; Roots = $rootRows; Cohort = $cohortRows; Sdk = $sdkInputs;
    Membership = $script:MomBuildMembership;
    Env = @($env:INCLUDE, ($script:MomBuildLibraryDirectories -join ';'), $env:LIBPATH, ($searchPath -join ';'), $env:CUDA_PATH, $env:HIP_PATH, $env:ROCM_PATH) }
}

function Assert-MomBuildInputs {
  $snapshot = Get-Variable -Name MomBuildSnapshot -Scope Script -ErrorAction SilentlyContinue
  $checked = @{}
  if ($snapshot) { foreach ($inputFile in $snapshot.Value) {
    if ($checked.ContainsKey($inputFile.Path)) {
      if ($checked[$inputFile.Path] -ne $inputFile.Hash) { throw 'Dependency changed between task admissions.' }
      continue
    }
    if ($inputFile.Hash -ne (Get-MomBuildHash $inputFile.Path)) { throw "Build input changed during compilation: $($inputFile.Path)" }
    $checked[$inputFile.Path] = $inputFile.Hash
  } }
  $directories = Get-Variable -Name MomBuildDirectories -Scope Script -ErrorAction SilentlyContinue
  if ($directories) {
    $membership = @($directories.Value | ForEach-Object { Get-ChildItem -LiteralPath $_ -File -Recurse | Select-Object -ExpandProperty FullName } | Sort-Object -Unique)
    $membership += @($script:MomBuildLibraryDirectories | ForEach-Object {
      if (Test-Path -LiteralPath $_ -PathType Container) { Get-ChildItem -LiteralPath $_ -File -Filter '*.lib' | Select-Object -ExpandProperty FullName }
    })
    $membership = @($membership | Sort-Object -Unique)
    if ((Get-MomBuildKey $membership) -ne $script:MomBuildMembership) { throw 'Include/source membership changed during compilation' }
  }
}

function Read-MomBuildState([string]$Path) {
  try { return Get-Content -LiteralPath $Path -Raw -Encoding UTF8 | ConvertFrom-Json } catch { return $null }
}

function Write-MomBuildState([string]$Path, $Value) {
  $temp = "$Path.$([Guid]::NewGuid().ToString('N')).tmp"
  try {
    $Value | ConvertTo-Json -Compress -Depth 12 | Set-Content -LiteralPath $temp -Encoding UTF8
    Move-Item -LiteralPath $temp -Destination $Path -Force
  } finally { Remove-Item -LiteralPath $temp -Force -ErrorAction SilentlyContinue }
}

function Get-MomCompilerDependencies([string]$Path) {
  # -MT mom avoids drive-letter colons in the make target. Clang escapes spaces,
  # backslashes, hashes and dollars; only backslash-newline joins physical lines.
  $text = (Get-Content -LiteralPath $Path -Raw -Encoding UTF8) -replace '\\\r?\n', ' '
  if ($text -notmatch '^mom:\s*') { throw "Invalid compiler dependency file: $Path" }
  # SYCL can emit separate host/device rules with the same requested target.
  $body = ($text -split '\r?\n' | ForEach-Object {
    if ($_ -match '^\s*$') { return }
    if ($_ -notmatch '^mom:\s*') { throw "Unexpected compiler dependency target: $Path" }
    $_ -replace '^mom:\s*', ''
  }) -join ' '
  $tokens = [regex]::Matches($body, '(?:\\.|[^\s\\])+')
  $paths = @($tokens | ForEach-Object {
    $p = $_.Value -replace '\\([ #\\])', '$1' -replace '\$\$', '$'
    Resolve-MomBuildPath $p
  } | Sort-Object -Unique)
  if ($paths.Count -eq 0) { throw "Empty compiler dependency file: $Path" }
  return @($paths | ForEach-Object { @{ Path = $_; Hash = Get-MomBuildHash $_ } })
}

function New-MomCompileTask([string]$Exe, [string[]]$CompilerArgs, [string]$Source,
    [string]$Object, [string]$What, [string]$Context) {
  $dep = "$Object.d"
  $actualArgs = @($CompilerArgs) + @('-MD', '-MF', $dep, '-MT', 'mom')
  $depFiles = @($dep)
  if ('-fsycl' -in $CompilerArgs) {
    # Host -MD can omit device-only includes. Give each frontend its own file
    # during the same compilation, and require all of them before caching.
    $targetArgs = @($CompilerArgs | Where-Object { $_ -like '-fsycl-targets=*' })
    if ($targetArgs.Count -ne 1) { throw 'Incremental SYCL compilation requires explicit targets.' }
    $targets = @($targetArgs[0].Substring('-fsycl-targets='.Length) -split ',' | ForEach-Object {
      if ($_ -match '^nvidia_gpu_sm_[0-9]+$') { 'nvptx64-nvidia-cuda' } else { $_ }
    } | Sort-Object -Unique)
    foreach ($target in $targets) {
      if ($target -notin @('spir64', 'nvptx64-nvidia-cuda')) { throw 'Unsupported incremental SYCL target.' }
      $deviceDep = "$Object.$target.d"
      $depFiles += $deviceDep
      $actualArgs += @("-Xsycl-target-frontend=$target", "-MF$deviceDep")
    }
  }
  $key = Get-MomBuildKey @{ Context = $Context; Exe = $Exe; Args = $actualArgs; WorkDir = (Get-Location).Path }
  $task = [PSCustomObject]@{ Exe = $Exe; Args = $actualArgs; What = $What; Object = $Object;
    Dep = $dep; DepFiles = $depFiles; State = "$Object.json"; Key = $key; Source = Resolve-MomBuildPath $Source; Skip = $false;
    Started = [DateTime]::UtcNow.Ticks; Before = @() }
  $state = Read-MomBuildState $task.State
  try {
    if ($env:MOM_FORCE_REBUILD -ne '1' -and $state -and $state.Key -eq $key -and $state.Output -eq (Get-MomBuildHash $Object)) {
      $deps = @($state.Dependencies)
      if ($deps.Count -gt 0 -and @($deps | Where-Object { $_.Path -eq $task.Source }).Count -gt 0) {
        $task.Skip = @($deps | Where-Object { $_.Hash -ne (Get-MomBuildHash $_.Path) }).Count -eq 0
      }
    }
  } catch { $task.Skip = $false }
  if ($task.Skip) { $script:MomBuildSnapshot = @($script:MomBuildSnapshot) + $deps }
  if (-not $task.Skip) {
    $known = @($task.Source)
    if ($state -and $state.PSObject.Properties['Dependencies']) { $known += @($state.Dependencies | ForEach-Object { $_.Path }) }
    $task.Before = @($known | Sort-Object -Unique | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | ForEach-Object {
      @{ Path = $_; Hash = Get-MomBuildHash $_ }
    })
    # Remove receipts before launching: a killed compiler cannot reuse its output.
    Remove-Item -LiteralPath (@($task.State, $Object) + $depFiles) -Force -ErrorAction SilentlyContinue
  }
  return $task
}

function Complete-MomCompileTask($Task) {
  $deps = @($Task.DepFiles | ForEach-Object { Get-MomCompilerDependencies $_ } | Sort-Object { $_.Path } -Unique)
  if (@($deps | Where-Object { $_.Path -eq $Task.Source }).Count -eq 0) {
    throw "Compiler dependency file omitted source: $($Task.Source)"
  }
  foreach ($dependency in $deps) {
    $before = @($Task.Before | Where-Object { $_.Path -eq $dependency.Path })
    if (($before.Count -gt 0 -and $before[0].Hash -ne $dependency.Hash) -or
        ($before.Count -eq 0 -and (Get-Item -LiteralPath $dependency.Path).LastWriteTimeUtc.Ticks -gt $Task.Started)) {
      throw "Dependency changed during compilation: $($dependency.Path)"
    }
  }
  $script:MomBuildSnapshot = @($script:MomBuildSnapshot) + $deps
  Write-MomBuildState $Task.State @{ Key = $Task.Key; Dependencies = $deps; Output = Get-MomBuildHash $Task.Object }
}

function Invoke-MomIncrementalLink([string]$Exe, [string[]]$LinkArgs, [string[]]$Objects,
    [string]$Output, [string]$StatePath, [string]$Context) {
  Assert-MomBuildInputs
  $key = Get-MomBuildKey @{ Context = $Context; Exe = $Exe; Args = $LinkArgs;
    WorkDir = (Get-Location).Path; Objects = @($Objects | ForEach-Object { @{ Path = $_; Hash = Get-MomBuildHash $_ } }) }
  $state = Read-MomBuildState $StatePath
  try {
    if ($env:MOM_FORCE_REBUILD -ne '1' -and $state -and $state.Key -eq $key -and $state.Output -eq (Get-MomBuildHash $Output)) {
      Write-Host "  [cached link] $Output"; return
    }
  } catch { }
  Remove-Item -LiteralPath $StatePath, $Output -Force -ErrorAction SilentlyContinue
  $global:LASTEXITCODE = $null
  & $Exe @LinkArgs
  $linkExit = $global:LASTEXITCODE
  if ($null -eq $linkExit -or $linkExit -ne 0) { throw "Link failed ($linkExit): $Output" }
  Assert-MomBuildInputs
  Write-MomBuildState $StatePath @{ Key = $key; Output = Get-MomBuildHash $Output }
}
