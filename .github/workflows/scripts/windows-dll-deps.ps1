$ErrorActionPreference = "Stop"

function Get-MominerDumpBin {
  $command = Get-Command dumpbin.exe -ErrorAction SilentlyContinue
  if ($command) {
    return $command.Source
  }

  $candidates = Get-ChildItem `
    -Path @(
      "C:/Program Files/Microsoft Visual Studio/2022/*/VC/Tools/MSVC/*/bin/Hostx64/x64/dumpbin.exe",
      "C:/BuildTools/VC/Tools/MSVC/*/bin/Hostx64/x64/dumpbin.exe"
    ) `
    -File `
    -ErrorAction SilentlyContinue |
    Sort-Object FullName -Descending

  if (-not $candidates) {
    throw "dumpbin.exe was not found. Install Visual Studio Build Tools or run from a VS developer shell."
  }

  return $candidates[0].FullName
}

function Get-MominerDllDependencies {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Path,

    [Parameter(Mandatory = $true)]
    [string]$DumpBin
  )

  $output = & $DumpBin /nologo /dependents $Path 2>&1
  if ($LASTEXITCODE -ne 0) {
    throw "dumpbin failed for ${Path}: $output"
  }

  $deps = New-Object 'System.Collections.Generic.List[string]'
  foreach ($line in $output) {
    if ($line -match '^\s*([A-Za-z0-9_.+-]+\.dll)\s*$') {
      $deps.Add($Matches[1].ToLowerInvariant())
    }
  }
  return $deps
}

function Test-MominerRedistributableDllName {
  param([Parameter(Mandatory = $true)][string]$Name)

  return $Name -match '^(msvcp|vcruntime|concrt|vcamp|vcomp|vccorlib)[0-9].*\.dll$'
}

function Test-MominerSystemDllName {
  param([Parameter(Mandatory = $true)][string]$Name)

  if ($Name -like 'api-ms-win-*.dll' -or $Name -like 'ext-ms-win-*.dll') {
    return $true
  }

  $systemNames = @(
    'advapi32.dll',
    'bcrypt.dll',
    'cfgmgr32.dll',
    'combase.dll',
    'crypt32.dll',
    'dbghelp.dll',
    'dnsapi.dll',
    'dwmapi.dll',
    'dxcore.dll',
    'gdi32.dll',
    'gdi32full.dll',
    'imm32.dll',
    'iphlpapi.dll',
    'kernel32.dll',
    'msvcrt.dll',
    'ncrypt.dll',
    'ntdll.dll',
    'ole32.dll',
    'oleaut32.dll',
    'powrprof.dll',
    'psapi.dll',
    'rpcrt4.dll',
    'sechost.dll',
    'setupapi.dll',
    'shell32.dll',
    'shlwapi.dll',
    'user32.dll',
    'userenv.dll',
    'ucrtbase.dll',
    'version.dll',
    'win32u.dll',
    'winhttp.dll',
    'winmm.dll',
    'ws2_32.dll'
  )
  return $systemNames -contains $Name
}

function Test-MominerWindowsPath {
  param([Parameter(Mandatory = $true)][string]$Path)

  $windows = [System.IO.Path]::GetFullPath($env:WINDIR).TrimEnd('\')
  $full = [System.IO.Path]::GetFullPath($Path)
  return $full.StartsWith($windows, [System.StringComparison]::OrdinalIgnoreCase)
}

function Get-MominerDllSearchRoots {
  param([Parameter(Mandatory = $true)][string]$PackageDir)

  $roots = New-Object 'System.Collections.Generic.List[string]'
  $addRoot = {
    param([string]$Path)
    if ($Path -and (Test-Path $Path)) {
      $resolved = (Resolve-Path $Path).Path
      if (-not $roots.Contains($resolved)) {
        $roots.Add($resolved)
      }
    }
  }

  & $addRoot $PackageDir
  & $addRoot "build/Release"

  $node = Get-Command node.exe -ErrorAction SilentlyContinue
  if ($node) {
    & $addRoot (Split-Path -Parent $node.Source)
  }

  if ($env:ONEAPI_ROOT) {
    & $addRoot (Join-Path $env:ONEAPI_ROOT "compiler/latest/bin")
    & $addRoot (Join-Path $env:ONEAPI_ROOT "compiler/latest/bin/compiler")
  }

  Get-ChildItem `
    -Path @(
      "C:/Program Files/Microsoft Visual Studio/2022/*/VC/Redist/MSVC/*/x64/Microsoft.VC*.CRT",
      "C:/BuildTools/VC/Redist/MSVC/*/x64/Microsoft.VC*.CRT"
    ) `
    -Directory `
    -ErrorAction SilentlyContinue |
    ForEach-Object { & $addRoot $_.FullName }

  & $addRoot (Join-Path $env:WINDIR "System32")

  return $roots
}

function New-MominerDllIndex {
  param([Parameter(Mandatory = $true)][string[]]$Roots)

  $index = @{}
  foreach ($root in $Roots) {
    $isWindowsRoot = Test-MominerWindowsPath $root
    $files = Get-ChildItem `
      -Path $root `
      -Filter '*.dll' `
      -File `
      -Recurse:(!$isWindowsRoot) `
      -ErrorAction SilentlyContinue

    foreach ($file in $files) {
      $key = $file.Name.ToLowerInvariant()
      if (-not $index.ContainsKey($key)) {
        $index[$key] = $file.FullName
      }
    }
  }
  return $index
}

function Resolve-MominerDependency {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Name,

    [Parameter(Mandatory = $true)]
    [string]$PackageDir,

    [Parameter(Mandatory = $true)]
    [hashtable]$Index
  )

  $packagePath = Join-Path $PackageDir $Name
  if (Test-Path $packagePath) {
    return @{ Kind = 'package'; Path = (Resolve-Path $packagePath).Path }
  }

  if (Test-MominerSystemDllName $Name) {
    return @{ Kind = 'system'; Path = $null }
  }

  if ($Index.ContainsKey($Name)) {
    $source = $Index[$Name]
    if ((Test-MominerWindowsPath $source) -and -not (Test-MominerRedistributableDllName $Name)) {
      return @{ Kind = 'system'; Path = $source }
    }
    return @{ Kind = 'copy'; Path = $source }
  }

  return @{ Kind = 'missing'; Path = $null }
}

function Copy-MominerOptionalRuntimeFiles {
  param(
    [Parameter(Mandatory = $true)]
    [string]$PackageDir
  )

  if (-not $env:ONEAPI_ROOT) {
    return
  }

  $roots = @(
    (Join-Path $env:ONEAPI_ROOT "compiler/latest/bin"),
    (Join-Path $env:ONEAPI_ROOT "compiler/latest/bin/compiler")
  )
  $patterns = @(
    'sycl*.dll',
    'ur_*.dll',
    'ur_adapter*.dll',
    'pi_*.dll',
    'OpenCL.dll',
    'intelocl*.dll',
    'ocl*.dll',
    'libocl_svml_*.dll',
    'libiomp5*.dll',
    'libmmd*.dll',
    'libirc*.dll',
    'tbb*.dll',
    'umf*.dll',
    'ze_loader*.dll',
    'clbltfn*.rtl',
    'cllibrary*.rtl',
    'cllibrary*.o'
  )

  foreach ($root in $roots) {
    if (-not (Test-Path $root)) {
      continue
    }

    foreach ($pattern in $patterns) {
      Get-ChildItem -Path $root -Filter $pattern -File -ErrorAction SilentlyContinue |
        ForEach-Object {
          $dest = Join-Path $PackageDir $_.Name
          if (-not (Test-Path $dest)) {
            Copy-Item $_.FullName $dest
          }
        }
    }
  }
}

function Invoke-MominerDllClosure {
  param(
    [Parameter(Mandatory = $true)]
    [string]$PackageDir,

    [Parameter(Mandatory = $true)]
    [string[]]$EntryPaths,

    [switch]$CopyMissing
  )

  $packageFull = (Resolve-Path $PackageDir).Path
  $dumpBin = Get-MominerDumpBin
  $roots = Get-MominerDllSearchRoots $packageFull
  $index = New-MominerDllIndex $roots
  $queue = New-Object 'System.Collections.Generic.Queue[string]'
  $seen = New-Object 'System.Collections.Generic.HashSet[string]'
  $missing = New-Object 'System.Collections.Generic.List[string]'

  foreach ($entry in $EntryPaths) {
    if (Test-Path $entry) {
      $queue.Enqueue((Resolve-Path $entry).Path)
    }
  }

  while ($queue.Count -gt 0) {
    $current = $queue.Dequeue()
    if (-not $seen.Add($current.ToLowerInvariant())) {
      continue
    }

    foreach ($dep in (Get-MominerDllDependencies -Path $current -DumpBin $dumpBin)) {
      $resolution = Resolve-MominerDependency -Name $dep -PackageDir $packageFull -Index $index
      switch ($resolution.Kind) {
        'package' {
          $queue.Enqueue($resolution.Path)
        }
        'copy' {
          if ($CopyMissing) {
            $dest = Join-Path $packageFull $dep
            Copy-Item $resolution.Path $dest -Force
            $resolvedDest = (Resolve-Path $dest).Path
            $index[$dep] = $resolvedDest
            $queue.Enqueue($resolvedDest)
          } else {
            $missing.Add("$dep required by $current")
          }
        }
        'missing' {
          $missing.Add("$dep required by $current")
        }
      }
    }
  }

  if ($missing.Count -gt 0) {
    $prefix = if ($CopyMissing) {
      "Unresolved non-system DLL dependencies:"
    } else {
      "Release package is missing non-system DLL dependencies:"
    }
    throw "$prefix`n$($missing -join "`n")"
  }
}

function Copy-MominerDllClosure {
  param(
    [Parameter(Mandatory = $true)]
    [string]$PackageDir,

    [Parameter(Mandatory = $true)]
    [string[]]$EntryPaths
  )

  Invoke-MominerDllClosure -PackageDir $PackageDir -EntryPaths $EntryPaths -CopyMissing
}

function Test-MominerDllClosure {
  param(
    [Parameter(Mandatory = $true)]
    [string]$PackageDir,

    [Parameter(Mandatory = $true)]
    [string[]]$EntryPaths
  )

  Invoke-MominerDllClosure -PackageDir $PackageDir -EntryPaths $EntryPaths
}
