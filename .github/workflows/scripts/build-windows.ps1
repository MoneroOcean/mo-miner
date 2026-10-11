$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

trap {
  if ($env:GITHUB_ACTIONS -eq 'true') {
    # GitHub Actions workflow commands require %/CR/LF percent-encoded in the message.
    $message = $_.Exception.Message.Replace('%', '%25').Replace("`r", '%0D').Replace("`n", '%0A')
    Write-Host "::error title=Windows build failed::$message"
  }
  # A trap is not a catch: break rethrows the original exception instead of replacing it.
  break
}

function Invoke-MominerNative {
  param([scriptblock]$Command, [string]$Name)
  & $Command
  if ($LASTEXITCODE -ne 0) {
    throw "$Name failed with exit code $LASTEXITCODE."
  }
}

function Get-MominerBuildJobs {
  $jobs = 0
  if ($env:MOM_BUILD_JOBS) {
    [void][int]::TryParse($env:MOM_BUILD_JOBS, [ref]$jobs)
  }
  if ($jobs -lt 1) {
    $jobs = [Environment]::ProcessorCount
  }
  return [Math]::Max(1, $jobs)
}

function Get-MomNodeHeaderInputs([string]$Root) {
  $files = @(Get-ChildItem -LiteralPath $Root -File)
  # Downloaded headers use include/node + architecture import libraries; an
  # explicit Node source tree uses src and the named dependency include leaves.
  foreach ($relative in @('include','x64','ia32','arm64','src','deps\uv\include',
      'deps\v8\include','deps\openssl\config','deps\openssl\openssl\include','deps\zlib')) {
    $leaf = Join-Path $Root $relative
    if (Test-Path -LiteralPath $leaf -PathType Container) {
      $files += @(Get-ChildItem -LiteralPath $leaf -File -Recurse -Depth 4 | Where-Object {
        $_.Extension -in @('.h','.hpp','.inc','.inl','.gypi','.lib')
      })
    }
  }
  return @($files | Sort-Object FullName -Unique | ForEach-Object {
    @{ Path = $_.FullName; Hash = Get-MomBuildHash $_.FullName }
  })
}

function Find-MominerNodeGyp {
  $candidates = @()

  $nodeRoot = Split-Path (Get-Command node -ErrorAction Stop | Select-Object -First 1).Source -Parent
  $candidates += Join-Path $nodeRoot "node_modules\npm\node_modules\node-gyp\bin\node-gyp.js"

  $npmCommand = Get-Command npm -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($npmCommand) {
    $npmRoot = Split-Path $npmCommand.Source -Parent
    $candidates += Join-Path $npmRoot "node_modules\npm\node_modules\node-gyp\bin\node-gyp.js"
  }

  # Bundled node-gyp needs no optional global query, which can emit fatal stderr on PS5.1.
  foreach ($candidate in ($candidates | Select-Object -Unique)) {
    if (Test-Path -LiteralPath $candidate -PathType Leaf) {
      return (Resolve-Path -LiteralPath $candidate).Path
    }
  }
  $candidates = @()
  $globalRootOutput = & npm root -g 2>$null
  if ($LASTEXITCODE -eq 0 -and $globalRootOutput) {
    $globalRoot = ($globalRootOutput | Select-Object -First 1).Trim()
    $candidates += Join-Path $globalRoot "npm\node_modules\node-gyp\bin\node-gyp.js"
    $candidates += Join-Path $globalRoot "node-gyp\bin\node-gyp.js"
  }

  foreach ($candidate in ($candidates | Select-Object -Unique)) {
    if (Test-Path -LiteralPath $candidate -PathType Leaf) {
      return (Resolve-Path -LiteralPath $candidate).Path
    }
  }
  return $null
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "../../..")).Path
Set-Location $repoRoot
. (Join-Path $repoRoot 'scripts\windows-incremental-build.ps1')
Initialize-MomBuildEnvironment $repoRoot
. (Join-Path $repoRoot 'scripts\windows-msbuild-cache.ps1')
if (-not $env:ONEAPI_ROOT) {
  $env:ONEAPI_ROOT = "C:\Program Files (x86)\Intel\oneAPI"
}
if (-not $env:VS2022INSTALLDIR -and (Test-Path "C:\BuildTools")) {
  $env:VS2022INSTALLDIR = "C:\BuildTools"
}

Invoke-MominerNative { node -v } "node"
Invoke-MominerNative { npm -v } "npm"
Invoke-MominerNative { python --version } "python"

$nodeGyp = Find-MominerNodeGyp
if (-not $nodeGyp) {
  Invoke-MominerNative { npm install -g node-gyp@12.2.0 } "npm install node-gyp"
  $nodeGyp = Find-MominerNodeGyp
}
if (-not $nodeGyp) {
  throw "Unable to locate node-gyp."
}
Write-Host "Using node-gyp at $nodeGyp"
Invoke-MominerNative { node $nodeGyp --version } "node-gyp version"

$setvars = Join-Path $env:ONEAPI_ROOT "setvars.bat"
$envLines = & cmd.exe /d /s /c "call `"$setvars`" intel64 --force >nul && set"
if ($LASTEXITCODE -ne 0) {
  throw "oneAPI setvars failed with exit code $LASTEXITCODE."
}
foreach ($line in $envLines) {
  if ($line -match '^([^=]+)=(.*)$') {
    [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], "Process")
  }
}
# The Intel MSBuild toolsets read $(ICInstallDir)/$(IDPCInstallDir) to locate icx. Resolve the REAL
# compiler dir and pass it to MSBuild as a GLOBAL property below. Two CI-specific traps this avoids:
#   1. Prefer the VERSIONED dir (compiler\<ver>) over the `latest` junction -- actions/cache flattens the
#      junction, and the "Intel C++ Compiler" toolset validates the version embedded in the path.
#   2. The classic-icx toolset's own props otherwise set ICInstallDir from a stale integration-registry
#      value, overriding the env var ("Could not expand ICInstallDir"); only a /p: global property wins.
$compilerBase = Join-Path $env:ONEAPI_ROOT "compiler"
$compilerDir = Get-ChildItem $compilerBase -Directory -ErrorAction SilentlyContinue |
  Where-Object { $_.Name -match '^[0-9]+\.[0-9]' -and (Test-Path (Join-Path $_.FullName "bin")) } |
  Sort-Object Name -Descending | Select-Object -First 1 | ForEach-Object { $_.FullName }
if (-not $compilerDir) {
  $latest = Join-Path $compilerBase "latest"
  if (Test-Path (Join-Path $latest "bin")) { $compilerDir = $latest }
}
if (-not $compilerDir) { throw "Could not locate the oneAPI compiler directory under $env:ONEAPI_ROOT." }
# Forward slashes + trailing slash: dodges the MSBuild /p: trailing-backslash quoting trap, and the
# toolset expects a trailing separator (it builds $(ICInstallDir)bin\...).
$compilerDir = ($compilerDir -replace '\\', '/').TrimEnd('/') + "/"
$env:ICInstallDir = $compilerDir
$env:IDPCInstallDir = $compilerDir
Write-Host "ICInstallDir = $compilerDir"

Invoke-MominerNative { icx --version } "icx"

$buildJobs = Get-MominerBuildJobs
Write-Host "MOM_BUILD_JOBS = $buildJobs"

# MSBuild is on PATH inside a VS Developer/Native-Tools shell (and on CI runners), but not on a bare
# VS Build Tools box. Fall back to locating it via vswhere so a local Windows build works either way.
$msbuildCmd = Get-Command MSBuild.exe -ErrorAction SilentlyContinue | Select-Object -First 1
if ($msbuildCmd) {
  $msbuild = $msbuildCmd.Source
} else {
  $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
  if (Test-Path $vswhere) {
    # NB: do not add -requires Microsoft.Component.MSBuild -- VS Build Tools doesn't report that exact
    # component id, and it makes -find return nothing. -products * covers BuildTools + full VS.
    $msbuild = & $vswhere -latest -products * -find "MSBuild\**\Bin\MSBuild.exe" |
      Select-Object -First 1
  }
  if (-not $msbuild) {
    throw "MSBuild.exe not found on PATH or via vswhere; run from a VS Developer/Native Tools prompt."
  }
}
# Keep node-gyp's generated projects, objects and tracker logs separate from the
# other workers. Configure does not run on a normal source/header edit.
$oneapiCache = Join-Path $repoRoot 'build\.incremental\oneapi'
$configureState = Join-Path $oneapiCache 'configure.json'
$nodeIdentity = & node -p 'JSON.stringify({version:process.version,abi:process.versions.modules,arch:process.arch})'
if ($LASTEXITCODE -ne 0) { throw 'Unable to read Node build identity.' }
$nodeHeaders = $env:npm_config_nodedir
if ($nodeHeaders) {
  $nodeHeaders = Resolve-MomBuildPath $nodeHeaders
  if (-not (Test-Path -LiteralPath $nodeHeaders -PathType Container)) {
    throw "Explicit Node header directory is missing: $nodeHeaders"
  }
  $configureHeaderArgs = @("--nodedir=$nodeHeaders")
  if ($env:npm_config_target) { $configureHeaderArgs += "--target=$($env:npm_config_target)" }
} else {
  $nodeVersion = if ($env:npm_config_target) { $env:npm_config_target.TrimStart('v') }
    else { ($nodeIdentity | ConvertFrom-Json).version.TrimStart('v') }
  $nodeCacheRoot = if ($env:npm_config_devdir) { Resolve-MomBuildPath $env:npm_config_devdir }
    else { Join-Path $env:LOCALAPPDATA 'node-gyp\Cache' }
  $nodeHeaders = Join-Path $nodeCacheRoot $nodeVersion
  $configureHeaderArgs = @("--devdir=$nodeCacheRoot", "--target=$nodeVersion")
  # Configure installs headers on a fresh machine. Admit the same selected cache
  # first so the cold fingerprint includes exactly the headers the build uses.
  Invoke-MominerNative { node $nodeGyp install --ensure @configureHeaderArgs } 'node-gyp install headers'
}
$nodeHeaderInputs = @(Get-MomNodeHeaderInputs $nodeHeaders)
$nodeHeaderKey = Get-MomBuildKey $nodeHeaderInputs
# Fresh VM installs may give installVersion/import libraries new mtimes. Node
# SDK identity is content-only, not installation time, so round-trips stay warm.
$toolchainRoots = @($compilerDir, $env:VCToolsInstallDir, $env:WindowsSdkDir) |
  Where-Object { $_ -and (Test-Path -LiteralPath $_ -PathType Container) }
$context = Initialize-MomIncrementalBuild $oneapiCache @(
  $nodeGyp, (Get-Command node).Source, (Get-Command icx).Source, $msbuild,
  (Join-Path $repoRoot 'scripts\windows-incremental-build.ps1'),
  (Join-Path $repoRoot 'scripts\windows-msbuild-cache.ps1')
) $toolchainRoots
# Inventory changes can alter include resolution even when an existing header
# is unchanged (e.g. adding a higher-priority include or removing an .inc file).
$sourceInventory = @(@('native','sycl','xmrig') | ForEach-Object {
  $root = Join-Path $repoRoot $_
  if (Test-Path -LiteralPath $root) {
    Get-ChildItem -LiteralPath $root -File -Recurse | Where-Object {
      $_.Extension -in @('.c','.cpp','.cc','.h','.hpp','.inc','.asm')
    } | ForEach-Object { $_.FullName.Substring($repoRoot.Length) }
  }
} | Sort-Object)
$configureKey = Get-MomBuildKey @{
  Version = 1; Context = $context; Node = $nodeIdentity; Headers = $nodeHeaderKey; Inventory = $sourceInventory
  Binding = Get-MomBuildHash (Join-Path $repoRoot 'binding.gyp')
  Script = Get-MomBuildHash $PSCommandPath
  Options = @('msvs_version=2022','Release','x64','TrackFileAccess=true',
    'UseMultiToolTask=false','DPCPPLINKOptions=/clang:-fsycl-device-code-split=per_kernel')
  Env = @($env:npm_config_target, $env:npm_config_arch, $env:npm_config_disturl,
    $env:npm_config_nodedir, $env:npm_config_devdir, $env:GYP_DEFINES, $env:CL, $env:_CL_, $env:LINK)
}
$cachedConfigure = Read-MomBuildState $configureState
$generatedGraph = @((Get-ChildItem -LiteralPath (Join-Path $repoRoot 'build') -File -ErrorAction SilentlyContinue) |
  Where-Object { $_.Extension -in @('.vcxproj','.props','.targets','.gypi','.sln') } |
  Sort-Object FullName | ForEach-Object { @{ Path = $_.FullName; Hash = Get-MomBuildHash $_.FullName } })
$generatedGraphKey = Get-MomBuildKey $generatedGraph
$configureNeeded = $env:MOM_FORCE_REBUILD -eq '1' -or -not $cachedConfigure -or
  $cachedConfigure.Key -ne $configureKey -or $cachedConfigure.Graph -ne $generatedGraphKey
if (-not (Test-Path -LiteralPath 'build\mom.vcxproj')) { $configureNeeded = $true }
if ($configureNeeded) {
  Clear-MomMsbuildRelease (Join-Path $repoRoot 'build')
  Remove-Item -LiteralPath $configureState -Force -ErrorAction SilentlyContinue
  Invoke-MominerNative { node $nodeGyp configure --msvs_version=2022 @configureHeaderArgs } 'node-gyp configure'
  # Configure identity is independent of compilation success. Retain it on an
  # ordinary compile failure so retrying preserves the remaining object graph.
  $generatedGraph = @((Get-ChildItem -LiteralPath (Join-Path $repoRoot 'build') -File) |
    Where-Object { $_.Extension -in @('.vcxproj','.props','.targets','.gypi','.sln') } |
    Sort-Object FullName | ForEach-Object { @{ Path = $_.FullName; Hash = Get-MomBuildHash $_.FullName } })
  $generatedGraphKey = Get-MomBuildKey $generatedGraph
  Write-MomBuildState $configureState @{ Key = $configureKey; Graph = $generatedGraphKey }
}
$msbuildTarget = if ($configureNeeded) { '/t:Rebuild' } else { '/t:Build' }
$trackedStatePath = Join-Path $oneapiCache 'outputs.json'
$trackedState = Read-MomBuildState $trackedStatePath
if (-not $trackedState) { $msbuildTarget = '/t:Rebuild' }
if (-not $configureNeeded) { Repair-MomTrackedOutputs $trackedState (Join-Path $repoRoot 'build') }
$knownInputsBefore = @($trackedState.Outputs | ForEach-Object { $_.Inputs } |
  ForEach-Object { $_.Path } | Sort-Object -Unique | Where-Object {
    $_ -and -not $_.StartsWith((Join-Path $repoRoot 'build') + '\', [StringComparison]::OrdinalIgnoreCase) -and
    (Test-Path -LiteralPath $_ -PathType Leaf)
  } | ForEach-Object { @{ Path = $_; Hash = Get-MomBuildHash $_ } })
# Source content protects against restored/backdated timestamps. Untracked
# inputs require a conservative rebuild; tracked changes delete only matched
# objects above. Once native tracking covers an input, edits remain per-file.
$sourceHashes = @($sourceInventory | ForEach-Object {
  $path = $repoRoot + $_
  @{ Path = $path; Hash = Get-MomBuildHash $path }
})
if (-not $configureNeeded -and $trackedState) {
  $trackedInputs = @($trackedState.Outputs | Where-Object {
    [IO.Path]::GetExtension($_.Path) -in @('.obj','.pch','.res')
  } | ForEach-Object { $_.Inputs } | ForEach-Object { $_.Path })
  foreach ($inputRow in @($trackedState.Sources)) {
    if ($inputRow.Hash -ne (Get-MomBuildHash $inputRow.Path) -and $inputRow.Path -notin $trackedInputs) {
      $msbuildTarget = '/t:Rebuild'; break
    }
  }
}
# Capture output (rather than letting it stream) so the tail can be re-shown in the failure message.
# ICInstallDir/IDPCInstallDir passed as global /p: properties so the Intel toolset can't override them
# from a stale integration registry (see the compiler-dir resolution above).
$msbuildCapturePreference = $ErrorActionPreference
$global:LASTEXITCODE = $null
try {
  # PowerShell 5.1 turns redirected native stderr into error records, even on success.
  $ErrorActionPreference = 'Continue'
  # Intel MultiToolTask workers collided saving a shared write tlog with tracking enabled.
  # Standard CL batching retains native dependency tracking and project/compiler parallelism.
  $msbuildOutput = & $msbuild build\mom.vcxproj /clp:Verbosity=minimal /nologo /nodeReuse:false `
    $msbuildTarget `
    "/m:$buildJobs" /p:Configuration=Release /p:Platform=x64 `
    "/p:CL_MPCount=$buildJobs" /p:UseMultiToolTask=false /p:TrackFileAccess=true `
    "/p:DPCPPLINKOptions=/clang:-fsycl-device-code-split=per_kernel" `
    "/p:ICInstallDir=$($env:ICInstallDir)" "/p:IDPCInstallDir=$($env:IDPCInstallDir)" 2>&1
  $msbuildExitCode = $global:LASTEXITCODE
} finally {
  $ErrorActionPreference = $msbuildCapturePreference
}
$msbuildOutput | ForEach-Object { Write-Host $_ }
if ($null -eq $msbuildExitCode -or $msbuildExitCode -ne 0) {
  $tail = ($msbuildOutput | Select-Object -Last 80) -join "`n"
  throw "MSBuild failed with exit code $msbuildExitCode.`n$tail"
}
# An ignored linker option can silently drop a required SYCL device-link policy.
$ignoredOptions = $msbuildOutput | Where-Object { [string]$_ -match '(?i)\bLNK4044\b' }
if ($ignoredOptions) {
  throw ("MSBuild ignored a compiler/linker option.`n" + ($ignoredOptions -join "`n"))
}

New-Item -ItemType Directory -Force build\Release | Out-Null
$releasePath = (Resolve-Path build\Release).Path
$compilerCachePath = Join-Path $repoRoot 'build\compilers'
$incrementalCachePath = Join-Path $repoRoot 'build\.incremental'
Get-ChildItem build -Recurse -Include *.node,*.dll |
  Where-Object {
    $_.FullName -ne (Join-Path $releasePath $_.Name) -and
    -not $_.FullName.StartsWith($incrementalCachePath + '\', [StringComparison]::OrdinalIgnoreCase) -and
    -not $_.FullName.StartsWith($compilerCachePath + '\', [StringComparison]::OrdinalIgnoreCase)
  } |
  Copy-Item -Destination build\Release -Force

if (-not (Test-Path build\Release\mom.node)) {
  throw "build\Release\mom.node was not created."
}
try {
  Assert-MomBuildInputs
  if ((Get-MomBuildKey @(Get-MomNodeHeaderInputs $nodeHeaders)) -ne $nodeHeaderKey) {
    throw 'Node SDK changed during MSBuild.'
  }
  foreach ($inputRow in $knownInputsBefore) {
    if ($inputRow.Hash -ne (Get-MomBuildHash $inputRow.Path)) {
      throw 'Tracked dependency changed during MSBuild.'
    }
  }
} catch {
  Remove-Item -LiteralPath $trackedStatePath -Force -ErrorAction SilentlyContinue
  throw
}
$currentSources = @($sourceInventory | ForEach-Object {
  $path = $repoRoot + $_
  @{ Path = $path; Hash = Get-MomBuildHash $path }
})
if ((Get-MomBuildKey $currentSources) -ne (Get-MomBuildKey $sourceHashes)) {
  Remove-Item -LiteralPath $trackedStatePath -Force -ErrorAction SilentlyContinue
  throw 'Build inputs changed during MSBuild; retry from a stable source tree.'
}
try {
  $trackedOutputs = @(Get-MomTrackedOutputs (Join-Path $repoRoot 'build'))
  Assert-MomTrackedOutputCoverage $trackedOutputs (Join-Path $repoRoot 'build')
} catch {
  Remove-Item -LiteralPath $trackedStatePath -Force -ErrorAction SilentlyContinue
  throw
}
$trackedOutputs += @('mom.node','sycl.dll') | ForEach-Object {
  $path = Join-Path $repoRoot "build\Release\$_"
  @{ Path = $path; Hash = Get-MomBuildHash $path; Inputs = $currentSources }
}
Write-MomBuildState $trackedStatePath @{ Sources = $currentSources; Outputs = $trackedOutputs }
