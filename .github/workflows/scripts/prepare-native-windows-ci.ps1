[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)]
  [ValidateSet('core', 'io', 'hermes')]
  [string]$Role
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..\..')).Path
. (Join-Path $repoRoot 'scripts\import-vcvars.ps1')
Import-MomVcVars64

function Resolve-RequiredFile([string]$Path, [string]$Name) {
  if ([string]::IsNullOrWhiteSpace($Path) -or $Path.Length -gt 4096 -or
      $Path -match '[\x00-\x1F\x7F]' -or -not [IO.Path]::IsPathRooted($Path) -or
      -not (Test-Path -LiteralPath $Path -PathType Leaf)) {
    throw "Required $Name file is missing: $Path"
  }
  $fullPath = [IO.Path]::GetFullPath($Path)
  $nodePath = 'C:\Program Files\nodejs\node.exe'
  if (-not (Test-Path -LiteralPath $nodePath -PathType Leaf)) {
    throw "Pinned Node.js is required to canonicalize $($Name): $Path"
  }
  $resolvedRows = @(& $nodePath -e 'process.stdout.write(require(process.argv[1]).realpathSync(process.argv[2]))' 'node:fs' $fullPath 2>$null)
  $nodeExit = $LASTEXITCODE
  if ($nodeExit -ne 0 -or $resolvedRows.Count -ne 1) {
    throw "Unable to canonicalize $($Name): $Path"
  }
  $canonicalPath = [string]$resolvedRows[0]
  if ([string]::IsNullOrWhiteSpace($canonicalPath) -or $canonicalPath.Length -gt 4096 -or
      $canonicalPath -match '[\x00-\x1F\x7F]' -or
      -not [IO.Path]::IsPathRooted($canonicalPath) -or
      -not (Test-Path -LiteralPath $canonicalPath -PathType Leaf)) {
    throw "Canonical $Name path is not one absolute regular file path: $Path"
  }
  return $canonicalPath
}

function Resolve-Application([string]$Name) {
  $application = Get-Command -Name $Name -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
  if (-not $application) { throw "Required $Name application is not on PATH" }
  return Resolve-RequiredFile $application.Source $Name
}

function Get-OneApiCompiler {
  if ([string]::IsNullOrWhiteSpace($env:ONEAPI_ROOT) -or -not (Test-Path -LiteralPath $env:ONEAPI_ROOT -PathType Container)) {
    throw 'The pinned oneAPI installation did not export a valid ONEAPI_ROOT'
  }
  $compiler = Resolve-RequiredFile (Join-Path $env:ONEAPI_ROOT 'compiler\latest\bin\icx-cl.exe') 'Intel icx-cl'
  $version = (Get-Item -LiteralPath $compiler).VersionInfo.ProductVersion
  if ($version -notmatch '^2026\.2(\.|\s|$)') { throw "Expected Intel compiler 2026.2, found '$version'" }
  $compilerRoot = Split-Path -Parent (Split-Path -Parent $compiler)
  # vcvars resets LIB to MSVC/Windows SDK paths. Restore only this selected Intel version's
  # static runtime for real compiler/link probes, without importing another oneAPI environment.
  $runtimeLibrary = Resolve-RequiredFile (Join-Path $compilerRoot 'lib\libircmt.lib') 'Intel libircmt.lib'
  $runtimeLibraryDir = [IO.Path]::GetDirectoryName($runtimeLibrary)
  if (-not [string]::Equals($runtimeLibraryDir, (Join-Path $compilerRoot 'lib'), [StringComparison]::OrdinalIgnoreCase)) {
    throw 'The selected Intel static runtime redirects outside its compiler version'
  }
  $runtimeLibraryKey = $runtimeLibraryDir.TrimEnd([char]'\')
  $libEntries = [System.Collections.Generic.List[string]]::new()
  foreach ($entry in ($env:LIB -split ';')) {
    if ([string]::IsNullOrWhiteSpace($entry)) { continue }
    $normalizedEntry = $null
    try {
      $normalizedEntry = [IO.Path]::GetFullPath($entry.Trim().Trim([char[]]@('"'))).TrimEnd([char]'\')
    } catch {
      # Preserve unusual existing LIB entries; only remove normalized matches for our runtime path.
    }
    if (-not $normalizedEntry -or
        -not [string]::Equals($normalizedEntry, $runtimeLibraryKey, [StringComparison]::OrdinalIgnoreCase)) {
      $libEntries.Add($entry)
    }
  }
  $env:LIB = (@($runtimeLibraryDir) + @($libEntries.ToArray())) -join ';'
  return $compiler
}

function Get-NativePerl {
  $root = if ([string]::IsNullOrWhiteSpace($env:MOM_NATIVE_PERL_ROOT)) { 'C:\Strawberry' } else { $env:MOM_NATIVE_PERL_ROOT }
  $perlBin = Join-Path $root 'perl\bin'
  $dllBin = Join-Path $root 'c\bin'
  if (-not (Test-Path -LiteralPath $dllBin -PathType Container)) { throw "Native Perl runtime DLL directory is missing: $dllBin" }
  $perl = Resolve-RequiredFile (Join-Path $perlBin 'perl.exe') 'native Perl'
  $oldPath = $env:Path
  try {
    $env:Path = "$dllBin;$perlBin;$oldPath"
    $reported = @(& $perl -e 'print $^O' 2>$null)
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0 -or ([string]::Join('', $reported)).Trim() -cne 'MSWin32') {
      throw 'Selected Perl is not a working native Windows (MSWin32) runtime'
    }
  } finally {
    $env:Path = $oldPath
  }
  # Keep the selected runtime's own DLLs visible to the public OpenSSL producer too.
  $env:Path = "$dllBin;$perlBin;$env:Path"
  return $perl
}

function Get-VcTool([string]$Name) {
  if ([string]::IsNullOrWhiteSpace($env:VCToolsInstallDir)) { throw 'vcvars64 did not expose VCToolsInstallDir' }
  return Resolve-RequiredFile (Join-Path $env:VCToolsInstallDir ("bin\Hostx64\x64\$Name")) $Name
}

function Get-WindowsSdkTool([string]$Name) {
  if ([string]::IsNullOrWhiteSpace($env:WindowsSdkDir) -or [string]::IsNullOrWhiteSpace($env:WindowsSDKVersion)) {
    throw 'vcvars64 did not expose the selected Windows SDK directory and version'
  }
  $version = $env:WindowsSDKVersion.Trim().TrimEnd([char]'\')
  return Resolve-RequiredFile (Join-Path $env:WindowsSdkDir ("bin\$version\x64\$Name")) $Name
}

function Get-VisualStudioRoot([switch]$RequireCoreComponents) {
  $programFilesX86 = [Environment]::GetFolderPath('ProgramFilesX86')
  $vswhere = Resolve-RequiredFile (Join-Path $programFilesX86 'Microsoft Visual Studio\Installer\vswhere.exe') 'vswhere.exe'
  $requirements = @('-requires', 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64')
  if ($RequireCoreComponents) {
    $requirements += @('-requires', 'Microsoft.VisualStudio.Component.Windows11SDK.26100',
      '-requires', 'Microsoft.VisualStudio.Component.VC.CMake.Project')
  }
  $rows = & $vswhere -latest -products '*' @requirements -property installationPath
  $exitCode = $LASTEXITCODE
  $root = ([string]($rows | Select-Object -First 1)).Trim()
  if ($exitCode -ne 0 -or [string]::IsNullOrWhiteSpace($root)) { throw 'Required Visual Studio Build Tools were not found' }

  # Keep the tools imported above and the separately queried MSBuild installation in one VS root.
  $vcRoot = [IO.Path]::GetFullPath($env:VCToolsInstallDir).TrimEnd([char]'\')
  $expected = [IO.Path]::GetFullPath((Join-Path $root 'VC\Tools\MSVC')).TrimEnd([char]'\') + '\'
  if (-not $vcRoot.StartsWith($expected, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'vswhere and vcvars64 selected different Visual Studio installations'
  }
  return [IO.Path]::GetFullPath($root)
}

function Invoke-NativeStage([string]$Stage) {
  $node = Resolve-Application 'node.exe'
  & $node (Join-Path $repoRoot 'scripts\prepare-native-ci.js') stage $Stage
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

switch ($Role) {
  'core' {
    $vsRoot = Get-VisualStudioRoot -RequireCoreComponents
    $npm = Resolve-Application 'npm.cmd'
    $prefixRows = & $npm prefix -g
    $npmExit = $LASTEXITCODE
    $npmPrefix = ([string]($prefixRows | Select-Object -Last 1)).Trim()
    if ($npmExit -ne 0 -or [string]::IsNullOrWhiteSpace($npmPrefix)) {
      throw 'Unable to resolve the pinned global node-gyp installation'
    }
    $env:MOM_NATIVE_WINDOWS_CORE_NODE_GYP = Resolve-RequiredFile (Join-Path $npmPrefix 'node_modules\node-gyp\bin\node-gyp.js') 'node-gyp.js'
    $env:MOM_NATIVE_WINDOWS_CORE_MSBUILD = Resolve-RequiredFile (Join-Path $vsRoot 'MSBuild\Current\Bin\MSBuild.exe') 'MSBuild.exe'
    $env:MOM_NATIVE_WINDOWS_CORE_DUMPBIN = Get-VcTool 'dumpbin.exe'
    $env:MOM_NATIVE_WINDOWS_CORE_PYTHON = Resolve-RequiredFile 'C:\Tools\mom\python-3.12.10\python.exe' 'pinned Python'
    $env:MOM_NATIVE_WINDOWS_CORE_COMPILER = Get-OneApiCompiler
    Invoke-NativeStage 'windows-native-core'
  }
  'io' {
    $openssl = [ordered]@{
      ar = Get-VcTool 'lib.exe'; cc = Get-OneApiCompiler; link = Get-VcTool 'link.exe'
      make = Get-VcTool 'nmake.exe'; perl = Get-NativePerl
      rc = Get-WindowsSdkTool 'rc.exe'; tar = Resolve-RequiredFile (Join-Path $env:SystemRoot 'System32\tar.exe') 'Windows tar.exe'
    }
    $env:MOM_NATIVE_WINDOWS_CONTROLLER_TOOLS = [ordered]@{
      cmake = Resolve-Application 'cmake.exe'; cc = Get-OneApiCompiler; ninja = Resolve-Application 'ninja.exe'; openssl = $openssl
    } | ConvertTo-Json -Compress -Depth 4
    Invoke-NativeStage 'windows-controller-io'
  }
  'hermes' {
    $compiler = Get-OneApiCompiler
    $env:MOM_NATIVE_WINDOWS_CONTROLLER_TOOLS = [ordered]@{
      git = Resolve-Application 'git.exe'; cmake = Resolve-Application 'cmake.exe'; cc = $compiler
      cxx = $compiler; ninja = Resolve-Application 'ninja.exe'
    } | ConvertTo-Json -Compress -Depth 3
    Invoke-NativeStage 'windows-controller-hermes'
  }
}
