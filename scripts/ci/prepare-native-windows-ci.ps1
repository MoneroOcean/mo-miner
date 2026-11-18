[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)]
  [ValidateSet('core', 'io', 'hermes')]
  [string]$Role
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
. (Join-Path $repoRoot 'scripts\native-windows-tools.ps1')

function Invoke-NativeStage([string]$Stage) {
  $node = Resolve-Application 'node.exe'
  & $node (Join-Path $repoRoot 'scripts\prepare-native-ci.js') stage $Stage
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

$tools = Get-MomNativeWindowsBuildTools -Role $Role
switch ($Role) {
  'core' {
    $env:MOM_NATIVE_WINDOWS_CORE_NODE_GYP = $tools.nodeGyp
    $env:MOM_NATIVE_WINDOWS_CORE_MSBUILD = $tools.msbuild
    $env:MOM_NATIVE_WINDOWS_CORE_DUMPBIN = $tools.dumpbin
    $env:MOM_NATIVE_WINDOWS_CORE_PYTHON = $tools.python
    $env:MOM_NATIVE_WINDOWS_CORE_COMPILER = $tools.compiler
    Invoke-NativeStage 'windows-native-core'
  }
  'io' {
    $env:MOM_NATIVE_WINDOWS_CONTROLLER_TOOLS = $tools | ConvertTo-Json -Compress -Depth 4
    Invoke-NativeStage 'windows-controller-io'
  }
  'hermes' {
    $env:MOM_NATIVE_WINDOWS_CONTROLLER_TOOLS = $tools | ConvertTo-Json -Compress -Depth 3
    Invoke-NativeStage 'windows-controller-hermes'
  }
}
