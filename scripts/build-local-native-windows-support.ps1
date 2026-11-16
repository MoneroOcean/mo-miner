[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)]
  [ValidateNotNullOrEmpty()]
  [string]$InputFile,

  [Parameter(Mandatory = $true)]
  [ValidateNotNullOrEmpty()]
  [string]$OutputFile
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot 'native-windows-tools.ps1')

$nodePath = Get-MomNativeBuildNode

# Each selection imports vcvars64 and the pinned Intel environment into this PowerShell
# process; the Node child below inherits the selected LIB and PATH without a shell relay.
$coreTools = Get-MomNativeWindowsBuildTools -Role core
$ioTools = Get-MomNativeWindowsBuildTools -Role io
$hermesTools = Get-MomNativeWindowsBuildTools -Role hermes
$tools = [ordered]@{
  core = $coreTools
  io = [ordered]@{ cmake = $ioTools.cmake; cc = $ioTools.cc; ninja = $ioTools.ninja }
  openssl = $ioTools.openssl
  hermes = $hermesTools
}

$utf8 = [Text.UTF8Encoding]::new($false, $true)
$oldOutputEncoding = $OutputEncoding
$oldConsoleOutputEncoding = [Console]::OutputEncoding
try {
  $OutputEncoding = $utf8
  [Console]::OutputEncoding = $utf8
  $toolsJson = ConvertTo-Json -InputObject $tools -Depth 8 -Compress
  $commandScript = Join-Path $PSScriptRoot 'build-local-native-windows-support-command.js'
  $toolsJson | & $nodePath $commandScript --run $InputFile $OutputFile
  $commandExitCode = $LASTEXITCODE
  if ($commandExitCode -ne 0) {
    throw "Local Windows support command failed with exit code $commandExitCode"
  }
} finally {
  $OutputEncoding = $oldOutputEncoding
  [Console]::OutputEncoding = $oldConsoleOutputEncoding
}
