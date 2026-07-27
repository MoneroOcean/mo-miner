$ErrorActionPreference = 'Continue'

# Windows hosted runners carry SDK caches unrelated to mom. AdaptiveCpp/LLVM, oneAPI, CUDA, HIP,
# and their packaging closure need substantially more than the runner's initial free space.
$paths = @(
  $env:AGENT_TOOLSDIRECTORY,
  'C:\Android',
  'C:\ghcup',
  "$env:ProgramFiles\Android",
  "$env:ProgramFiles\dotnet",
  "${env:ProgramFiles(x86)}\Android",
  "${env:ProgramFiles(x86)}\Microsoft SDKs\Azure"
) | Where-Object { $_ -and (Test-Path $_) } | Select-Object -Unique

foreach ($path in $paths) {
  Write-Host "Removing unused hosted-runner payload: $path"
  Remove-Item $path -Recurse -Force -ErrorAction SilentlyContinue
}

Get-PSDrive -Name C | Format-Table Name, Used, Free -AutoSize
