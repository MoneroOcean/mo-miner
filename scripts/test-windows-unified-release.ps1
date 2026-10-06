param(
  [switch]$SkipBuild,
  [switch]$SkipCompilerGates,
  [switch]$SkipGpuGates,
  [string]$OutputDirectory = 'windows-unified-results',
  [int]$LifecycleTimeoutMs = 240000,
  [string]$DpcppDir = $(if ($env:MOM_DPCPP_DIR) { $env:MOM_DPCPP_DIR } else { 'C:\Tools\dpcpp' }),
  [string]$AcppCudaDir = 'C:\Tools\acpp-cuda',
  [string]$AcppHipDir = 'C:\Tools\acpp-amd',
  [string]$HipPath = 'C:\Program Files\AMD\ROCm\7.1',
  [string]$CudaPath = $env:CUDA_PATH
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
. (Join-Path $repo 'scripts\windows-install-helpers.ps1')
$repoCanonical = [IO.Path]::GetFullPath($repo).TrimEnd([char[]]@('\', '/'))
$outputInput = if ([IO.Path]::IsPathRooted($OutputDirectory)) { $OutputDirectory } else {
  Join-Path $repo $OutputDirectory
}
$output = [IO.Path]::GetFullPath($outputInput).TrimEnd([char[]]@('\', '/'))
if ($output.Equals($repoCanonical, [StringComparison]::OrdinalIgnoreCase) -or
    -not $output.StartsWith("$repoCanonical\", [StringComparison]::OrdinalIgnoreCase)) {
  throw "OutputDirectory must stay inside the repository: $OutputDirectory"
}
Assert-NoReparseAncestor $output 'OutputDirectory reparse-point ancestor'
$outputMarker = Join-Path $output '.mom-windows-unified-results'
$outputItem = Get-Item -LiteralPath $output -Force -ErrorAction SilentlyContinue
if ($null -ne $outputItem) {
  if (($outputItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
    throw "Refusing OutputDirectory reparse point: $OutputDirectory"
  }
  if (-not $outputItem.PSIsContainer) {
    throw "Refusing OutputDirectory that is not a directory: $OutputDirectory"
  }
  $outputChildren = @(Get-ChildItem -LiteralPath $output -Force -ErrorAction Stop)
  if ($outputChildren.Count -gt 0) {
    $markerItem = Get-Item -LiteralPath $outputMarker -Force -ErrorAction SilentlyContinue
    if ($null -eq $markerItem) {
      throw "Refusing to replace unowned OutputDirectory: $OutputDirectory"
    }
    if (($markerItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0 -or
        $markerItem.PSIsContainer) {
      throw "Refusing invalid OutputDirectory ownership marker: $outputMarker"
    }
    if (-not (Test-MomMarker $outputMarker 'mom-windows-unified-results')) {
      throw "Refusing OutputDirectory with an invalid ownership marker: $outputMarker"
    }
    Assert-NoReparseAncestor $output 'OutputDirectory reparse-point ancestor'
    $preRemoveItem = Get-Item -LiteralPath $output -Force -ErrorAction Stop
    if (-not $preRemoveItem.PSIsContainer -or
        ($preRemoveItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0 -or
        -not (Test-MomMarker $outputMarker 'mom-windows-unified-results')) {
      throw "OutputDirectory changed before recursive removal: $output"
    }
    Remove-Item -LiteralPath $output -Recurse -Force
  }
}
$ownershipValue = 'mom-windows-unified-results'
[IO.Directory]::CreateDirectory($output) | Out-Null
$ownershipValue | Set-Content -LiteralPath $outputMarker -Encoding ascii -NoNewline
$pausedUpdateServices = @()
Set-Location $repo
$version = (& node.exe -p "require('./package.json').version").Trim()
if ($LASTEXITCODE -ne 0 -or -not $version) { throw 'Unable to read package version.' }

function Clear-CompilerEnvironment {
  foreach ($name in @(
    'MOM_GPU_BACKEND', 'MOM_NATIVE_DIR', 'MOM_NATIVE_PATH', 'MOM_SYCL_COMPILER',
    'MOM_OPENCL_DEVICE_TYPE', 'ONEAPI_DEVICE_SELECTOR', 'ZE_AFFINITY_MASK',
    'ACPP_VISIBILITY_MASK'
  )) { Remove-Item "Env:$name" -ErrorAction SilentlyContinue }
  $env:Path = $script:basePath
}

function Set-CompilerEnvironment(
  [string]$Compiler,
  [string]$SelectorName,
  [string]$SelectorValue
) {
  Clear-CompilerEnvironment
  $runtime = Join-Path $repo "build\win\compilers\$Compiler"
  $env:MOM_NATIVE_PATH = Join-Path $runtime 'mom.node'
  $env:Path = "$runtime;$runtime\hipSYCL;$script:basePath"
  Set-Item "Env:$SelectorName" $SelectorValue
}

$basePath = $env:Path
$env:SYCL_CACHE_PERSISTENT = '1'

try {
  & "$repo\.github\workflows\scripts\test-powershell-syntax.ps1"

  & npm.cmd install --ignore-scripts --no-audit --no-fund
  if ($LASTEXITCODE -ne 0) { throw "JavaScript tooling install failed: $LASTEXITCODE" }

  # A servicing restart invalidates an hours-long multi-GPU gate and can strand its throwaway disk.
  # Cancel only a pending restart and resume any service that this script actually stopped.
  $abort = Start-Process shutdown.exe -ArgumentList '/a' -Wait -PassThru -WindowStyle Hidden
  if ($abort.ExitCode -notin @(0, 1116)) { throw "Unable to cancel shutdown: $($abort.ExitCode)" }
  foreach ($serviceName in @('UsoSvc', 'wuauserv')) {
    $service = Get-Service $serviceName -ErrorAction SilentlyContinue
    if ($service -and $service.Status -eq 'Running') {
      Stop-Service $serviceName -Force -ErrorAction SilentlyContinue
      if ((Get-Service $serviceName -ErrorAction SilentlyContinue).Status -eq 'Stopped') {
        $pausedUpdateServices += $serviceName
      }
    }
  }

  if (-not $SkipBuild) {
    & "$repo\.github\workflows\scripts\build-windows-multicompiler.ps1" `
      -Backend all -DpcppDir $DpcppDir -AcppCudaDir $AcppCudaDir -AcppHipDir $AcppHipDir `
      -HipPath $HipPath -CudaPath $CudaPath
  }

  if (-not $SkipCompilerGates) {
    foreach ($vendor in @('intel', 'nvidia', 'amd')) {
      Write-Host "=== Sequential $vendor compiler-vector gate ==="
      & "$repo\scripts\test-windows-current-multicompiler.ps1" -Backend $vendor -SkipBuild
    }

    # The vector suites leave all small-state algorithms resident in one process. These independent
    # processes additionally prove that normal benchmark close executes ordered N-API cleanup before
    # AdaptiveCpp/CUDA unloads cudart. benchmark-gpu-algos rejects CUDA:4 even if the runtime exits 0.
    Set-CompilerEnvironment 'acpp-cuda' 'ACPP_VISIBILITY_MASK' 'cuda'
    & node.exe scripts\benchmark-gpu-algos.js `
      --label windows-nvidia-acpp-lifecycle `
      --algos autolykos2,beamhash3,c29,cn/gpu,equihash192_7,zelhash,zhash,etchash,fishhash,karlsenhashv2,xelishashv3 `
      --samples 1 --warmup-samples 0 --timeout-ms $LifecycleTimeoutMs `
      --output (Join-Path $output 'windows-nvidia-acpp-lifecycle.json')
    if ($LASTEXITCODE -ne 0) { throw "AdaptiveCpp CUDA lifecycle gate failed: $LASTEXITCODE" }
  }

  $archiveName = "mom-v$version-win.zip"
  $archive = Join-Path $output $archiveName
  & "$repo\.github\workflows\scripts\package-windows.ps1" -Archive $archive

  Clear-CompilerEnvironment
  foreach ($suite in @('cpu', 'gpu-portable-cpu')) {
    Remove-Item Env:MOM_GPU_BACKEND -ErrorAction SilentlyContinue
    Remove-Item Env:MOM_OPENCL_DEVICE_TYPE -ErrorAction SilentlyContinue
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
      "$repo\.github\workflows\scripts\test-release-windows.ps1" -Archive $archive -Suite $suite
    if ($LASTEXITCODE -ne 0) { throw "Extracted Windows $suite gate failed: $LASTEXITCODE" }
  }
  Remove-Item Env:MOM_OPENCL_DEVICE_TYPE -ErrorAction SilentlyContinue
  if (-not $SkipGpuGates) {
    foreach ($vendor in @('intel', 'nvidia', 'amd')) {
      $env:MOM_GPU_BACKEND = $vendor
      & powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
        "$repo\.github\workflows\scripts\test-release-windows.ps1" -Archive $archive -Suite gpu
      if ($LASTEXITCODE -ne 0) { throw "Extracted Windows $vendor GPU gate failed: $LASTEXITCODE" }
    }
  }
  Remove-Item Env:MOM_GPU_BACKEND -ErrorAction SilentlyContinue
  $digest = (Get-FileHash -Algorithm SHA256 $archive).Hash.ToLowerInvariant()
  "$digest  $archiveName" | Set-Content -Encoding ascii `
    (Join-Path $output "$archiveName.sha256")
  Write-Host "Windows unified release gate passed: $digest"
} finally {
  Clear-CompilerEnvironment
  foreach ($serviceName in $pausedUpdateServices) {
    Start-Service $serviceName -ErrorAction SilentlyContinue
  }
}
