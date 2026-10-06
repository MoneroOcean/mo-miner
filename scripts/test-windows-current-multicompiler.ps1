param(
  [Parameter(Mandatory = $true)]
  [ValidateSet('intel','nvidia','amd','opencl')]
  [string]$Backend,
  [ValidateSet('all','oneapi','dpcpp','portable','adaptivecpp')]
  [string]$Compiler = 'all',
  [switch]$SkipBuild
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $repo

if (-not $SkipBuild) {
  & "$repo\.github\workflows\scripts\build-windows-multicompiler.ps1" -Backend $Backend -Compiler $Compiler
}

$basePath = $env:Path
$nativeDir = Join-Path $repo 'build\win\compilers'
$env:MOM_NATIVE_DIR = $nativeDir
$env:MOM_REQUIRE_GPU_TESTS = '1'
$env:MOM_GPU_BACKEND = $Backend
$env:MOM_GPU_TEST_VENDORS = $Backend
$env:NODE_TEST_FLUSH_BUFFERED_OUTPUT = '1'
$env:SYCL_CACHE_PERSISTENT = '1'

function Clear-SelectorEnvironment {
  Remove-Item Env:ONEAPI_DEVICE_SELECTOR -ErrorAction SilentlyContinue
  Remove-Item Env:ZE_AFFINITY_MASK -ErrorAction SilentlyContinue
  Remove-Item Env:ACPP_VISIBILITY_MASK -ErrorAction SilentlyContinue
}

function Invoke-GpuSuite(
  [string]$Label,
  [string]$Compiler,
  [string]$SelectorName = '',
  [string]$SelectorValue = ''
) {
  Clear-SelectorEnvironment
  $runtime = Join-Path $nativeDir $Compiler
  $env:MOM_NATIVE_PATH = Join-Path $runtime 'mom.node'
  if (-not (Test-Path $env:MOM_NATIVE_PATH)) {
    throw "$Label worker is missing: $env:MOM_NATIVE_PATH"
  }
  if ($Compiler -like 'acpp-*') {
    $accelerator = if ($Compiler -eq 'acpp-cuda') { 'cuda' } else { 'hip' }
    foreach ($required in @('libomp.dll', 'hipSYCL\rt-backend-omp.dll',
                            "hipSYCL\rt-backend-$accelerator.dll")) {
      if (-not (Test-Path (Join-Path $runtime $required))) {
        throw "$Label worker is missing required AdaptiveCpp runtime: $required"
      }
    }
  }
  $env:Path = "$runtime;$runtime\hipSYCL;$basePath"
  if ($SelectorName) { Set-Item "Env:$SelectorName" $SelectorValue }

  $log = Join-Path $env:TEMP ("mom-gpu-{0}.log" -f ($Label -replace '[^A-Za-z0-9_.-]', '-'))
  Write-Host "=== $Label ==="
  & node.exe --require .\tests\common\test_output_buffer.js --test --test-concurrency=1 .\tests\discrete_gpu.js *> $log
  $status = $LASTEXITCODE
  if ($status -ne 0) {
    Get-Content $log
    throw "$Label failed: $status"
  }
  Get-Content $log | Select-Object -Last 12
  Start-Sleep -Seconds 3
}

function Invoke-PortableSuite(
  [string]$DisplayVendorName
) {
  Clear-SelectorEnvironment
  $portable = Join-Path $nativeDir 'dpcpp-opencl'
  $portableAddon = Join-Path $portable 'mom.node'
  # A targeted portable build stages its matching runtime locally, without a CUDA sibling.
  foreach ($required in @($portableAddon, (Join-Path $portable 'sycl.dll'),
                          (Join-Path $portable 'sycl9.dll'), (Join-Path $portable 'ur_adapter_opencl.dll'))) {
    if (-not (Test-Path $required)) { throw "Portable DPC++ worker dependency is missing: $required" }
  }
  $env:Path = "$portable;$basePath"
  # Let compiler-policy.js set MOM_NATIVE_PATH together with the per-subtest OpenCL CPU/GPU
  # selector. Pinning the addon here would intentionally disable policy environment selection and
  # let an Intel Level Zero device leak into what is meant to be the OpenCL compatibility gate.
  Remove-Item Env:MOM_NATIVE_PATH -ErrorAction SilentlyContinue
  $env:MOM_COMPILER_POLICY_STRICT = '1'
  $env:MOM_REQUIRE_PORTABLE_CPU_TESTS = '1'

  Write-Host "=== Portable OpenCL: CPU and $DisplayVendorName GPU ==="
  $env:MOM_GPU_BACKEND = 'opencl'
  Remove-Item Env:MOM_GPU_TEST_VENDORS -ErrorAction SilentlyContinue
  & node.exe .\tests\run_hash.js gpu
  if ($LASTEXITCODE -ne 0) { throw "Portable OpenCL compatibility suite failed: $LASTEXITCODE" }
}

$compilerLanes = 0
$portableLanes = 0

switch ($Backend) {
  'intel' {
    if ($Compiler -in @('all','oneapi')) {
      Invoke-GpuSuite 'Intel oneAPI' 'oneapi' 'ONEAPI_DEVICE_SELECTOR' 'level_zero:gpu'
      $compilerLanes++
    }
    if ($Compiler -in @('all','dpcpp')) {
      Invoke-GpuSuite 'Intel open DPC++' 'dpcpp' 'ONEAPI_DEVICE_SELECTOR' 'level_zero:gpu'
      $compilerLanes++
    }
    if ($Compiler -in @('all','portable')) {
      Invoke-PortableSuite 'Intel'
      $portableLanes = 1
    }
  }
  'nvidia' {
    if ($Compiler -in @('all','dpcpp')) {
      Invoke-GpuSuite 'NVIDIA open DPC++' 'dpcpp' 'ONEAPI_DEVICE_SELECTOR' 'cuda:gpu'
      $compilerLanes++
    }
    if ($Compiler -in @('all','adaptivecpp')) {
      Invoke-GpuSuite 'NVIDIA AdaptiveCpp' 'acpp-cuda' 'ACPP_VISIBILITY_MASK' 'cuda'
      $compilerLanes++
    }
    if ($Compiler -in @('all','portable')) {
      Invoke-PortableSuite 'NVIDIA'
      $portableLanes = 1
    }
  }
  'amd' {
    if ($Compiler -in @('all','adaptivecpp')) {
      Invoke-GpuSuite 'AMD AdaptiveCpp' 'acpp-hip' 'ACPP_VISIBILITY_MASK' 'hip'
      $compilerLanes++
    }
    if ($Compiler -in @('all','portable')) {
      Invoke-PortableSuite 'AMD'
      $portableLanes = 1
    }
  }
  'opencl' {
    if ($Compiler -in @('all','portable')) {
      Invoke-PortableSuite 'OpenCL'
      $portableLanes = 1
    }
  }
}

if ($compilerLanes -eq 0 -and -not $portableLanes) {
  throw "Compiler '$Compiler' is not available for backend '$Backend'."
}

Clear-SelectorEnvironment
$env:Path = $basePath
if ($portableLanes) {
  Write-Host "Windows $Backend multi-compiler gate passed: $compilerLanes native compiler lanes and $portableLanes portable compatibility lanes."
} else {
  Write-Host "Windows $Backend multi-compiler GPU gate passed: $compilerLanes compiler lanes."
}
