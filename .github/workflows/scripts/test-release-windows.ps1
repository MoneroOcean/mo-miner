param(
  [Parameter(Mandatory = $true)]
  [string]$Archive,

  [string]$Suite = "all"
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
if ($PSVersionTable.PSVersion.Major -ge 7) {
  $PSNativeCommandUseErrorActionPreference = $true
}

trap {
  if ($env:GITHUB_ACTIONS) {
    $message = $_.Exception.Message.Replace('%', '%25').Replace("`r", '%0D').Replace("`n", '%0A')
    Write-Host "::error title=Windows release test failed::$message"
  }
  break
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "../../..")).Path
Set-Location $repoRoot
if ($Suite -eq 'opencl') { $env:MOM_GPU_BACKEND = 'opencl' }

$workDir = if ($env:MOM_RELEASE_TEST_DIR) { $env:MOM_RELEASE_TEST_DIR } else { "release-test" }

Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip = [System.IO.Compression.ZipFile]::OpenRead((Resolve-Path $Archive).Path)
try {
  $rootEntry = $zip.Entries | Where-Object { $_.FullName -match '^[^/\\]+[/\\]$' } | Select-Object -First 1
  $root = if ($rootEntry) { $rootEntry.FullName.TrimEnd('/', '\') } else { "" }
  if (-not $root) {
    $root = (($zip.Entries | Select-Object -First 1).FullName -split '[/\\]')[0]
  }
  if ($zip.Entries | Where-Object { $_.FullName -match '(^|[/\\])tests([/\\]|$)' }) {
    throw "Release archive must not contain tests/."
  }
} finally {
  $zip.Dispose()
}

Remove-Item -Recurse -Force $workDir -ErrorAction SilentlyContinue
Expand-Archive $Archive $workDir
$packageDir = (Resolve-Path (Join-Path $workDir $root)).Path
$libsDir = Join-Path $packageDir "libs"
$node = (Resolve-Path (Join-Path $packageDir 'mom-node.exe')).Path
if (Test-Path (Join-Path $packageDir "tests")) {
  throw "Extracted release package unexpectedly contains tests/."
}

foreach ($compiler in @('oneapi','dpcpp','dpcpp-opencl','acpp-cuda','acpp-hip')) {
  if (-not (Test-Path (Join-Path $libsDir "$compiler\mom.node"))) {
    throw "Windows release package is missing libs/$compiler/mom.node."
  }
}
if (-not (Test-Path (Join-Path $libsDir 'dpcpp\ur_adapter_opencl.dll'))) {
  throw 'Windows release package is missing the generic DPC++ Unified Runtime OpenCL adapter.'
}
foreach ($runtime in @(
  'dpcpp\ur_loader.dll',
  'dpcpp-opencl\ur_loader.dll',
  'dpcpp-opencl\ur_adapter_opencl.dll',
  'dpcpp-opencl\ur_adapter_level_zero_v2.dll'
)) {
  if (-not (Test-Path (Join-Path $libsDir $runtime))) {
    throw "Windows release package is missing required Unified Runtime file libs/$runtime."
  }
}
foreach ($runtime in @(
  'acpp-cuda\libomp.dll',
  'acpp-cuda\hipSYCL\rt-backend-cuda.dll',
  'acpp-cuda\hipSYCL\rt-backend-omp.dll',
  'acpp-cuda\hipSYCL\bitcode\libkernel-sscp-host-full.bc',
  'acpp-cuda\hipSYCL\ext\bitcode\ptx\libdevice.10.bc',
  'acpp-hip\libomp.dll',
  'acpp-hip\hipSYCL\rt-backend-hip.dll',
  'acpp-hip\hipSYCL\rt-backend-omp.dll',
  'acpp-hip\hipSYCL\bitcode\libkernel-sscp-host-full.bc'
)) {
  if (-not (Test-Path (Join-Path $libsDir $runtime))) {
    throw "Windows release package is missing required AdaptiveCpp runtime libs/$runtime."
  }
}
$hiprtcCompiler = Get-ChildItem (Join-Path $libsDir 'acpp-hip') -Filter 'hiprtc*.dll' -File `
  -ErrorAction SilentlyContinue | Where-Object { $_.Name -notlike 'hiprtc-builtins*' } |
  Select-Object -First 1
if (-not $hiprtcCompiler) {
  throw 'Windows release package is missing the AMD source-JIT HIPRTC compiler DLL.'
}
foreach ($pattern in @('hiprtc-builtins*.dll', 'amd_comgr0*.dll')) {
  if (-not (Get-ChildItem (Join-Path $libsDir 'acpp-hip') -Filter $pattern -File `
      -ErrorAction SilentlyContinue | Select-Object -First 1)) {
    throw "Windows release package is missing the AMD source-JIT runtime $pattern."
  }
}

# Exercise the installer from the extracted artifact, not the repository copy. DryRun validates the
# batch wrapper, argument forwarding, PowerShell syntax, and packaged-runtime discovery without
# mutating a GPU-less hosted runner.
$installer = Join-Path $packageDir 'install.bat'
if (-not (Test-Path $installer)) {
  throw 'Windows release package is missing install.bat.'
}
if ($Suite -in @('all', 'cpu')) {
  & $installer -DryRun
  if ($LASTEXITCODE -ne 0) {
    throw "Packaged install.bat dry run failed with exit code $LASTEXITCODE."
  }
}

# package-windows.ps1 performs the mandatory dumpbin dependency-closure audit while constructing
# each isolated runtime directory. This extracted-archive gate validates that closure dynamically;
# repeating the flat audit from libs/ would confuse same-named DLLs in the isolated workers and
# would also require Visual Studio tools on an otherwise clean deployment machine.

Copy-Item tests (Join-Path $packageDir "tests") -Recurse
New-Item -ItemType Directory -Force (Join-Path $packageDir 'scripts') | Out-Null
Copy-Item scripts\validate-portable-opencl.js (Join-Path $packageDir 'scripts\validate-portable-opencl.js')

# A developer image may already point OCL_ICD_FILENAMES at oneAPI's GPU ICD. CPU deployment gates
# deliberately select Intel's separately installed CPU ICD so a headless runner cannot silently
# enumerate zero devices while ignoring the system-wide Khronos registry entry.
if ($Suite -eq 'sycl-cpu' -or ($Suite -eq 'opencl' -and $env:MOM_OPENCL_DEVICE_TYPE -eq 'cpu')) {
  $cpuIcdPaths = New-Object 'System.Collections.Generic.List[string]'
  $vendorKey = Get-Item 'HKLM:\SOFTWARE\Khronos\OpenCL\Vendors' -ErrorAction SilentlyContinue
  if ($vendorKey) {
    foreach ($name in $vendorKey.GetValueNames()) {
      if ([IO.Path]::GetFileName($name) -ieq 'intelocl64.dll' -and (Test-Path $name)) {
        $cpuIcdPaths.Add((Resolve-Path $name).Path)
      }
    }
  }
  foreach ($root in @(
    'C:\Program Files (x86)\Common Files\Intel\Shared Libraries',
    'C:\Program Files (x86)\Intel\oneAPI',
    "$env:WINDIR\System32\DriverStore\FileRepository"
  )) {
    if (-not (Test-Path $root)) { continue }
    $candidate = Get-ChildItem $root -Filter 'intelocl64.dll' -File -Recurse `
      -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($candidate) { $cpuIcdPaths.Add($candidate.FullName) }
  }
  $cpuIcd = $cpuIcdPaths | Select-Object -Unique -First 1
  if (-not $cpuIcd) {
    throw 'Intel CPU OpenCL ICD is missing; run scripts\install-dev.bat -Component opencl-cpu.'
  }
  $env:OCL_ICD_FILENAMES = $cpuIcd
}

# Minimal PATH: package/libs first, then the Windows system dirs the EXE needs.
$externalOpenClDir = if ($env:OCL_ICD_FILENAMES) {
  Split-Path -Parent (($env:OCL_ICD_FILENAMES -split ';')[0])
} else { $null }
$env:Path = @(
  $libsDir,
  $packageDir,
  $externalOpenClDir,
  "$env:WINDIR\System32",
  $env:WINDIR,
  "$env:WINDIR\System32\Wbem",
  "$env:WINDIR\System32\WindowsPowerShell\v1.0"
) -join ';'

# The test runner invokes the packaged Node executable directly so it can report each vector as an
# individual test.  Mirror the small part of mom.cmd's launcher environment that selects the initial
# worker; compiler-policy.js still performs any per-algorithm worker override in child processes.
$env:MOM_NATIVE_DIR = $libsDir
if ($Suite -eq 'sycl-cpu') {
  # Development/release jobs install Intel's redistributable CPU OpenCL implementation. Use the
  # portable SPIR-V worker so this archive gate needs no GPU and exercises the generic fallback ABI.
  $env:MOM_GPU_BACKEND = 'opencl'
  $env:MOM_OPENCL_DEVICE_TYPE = 'cpu'
  $env:ONEAPI_DEVICE_SELECTOR = 'opencl:cpu'
}
$defaultWorker = if ($Suite -eq 'sycl-cpu') { 'dpcpp-opencl' } else { switch ($env:MOM_GPU_BACKEND) {
  'nvidia' { 'dpcpp' }
  'amd' { 'acpp-hip' }
  'opencl' { 'dpcpp-opencl' }
  default { 'oneapi' }
} }
$env:MOM_NATIVE_PATH = Join-Path $libsDir "$defaultWorker\mom.node"
$env:MOM_NATIVE_PATH_LAUNCHER_DEFAULT = $env:MOM_NATIVE_PATH
$workerDir = Join-Path $libsDir $defaultWorker
$env:MOM_RUNTIME_DIR = $workerDir
$sharedDpcppDir = if ($defaultWorker -eq 'dpcpp-opencl') { Join-Path $libsDir 'dpcpp' } else { $null }
# For the CPU suite, an explicitly supplied CPU OpenCL runtime must provide OpenCL.dll ahead of the
# bundled generic loader. The shared DPC++ directory still follows it for sycl9/UR dependencies.
$env:Path = @($workerDir, (Join-Path $workerDir 'hipSYCL'), $externalOpenClDir,
  $sharedDpcppDir, $env:Path) -join ';'

function Enable-IntelOpenCL {
  if ($Suite -eq 'sycl-cpu' -or $env:MOM_OPENCL_DEVICE_TYPE -eq 'cpu') {
    return
  }
  if ($env:OCL_ICD_FILENAMES) {
    return
  }

  $intelOcl = Get-ChildItem -Path (Join-Path $libsDir 'oneapi') -Filter "intelocl*.dll" `
    -File -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($intelOcl) {
    $env:OCL_ICD_FILENAMES = $intelOcl.FullName
  }
}

function Get-SyclCpuDevicesFromOutput {
  param([Parameter(Mandatory = $true)][string[]]$Output)

  $devices = New-Object 'System.Collections.Generic.List[string]'
  foreach ($line in $Output) {
    # "cpuN: <description>" lines name an available CPU SYCL device.
    if ($line -match '^(cpu\d+):\s+.+$') {
      $devices.Add($Matches[1])
    }
  }
  return $devices
}

Remove-Item Env:MOM_ASSUME_SYCL_CPU -ErrorAction SilentlyContinue
Enable-IntelOpenCL
Push-Location $packageDir
try {
  # Run mom.cmd without aborting on a non-zero exit so we can inspect output/code.
  function Invoke-AlgoParams {
    $previous = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
      $output = & .\mom.cmd algo_params 2>&1
      return [pscustomobject]@{ Output = $output; Exit = $LASTEXITCODE }
    } finally {
      $ErrorActionPreference = $previous
    }
  }

  $smoke = Invoke-AlgoParams
  $smokeOutput = $smoke.Output
  if ($smoke.Exit -ne 0) {
    $env:MOM_DEBUG_STARTUP = "1"
    $debug = Invoke-AlgoParams
    Remove-Item Env:MOM_DEBUG_STARTUP -ErrorAction SilentlyContinue
    throw "Direct executable smoke test failed with exit code $($smoke.Exit). Output: $($smokeOutput -join ' | '). Debug exit code: $($debug.Exit). Debug output: $($debug.Output -join ' | ')"
  }
  $marker = $smokeOutput | Where-Object { $_ -match '^MOM_ALGO_PARAMS ' } | Select-Object -First 1
  if (-not $marker) {
    throw "Direct executable smoke test did not print algo params marker.`n$($smokeOutput -join "`n")"
  }
  $params = ($marker -replace '^MOM_ALGO_PARAMS ', '') | ConvertFrom-Json
  foreach ($prop in $params.PSObject.Properties) {
    $dev = [string]$prop.Value
    if (-not $dev -or $dev -match '(^|,)[^,]*(\*0|\^0)(,|$)') {
      throw "Invalid algo params for $($prop.Name): $dev"
    }
  }
  $openclGpuSuite = $Suite -eq 'opencl' -and $env:MOM_OPENCL_DEVICE_TYPE -ne 'cpu'
  if ($Suite -eq 'gpu' -or $openclGpuSuite) {
    $gpuParam = $params.PSObject.Properties | Where-Object { [string]$_.Value -match '(^|,)gpu\d+' } |
      Select-Object -First 1
    if (-not $gpuParam) {
      throw "Windows $Suite release test requires launcher-time GPU discovery, but algo_params returned no GPU job."
    }
  }
  $syclCpuDevices = Get-SyclCpuDevicesFromOutput $smokeOutput
  if (($Suite -eq "all" -or $Suite -eq "sycl-cpu") -and $syclCpuDevices.Count -eq 0) {
    throw "Windows $Suite release test requires a CPU SYCL device, but algo_params did not report one.`n$($smokeOutput -join "`n")"
  }

  if ($Suite -notin @("all", "cpu", "sycl-cpu", "gpu", "opencl")) {
    throw "Unknown release test suite: $Suite"
  }
  if ($Suite -eq 'gpu') { $env:MOM_REQUIRE_GPU_TESTS = '1' }
  if ($Suite -eq 'sycl-cpu' -or
      ($Suite -eq 'opencl' -and $env:MOM_OPENCL_DEVICE_TYPE -eq 'cpu')) {
    # Keep the archive gate fail-closed outside GitHub Actions as well. A missing or broken CPU
    # OpenCL device must never turn the per-algorithm portable kernel coverage into skipped tests.
    $env:MOM_REQUIRE_SYCL_CPU_TESTS = '1'
  }
  & $node tests/run_hash.js $Suite
  if ($LASTEXITCODE -ne 0) {
    throw "Hash suite failed: $Suite"
  }
} finally {
  Pop-Location
}
