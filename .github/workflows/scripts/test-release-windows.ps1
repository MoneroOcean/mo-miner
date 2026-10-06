param(
  [Parameter(Mandatory = $true)]
  [string]$Archive,

  [string]$Suite = "all"
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
if ($Suite -notin @("all", "cpu", "gpu", "gpu-discrete", "gpu-integrated", "gpu-multi", "gpu-portable-cpu")) {
  throw "Unknown release test suite: $Suite"
}
$env:MOM_SKIP_MSR = "1"
if ($Suite -eq 'gpu-portable-cpu') {
  # The missing-ICD path must fail closed before it can emit a successful SKIP.
  $env:MOM_REQUIRE_PORTABLE_CPU_TESTS = '1'
}
if ($PSVersionTable.PSVersion.Major -ge 7) {
  $PSNativeCommandUseErrorActionPreference = $true
}

trap {
  if ($env:GITHUB_ACTIONS -eq 'true') {
    $message = $_.Exception.Message.Replace('%', '%25').Replace("`r", '%0D').Replace("`n", '%0A')
    Write-Host "::error title=Windows release test failed::$message"
  }
  throw $_.Exception
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "../../..")).Path
Set-Location $repoRoot
. (Join-Path $repoRoot 'scripts\windows-install-helpers.ps1')

$workDir = if ($env:MOM_RELEASE_TEST_DIR) { $env:MOM_RELEASE_TEST_DIR } else { "mom-release-test" }
if (-not [IO.Path]::IsPathRooted($workDir)) { $workDir = Join-Path $repoRoot $workDir }
$workDir = [IO.Path]::GetFullPath($workDir).TrimEnd([char[]]@('\', '/'))
$workLeaf = [IO.Path]::GetFileName($workDir)
if ($workLeaf -notmatch '^mom-release-[A-Za-z0-9_.-]+$') {
  throw 'MOM_RELEASE_TEST_DIR must name a dedicated mom-release-* directory'
}
Assert-NoReparseAncestor $workDir 'Release test workspace'
$workspaceMarker = Join-Path $workDir '.mom-release-test-workspace'
$packageVersion = (Get-Content -LiteralPath (Join-Path $repoRoot 'package.json') -Raw |
  ConvertFrom-Json).version
if ([string]::IsNullOrWhiteSpace([string]$packageVersion) -or
    [string]$packageVersion -notmatch '^[0-9][0-9A-Za-z.-]*$') {
  throw "Invalid package version: $packageVersion"
}
$expectedRoot = "mom-v$packageVersion"
Assert-NoReparseAncestor $Archive 'Release archive'
$archiveItem = Get-Item -LiteralPath $Archive -Force -ErrorAction Stop
if ($archiveItem.PSIsContainer -or
    ($archiveItem.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
  throw 'Release archive must be a regular non-reparse file.'
}
$archivePath = $archiveItem.FullName
$privateArchive = Join-Path $workDir '.mom-release-input.zip'
if ($archivePath.Equals($workDir, [StringComparison]::OrdinalIgnoreCase) -or
    $archivePath.StartsWith("$workDir\", [StringComparison]::OrdinalIgnoreCase)) {
  throw 'Release archive must be outside the test workspace.'
}

$workItem = Get-Item -LiteralPath $workDir -Force -ErrorAction SilentlyContinue
if ($workItem) {
  if (-not $workItem.PSIsContainer) {
    throw 'MOM_RELEASE_TEST_DIR must be an owned directory'
  }
  if ($workItem.Attributes -band [IO.FileAttributes]::ReparsePoint) {
    throw 'MOM_RELEASE_TEST_DIR must not be a reparse point'
  }
  if (-not (Test-MomMarker $workspaceMarker 'mom release test workspace')) {
    throw 'MOM_RELEASE_TEST_DIR exists without the .mom-release-test-workspace marker'
  }
  Assert-NoReparseAncestor $workDir 'Release test workspace'
  Remove-Item -LiteralPath $workDir -Recurse -Force
}
[IO.Directory]::CreateDirectory($workDir) | Out-Null
Set-Content -LiteralPath $workspaceMarker -Value 'mom release test workspace' -NoNewline
Copy-Item -LiteralPath $archivePath -Destination $privateArchive -ErrorAction Stop

Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip = [System.IO.Compression.ZipFile]::OpenRead($privateArchive)
try {
  $members = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
  $hasRootEntry = $false
  $hasRootPayload = $false
  foreach ($entry in $zip.Entries) {
    $rawName = $entry.FullName
    if ($rawName -match '(^[/\\]|(^|[/\\])\.\.([/\\]|$)|:|[/\\]{2}|(^|[/\\])\.([/\\]|$))') {
      throw "Release archive contains an unsafe path: $($entry.FullName)"
    }
    $normalizedName = $rawName.Replace('\', '/')
    $isDirectory = $normalizedName.EndsWith('/')
    $member = $normalizedName.TrimEnd('/')
    if (-not $member) {
      throw 'Release archive contains an empty member name.'
    }
    if (-not $members.Add($member)) {
      throw "Release archive contains a duplicate member: $($entry.FullName)"
    }
    if ($member -cne $expectedRoot -and
        -not $member.StartsWith("$expectedRoot/", [StringComparison]::Ordinal)) {
      throw 'Release archive must contain only the expected package root.'
    }
    if ($member -ceq $expectedRoot) {
      if (-not $isDirectory -or $normalizedName -cne "$expectedRoot/") {
        throw 'Release archive must contain an explicit package root directory.'
      }
      $hasRootEntry = $true
    } else {
      $hasRootPayload = $true
    }
    $unixType = ($entry.ExternalAttributes -shr 16) -band 0xF000
    if ($unixType -eq 0xA000) {
      throw "Release archive contains a symbolic link: $($entry.FullName)"
    }
  }
  if (-not $hasRootEntry) {
    throw 'Release archive must contain an explicit package root directory.'
  }
  if (-not $hasRootPayload) {
    throw 'Release archive must contain the expected package root.'
  }
  $root = $expectedRoot
  if ($zip.Entries | Where-Object { $_.FullName -match '(^|[/\\])tests([/\\]|$)' }) {
    throw "Release archive must not contain tests/."
  }
  if ($zip.Entries | Where-Object { $_.FullName -match '(^|[/\\])DEVELOPMENT\.md$' }) {
    throw "Release archive must not contain DEVELOPMENT.md."
  }
} finally {
  $zip.Dispose()
}

# Focused archive tests set this after the structural checks; release CI leaves it unset.
if ($env:MOM_RELEASE_ARCHIVE_VALIDATION_ONLY -eq '1') {
  exit 0
}

Expand-Archive -LiteralPath $privateArchive -DestinationPath $workDir
$pending = [Collections.Generic.Stack[string]]::new()
$pending.Push($workDir)
while ($pending.Count -gt 0) {
  $current = $pending.Pop()
  $attributes = [IO.File]::GetAttributes($current)
  if ($attributes -band [IO.FileAttributes]::ReparsePoint) {
    throw 'Release extraction produced a reparse point.'
  }
  if (-not ($attributes -band [IO.FileAttributes]::Directory)) { continue }
  foreach ($child in [IO.Directory]::EnumerateFileSystemEntries($current)) {
    $pending.Push($child)
  }
}
$packageDir = Join-Path $workDir $root
$packageItem = Get-Item -LiteralPath $packageDir -Force -ErrorAction Stop
if (-not $packageItem.PSIsContainer) {
  throw 'Extracted release package root is not a directory.'
}
$libsDir = Join-Path $packageDir "libs"
$node = Join-Path $packageDir 'mom-node.exe'
foreach ($sidecar in @('kawpow_device.inc', 'kawpow_keccak.inc')) {
  $sidecarPath = Join-Path $libsDir "dpcpp\$sidecar"
  if (-not (Test-Path -LiteralPath $sidecarPath -PathType Leaf) -or
      (Get-Item -LiteralPath $sidecarPath -Force).Length -le 0) {
    throw "Windows release package is missing nonempty libs/dpcpp/$sidecar."
  }
}
if (Test-Path (Join-Path $packageDir "tests")) {
  throw "Extracted release package unexpectedly contains tests/."
}
if (-not (Test-Path (Join-Path $packageDir "GPU-CONFIG.md"))) {
  throw "Extracted release package is missing GPU-CONFIG.md."
}
if (-not (Test-Path (Join-Path $packageDir "gpu-tuning.js"))) {
  throw "Extracted release package is missing gpu-tuning.js."
}
if (-not (Test-Path (Join-Path $packageDir "helper/hash.js"))) {
  throw "Extracted release package is missing helper/hash.js."
}
if (Test-Path (Join-Path $packageDir "DEVELOPMENT.md")) {
  throw "Extracted release package unexpectedly contains DEVELOPMENT.md."
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
Copy-Item scripts\windows-command.js (Join-Path $packageDir 'scripts\windows-command.js')

# A developer image may already point OCL_ICD_FILENAMES at oneAPI's GPU ICD. CPU deployment gates
# deliberately select Intel's separately installed CPU ICD so a headless runner cannot silently
# enumerate zero devices while ignoring the system-wide Khronos registry entry.
if ($Suite -eq 'gpu-portable-cpu') {
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
    $message = 'Intel CPU OpenCL ICD is missing; run the release package install.bat.'
    if ($env:GITHUB_ACTIONS -or $env:MOM_REQUIRE_PORTABLE_CPU_TESTS -eq "1") {throw $message}
    Write-Host "SKIP gpu-portable-cpu: $message"
    exit 0
  }
  $env:OCL_ICD_FILENAMES = $cpuIcd
}

# Minimal PATH: package/libs first, then the Windows system dirs the EXE needs.
$openClIcdFiles = @($env:OCL_ICD_FILENAMES -split ';' |
  Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
$externalOpenClDir = if ($openClIcdFiles.Count -gt 0) {
  $env:OCL_ICD_FILENAMES = $openClIcdFiles -join ';'
  Split-Path -Parent $openClIcdFiles[0]
} else {
  Remove-Item Env:OCL_ICD_FILENAMES -ErrorAction SilentlyContinue
  $null
}
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
$restoreUnsetGpuBackendAfterSmoke = $false
if ($Suite -in @('gpu', 'gpu-discrete', 'gpu-multi') -and
    $null -eq [Environment]::GetEnvironmentVariable('MOM_GPU_BACKEND', 'Process')) {
  # The launcher can auto-select only a single detected vendor. Select one present GPU for this
  # pre-matrix smoke check; the matrix itself must see the original unset value and all vendors.
  $knownGpuVendorIds = @(Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue |
    ForEach-Object {
      if ([string]$_.PNPDeviceID -match 'VEN_(10DE|1002|8086)') { $Matches[1] }
    } | Sort-Object -Unique)
  $smokeBackend = if ($knownGpuVendorIds -contains '10DE') { 'nvidia' } elseif (
    $knownGpuVendorIds -contains '1002') { 'amd' } elseif (
    $knownGpuVendorIds -contains '8086') { 'intel' } else { $null }
  if ($smokeBackend) {
    $env:MOM_GPU_BACKEND = $smokeBackend
    $restoreUnsetGpuBackendAfterSmoke = $true
  }
}
if ($Suite -eq 'gpu-portable-cpu') {
  # Development/release jobs install Intel's redistributable CPU OpenCL implementation. Use the
  # portable SPIR-V worker so this archive gate needs no GPU and exercises the generic fallback ABI.
  $env:MOM_GPU_BACKEND = 'opencl'
  $env:MOM_OPENCL_DEVICE_TYPE = 'cpu'
  $env:ONEAPI_DEVICE_SELECTOR = 'opencl:cpu'
}
$defaultWorker = if ($Suite -eq 'gpu-portable-cpu') { 'dpcpp-opencl' } else { switch ($env:MOM_GPU_BACKEND) {
  'nvidia' { 'dpcpp' }
  'amd' { 'acpp-hip' }
  'opencl' { 'dpcpp-opencl' }
  default { 'oneapi' }
} }
$env:MOM_NATIVE_PATH = Join-Path $libsDir "$defaultWorker\mom.node"
$env:MOM_NATIVE_PATH_LAUNCHER_DEFAULT = $env:MOM_NATIVE_PATH
$workerDir = Join-Path $libsDir $defaultWorker
$env:MOM_RUNTIME_DIR = $workerDir
if ($Suite -eq 'gpu-portable-cpu' -and -not $env:UR_ADAPTERS_FORCE_LOAD) {
  # Direct Node tests bypass mom.cmd; mirror its isolated OpenCL adapter binding.
  # UR requires quotes inside the value for Windows drive-letter paths.
  $env:UR_ADAPTERS_FORCE_LOAD = '"' + (Join-Path $workerDir 'ur_adapter_opencl.dll') + '"'
}
$sharedOneApiDir = if ($defaultWorker -eq 'dpcpp-opencl') { Join-Path $libsDir 'oneapi' } else { $null }
# For the CPU suite, an explicitly supplied CPU OpenCL runtime must provide OpenCL.dll ahead of the
# bundled generic loader. Shared oneAPI follows it for the matching JIT library.
$env:Path = @($workerDir, (Join-Path $workerDir 'hipSYCL'), $externalOpenClDir,
  $sharedOneApiDir, $env:Path) -join ';'

function Enable-IntelOpenCL {
  if ($Suite -eq 'gpu-portable-cpu' -or $env:MOM_OPENCL_DEVICE_TYPE -eq 'cpu') {
    return
  }
  # Generic OpenCL normally uses the passed-through vendor's system ICD. Some Intel driver packages
  # have no Khronos registry entry, so their ICD disappears when the archive test minimizes PATH.
  # Use the packaged Intel ICD only when Intel is the sole hardware GPU vendor; leave mixed, AMD,
  # and NVIDIA environments untouched.
  if ($env:MOM_GPU_BACKEND -eq 'opencl') {
    $gpuVendors = @(Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue |
      ForEach-Object {
        if ([string]$_.PNPDeviceID -match 'VEN_(8086|10DE|1002)') { $Matches[1] }
      } | Sort-Object -Unique)
    if ($gpuVendors.Count -ne 1 -or $gpuVendors[0] -ne '8086') { return }
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
      $output = & .\mom.cmd algorithms 2>&1
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
  $marker = $smokeOutput | Where-Object { $_ -match '^MOM_ALGORITHMS ' } | Select-Object -First 1
  if (-not $marker) {
    throw "Direct executable smoke test did not print algorithms marker.`n$($smokeOutput -join "`n")"
  }
  $params = ($marker -replace '^MOM_ALGORITHMS ', '') | ConvertFrom-Json
  foreach ($prop in $params.PSObject.Properties) {
    $dev = [string]$prop.Value
    if (-not $dev -or $dev -match '(^|,)[^,]*(\*0|\^0)(,|$)') {
      throw "Invalid algorithms entry for $($prop.Name): $dev"
    }
  }
  if ($Suite -in @('gpu', 'gpu-discrete', 'gpu-integrated', 'gpu-multi')) {
    $gpuParam = $params.PSObject.Properties | Where-Object { [string]$_.Value -match '(^|,)gpu\d+' } |
      Select-Object -First 1
    if (-not $gpuParam) {
      throw "Windows $Suite release test requires launcher-time GPU discovery, but algorithms returned no GPU job."
    }
  }
  if ($Suite -eq 'gpu-integrated') {
    $integratedGpu = $smokeOutput | Where-Object {
      $_ -match '^gpu\d+: .*Intel.*\[integrated\]$'
    } |
      Select-Object -First 1
    if (-not $integratedGpu) {
      throw "Windows gpu-integrated release test requires an Intel integrated GPU, but algorithms reported none."
    }
  }
  $syclCpuDevices = Get-SyclCpuDevicesFromOutput $smokeOutput
  if (($Suite -eq "all" -or $Suite -eq "gpu-portable-cpu") -and $syclCpuDevices.Count -eq 0) {
    throw "Windows $Suite release test requires a CPU SYCL device, but algorithms did not report one.`n$($smokeOutput -join "`n")"
  }

  if ($restoreUnsetGpuBackendAfterSmoke) {
    Remove-Item Env:MOM_GPU_BACKEND -ErrorAction SilentlyContinue
  }
  if ($Suite -in @('gpu', 'gpu-discrete', 'gpu-integrated', 'gpu-multi')) { $env:MOM_REQUIRE_GPU_TESTS = '1' }
  if ($Suite -eq 'gpu-integrated') { $env:MOM_REQUIRE_INTEGRATED_GPU_TESTS = '1' }
  if ($Suite -eq 'gpu-multi') { $env:MOM_REQUIRE_MULTI_GPU_TESTS = '1' }
  & $node tests/run_hash.js $Suite
  if ($LASTEXITCODE -ne 0) {
    throw "Hash suite failed: $Suite"
  }
} finally {
  Pop-Location
}
