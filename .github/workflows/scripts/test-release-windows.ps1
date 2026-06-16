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

# Log the runner CPU vendor: RandomX (rx/*, panthera) crashed with 0xC0000005 on AMD CI runners until
# XMRIG_FIX_RYZEN was defined (registers the JIT main-loop bounds the RxFix exception handler recovers
# from). Print it so an rx access-violation can be correlated with the CPU at a glance.
Write-Host "Runner CPU: $env:PROCESSOR_IDENTIFIER"

. "$PSScriptRoot/windows-dll-deps.ps1"

$node = (Get-Command node.exe).Source
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
$packageDir = Join-Path $workDir $root
$libsDir = Join-Path $packageDir "libs"
if (Test-Path (Join-Path $packageDir "tests")) {
  throw "Extracted release package unexpectedly contains tests/."
}

foreach ($lib in @("sycl.dll", "mom.node")) {
  if (-not (Test-Path (Join-Path $libsDir $lib))) {
    throw "Windows release package is missing libs/$lib."
  }
}
$entryPaths = @(
  (Join-Path $packageDir "mom-node.exe"),
  (Join-Path $libsDir "mom.node"),
  (Join-Path $libsDir "sycl.dll")
)
Test-MominerDllClosure -PackageDir $libsDir -EntryPaths $entryPaths

Copy-Item tests (Join-Path $packageDir "tests") -Recurse

# Minimal PATH: package/libs first, then the Windows system dirs the EXE needs.
$env:Path = @(
  $libsDir,
  $packageDir,
  "$env:WINDIR\System32",
  $env:WINDIR,
  "$env:WINDIR\System32\Wbem",
  "$env:WINDIR\System32\WindowsPowerShell\v1.0"
) -join ';'

function Enable-IntelOpenCL {
  if ($env:OCL_ICD_FILENAMES) {
    return
  }

  $intelOcl = Get-ChildItem -Path $libsDir -Filter "intelocl*.dll" -File -ErrorAction SilentlyContinue | Select-Object -First 1
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
  $syclCpuDevices = Get-SyclCpuDevicesFromOutput $smokeOutput
  if (($Suite -eq "all" -or $Suite -eq "sycl-cpu") -and $syclCpuDevices.Count -eq 0) {
    throw "Windows $Suite release test requires a CPU SYCL device, but algo_params did not report one.`n$($smokeOutput -join "`n")"
  }

  if ($Suite -notin @("all", "cpu", "sycl-cpu", "gpu")) {
    throw "Unknown release test suite: $Suite"
  }
  & $node tests/run_hash.js $Suite
  if ($LASTEXITCODE -ne 0) {
    throw "Hash suite failed: $Suite"
  }
} finally {
  Pop-Location
}
