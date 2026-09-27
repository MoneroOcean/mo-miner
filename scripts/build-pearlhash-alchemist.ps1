param(
  [switch]$Run,
  [switch]$RunOnly,
  [switch]$Benchmark,
  [switch]$ValidateOnly,
  [ValidateRange(10,200)][int]$Samples = 40,
  [ValidateSet(2048,4096,8192,16384,32768,65536)][int]$MatrixM = 16384,
  [ValidateSet(2048,4096,8192,16384,32768,65536)][int]$MatrixN = 16384,
  [string]$DeviceSelector = 'level_zero:gpu'
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
. (Join-Path $PSScriptRoot 'import-vcvars.ps1')
Import-MomVcVars64

$compilerRoot = if ($env:CMPLR_ROOT) { $env:CMPLR_ROOT } else {
  Join-Path ${env:ProgramFiles(x86)} 'Intel\oneAPI\compiler\latest'
}
$compiler = Join-Path $compilerRoot 'bin\icx.exe'
if (!(Test-Path $compiler)) { throw "Intel compiler not found: $compiler" }
$vars = Join-Path $compilerRoot 'env\vars.bat'
$envLines = & cmd.exe /d /s /c "call `"$vars`" >nul && set"
if ($LASTEXITCODE -ne 0) { throw 'Intel compiler environment initialization failed' }
foreach ($line in $envLines) {
  if ($line -match '^([^=]+)=(.*)$') {
    [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], 'Process')
  }
}

$oneApiRoot = Split-Path (Split-Path $compilerRoot -Parent) -Parent
foreach ($component in @('tcm','umf')) {
  $componentRoot = Join-Path $oneApiRoot $component
  if (Test-Path $componentRoot) {
    $bin = Get-ChildItem $componentRoot -Directory | Sort-Object LastWriteTime -Descending |
      ForEach-Object { Join-Path $_.FullName 'bin' } |
      Where-Object { Test-Path $_ } | Select-Object -First 1
    if ($bin) { $env:Path = "$bin;$env:Path" }
  }
}
# Large matrices can exceed the driver's default per-allocation limit.
$env:UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS = '1'

$buildDir = Join-Path $repo 'build\alchemist'
New-Item -ItemType Directory -Force $buildDir | Out-Null
# Intel creates private subdirectories below TEMP. Reusing a TEMP root across
# the interactive user and an automation account can reuse an inaccessible
# compiler directory even when the build directory itself is writable.
$compilerTemp = Join-Path $buildDir ('tmp-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory $compilerTemp | Out-Null
$env:TEMP = $compilerTemp
$env:TMP = $compilerTemp
$env:TMPDIR = $compilerTemp
$executable = Join-Path $buildDir 'pearlhash-alchemist-test.exe'
Push-Location $buildDir
try {
  if (!$RunOnly) {
  & $compiler -fsycl -fsycl-device-code-split=per_kernel /std:c++20 /O2 /EHsc /D_CRT_SECURE_NO_WARNINGS `
    (Join-Path $repo 'tests\pearlhash_alchemist.cpp') "/Fe:$executable"
  if ($LASTEXITCODE -ne 0) { throw "Intel compilation failed ($LASTEXITCODE)" }
  Write-Host "Built $executable"
  }
  if ($Run -or $RunOnly -or $Benchmark -or $ValidateOnly) {
    if (!(Test-Path $executable)) { throw 'Build the Alchemist test first (Ctrl+Shift+B).' }
    $env:ONEAPI_DEVICE_SELECTOR = $DeviceSelector
    # Stage statistics add waits to the measured pipeline; never inherit them.
    $env:MOM_PEARLHASH_STATS = $null
    $env:MOM_PEARLHASH_CHK = $null
    if ($ValidateOnly) { & $executable }
    elseif ($Benchmark) { & $executable --benchmark $Samples $MatrixM $MatrixN } else { & $executable }
    if ($LASTEXITCODE -ne 0) { throw "Alchemist validation failed ($LASTEXITCODE)" }
    Write-Host 'GPU validation completed successfully.'
  } else {
    Write-Host 'BUILD SUCCEEDED. GPU tests were not run.'
    Write-Host 'To run them in VS Code: Terminal > Run Task > Intel: Build and validate Alchemist'
  }
} finally {
  Pop-Location
  # Only remove the unique temporary directory created by this invocation.
  $resolvedTemp = [IO.Path]::GetFullPath($compilerTemp)
  $resolvedBuild = [IO.Path]::GetFullPath($buildDir).TrimEnd('\') + '\'
  if ($resolvedTemp.StartsWith($resolvedBuild, [StringComparison]::OrdinalIgnoreCase) -and
      (Split-Path $resolvedTemp -Leaf) -match '^tmp-[0-9a-f]{32}$') {
    Remove-Item -LiteralPath $resolvedTemp -Recurse -Force -ErrorAction SilentlyContinue
  }
}
