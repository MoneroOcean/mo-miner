function Invoke-Checked([string]$File, [string[]]$Arguments) {
  & $File @Arguments
  if ($LASTEXITCODE -ne 0) { throw "$File failed with exit code $LASTEXITCODE" }
}

function Download([string]$Url, [string]$OutFile) {
  New-Item -ItemType Directory -Force (Split-Path -Parent $OutFile) | Out-Null
  Invoke-Checked 'curl.exe' @(
    '-fL','--retry','5','--retry-all-errors','--retry-delay','5',
    '--continue-at','-','-o',$OutFile,$Url
  )
}

function Assert-Sha256([string]$Path, [string]$Expected) {
  $actual = (Get-FileHash -Algorithm SHA256 $Path).Hash.ToLowerInvariant()
  if ($actual -ne $Expected.ToLowerInvariant()) {
    Remove-Item $Path -Force -ErrorAction SilentlyContinue
    throw "SHA256 mismatch for $Path`: expected $Expected, got $actual"
  }
}

function Assert-Authenticode([string]$Path) {
  $signature = Get-AuthenticodeSignature -LiteralPath $Path
  if ($signature.Status -ne [System.Management.Automation.SignatureStatus]::Valid) {
    throw "Authenticode verification failed for $Path`: $($signature.Status)"
  }
}

function Test-MomMarker([string]$Path, [string]$Expected) {
  $item = Get-Item -LiteralPath $Path -Force -ErrorAction SilentlyContinue
  if (-not $item -or $item.PSIsContainer -or
      ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
    return $false
  }
  try {
    [string]::Equals((Get-Content -LiteralPath $Path -Raw), $Expected,
      [StringComparison]::Ordinal)
  } catch {
    $false
  }
}

function Assert-NoReparseAncestor([string]$Path, [string]$Label) {
  $current = [IO.Path]::GetFullPath($Path)
  while ($current) {
    $item = Get-Item -LiteralPath $current -Force -ErrorAction SilentlyContinue
    if ($item -and ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
      throw "$Label must not use a reparse point: $current"
    }
    $parent = [IO.Path]::GetDirectoryName($current.TrimEnd([char[]]@('\', '/')))
    if (-not $parent -or $parent -eq $current) { break }
    $current = $parent
  }
}

function Test-MomCudaCompilerSdk([string]$Root) {
  if ([string]::IsNullOrWhiteSpace($Root)) { return $false }
  foreach ($relative in @(
    'bin\nvcc.exe',
    'bin\ptxas.exe',
    'nvvm\libdevice\libdevice.10.bc',
    'include\cuda.h',
    'include\cuda_runtime.h',
    'include\nvrtc.h',
    'lib\x64\cuda.lib',
    'lib\x64\cudart_static.lib',
    'lib\x64\nvrtc.lib'
  )) {
    if (-not (Test-Path -LiteralPath (Join-Path $Root $relative) -PathType Leaf)) {
      return $false
    }
  }
  return $true
}

function Get-MomHipDeviceLib([string]$Root) {
  if ([string]::IsNullOrWhiteSpace($Root)) { return $null }
  $deviceLib = Get-ChildItem -LiteralPath $Root -Filter 'ockl.bc' -File -Recurse `
    -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($deviceLib) { return $deviceLib.FullName }
  return $null
}

function Test-MomHipSdk([string]$Root) {
  if ([string]::IsNullOrWhiteSpace($Root)) { return $false }
  foreach ($relative in @(
    'lib\amdhip64.lib',
    'include\hip\hip_runtime_api.h',
    'include\hip\hiprtc.h',
    'bin\clang++.exe',
    'bin\llvm-link.exe',
    'bin\opt.exe',
    'bin\llc.exe',
    'bin\lld-link.exe'
  )) {
    if (-not (Test-Path -LiteralPath (Join-Path $Root $relative) -PathType Leaf)) {
      return $false
    }
  }
  return [bool](Get-MomHipDeviceLib $Root)
}

function Test-MomDpcppToolchain([string]$Root) {
  if ([string]::IsNullOrWhiteSpace($Root)) { return $false }
  foreach ($relative in @(
    'bin\clang++.exe',
    'bin\clang.exe',
    'bin\lld-link.exe',
    'bin\sycl-post-link.exe',
    'bin\llvm-spirv.exe',
    'bin\llvm-link.exe',
    'bin\clang-linker-wrapper.exe',
    'bin\clang-offload-wrapper.exe',
    'bin\sycl9.dll',
    'bin\sycl-jit.dll',
    # The open toolchain intentionally carries the proxy; package-windows.ps1 supplies oneAPI's
    # ABI-compatible ur_loader.dll beside each released worker.
    'bin\ur_win_proxy_loader.dll',
    'bin\ur_adapter_cuda.dll',
    'bin\ur_adapter_opencl.dll',
    'bin\ur_adapter_level_zero_v2.dll'
  )) {
    if (-not (Test-Path -LiteralPath (Join-Path $Root $relative) -PathType Leaf)) {
      return $false
    }
  }
  return $true
}
