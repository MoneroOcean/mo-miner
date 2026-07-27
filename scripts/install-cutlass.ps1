$MomCutlassVersion = "v4.6.1"
$MomCutlassSha256 = "455d9ba37d57cb214d67b5d1a6070441244b378bcacb2e916c3b86f2a9b02e1c"
$MomCutlassUrl = "https://github.com/NVIDIA/cutlass/archive/refs/tags/$MomCutlassVersion.tar.gz"
$MomCcclVersion = "12.6.37"
$MomCcclSha256 = "4fe0460b101887a62fd8ceb1e518926439148c23ec95ef41b694c583031392e9"
$MomCcclUrl = "https://developer.download.nvidia.com/compute/cuda/redist/cuda_cccl/windows-x86_64/cuda_cccl-windows-x86_64-$MomCcclVersion-archive.zip"

function Test-MomCccl {
  param([Parameter(Mandatory = $true)][string]$CudaRoot)
  (Test-Path (Join-Path $CudaRoot "include\cuda\std\cstdint")) -or
    (Test-Path (Join-Path $CudaRoot "include\cccl\cuda\std\cstdint"))
}

function Install-MomCccl {
  param([Parameter(Mandatory = $true)][string]$CudaRoot)
  if (Test-MomCccl $CudaRoot) { return }

  $tempBase = if ($env:TEMP) { $env:TEMP } else { [IO.Path]::GetTempPath() }
  $work = Join-Path $tempBase "mom-cccl"
  $archive = Join-Path $work "cccl.zip"
  $extract = Join-Path $work "extract"
  Remove-Item $work -Recurse -Force -ErrorAction SilentlyContinue
  New-Item -ItemType Directory -Force $extract | Out-Null
  try {
    Invoke-WebRequest -UseBasicParsing -Uri $MomCcclUrl -OutFile $archive
    $actual = (Get-FileHash -Algorithm SHA256 $archive).Hash.ToLowerInvariant()
    if ($actual -ne $MomCcclSha256) {
      throw "CCCL archive SHA256 mismatch: expected $MomCcclSha256, got $actual"
    }
    Expand-Archive -Path $archive -DestinationPath $extract
    $include = Join-Path $extract "cuda_cccl-windows-x86_64-$MomCcclVersion-archive\include"
    if (-not (Test-Path (Join-Path $include "cuda\std\cstdint"))) {
      throw "CCCL archive does not contain include\cuda\std\cstdint"
    }
    New-Item -ItemType Directory -Force (Join-Path $CudaRoot "include") | Out-Null
    Copy-Item (Join-Path $include "*") (Join-Path $CudaRoot "include") -Recurse -Force
  } finally {
    Remove-Item $work -Recurse -Force -ErrorAction SilentlyContinue
  }
  Write-Host "Installed NVIDIA CCCL $MomCcclVersion headers."
}

function Test-MomCutlass {
  param([string]$Destination = (Join-Path $env:ProgramData "mom\cutlass"))
  $marker = Join-Path $Destination ".mom-version"
  (Test-Path (Join-Path $Destination "include\cute\tensor.hpp")) -and
    (Test-Path $marker) -and
    ((Get-Content $marker -Raw).Trim() -eq $MomCutlassSha256)
}

function Install-MomCutlass {
  param([string]$Destination = (Join-Path $env:ProgramData "mom\cutlass"))

  if (Test-MomCutlass $Destination) {
    Write-Host "CUTLASS $MomCutlassVersion headers are already installed."
    return
  }

  $tempBase = if ($env:TEMP) { $env:TEMP } else { [IO.Path]::GetTempPath() }
  $work = Join-Path $tempBase "mom-cutlass"
  $archive = Join-Path $work "cutlass.tar.gz"
  $extract = Join-Path $work "extract"
  Remove-Item $work -Recurse -Force -ErrorAction SilentlyContinue
  New-Item -ItemType Directory -Force $extract | Out-Null
  try {
    Invoke-WebRequest -UseBasicParsing -Uri $MomCutlassUrl -OutFile $archive
    $actual = (Get-FileHash -Algorithm SHA256 $archive).Hash.ToLowerInvariant()
    if ($actual -ne $MomCutlassSha256) {
      throw "CUTLASS archive SHA256 mismatch: expected $MomCutlassSha256, got $actual"
    }
    & "$env:SystemRoot\System32\tar.exe" -xf $archive -C $extract `
      "cutlass-$($MomCutlassVersion.TrimStart('v'))/include"
    if ($LASTEXITCODE -ne 0) { throw "CUTLASS archive extraction failed: $LASTEXITCODE" }
    $include = Join-Path $extract "cutlass-$($MomCutlassVersion.TrimStart('v'))\include"
    if (-not (Test-Path (Join-Path $include "cute\tensor.hpp"))) {
      throw "CUTLASS archive does not contain include\cute\tensor.hpp"
    }
    Remove-Item $Destination -Recurse -Force -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force $Destination | Out-Null
    Move-Item $include (Join-Path $Destination "include")
    Set-Content -Path (Join-Path $Destination ".mom-version") `
      -Value $MomCutlassSha256 -NoNewline
  } finally {
    Remove-Item $work -Recurse -Force -ErrorAction SilentlyContinue
  }
  Write-Host "Installed CUTLASS $MomCutlassVersion headers."
}
