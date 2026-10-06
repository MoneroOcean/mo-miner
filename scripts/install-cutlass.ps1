$MomCutlassVersion = "v4.6.1"
$MomCutlassSha256 = "455d9ba37d57cb214d67b5d1a6070441244b378bcacb2e916c3b86f2a9b02e1c"
$MomCutlassUrl = "https://github.com/NVIDIA/cutlass/archive/refs/tags/$MomCutlassVersion.tar.gz"
$MomCcclVersion = "12.6.37"
$MomCcclSha256 = "4fe0460b101887a62fd8ceb1e518926439148c23ec95ef41b694c583031392e9"
$MomCcclUrl = "https://developer.download.nvidia.com/compute/cuda/redist/cuda_cccl/windows-x86_64/cuda_cccl-windows-x86_64-$MomCcclVersion-archive.zip"

function Assert-MomNoReparseAncestor {
  param(
    [Parameter(Mandatory = $true)][string]$Path,
    [Parameter(Mandatory = $true)][string]$Description
  )
  $current = [IO.Path]::GetFullPath($Path)
  while ($current) {
    $item = Get-Item -LiteralPath $current -Force -ErrorAction SilentlyContinue
    if ($item -and ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
      throw "Refusing $Description with a reparse point: $current"
    }
    $parent = [IO.Path]::GetDirectoryName($current.TrimEnd([char[]]@('\', '/')))
    if (-not $parent -or $parent -eq $current) { break }
    $current = $parent
  }
}

function Test-MomDirectoryDestination {
  param(
    [Parameter(Mandatory = $true)][string]$Path,
    [Parameter(Mandatory = $true)][string]$Description
  )
  Assert-MomNoReparseAncestor $Path $Description
  $item = Get-Item -LiteralPath $Path -Force -ErrorAction SilentlyContinue
  if ($null -eq $item) { return $false }
  if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
    throw "Refusing $Description that is a reparse point: $Path"
  }
  if (-not $item.PSIsContainer) {
    throw "Refusing $Description that is not a directory: $Path"
  }
  return $true
}

function Test-MomCccl {
  param([Parameter(Mandatory = $true)][string]$CudaRoot)
  (Test-Path (Join-Path $CudaRoot "include\cuda\std\cstdint")) -or
    (Test-Path (Join-Path $CudaRoot "include\cccl\cuda\std\cstdint"))
}

function Get-MomVersionMarker {
  param([Parameter(Mandatory = $true)][string]$Destination)
  $marker = Join-Path $Destination ".mom-version"
  $item = Get-Item -LiteralPath $marker -Force -ErrorAction SilentlyContinue
  if ($null -eq $item) { return $null }
  if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
    throw "Refusing CUTLASS version marker that is a reparse point: $marker"
  }
  if ($item -isnot [IO.FileInfo]) {
    throw "Refusing CUTLASS version marker that is not a regular file: $marker"
  }
  return $item
}

function Install-MomCccl {
  param([Parameter(Mandatory = $true)][string]$CudaRoot)
  Test-MomDirectoryDestination $CudaRoot 'CCCL destination' | Out-Null
  if (Test-MomCccl $CudaRoot) { return }

  $tempBase = if ($env:TEMP) { $env:TEMP } else { [IO.Path]::GetTempPath() }
  $work = Join-Path $tempBase ("mom-cccl-{0}" -f [Guid]::NewGuid().ToString('N'))
  $archive = Join-Path $work "cccl.zip"
  $extract = Join-Path $work "extract"
  $workCreated = $false
  try {
    [IO.Directory]::CreateDirectory($work) | Out-Null
    $workCreated = $true
    [IO.Directory]::CreateDirectory($extract) | Out-Null
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
    if (-not (Test-MomDirectoryDestination $CudaRoot 'CCCL destination')) {
      [IO.Directory]::CreateDirectory($CudaRoot) | Out-Null
    }
    $includeDestination = Join-Path $CudaRoot "include"
    if (-not (Test-MomDirectoryDestination $includeDestination 'CCCL include destination')) {
      [IO.Directory]::CreateDirectory($includeDestination) | Out-Null
    }
    Copy-Item -Path (Join-Path $include "*") -Destination $includeDestination -Recurse -Force
  } finally {
    if ($workCreated) {
      Remove-Item -LiteralPath $work -Recurse -Force -ErrorAction SilentlyContinue
    }
  }
  Write-Host "Installed NVIDIA CCCL $MomCcclVersion headers."
}

function Test-MomCutlass {
  param([string]$Destination = (Join-Path $env:ProgramData "mom\cutlass"))
  $markerItem = Get-MomVersionMarker $Destination
  (Test-Path (Join-Path $Destination "include\cute\tensor.hpp")) -and
    ($null -ne $markerItem) -and
    ((Get-Content -LiteralPath $markerItem.FullName -Raw).Trim() -eq $MomCutlassSha256)
}

function Install-MomCutlass {
  param([string]$Destination = (Join-Path $env:ProgramData "mom\cutlass"))

  Test-MomDirectoryDestination $Destination 'CUTLASS destination' | Out-Null
  if (Test-MomCutlass $Destination) {
    Write-Host "CUTLASS $MomCutlassVersion headers are already installed."
    return
  }

  $tempBase = if ($env:TEMP) { $env:TEMP } else { [IO.Path]::GetTempPath() }
  $work = Join-Path $tempBase ("mom-cutlass-{0}" -f [Guid]::NewGuid().ToString('N'))
  $archive = Join-Path $work "cutlass.tar.gz"
  $extract = Join-Path $work "extract"
  $workCreated = $false
  try {
    [IO.Directory]::CreateDirectory($work) | Out-Null
    $workCreated = $true
    [IO.Directory]::CreateDirectory($extract) | Out-Null
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
    if (-not (Test-MomDirectoryDestination $Destination 'CUTLASS destination')) {
      [IO.Directory]::CreateDirectory($Destination) | Out-Null
    }
    $includeDestination = Join-Path $Destination "include"
    if (-not (Test-MomDirectoryDestination $includeDestination 'CUTLASS include destination')) {
      [IO.Directory]::CreateDirectory($includeDestination) | Out-Null
    }
    Copy-Item -Path (Join-Path $include "*") -Destination $includeDestination -Recurse -Force
    Get-MomVersionMarker $Destination | Out-Null
    Set-Content -LiteralPath (Join-Path $Destination ".mom-version") `
      -Value $MomCutlassSha256 -NoNewline
  } finally {
    if ($workCreated) {
      Remove-Item -LiteralPath $work -Recurse -Force -ErrorAction SilentlyContinue
    }
  }
  Write-Host "Installed CUTLASS $MomCutlassVersion headers."
}
