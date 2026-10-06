# Package the from-source intel/llvm `--cuda` DPC++ toolchain (Windows) into a single tarball that CI
# restores to build the unified spir64+nvptx sycl.dll, instead of rebuilding the ~1.5h LLVM toolchain on
# every run (which cannot finish inside a GitHub-hosted Windows job anyway: 6h timeout, 2-4 vCPU). Run
# this ONCE on the build VM following scripts/windows-multicompiler.md; the
# resulting asset is uploaded to a GitHub release (NOT committed to git). gzip is used so the CI runner's
# built-in tar.exe can extract it with no extra tooling.
param(
  [string]$BuildDir = "C:\llvm\build",
  [string]$OutFile  = "C:\mom\dpcpp-cuda-win-v7.1.1.tar.gz"
)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
. (Join-Path $PSScriptRoot '../../../scripts/windows-install-helpers.ps1')

if (-not (Test-Path (Join-Path $BuildDir "bin\clang++.exe"))) {
  throw "clang++.exe not found under $BuildDir\bin - point -BuildDir at the from-source --cuda build dir."
}
foreach ($d in @("bin", "lib", "include")) {
  if (-not (Test-Path (Join-Path $BuildDir $d))) { throw "$BuildDir\$d is missing." }
}
if (-not (Test-MomDpcppToolchain $BuildDir)) {
  throw 'DPC++ toolchain is incomplete; package its matching real UR loader with the compiler.'
}

$outDir = Split-Path $OutFile -Parent
if ($outDir) { New-Item -ItemType Directory -Force $outDir | Out-Null }
Remove-Item -Force $OutFile, ($OutFile + ".sha256") -ErrorAction SilentlyContinue

# bin/   : clang++.exe + the full -fsycl tool pipeline (sycl-post-link, llvm-spirv, llc, lld-link, ...)
#          and the runtime DLLs we ship (sycl9.dll, sycl-jit.dll, ur_adapter_*.dll).
# lib/   : SYCL device libs (*.bc), clang builtins (lib\clang\), import libs (sycl9.lib, ze_loader.lib).
#          lib\LLVM*.lib are pure LLVM component archives used only to LINK llvm tools, never referenced
#          by `clang++ -fsycl` on user code -> excluded (~0.7 GB). clang*.lib kept (avoids matching the
#          clang_rt.*.lib builtins under lib\clang\, which a `clang*.lib` glob would wrongly drop).
# include/: the SYCL headers.
$tar = Join-Path $env:SystemRoot "system32\tar.exe"
Write-Host "Packing $BuildDir (bin, lib, include; excluding lib/LLVM*.lib) -> $OutFile ..."
& $tar -C $BuildDir --exclude "lib/LLVM*.lib" -czf $OutFile bin lib include
if ($LASTEXITCODE -ne 0) { throw "tar failed with exit code $LASTEXITCODE." }

$item = Get-Item $OutFile
$hash = (Get-FileHash $OutFile -Algorithm SHA256).Hash.ToLower()
"$hash  $($item.Name)" | Set-Content -Encoding ascii ($OutFile + ".sha256")
Write-Host ("DONE  {0}  {1:N0} MB  sha256={2}" -f $item.Name, ($item.Length / 1MB), $hash)
