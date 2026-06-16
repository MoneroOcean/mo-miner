# Windows unified Intel + NVIDIA build (spir64 + nvptx)

This reproduces the **unified Windows `mom.node`** that runs GPU algos on **both** Intel (Level-Zero/
OpenCL, spir64) and **NVIDIA** (CUDA, nvptx) GPUs from one binary — the Windows counterpart of the Linux
combined build (`scripts/combined-build.sh`). Verified on an NVIDIA L4 (sm_89, Windows Server 2022):
`gpu` hash-vector suite **7/7** (cn/gpu, kawpow incl. runtime JIT, etchash, autolykos2, c29×2, pearl).

The released Windows package today is Intel-only (`mom_sycl_impl=dpcpp`, MSBuild + Intel oneAPI DPC++ →
`sycl.dll` with spir64+ESIMD). NVIDIA support needs a SYCL CUDA backend, which the **prebuilt** intel/llvm
Windows nightly does **not** ship (its `clang++.exe` is built without `--cuda`: no NVPTX target, no CUDA UR
adapter). So the toolchain must be built from source with `--cuda`, then used to build a `sycl.dll` that
carries both device images. `mom.node` itself stays MSVC-built and only delay-loads `sycl.dll`, so it is
unchanged — the unified part is entirely in `sycl.dll` + the from-source SYCL runtime DLLs.

## Prerequisites (Windows Server 2022, x64)
- VS 2022 Build Tools (MSVC + Windows SDK), CMake, Ninja, Python 3, Git, Node.js LTS.
- CUDA Toolkit 12.x (e.g. 12.6) + NVIDIA driver. A single **sm_80** PTX image is driver-JIT
  forward-compatible to Ampere/Ada/Hopper (same as Linux). `%CUDA_PATH%` must point at the toolkit
  (`...\nvvm\libdevice\libdevice.10.bc` is the Windows analog of Linux `/usr/local/cuda`; the SYCL CUDA
  backend and the kawpow source-JIT both need it).
- Intel oneAPI DPC++ (`icx`) only if you also want the MSBuild Intel build for comparison.

## 1. Build intel/llvm with the CUDA backend (one clang → spir64 + nvptx)
Match the nightly the Linux build pins (`scripts/build-combined.dockerfile`, `DPCPP_RELEASE`) for kernel
parity. From an x64 Native Tools env (`vcvars64.bat`), with CMake/Ninja/Python/`%CUDA_PATH%` on PATH:
```
git clone --depth 1 --branch nightly-2026-06-13 https://github.com/intel/llvm C:\llvm
python C:\llvm\buildbot\configure.py --cuda --cmake-gen "Ninja" -o C:\llvm\build
python C:\llvm\buildbot\compile.py  -o C:\llvm\build -j <N>
```
Three Windows-only snags (all worked around without touching the SYCL kernels):
- **`UR_USE_EXTERNAL_UMF`** auto-turns ON if oneAPI's UMF is discoverable, linking oneAPI's UMF 1.1
  (no CUDA memory provider) → the CUDA adapter fails `umf::CreateProviderPool`. Reconfigure with
  `cmake C:\llvm\build -DUR_USE_EXTERNAL_UMF=OFF` so UR builds its own CUDA-aware UMF.
- A benign `cmake -E copy ... umfd.lib` step (Debug UMF, never produced) stops ninja; build with
  keep-going (`cmake --build C:\llvm\build -- deploy-sycl-toolchain -k 0 -j <N>`) — the Release
  `clang++.exe`, `sycl9.dll`, `sycl-jit.dll`, and `ur_adapter_{cuda,level_zero,opencl}.dll` all build.
- **UMF CUDA provider requires CUDA VMM** (`cuMemGetAllocationGranularity`), which some cloud L4s report
  as unsupported (`CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED=0`). UMF only uses VMM for a
  min-alignment query; its allocations use `cuMemAlloc`. Patch
  `_deps/unified-memory-framework-src/src/provider/provider_cuda.c` `cu_memory_provider_post_initialize`
  to fall back to `min_alignment = 512` instead of failing when the granularity query is not supported,
  then rebuild `ur_adapter_cuda`. After this, `sycl-ls` shows `[cuda:gpu] NVIDIA ... [CUDA 12.x]`.

## 2. Build sycl.dll (spir64 + nvptx) with the from-source clang
Build `mom.node` + the Intel `sycl.dll` first via the normal MSBuild flow
(`.github/workflows/scripts/build-windows.ps1`, run inside `vcvars64`), then rebuild ONLY `sycl.dll` with
the from-source clang so it carries both device images (mom.node delay-loads it by name; the C-ABI exports
match because both compilers use MSVC mangling). The flags mirror the Linux dpcpp-combined build, plus
Windows-specific `-DNOMINMAX -DWIN32_LEAN_AND_MEAN` and `-std=c++20`:
```
set CLANG=C:\llvm\build\bin\clang++.exe
set F=-std=c++20 -O3 -ffp-contract=off -fsycl-embed-ir -DNDEBUG -DMOM_SYCL_BUILD ^
      -DNOMINMAX -DWIN32_LEAN_AND_MEAN -DMOM_SYCL_HAS_CUDA -DMOM_PEARL_HAS_ESIMD ^
      -fno-strict-aliasing -IC:\mom\xmrig
rem main SYCL TUs (spir64 + nvptx):
%CLANG% -fsycl -fsycl-targets=spir64,nvidia_gpu_sm_80 %F% -c sycl\{lib,ethash,etchash,autolykos2,pearl,c29,cn-gpu,kawpow,blake2b}.cpp
rem ESIMD TU (spir64 only):
%CLANG% -fsycl -fsycl-targets=spir64 %F% -DPEARL_ESIMD -c sycl\pearl_esimd.cpp
rem host helpers the sycl target also needs (sha3/keccak/blake2b):
%CLANG% %F:-fsycl-embed-ir=% -c xmrig\base\crypto\{sha3,keccak}.cpp ; clang -c xmrig\crypto\randomx\blake2\blake2b.c
rem link:
%CLANG% -fsycl -fsycl-targets=spir64,nvidia_gpu_sm_80 -shared *.obj -o build\Release\sycl.dll
```
(See the inline build script used during bring-up for the exact per-file invocation.) The only repo source
change required is `sycl/kawpow_jit.inc`: its CUDA source-JIT used POSIX `dladdr`/`<dlfcn.h>` to locate the
module dir; it now uses `GetModuleHandleEx`/`GetModuleFileName` on Windows (declared directly, NOT via
`<windows.h>`, whose `_Interlocked*` intrinsics clash with MSVC `<atomic>` under clang). The Linux path is
unchanged (verified: B580 `gpu` suite still 7/7).

## 3. Runtime DLLs
`sycl.dll` (clang) links the from-source SYCL runtime, so ship these from `C:\llvm\build\bin` beside
`mom.node`/`sycl.dll` (replacing the oneAPI runtime used by the Intel-only package):
`sycl9.dll`, `sycl-jit.dll` (kawpow JIT), `ur_win_proxy_loader.dll`,
`ur_adapter_cuda.dll`, `ur_adapter_level_zero.dll`, `ur_adapter_opencl.dll`, and `kawpow_device.inc`
(beside the module, for the kawpow JIT). `nvcuda.dll`/`nvml.dll` come from the NVIDIA driver; the CUDA
toolkit (libdevice) must be present for the kawpow JIT (else it falls back to the slower AOT kernel).
