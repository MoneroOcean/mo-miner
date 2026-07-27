# Windows multi-compiler development image

The Windows development environment is one consolidated qcow2 overlay,
`win-mom-dev.qcow2`, backed directly by the GPU-driver layer and recreated by running the repository-owned
`scripts\install-dev.bat -Component all`. It contains all compiler-side dependencies and is the
default source-build base for `~/win/run.sh`; `--release` still selects the driver-only base. The host
VM helper owns only Windows installation, drivers, passthrough, and layer lifecycle—miner/compiler
provisioning lives in this repository.
The helper's default is a cold-present `GPU_GROUP=all` VM with Arc, NVIDIA, and AMD attached together,
which lets one build feed consecutive cross-vendor gates without reboot gaps. Set
`GPU_GROUP=intel|nvidia|amd` to isolate a vendor while diagnosing it. For packaging, cleanup, and
other work that does not execute GPU code, use `GPU_GROUP=none` and the VM's virtio display.

The host helper owns the awkward hardware details: dGPUs live on a private desktop seat, passthrough
uses `managed=no`, and the all-GPU boot temporarily presents the NVIDIA card with an 8 GiB BAR1 so
Arc BAR2 plus the other 64-bit BARs fit OVMF's resource layout. VRAM remains 16 GiB and BAR1 returns
to 16 GiB before the Linux NVIDIA driver is rebound. A health gate waits until all three Windows PnP
devices report `CM_PROB_NONE` before it uploads or executes source.

| Compiler family       | Installed path                               | Runtime use                                                  |
| --------------------- | -------------------------------------------- | ------------------------------------------------------------ |
| Intel oneAPI 2026     | `C:\Program Files (x86)\Intel\oneAPI`        | Intel Level Zero default                                     |
| Open-source DPC++     | `C:\Tools\dpcpp`                             | NVIDIA CUDA default; Intel Level Zero comparison/fallback    |
| AdaptiveCpp CUDA      | `C:\Tools\acpp-cuda`                         | NVIDIA algorithm overrides                                   |
| AdaptiveCpp HIP       | `C:\Tools\acpp-amd`                          | AMD default                                                  |

`C:\Tools\mom-toolchains.txt` records the installed paths. The AdaptiveCpp HIP tree is the pinned
post-`c69d230e` develop build with `rt-backend-hip.dll` and generic AMDGPU libkernel bitcode; it is not the
older CPU/CUDA-only nightly by itself.

Build every miner worker in one command:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  .github\workflows\scripts\build-windows-multicompiler.ps1
```

The default `-Backend all` builds every worker once. For an isolated source run, add `-Backend intel`,
`-Backend nvidia`, or `-Backend amd`; Intel emits both oneAPI and a genuine open-source DPC++ SPIR-V
worker, NVIDIA emits the combined SPIR-V/NVPTX DPC++ worker plus AdaptiveCpp CUDA, and AMD emits
AdaptiveCpp HIP. The local helper derives the narrow choice from a single-vendor `GPU_GROUP`, sets
`MOM_NATIVE_DIR` to the isolated worker tree, and skips its automatic prebuild when the requested
PowerShell command already manages the build or tests.

The final local promotion gate deliberately builds once and then tests vendors sequentially:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  scripts\test-windows-unified-release.ps1
```

It runs every compiler's complete current GPU vector suite, checks AdaptiveCpp/CUDA process teardown, creates the one
unified Windows zip, then tests the extracted archive through CPU, a real SYCL-CPU dependency smoke,
and Intel, NVIDIA, and AMD GPU paths. `-SkipBuild` reuses an already validated `build\win\compilers` tree;
`-SkipCompilerGates` is for a packaging-only rerun after those exact worker binaries already passed.

The output is deliberately isolated:

```text
build\win\compilers\
  oneapi\       mom.node, sycl.dll, oneAPI runtime closure
  dpcpp\        mom.node, sycl.dll, DPC++ Level Zero/CUDA runtime
  acpp-cuda\    mom.node, sycl.dll, CUDA + required OMP host plugins, bitcode, relocatable opt/llc
  acpp-hip\     mom.node, sycl.dll, HIP + required OMP host plugins, device bitcode, ROCm opt/llc
```

`package-windows.ps1` preserves those directories. `mom.cmd` auto-detects a single GPU vendor and
loads its default addon before `algo_params` discovery; `GPU-COMPILERS.md` then selects each worker
before it is spawned, so incompatible `sycl9.dll`/AdaptiveCpp runtimes never enter the same process.
Set `MOM_GPU_BACKEND=intel|nvidia|amd` on a mixed-vendor host.

Release builds always discard a copied `build/` tree before compiling, preventing a Linux ELF addon
from surviving an incremental Windows build. Packaging computes each worker's DLL closure separately.
The DPC++ release snapshot contains only its worker and production runtimes: compiler executables and
debug adapters are stripped even when packaging an older cached build directory.
`kawpow_device.inc` is copied beside the isolated DPC++ worker because the production CUDA
source-JIT resolves it relative to the loaded module.
The AMD worker additionally carries the HIPRTC-builtins and versioned COMGR DLL used by AdaptiveCpp's
HIP backend; `amdhip64_7.dll` deliberately comes from the installed display driver, because loading a bundled
SDK runtime alongside the driver runtime can crash the process.

For a fresh machine, `scripts\install-dev.bat -Component all` installs the common build tools, Node,
oneAPI, CUDA/HIP compiler SDKs, open DPC++, and both AdaptiveCpp trees without touching GPU display
drivers. `-Component dpcpp`, `acpp-cuda`, or `acpp-hip` provides the same narrow staging used by CI.
`-Component dpcpp-hip` is retained as an explicit research/reproducer setup and is not part of `all`
or release artifacts.
CI caches the independently selected trees and SDKs; `validate-windows-toolchains.ps1` checks each
backend against its passthrough GPU.
