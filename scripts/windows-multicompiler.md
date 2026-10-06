# Windows multi-compiler development image

The Windows development environment is `win-mom-dev-base.qcow2`, a child of the
host's GPU-driver base. Recreate it from the repository root with:

```bash
~/win/run.sh --rebuild-dev scripts/install-dev.bat -- -Component all -Jobs 8
```

The host helper creates a temporary child of the GPU-driver layer, runs the
explicitly supplied repository installer, validates the overlay, then atomically
replaces the development base. It remains a QCOW2 child so the Windows, OS, and
GPU-driver storage is shared. `win-mom-release-base.qcow2` is an independent
sibling provisioned with `scripts/install.bat`; it is selected by `--release`.
The host VM helper owns only Windows installation, policy, drivers, passthrough,
and layer lifecycle. Miner/compiler provisioning stays in this repository.

The safe ordinary default is `GPU_GROUP=intel`. Set
`GPU_GROUP=intel|arc|nvidia|amd` for a single cold-present vendor; `arc` is an alias for `intel`.
For provisioning,
packaging, cleanup, and other work that does not execute GPU code, use
`GPU_GROUP=none` and the VM's virtio display. `GPU_GROUP=all` remains an explicit
diagnostic mode rather than the default.

The host helper owns the awkward hardware details: dGPUs live on a private desktop seat, passthrough
uses `managed=no`, and the all-GPU boot temporarily presents the NVIDIA card with an 8 GiB BAR1 so
Arc BAR2 plus the other 64-bit BARs fit OVMF's resource layout. VRAM remains 16 GiB and BAR1 returns
to 16 GiB before the Linux NVIDIA driver is rebound. A health gate checks each selected GPU before
it uploads or executes source.

| Compiler family       | Installed path                               | Runtime use                                                  |
| --------------------- | -------------------------------------------- | ------------------------------------------------------------ |
| Intel oneAPI 2026     | `C:\Program Files (x86)\Intel\oneAPI`        | Intel Level Zero default                                     |
| Open-source DPC++     | `C:\Tools\dpcpp`                             | NVIDIA CUDA default; Intel Level Zero comparison/fallback    |
| AdaptiveCpp CUDA      | `C:\Tools\acpp-cuda`                         | NVIDIA algorithm overrides                                   |
| AdaptiveCpp HIP       | `C:\Tools\acpp-amd`                          | AMD default                                                  |

`C:\Tools\mom-toolchains.txt` records the installed paths. The AdaptiveCpp HIP tree is a pinned build
with `rt-backend-hip.dll` and generic AMDGPU libkernel bitcode.

Build every miner worker in one command:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  .github/workflows/scripts/build-windows-multicompiler.ps1
```

The default `-Backend all` builds every worker once. For an isolated source run, add `-Backend intel`,
`-Backend nvidia`, `-Backend amd`, or `-Backend opencl`. Intel emits oneAPI and open-source DPC++
SPIR-V workers; NVIDIA emits the combined SPIR-V/NVPTX DPC++ worker plus AdaptiveCpp CUDA; AMD emits
AdaptiveCpp HIP. The local helper derives the narrow choice from a single-vendor `GPU_GROUP`, sets
`MOM_NATIVE_DIR` to the isolated worker tree, and skips its automatic prebuild when the requested
PowerShell command already manages the build or tests.

Build the archive once in the development VM, then test the extracted package in the runtime VM
for each vendor:

```bash
GPU_GROUP=none ~/win/run.sh --download build/win -- \
  powershell -NoProfile -ExecutionPolicy Bypass -File \
  .github\workflows\scripts\build-windows-multicompiler.ps1
GPU_GROUP=none ~/win/run.sh --download build/win --download mom-v0.9.0-win.zip -- \
  powershell -NoProfile -ExecutionPolicy Bypass -Command \
  'npm install --ignore-scripts --no-audit --no-fund; if ($LASTEXITCODE) { exit $LASTEXITCODE }; & .github/workflows/scripts/package-windows.ps1'
mkdir -p build/win-package
ln -f mom-v0.9.0-win.zip build/win-package/mom-v0.9.0-win.zip
GPU_GROUP=none ~/win/run.sh --release --download build/win-package -- \
  powershell -NoProfile -ExecutionPolicy Bypass -File \
  .github/workflows/scripts/test-release-windows.ps1 -Archive build/win-package/mom-v0.9.0-win.zip -Suite cpu
GPU_GROUP=none ~/win/run.sh --release --download build/win-package -- \
  powershell -NoProfile -ExecutionPolicy Bypass -File \
  .github/workflows/scripts/test-release-windows.ps1 -Archive build/win-package/mom-v0.9.0-win.zip -Suite gpu-portable-cpu
GPU_GROUP=intel ~/win/run.sh --release --download build/win-package -- \
  powershell -NoProfile -ExecutionPolicy Bypass -File \
  .github/workflows/scripts/test-release-windows.ps1 -Archive build/win-package/mom-v0.9.0-win.zip -Suite gpu
GPU_GROUP=nvidia ~/win/run.sh --release --download build/win-package -- \
  powershell -NoProfile -ExecutionPolicy Bypass -File \
  .github/workflows/scripts/test-release-windows.ps1 -Archive build/win-package/mom-v0.9.0-win.zip -Suite gpu
GPU_GROUP=amd ~/win/run.sh --release --download build/win-package -- \
  powershell -NoProfile -ExecutionPolicy Bypass -File \
  .github/workflows/scripts/test-release-windows.ps1 -Archive build/win-package/mom-v0.9.0-win.zip -Suite gpu
```

Run the build and packaging commands in the development VM. The release VM is runtime-only and is
for testing the resulting archive. Use `-Suite cpu` and `-Suite gpu-portable-cpu` without a passed
through GPU, then `-Suite gpu` once for each Intel, NVIDIA, and AMD passthrough group. The package
test checks the archive layout, launcher, bundled dependencies, CPU paths, and selected GPU vectors.

The output is deliberately isolated:

```text
build\win\compilers\
  oneapi\       mom.node, sycl.dll, oneAPI runtime closure
  dpcpp\        mom.node, sycl.dll, DPC++ Level Zero/CUDA runtime
  dpcpp-opencl\ mom.node, sycl.dll, standards-only DPC++ OpenCL runtime
  acpp-cuda\    mom.node, sycl.dll, CUDA + required OMP host plugins, bitcode, relocatable opt/llc
  acpp-hip\     mom.node, sycl.dll, HIP + required OMP host plugins, device bitcode, ROCm opt/llc
```

`.github/workflows/scripts/package-windows.ps1` preserves those directories. `mom.cmd` auto-detects
a single GPU vendor and loads its default addon before `algorithms` discovery. `GPU-CONFIG.md` then
selects each worker before it is spawned, so incompatible `sycl9.dll`/AdaptiveCpp runtimes never
enter the same process.
Set `MOM_GPU_BACKEND=intel|nvidia|amd|opencl` on a mixed-vendor host.

Release builds move the copied `build/` tree aside, compile into a clean Windows tree, then restore
the preserved caches and replace only `build/win`; this prevents a Linux ELF addon from surviving an
incremental Windows build. Packaging computes each worker's DLL closure separately.
The DPC++ release snapshot contains only its worker and production runtimes; compiler executables and
debug adapters are stripped even when packaging an older cached build directory.
`sycl/kawpow/{device,keccak}.inc` is copied beside the isolated DPC++ worker because the production
CUDA source-JIT resolves it relative to the loaded module. The AMD worker also carries the
HIPRTC-builtins and versioned COMGR DLL used by AdaptiveCpp's HIP backend. `amdhip64_7.dll` comes from
the installed display driver because mixing a bundled SDK runtime with the driver runtime can crash.

For a fresh machine, `scripts\install-dev.bat -Component all` installs the common build tools, Node,
oneAPI, CUDA/HIP compiler SDKs, open DPC++, and both AdaptiveCpp trees without touching GPU display
drivers. `-Component dpcpp`, `acpp-cuda`, or `acpp-hip` provides the same narrow staging used by CI.
CI caches the source-built AdaptiveCpp trees; DPC++ is restored from its SHA256-pinned release asset.
`scripts\validate-windows-toolchains.ps1` checks the Intel, CUDA, and HIP toolchains against their
passthrough GPUs.
