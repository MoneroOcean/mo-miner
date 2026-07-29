# Development and releases

Generated native artifacts are platform-separated: Linux uses `build/lin`, while Windows uses
`build/win`; reusable compiler object trees live in `build/cache`.
`node-gyp` still creates a temporary top-level `build` workspace during compilation;
the entrypoints temporarily park the persistent tree and restore it after compilation.

## GPU compiler layout

mom uses isolated compiler runtimes selected per operating system, GPU vendor, and algorithm. The
human-readable [GPU-COMPILERS.md](GPU-COMPILERS.md) table is also the runtime configuration source;
changing compilers recreates the affected worker process.

| Compiler family           | Linux scope                                  | Windows scope                        |
| ------------------------- | -------------------------------------------- | ------------------------------------ |
| Intel oneAPI 2026         | Intel default                                | Intel default                        |
| Open-source DPC++ nightly | NVIDIA default; generic OpenCL fallback      | NVIDIA default; OpenCL fallback      |
| AdaptiveCpp generic/SSCP  | Selected NVIDIA overrides; AMD               | NVIDIA overrides; AMD                |

## Development environment

GPU display drivers are host prerequisites. All miner-specific development provisioning is
centralized in `scripts/install-dev.sh` and `scripts\install-dev.bat`:

```bash
sudo scripts/install-dev.sh --component all --jobs 8
sudo scripts/install-dev.sh --component dpcpp-hip --jobs 2
```

```bat
scripts\install-dev.bat -Component all -Jobs 8
scripts\install-dev.bat -Component dpcpp-hip -Jobs 2
```

Linux components are `base`, `node`, `oneapi`, `cuda`, `rocm`, `dpcpp`, `dpcpp-hip`, `acpp-cuda`,
and `acpp-hip`. Windows uses the same layout with `hip` in place of `rocm` and adds an independently
selectable `opencl-cpu` runtime for package tests. The Windows `oneapi` component includes that CPU
runtime. The prebuilt open-source `dpcpp` component carries SPIR-V, CUDA, and the Unified Runtime
OpenCL adapter. AdaptiveCpp generic/SSCP is the release AMD path and JITs for the detected GPU, so
releases contain no architecture-specific AMD code object. The explicitly selected `dpcpp-hip`
component remains available only to reproduce the documented open-DPC++ AMD workaround; `all`
deliberately excludes that experimental targeted build. The installers pin the toolchain versions
used for performance measurements.

On Linux, `r.sh` builds and runs one multicompiler development image. Docker buildx is required:

```bash
git clone https://github.com/MoneroOcean/mo-miner.git
cd mo-miner
MOM_GPU_BACKEND=intel ./r.sh node mom.js algo_params
MOM_GPU_BACKEND=nvidia ./r.sh node mom.js algo_params
MOM_GPU_BACKEND=amd ./r.sh node mom.js algo_params
MOM_GPU_BACKEND=opencl ./r.sh node mom.js algo_params
```

Normal runs reuse the installed compiler image and rebuild miner objects only. Set
`MOM_REBUILD_DEV_IMAGE=1` only after intentionally changing a compiler/runtime Docker stage.

On Windows, the multi-compiler builder accepts a targeted worker selector for fast iteration while
preserving the other workers already under `build\win`:

```powershell
powershell -File .github\workflows\scripts\build-windows-multicompiler.ps1 -Backend intel -Compiler portable
powershell -File scripts\test-windows-current-multicompiler.ps1 -Backend intel -Compiler portable -SkipBuild
```

Omit `-Compiler` for a clean release/CI build of every worker required by the selected backend.

## Testing

The `r.sh` development image is the supported Linux test path:

```bash
./r.sh npm test
./r.sh npm run test:github
MOM_GPU_TEST_VENDORS=amd ./r.sh npm run test:gpu-discrete
MOM_GPU_TEST_VENDORS=nvidia ./r.sh npm run test:gpu-multi
./r.sh npm run test:perf
MOM_PERF_SAMPLES=3 ./r.sh npm run test:perf -- etchash
npm run test:deploy
```

`npm test` groups GPU checks by algorithm, then by meaningful implementation (`sycl`,
`sycl-native`, and `native` where available). Discrete vendors run concurrently; the generic OpenCL
CPU and supported integrated Intel GPUs run sequentially. Unavailable devices are skipped.
`test:gpu-discrete` is the shorter native-device lane used by targeted compiler builds.
`test:gpu-multi` separately checks two workers on one GPU and all same-vendor discrete GPUs in one
job. `test:github` is the fast, network-guarded subset used by hardware-free hosted runners; it
contains no expected device skips. The performance runner accepts any supported algorithm after
`--`.

`test:deploy` builds and unpacks the unified archives, runs their installers and complete GPU
vector suites, then benchmarks every supported GPU algorithm. Each result must reach at least 95%
of its platform value in the README table. Windows checks use `~/win/run.sh` when it exists and
otherwise skip cleanly. Limit the run with `MOM_DEPLOY_TARGET` set to `linux`, `windows`,
`linux-intel`, `linux-nvidia`, `linux-amd`, `windows-intel`, `windows-nvidia`, or `windows-amd`.

For compiler or kernel A/B measurements, produce JSON reports with `scripts/benchmark-gpu-algos.js`
and compare any number of runs in normalized H/s with:

```bash
node scripts/compare-gpu-benchmarks.js baseline.json candidate.json
```

## Release artifacts

Tagged releases produce one unified Linux x86-64 `.tgz` and one unified Windows x86-64 `.zip`.
Each archive includes CPU mining plus Intel, NVIDIA, AMD, and generic OpenCL GPU support. Docker and
Node.js are not required on the target machine; the bundled installer provisions any missing host
runtime and NVIDIA source-JIT tools.

The source-built compiler stages are cached independently so normal two-core GitHub packaging jobs
assemble the miner instead of rebuilding LLVM. Release CI unpacks each final archive, runs the
native CPU suite, and requires one fast portable SYCL/OpenCL CPU vector for every GPU algorithm.
Hosted runners do not create unavailable GPU lanes; real-hardware validation runs those vector
suites per vendor.

For a GPU outside the tuned vendor paths, select the generic SPIR-V fallback with
`MOM_GPU_BACKEND=opencl`.
