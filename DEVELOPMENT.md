# Development and releases

Generated native artifacts are platform-separated: Linux uses `build/lin`, while Windows uses
`build/win`; reusable compiler object trees live in `build/cache`.
`node-gyp` still creates a temporary top-level `build` workspace during compilation;
the entrypoints temporarily park the persistent tree and restore it after compilation.

## GPU compiler layout

mom uses isolated compiler runtimes selected per operating system, GPU vendor, and algorithm. The
human-readable [GPU-CONFIG.md](GPU-CONFIG.md) tables are the runtime configuration source;
changing compilers recreates the affected worker process.

| Compiler family           | Linux scope                                  | Windows scope                        |
| ------------------------- | -------------------------------------------- | ------------------------------------ |
| Intel oneAPI 2026         | Intel default                                | Intel default                        |
| Open-source DPC++         | NVIDIA default; generic OpenCL fallback      | NVIDIA default; OpenCL fallback      |
| AdaptiveCpp generic/SSCP  | Selected NVIDIA overrides; AMD               | NVIDIA overrides; AMD                |

Linux pins a nightly DPC++ payload. Windows restores a SHA256-pinned SDK built from the
`intel/llvm` `v7.1.1` source tag; its compiler reports `7.1.0 (pre-release)`. The source tag and
archive name identify the build, not a different compiler banner.

### Runtime compatibility

These rules are implemented in the capability and memory checks, not parsed from this table.

| Condition         | Compiler  | Backend exceptions                        |
| ----------------- | --------- | ----------------------------------------- |
| NVIDIA SM < 80    | acpp-cuda | `cn/gpu`, `octopus`, `nexapow`: `sycl`    |
| NVIDIA SM 61–79   | acpp-cuda | `pearlhash`: `sycl-native`                |
| NVIDIA SM < 61    | acpp-cuda | `pearlhash`, `walahash`: `sycl`           |
| NVIDIA SM unknown | acpp-cuda | same exceptions as SM < 61                |
| NVIDIA < 8 GiB    | unchanged | compact automatic `pearlhash` profile     |
| Generic OpenCL    | SPIR-V IL | required by `sycl-opencl`                 |

NVIDIA devices below SM80, or with unknown capability, use the existing architecture-neutral
AdaptiveCpp CUDA worker. Keep its CUDA 12.x build toolchain:
[CUDA 13 removed pre-Turing offline build support](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html#deprecated-architectures).
Maxwell (SM50) is the preliminary compatibility baseline; older-device execution still
needs qualification. The modern DPC++ worker retains its single SM80 target and measured kernels.
Pearl/Octopus int8 MMA plus asynchronous copies require SM80; WalaHash DP4A requires SM61.
Those optional paths check the selected device and retain ordinary SYCL on missing capabilities.
An explicit `MOM_NATIVE_PATH` is binding: a missing addon fails instead of loading another worker.

AMD uses the same generic/SSCP worker on every architecture. Pearl, Octopus, and WalaHash select
their RDNA4 int8 WMMA kernels only for detected `gfx1200`/`gfx1201` devices; other or unknown targets
retain ordinary SYCL. Octopus also checks wave32 support before its cooperative fallback.
This avoids RX 9060 XT-specific binaries but does not replace older-GPU runtime qualification.

GPU work starts with a complete generic `sycl` implementation. Prefer `sycl-native` only when
vendor extensions or target-specific tuning give a repeatable, meaningful same-GPU performance
gain, and prefer `native` only when it improves materially again. Native kernels must be compiled
for the detected device at run time rather than shipped as architecture-specific binaries. Every
`sycl-native` or `native` path must retain a bit-exact `sycl` or `sycl-native` fallback and select
it automatically when required capabilities, compilation or loading are unavailable. A submitted
kernel/device fault is not unavailability: abort rather than retry on a potentially invalid queue.

Among toolchains capable of the target backend, prefer Intel oneAPI DPC++ 2026, then pinned open-source
DPC++, then AdaptiveCpp. A lower-priority compiler is an algorithm-specific override only
when counterbalanced measurements show a meaningful gain or a higher-priority compiler cannot
build or run the implementation reliably. Keep the fallback implementation in the same vector
suite so an optimized path cannot become the algorithm's only working path.

## Development environment

GPU display drivers are host prerequisites. All miner-specific development provisioning is
centralized in `scripts/install-dev.sh` and `scripts\install-dev.bat`:

```bash
sudo scripts/install-dev.sh --component all --jobs 8
```

```bat
scripts\install-dev.bat -Component all -Jobs 8
```

Linux components are `base`, `node`, `oneapi`, `cuda`, `rocm`, `dpcpp`, `acpp-cuda`,
and `acpp-hip`. Windows uses the same layout with `hip` in place of `rocm` and adds an independently
selectable `opencl-cpu` runtime for package tests. The Windows `oneapi` component includes that CPU
runtime. The prebuilt open-source `dpcpp` component carries SPIR-V, CUDA, and the Unified Runtime
OpenCL adapter. AdaptiveCpp generic/SSCP is the release AMD path and JITs for the detected GPU, so
releases contain no architecture-specific AMD code object. The installers pin the toolchain
versions used for performance measurements.

On Linux, `r.sh` builds and runs one multicompiler development image. Docker buildx is required:

```bash
git clone https://github.com/MoneroOcean/mo-miner.git
cd mo-miner
MOM_GPU_BACKEND=intel ./r.sh node mom.js algorithms
MOM_GPU_BACKEND=nvidia ./r.sh node mom.js algorithms
MOM_GPU_BACKEND=amd ./r.sh node mom.js algorithms
MOM_GPU_BACKEND=opencl ./r.sh node mom.js algorithms
```

Normal runs reuse the installed compiler image and rebuild miner objects only. Set
`MOM_REBUILD_DEV_IMAGE=1` only after intentionally changing a compiler/runtime Docker stage.
For a focused kernel iteration, set `MOM_LINUX_BUILD_COMPILER` to `oneapi`, `dpcpp`,
`dpcpp-opencl`, `acpp-cuda`, or `acpp-hip` to rebuild and publish only that compatible worker while
preserving the other cached workers. Omit it (or use `all`) for release-wide validation. For
example, NVIDIA C30 uses:

```bash
MOM_GPU_BACKEND=nvidia MOM_LINUX_BUILD_COMPILER=acpp-cuda ./r.sh npm run test:gpu-discrete
```

Linux GPU measurements can run concurrently after their workers have been built normally.
Reuse mode does not touch the shared build tree and rejects a stale worker. `r.sh` derives separate
container and SYCL-cache names from `MOM_GPU_BACKEND`, for example `mom-intel` and `mom-nvidia`:

```bash
MOM_GPU_BACKEND=intel MOM_REUSE_BUILT_WORKER=1 \
  ./r.sh node tests/run_perf.js nexapow >intel.log 2>&1 &
MOM_GPU_BACKEND=nvidia MOM_REUSE_BUILT_WORKER=1 \
  ./r.sh node tests/run_perf.js walahash >nvidia.log 2>&1 &
wait
```

Normal build-capable runs remain exclusive. Reuse runs share the build lock; GPU occupancy and
ownership must be coordinated from fresh device evidence before launch. No advisory GPU lock is
used. Bound concurrent measurements to the host's power and cooling capacity, and compare rates
only when other workloads do not contend for the measured device or its CPU support.

On Windows, the multi-compiler builder accepts a targeted worker selector for fast iteration while
preserving the other workers already under `build\win`:

```powershell
powershell -File .github\workflows\scripts\build-windows-multicompiler.ps1 `
  -Backend intel -Compiler portable
powershell -File scripts\test-windows-current-multicompiler.ps1 `
  -Backend intel -Compiler portable -SkipBuild
```

Omit `-Compiler` for a clean release/CI build of every worker required by the selected backend.

## Testing

The `r.sh` development image is the supported Linux test path:

```bash
./r.sh npm test
./r.sh npm run test:github
MOM_GPU_BACKEND=amd ./r.sh npm run test:gpu-discrete
MOM_GPU_BACKEND=all ./r.sh npm run test:gpu-multi
./r.sh npm run test:perf
MOM_PERF_SAMPLES=3 ./r.sh npm run test:perf -- etchash
npm run test:deploy
```

`npm test` groups GPU checks by algorithm, then by meaningful implementation (`sycl`,
`sycl-native`, and `native` where available). Each backend runs its discrete cases in batches of up
to two; backend groups, generic OpenCL CPU cases, and supported integrated Intel GPUs run
sequentially. Unavailable devices are skipped.
`test:gpu-discrete` is the shorter native-device lane used by targeted compiler builds.
`test:gpu-multi` separately checks two workers on one GPU, all same-vendor discrete GPUs in one job,
and two configured GPU vendors in concurrent processes. `test:github` is the fast, network-guarded
subset used by hardware-free hosted runners; it contains no expected device skips. The performance
runner accepts any supported algorithm after `--`.

`MOM_GPU_BACKEND=intel ./r.sh npm test -- gpu-fishhash-batch` compares real FishHash/Karlsen mining
batches with the scalar reference, including noninitial nonces and all three header layouts.
Select `intel`, `nvidia`, or `amd`; this opt-in check builds the full DAG and needs at least 6 GiB
VRAM.

Set `MOM_LOOP_STATS=1` for a GPU CPU-load A/B. Each `LOOPSTAT` line reports the compute worker's
process CPU percentage over the same roughly ten-second window as its dispatch timings; normal runs
do not collect these statistics. Compare it together with the steady hashrate on one quiet GPU.

`test:deploy` builds and unpacks the unified archives, runs their installers and complete GPU
vector suites, then benchmarks every supported GPU algorithm available in the test environment.
Verthash generates its data file on first use; `MOM_VERTHASH_DATA` is only an optional path
override. Each rate must reach at least 95% of its README platform value. Windows checks use
`~/win/run.sh` when it is executable, and otherwise skip cleanly. Limit the run with
`MOM_DEPLOY_TARGET` set to `linux`,
`windows`, `linux-intel`, `linux-nvidia`, `linux-amd`, `windows-intel`, `windows-nvidia`, or
`windows-amd`.
Set `MOM_DEPLOY_ALGO` to limit both the GPU vector and performance gates to one algorithm. Pure-GPU
cases fail on two consecutive post-setup windows at ≥80% of one CPU core. C29 and C30 remain in
the performance gate but are excluded from the CPU-load gate because they perform host cycle search.
The complete Windows target also runs NVIDIA and AMD package workers concurrently.

For compiler or kernel A/B measurements, produce JSON reports with `scripts/benchmark-gpu-algos.js`
and compare any number of runs in normalized H/s with:

```bash
node scripts/compare-gpu-benchmarks.js baseline.json candidate.json
```

## Release artifacts

Tagged releases produce one unified Linux x86-64 `.tgz` and one unified Windows x86-64 `.zip`.
Each archive includes CPU mining plus Intel, NVIDIA, AMD, and generic OpenCL GPU support. Docker and
Node.js are not required on the target machine, and application runtimes are bundled. Display
drivers remain host prerequisites; the installer checks the host and provisions supported missing
dependencies and NVIDIA source-JIT tools.

The source-built compiler stages are cached independently so normal two-core GitHub packaging jobs
assemble the miner instead of rebuilding LLVM. Release CI unpacks each final archive, runs the
native CPU suite, and requires one fast portable SYCL/OpenCL CPU vector for every GPU algorithm.
Hosted runners do not create unavailable GPU lanes; real-hardware validation runs those vector
suites per vendor.

Windows CI and development provisioning restore the SHA256-pinned
`dpcpp-cuda-win-v7.1.1.tar.gz` asset from the `toolchain-win-dpcpp-cuda` release. When changing
that pin, publish the matching asset before relying on a cold build.

For a GPU outside the tuned vendor paths, select the generic SPIR-V fallback with
`MOM_GPU_BACKEND=opencl`.
