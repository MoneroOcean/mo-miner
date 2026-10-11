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

On Linux, `npm run dev` builds and runs one multicompiler development image. Docker buildx is required:

```bash
git clone https://github.com/MoneroOcean/mo-miner.git
cd mo-miner
MOM_GPU_BACKEND=intel npm run dev -- algorithms
MOM_GPU_BACKEND=intel_igpu npm run dev -- algorithms
MOM_GPU_BACKEND=nvidia npm run dev -- algorithms
MOM_GPU_BACKEND=amd npm run dev -- algorithms
MOM_GPU_BACKEND=opencl npm run dev -- algorithms
```

The Docker implementation lives in `scripts/dev-container.sh`. CI invokes that helper
directly without host npm; developers normally use `npm run dev -- ...`.
The helper can also be invoked directly when host npm is unavailable.

`intel` selects discrete GPUs; `intel_igpu` selects integrated GPUs through the existing portable
worker. Use `npm test -- gpu-integrated` for its correctness suite. Device numbers are preserved
across these selectors; use the `gpuN` reported by the matching `algorithms` command.
The integrated policy uses Level Zero with the standards-only image because older Intel iGPUs
cannot link the discrete worker's ESIMD/DPAS image. No additional release runtime is needed.

Normal runs reuse the installed compiler image and rebuild miner objects only. Missing packaging
tools are added without rebuilding compilers. Set `MOM_REBUILD_DEV_IMAGE=1` after compiler/runtime
Docker changes or when a cached toolchain fails packaging validation.
For a focused kernel iteration, set `MOM_LINUX_BUILD_COMPILER` to `oneapi`, `dpcpp`,
`dpcpp-opencl`, `acpp-cuda`, or `acpp-hip` to rebuild and publish only that compatible worker while
preserving the other cached workers. Omit it (or use `all`) for release-wide validation. For
example, NVIDIA C30 uses:

```bash
MOM_GPU_BACKEND=nvidia MOM_LINUX_BUILD_COMPILER=acpp-cuda npm run dev -- npm test -- gpu-discrete
```

Linux GPU measurements can run concurrently after their workers have been built normally.
Reuse mode does not touch the shared build tree and rejects a stale worker. `npm run dev` derives separate
container and SYCL-cache names from `MOM_GPU_BACKEND`, for example `mom-intel` and `mom-nvidia`:

```bash
MOM_GPU_BACKEND=intel MOM_REUSE_BUILT_WORKER=1 \
  npm run dev -- node tests/run_perf.js nexapow >intel.log 2>&1 &
MOM_GPU_BACKEND=nvidia MOM_REUSE_BUILT_WORKER=1 \
  npm run dev -- node tests/run_perf.js walahash >nvidia.log 2>&1 &
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

Omit `-Compiler` to build every worker required by the selected backend.

### Native release build from a clean checkout

Linux packaging requires a standalone Node.js binary with SEA support, not a
distribution build linked to `libnode`. Use an official Node.js distribution;
`NODE_BIN=/path/to/node` selects the packaging binary when it is not on `PATH`.

Clone this repository and install the development toolchains with the commands above. On Windows,
run `scripts\install-dev.bat -Component all -Jobs 8` from an elevated terminal. Then, from the
checkout on either OS:

```sh
npm install --ignore-scripts
npm run build:release
```

The command detects Linux or Windows, builds all portable workers, and uses the same packagers as
release CI to produce `mom-v<VERSION>-lin.tgz` or `mom-v<VERSION>-win.zip` in the checkout. It does not
use Docker, run tests/mining, publish anything, or require a GPU. Linux native packaging needs the
official Node.js binary installed by `install-dev.sh`, `patchelf`, and `binutils`. Set
`MOM_BUILD_JOBS` to bound compilation; leave `MOM_CPU_MARCH` unset for portable releases.

The initial toolchain installation is substantial: oneAPI, DPC++, CUDA, ROCm/HIP, and AdaptiveCpp
are needed for a unified release. All npm build/test entrypoints share the checkout's `build/` cache,
with separate objects for each OS/compiler/backend, not each command. Source/header edits rebuild
dependent files; compiler, SDK, or build-option changes invalidate
incompatible objects. Each compiler/backend keeps only its current cache; successful builds prune
obsolete outputs and interrupted metadata, preserving other backends' reusable caches. Standard
packaging also discards prior partial archives when replacing its staging directory. Use
`npm run build:release -- --clean` to force a full rebuild. On Linux, `npm run dev`
defaults to Docker and provisions its toolchains from a clean clone; set `MOM_DEV_HOST=1` before npm
to use installed host toolchains instead. The native release command also uses installed host tools.
On Windows, `npm run dev` uses these same installed tools and the `build/win` cache without
packaging a release.
`npm run build:release -- --package-only` packages workers already built on that OS; it does not
compile them again. CI uses this same entrypoint, through `npm run dev` only when it needs the Linux
container's toolchains. Packaging itself has no Docker dependency.

### npm commands by host

| Command                 | Linux host                                                      | Windows host                              |
| ----------------------- | --------------------------------------------------------------- | ----------------------------------------- |
| `npm run dev`           | Incremental Docker build; `MOM_DEV_HOST=1` builds/runs natively | Incremental native build, then launch MoM |
| `npm run build:release` | Native build and TGZ packaging                                  | Native build and ZIP packaging            |
| `npm run check`         | Host lint and typecheck                                         | Host lint and typecheck                   |
| `npm run lint`          | Host ESLint                                                     | Host ESLint                               |
| `npm run typecheck`     | Host TypeScript/JSDoc checks                                    | Host TypeScript/JSDoc checks              |
| `npm test`              | Host checks, then correctness suites in Docker                  | Host checks and correctness suites        |
| `npm run test:perf`     | Hardware benchmarks in Docker                                   | Native hardware benchmarks                |
| `npm run test:deploy`   | Linux containers; optional Windows VM                           | Native Windows release checks             |

`npm test -- logic` and `npm test -- host` stay on the Linux host; the latter needs a C++
compiler. Tests launched through `MOM_DEV_HOST=1 npm run dev -- npm test`, already-containerized
tests, and tests of unpacked releases run directly, without starting another container.
The `npm test -- host` selector is a Linux-only source-check suite, not the native-development switch.

For native Linux development, install the SDKs once, then build and run directly:

```bash
sudo scripts/install-dev.sh --component all --jobs 8
npm install --ignore-scripts
MOM_DEV_HOST=1 npm run dev
MOM_DEV_HOST=1 npm run dev -- algorithms
MOM_DEV_HOST=1 npm run dev -- npm test
```

`MOM_DEV_HOST=1` selects native Linux development; unset, empty, or `0` selects Docker.
The `--` forwards arguments through npm. Native mode uses the same incremental cache as Docker,
but a different Node version or toolchain correctly triggers rebuilding. It installs no SDKs
automatically and does not package a release. For Intel-only development,
`--component base,node,oneapi,dpcpp` avoids the NVIDIA/AMD SDKs. Windows development always runs natively.

Older Docker builds may have left root-owned caches. With all builds stopped, repair those once
from the checkout using `sudo chown -R "$(id -u):$(id -g)" build/cache`. Successful Docker builds
now return cache ownership to the checkout owner.

On Windows, install the build tools once from an elevated terminal, then open a fresh terminal:

```bat
scripts\install-dev.bat -Component all -Jobs 8
npm install --ignore-scripts
npm run dev
```

With no arguments, `npm run dev` launches MoM's CLI help. Arguments normally go directly to MoM:
`npm run dev -- algorithms` lists devices and algorithms, and `npm run dev -- mine config.json`
mines using an existing configuration. Explicit Node.js and npm commands also run in the development
environment, for example `npm run dev -- npm test -- cpu` or `npm run dev -- node tests/run_perf.js etchash`.
Prefix any other command with `--exec`, for example `npm run dev -- --exec bash -c 'echo ready'`.
`MOM_GPU_BACKEND` selects the development backend (`intel` by default; `all` builds every vendor).
Set overrides before npm, for example `MOM_GPU_BACKEND=amd npm run dev -- algorithms` on Linux.
Selected `MOM_*` build, test, and runtime settings are forwarded into the development container;
arbitrary host environment variables are not. On Windows Command Prompt, use
`set "MOM_GPU_BACKEND=amd"` before `npm run dev -- algorithms` for AMD.
GPU drivers must already be installed. Native development always validates its incremental build;
`MOM_REUSE_BUILT_WORKER=1` is only a Docker shortcut, not supported by native Linux or Windows.

## Testing

Run source-only checks directly on the host; `host` is the Linux offline source gate and `logic`
runs the JavaScript logic suites on the current host. Full and hardware tests default to the
Linux development image; set `MOM_DEV_HOST=1` before npm in these `dev` commands to run natively:

```bash
npm test -- host
npm test -- logic
npm run dev -- npm test
MOM_GPU_BACKEND=amd npm run dev -- npm test -- gpu-discrete
MOM_GPU_BACKEND=all npm run dev -- npm test -- gpu-multi
npm run dev -- npm run test:perf
MOM_PERF_SAMPLES=3 npm run dev -- npm run test:perf -- etchash
npm run test:deploy
```

On Windows, run `npm test`, its `logic`/CPU/GPU selectors, and `npm run test:perf` directly
after building the native workers, or through `npm run dev --` to build them first.
The compiler policy discovers `build/win/compilers` automatically; no `MOM_NATIVE_DIR` is needed.

Every `npm test` selector first runs lint and strict typechecking, and all suites guard against
pool-network access. `host` runs the native host/reference fixtures once plus the complete logic,
HooHash, and Pearl claim fixtures without GPU hardware or vendor compiler SDKs. A real Pearl proof
remains opt-in with `MOM_NODE_POWHASH_PATH` and `MOM_PEARLHASH_CLAIM_DEVICE`. `logic` is source-only
and does not build workers.
These selectors do not enter the development container.
The host fixtures need Bash, a C++20 compiler, and Boost headers (`libboost-dev` on Ubuntu).

Default `npm test` runs the complete logic, native-addon, CPU, and GPU vector suites. It groups GPU
checks by algorithm, then by meaningful implementation (`sycl`,
`sycl-native`, and `native` where available). Each backend runs its discrete cases in batches of up
to two; backend groups, generic OpenCL CPU cases, and supported integrated Intel GPUs run
sequentially. Unavailable devices are skipped.
Use `npm test -- cpu`, `-- gpu`, or `-- gpu-discrete` for a focused hardware lane; `gpu-discrete`
is the shorter native-device lane used by targeted compiler builds. `npm test -- gpu-multi`
separately checks two workers on one GPU, all same-vendor discrete GPUs in one job, and two configured
GPU vendors in concurrent processes. CI uses `host` for source quality and separately tests built
release archives. The performance runner accepts any supported algorithm after `--`.

View command descriptions with `npm pkg get scriptDescriptions`; this informational field does not
change npm execution.

`MOM_GPU_BACKEND=intel npm run dev -- npm test -- gpu-fishhash-batch` compares real FishHash/Karlsen mining
batches with the scalar reference, including noninitial nonces and all three header layouts.
Select `intel`, `nvidia`, or `amd`; this opt-in check builds the full DAG and needs at least 6 GiB
VRAM.

Set `MOM_LOOP_STATS=1` for a GPU CPU-load A/B. Each `LOOPSTAT` line reports the compute worker's
process CPU percentage over the same roughly ten-second window as its dispatch timings; normal runs
do not collect these statistics. Compare it together with the steady hashrate on one quiet GPU.

`test:deploy` reuses `build:release`, building each selected OS once. On Linux it uses the
development container and optional Windows VM; on Windows it builds and tests natively, without
Docker or a VM. It then unpacks the archives, runs their installers and complete GPU
vector suites, and benchmarks every supported GPU algorithm available in the test environment.
Verthash generates its data file on first use; `MOM_VERTHASH_DATA` is only an optional path
override. Each rate must reach at least 95% of its README platform value. On a Linux host, Windows
checks use `~/win/run.sh` when it is executable, and otherwise skip cleanly. When selecting these VM
checks, use an interactive terminal to authenticate once before long-running work; noninteractive
refreshes keep authentication fresh while the command runs. Revoked credentials fail clearly rather
than prompt again. Root sessions and supplied `SUDO_PASSWORD` bypass this prompt and refresher.
Failed builds skip their dependent work while remaining failures in the summary. Limit the run with
`MOM_DEPLOY_TARGET` set to `linux`,
`windows`, `linux-intel`, `linux-nvidia`, `linux-amd`, `windows-intel`, `windows-nvidia`, or
`windows-amd`. A native Windows host accepts only `all`, `windows`, and `windows-<vendor>`;
missing GPU vendors are reported as skips, and no Linux checks are attempted there.
After successful Linux checks, retry only Windows with `MOM_DEPLOY_TARGET=windows npm run test:deploy`.
Deployment builds use the same incremental caches. Set `MOM_FORCE_REBUILD=1` for a clean build;
`MOM_DEPLOY_REUSE_ARCHIVE=1` instead tests the existing archive without incorporating source edits.
These controls cannot be combined. Windows caches persist in `build/win` across disposable VMs.
Set `MOM_DEPLOY_DEBUG=1` to retain Windows deployment logs and benchmark evidence under
`build/deploy-win/debug`; normal runs remove their auxiliary logs with the unpacked release.
Set `MOM_DEPLOY_ALGO` to limit both the GPU vector and performance gates to one algorithm. Pure-GPU
cases fail on two consecutive post-setup windows at ≥80% of one CPU core. C29 and C30 remain in
the performance gate but are excluded from the CPU-load gate because they perform host cycle search.
Both OS targets test Intel, NVIDIA, then AMD. Their mixed checks also run two vendors concurrently
on a small CN/GPU correctness test, checking runtime isolation rather than performance.
Linux and VM runs use NVIDIA + AMD; native Windows uses the first two available discrete-GPU
vendors. The mixed check skips without the required devices or when limited to one algorithm.

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
