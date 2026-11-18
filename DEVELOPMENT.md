# Development and native builds

This guide covers developer workflows for the `mo-miner` repository: Engineering Mode, tests, and
the standalone release build. Terms follow [README.md](README.md): *Engineering Mode* runs the
JavaScript controller from a checkout, the *Engineering archive* is the legacy Node.js package, and
the *standalone release* is the user-facing native `mom` package. GPU routing details live in
[GPU-CONFIG.md](GPU-CONFIG.md); the Windows development VM workflow is in
[scripts/windows-multicompiler.md](scripts/windows-multicompiler.md).

## Requirements and setup

Use the Node.js and npm versions declared by package.json (Node.js 24.15 or newer, npm 11.10 or
newer). Install the native development prerequisites for the components you intend to build.

Linux:

~~~sh
sudo ./scripts/install-dev.sh --component all --jobs 8
~~~

Windows, from an elevated PowerShell or Command Prompt:

~~~powershell
scripts\install-dev.bat -Component all -Jobs 8
~~~

Components can be selected individually (comma-separated or repeated). `scripts/install-dev.sh
--help` prints its options and component list: `base node oneapi cuda rocm dpcpp acpp-cuda
acpp-hip all`; `--validate-only` checks components without installing. The Windows installer has no
usage output; its components are `base node oneapi opencl-cpu cuda hip dpcpp acpp-cuda acpp-hip
perl all` (see the `param` block and `$valid` list in `scripts/install-dev.ps1`), and
`-ValidateOnly` checks without installing. Both accept a job count (`--jobs N` / `-Jobs N`, default
from `MOM_BUILD_JOBS` or the CPU count). Installing every vendor SDK is optional if you only edit
JavaScript or run source-only tests. GPU drivers are never installed. Keep vendor toolchains and
driver installations separate from project outputs.

The Windows `oneapi` component also installs the pinned CPU OpenCL runtime (`opencl-cpu` installs
only that runtime). On Linux the CPU runtime is installed separately by
`bash scripts/install-opencl-cpu.sh`. Windows static OpenSSL source builds additionally need native
Perl: the `perl` component installs a checksum-pinned Strawberry runtime (included in `all`)
without its C/C++ toolchain. It is a build prerequisite, not a miner dependency; ordinary
Engineering Mode does not need it.

Install the JavaScript dependencies, then run the source checks:

~~~sh
npm ci --ignore-scripts
npm run check
~~~

Skipping install scripts avoids an implicit node-gyp build; `npm run dev` builds native workers
explicitly. `npm run check` runs ESLint and strict JavaScript/JSDoc type checking; it does not
build or test native code.

## Engineering Mode

Engineering Mode runs the JavaScript controller under Node.js with the maintained native
workers/addons. JavaScript-only edits need no native rebuild; native C++/SYCL edits rebuild
affected inputs incrementally.

~~~sh
npm run dev -- algorithms
npm run dev -- mine config.json
~~~

On Linux, `npm run dev` uses the development Docker image by default (`scripts/dev-container.sh`),
which keeps the toolchain consistent and persists outputs through the repository's build
directories. To use SDKs installed on the host (Linux x86-64 only):

~~~sh
MOM_DEV_HOST=1 npm run dev -- algorithms
~~~

On Windows x64, the command first runs the native multicompiler builder, then runs MoM (or the
requested Node/npm command) using the Windows build tree:

~~~powershell
npm run dev -- npm test -- logic
npm run dev -- --exec where.exe node
~~~

`--exec` is for commands other than MoM, Node, or npm; do not use it to bypass the build.

### Selecting a GPU backend

`MOM_GPU_BACKEND` selects which workers `npm run dev` builds and runs: `intel`, `intel_igpu`,
`nvidia`, `amd`, `opencl`, or `all` (default `intel`). `all` is accepted only by the development
and build tooling; the standalone `mom` and Engineering archive launchers accept the first five.

~~~sh
MOM_GPU_BACKEND=nvidia npm run dev -- algorithms
~~~

Backend names select the compiler/factory configuration; they do not prove the hardware is
available. See [GPU-CONFIG.md](GPU-CONFIG.md) for the routing matrix and runtime requirements.
On Linux, `MOM_LINUX_BUILD_COMPILER` (`all`, `oneapi`, `dpcpp`, `dpcpp-opencl`, `acpp-cuda`,
`acpp-hip`) narrows the compiler family and must be compatible with the backend.
`MOM_REUSE_BUILT_WORKER=1` is a Docker-worker shortcut and is rejected by native development.

### Incremental outputs and caches

Linux outputs go to build/lin, Windows outputs to build/win, and reusable compiler objects to
build/cache. node-gyp temporarily creates a top-level build tree; the entrypoints park and restore
the persistent tree around it.

The shared native object cache is build-time state, not a runtime dependency. Keys include source
and included-header bytes, compiler/tool identity, build environment and flags. Changed inputs
rebuild affected objects; final links, audits and receipts are produced for every build. Controls:

- `MOM_NATIVE_NINJA=/absolute/path/to/ninja` selects a specific Ninja executable.
- `MOM_NATIVE_CACHE=0` disables shared object reuse (release/component builds).
- `MOM_FORCE_REBUILD=1` forces incremental outputs to rebuild.
- `MOM_BUILD_JOBS` bounds producer parallelism where supported.

Windows Core has a separate validated MSBuild cache at build/cache/native-windows-core/v1; the
application and GPU object batches use the shared object cache. Incomplete or stale incremental
state takes the conservative rebuild path. Do not delete or modify cache state during a build.

## Tests and runtime checks

`npm test` runs `npm run check` and then `tests/run-hash.js`; pass a suite after `--`. Start with
the source-only suites, which need no GPU:

~~~sh
npm test -- logic
npm test -- host    # Linux: source-built host configuration; needs Bash, a C++ compiler, Boost
npm test -- cpu     # CPU correctness
~~~

To run a suite under the selected Engineering build, go through the development entrypoint:

~~~sh
npm run dev -- npm test -- logic
~~~

The early hashing gate used by CI (CPU vectors, addon contracts and all portable SYCL CPU
algorithms) needs the CPU OpenCL runtime but no GPU; no standalone executable is built:

~~~sh
MOM_DEV_HOST=1 MOM_PORTABLE_BUILD=1 MOM_GPU_BACKEND=opencl MOM_LINUX_BUILD_COMPILER=dpcpp-opencl \
  MOM_OPENCL_DEVICE_TYPE=cpu ONEAPI_DEVICE_SELECTOR=opencl:cpu MOM_REQUIRE_PORTABLE_CPU_TESTS=1 \
  npm run dev -- node tests/run-hash.js engineering
~~~

Hardware suites are explicit selectors: `gpu`, `gpu-discrete`, `gpu-integrated`, `gpu-multi`,
`gpu-portable-cpu` and `gpu-fishhash-batch`. Run them only when the devices and drivers are
available and not in use; they are not substitutes for one another (portable CPU coverage is not
GPU coverage, and a skipped device is not a pass). No suite connects to a mining pool. For bounded
integrated-GPU checks set `MOM_INTEGRATED_PORTABLE_TEST=1` with `gpu-integrated` (compact
fixtures; some exercise generation or recovery substeps rather than full solvers), and
`MOM_REQUIRE_INTEGRATED_GPU_TESTS=1` to fail when no integrated GPU exists.

`npm run test:perf -- <algorithm>` measures performance on the available hardware; it is not a
correctness gate. `npm run test:deploy` builds and tests the Engineering archives, including GPU
correctness and README performance checks. Stop unrelated mining workloads before benchmarking.

The standalone release checker is separate from the test runner. It takes the `entrypoint.json`
build receipt of a build output and runs `audit`, `cpu`, `cli`, `gpu-portable-cpu` or `gpu` lanes:

~~~sh
node scripts/check-native-release.js /output/entrypoint.json cpu
node scripts/check-native-release.js /output/entrypoint.json cli
node scripts/check-native-release.js /output/entrypoint.json gpu --vendor intel --algo pearlhash
~~~

The GPU lane needs the named hardware, fails on missing devices, and does not build anything. A
custom GPU harness must unset `MOM_SKIP_SYCL_ALGO_PARAMS` (the CPU-probe flag checks presence, so
`0` still counts as set); the checker removes it for its GPU lanes. Run native builds and checks
in a resource-limited environment with swap disabled where practical; command timeouts and job
counts are not memory limits.

The standalone release does not load the MSR kernel module: it uses an already-enabled interface
(`allow_writes=on` or `default`) or an openable MSR device, otherwise MSR tuning is skipped.
Engineering Mode keeps its existing fallback; `MOM_SKIP_MSR=1` disables MSR use in CI.

## Native SDK inputs and builds

Standalone release builds require an explicit SDK descriptor and a new output directory. The
descriptor selects the application source, compiler/toolchain, runtime libraries, GPU component
receipts or source builds, and optional distribution data. The producer validates these inputs and
records their hashes; it does not provision a complete SDK on its own.

### Pinned source preparation

Acquisition and compilation are separate stages. `fetch-native-sdk-sources.js` fetches pinned Git
cohorts or source archives only with `--allow-network`, into a new output directory:

~~~sh
node scripts/fetch-native-sdk-sources.js compiler /absolute/path/to/git /new/compiler-sources --allow-network
node scripts/fetch-native-sdk-sources.js build /new/build-sources --allow-network
~~~

Archive cohorts have offline verifiers (`verify-build`, `verify-io`, `verify-zstd`,
`verify-sysroot`, `verify-libdrm`, `verify-nvrtc-*`):

~~~sh
node scripts/fetch-native-sdk-sources.js verify-build /absolute/path/to/build-sources
~~~

For SDK Git exports, `prepare-native-sdk-sources.js` materializes the pinned roots and maintained
patches from a roots descriptor (targets `linux`, `windows`, `compiler`, `acpp`):

~~~sh
node scripts/prepare-native-sdk-sources.js compiler /absolute/source-roots.json /new/compiler-sources
~~~

Other components have their own producers and input descriptors. Keep upstream revisions, patch
set, tool hashes and generated receipts together. Do not edit vendor checkouts in place, copy a
prebuilt library into a receipt, or rewrite a receipt to make stale bytes look current. After
acquisition, verify receipts and build offline where toolchains permit; give compilers and SDK
builders a fresh host-enforced memory cap, read-only source/SDK mounts and a distinct writable
output/cache directory. An SDK receipt is evidence for its named component only.

### Building a standalone release

On Linux x64 the no-argument entrypoint prepares pinned source-built SDKs, reuses admitted
incremental outputs, then builds, packs, audits and packages the executable:

~~~sh
npm run build:release
~~~

It uses `~/cache/mom/native-sdk`, an exclusive build lock, and an enforced aggregate memory limit
with no swap. `MOM_BUILD_JOBS` (1..16, default 2) sets the local build parallelism and sizes that
limit at 4 GiB plus 3 GiB per job beyond two, refusing a limit above host memory; SDKs cached with
another job count are reused. The archive and its SHA256 sidecar are copied into the checkout only after the
audits pass; successful private outputs are removed, and a failed one occupies one `release-work`
slot to inspect before retrying. The build installs no system packages: pinned Node, Intel
2026.2, CUDA and OS development prerequisites (plus `dpkg-deb` and `meson`) must already be
present, and missing ones fail before SDK acquisition. It is a build command, not a runtime test
or publication approval.

On Windows x64 the same command uses native Windows support producers together with an installed
WSL2 distribution for Linux source preparation and the remaining stages. Select the distribution
when more than one is installed:

~~~powershell
$env:MOM_WSL_DISTRO = "<installed WSL2 distribution>"
npm run build:release
~~~

The Windows side needs Node.js 24.15.0, Intel oneAPI 2026.2, Visual Studio C++ Build Tools,
Python and the pinned native Perl. The WSL2 distribution needs systemd user scopes, the pinned
Linux Node.js 24.15.0, the Intel toolchain and build tools. Nothing is installed or reconfigured
for you, and Linux dependencies stay separate from Windows `node_modules`. The WSL ext4 cache is
`~/cache/mom/native-sdk`; Windows support and export caches live under
`%LOCALAPPDATA%\MoM\native-wsl-windows\<checkout-hash>\{support,exports,bridge}`. One lock spans
the WSL build graph; its memory limit and disabled swap cover the Linux process tree only, and
native Windows stages run sequentially with two build jobs. Failed application output occupies one
`windows-release-work` slot, and failed bridge state blocks retry until inspected.

For advanced builds, pass a reviewed descriptor and a new absolute output directory:

~~~sh
npm run build:release -- --native /absolute/sdk.json /new/absolute/output
~~~

The equivalent environment selection is `MOM_NATIVE_SDK` plus `MOM_NATIVE_OUTPUT`. UPX packing
needs both `MOM_NATIVE_UPX` and `MOM_NATIVE_UPX_SHA256`, or `--pack /absolute/upx <sha256>`;
partial settings fail closed. Without UPX the output holds the raw executable and an
`entrypoint.json` build receipt; with UPX it holds the packed executable and receipt, and the raw
executable plus its producer receipt move to the sibling `<output>.unpacked`. The receipt records
`buildOnly: true` and `publicationQualified: false`: compilation, linking and static audits do not
mean the executable was run or approved for distribution. The legacy Engineering archive is
built with `npm run build:release -- --engineering-archive [--package-only | --clean]` or
`npm run test:deploy`.

On Windows the descriptor may select Core and GPU source producers or admitted component
receipts. On Linux the SDK composer accepts explicit source roots or receipts for the Hermes,
AdaptiveCpp, ROCm/COMGR, OpenMP, zstd and CUDA runtime inputs. Component receipts do not imply a
ready application. Windows static links keep MSVC's `oldnames.lib` for POSIX-style CRT names
(`open`, `read`) whose aliases may be absent from public-symbol maps: verify a complete link
before dropping a library.

### glibc 2.31 baseline

The Linux executable targets glibc 2.31 (Ubuntu 20.04+, Debian 11+, RHEL 9+). Every Linux SDK
producer, the Core/GPU factories and the final link compile against a pinned sysroot instead of the
build host's headers and libraries: focal glibc 2.31, focal kernel headers, focal `libelf` and
`libnuma` static archives, and the GCC 14 `libstdc++`/`libgcc` built for focal by the
ubuntu-toolchain-r PPA. The packages are mirrored in the `sysroot-inputs-v1` GitHub pre-release
(with source packages and signed indexes) and pinned by SHA-256 in
`scripts/native-linux-sysroot.js`; `fetch-native-sdk-sources.js sysroot|verify-sysroot` acquires
and checks them, and `native-linux-sysroot.js prepare|audit` extracts and admits the deterministic
tree. ROCm's `libdrm`/`libdrm_amdgpu` are built from pinned upstream 2.4.125 (`build-native-libdrm.js`,
fetched from the same release) because distro archives carry the host's glibc symbol versions.

`MOM_LINUX_SYSROOT` selects a prepared sysroot. `nativeSdkEnvironment` then sets `ICXCFG`/`ICPXCFG`
(a config file with `--sysroot` and `--gcc-install-dir`), `CMAKE_TOOLCHAIN_FILE` (which keeps
CMake's header and library searches inside the sysroot) and the pkg-config search paths; ambient
values of those variables are rejected. LLVM and DPC++ `clang` receive the same flags explicitly.
`npm run build:release` prepares the sysroot in its workspace, and every `native-job.yml` job
prepares it at `/tmp/mom-native-linux-sysroot-ci` before its producer runs. Every SDK receipt
records the sysroot receipt's SHA-256, and object-cache and Actions cache keys include it, so SDKs
built against another glibc are rebuilt rather than reused.

The final link does not use `-z pack-relative-relocs` (DT_RELR adds a `GLIBC_ABI_DT_RELR` need that
glibc older than 2.36 refuses to load) and links `librt` for `shm_open` (before glibc 2.34). The
release audit rejects any `GLIBC_` version above 2.31 and any symbol-version need outside the
allowed glibc libraries (for example a dynamic `libstdc++`). The release test workflow's
`glibc-floor` lane runs `mom algorithms` without network in pinned ubuntu:20.04, debian:11 and
rockylinux:9 images (`scripts/check-native-glibc-floor.sh`). GPU drivers remain external: an old
distribution can use a GPU only where its vendor still ships a runtime for it.

### CUDA images and in-process JIT

CUDA has two native SDK modes: static NVRTC libraries, or a generated CUDA image bundle. The image
producer, `scripts/build-native-cuda-images.js`, takes a pinned configuration and emits the
generated image include plus its receipt/sidecar, compiling the maintained kernel literals through
the selected NVRTC's public API without a GPU. When a descriptor selects image mode it names the
admitted `cudaImages` bundle and both CUDA factories must use it; the complete link rejects mixing
embedded images with static-NVRTC libraries. Algorithm JIT stays in process. Engineering Mode's
NVRTC-based worker is separate and unchanged.

The standalone release does not require Node/V8, `.node` addons, an extracted runtime directory or
a compiler installation to run; compiler SDKs are build-time inputs. Ordinary OS libraries and
system GPU drivers remain external. Do not copy application DLLs or shared libraries from the SDK
into the runtime directory.

### Packaging and notices

Release CI packages the checked-in `THIRD-PARTY-NOTICES.txt`. Before packaging, with the pinned
build Node:

~~~sh
node scripts/check-native-notices.js
~~~

This checks the reviewed notice bytes and semantic dependency pins in
`runtime/native-notice-review.json`. When dependencies change, review their actual license texts,
update the notice body if needed, then update the snapshot from `dependencyPins()` in the checker;
never refresh it just to silence a mismatch. It detects drift; it does not certify license
completeness.

To assemble complete license texts from admitted SDK sources, pass an ordered JSON array of
`{component, path, sha256}` inputs:

~~~sh
node scripts/prepare-native-notices.js /absolute/notice-inputs.json /new/notices
~~~

The assembler emits `THIRD-PARTY-NOTICES.txt` and `native-notices.json`, binding source hashes
without host paths or timestamps. Include each component's full license file (OpenMP adds
runtime-specific terms). For a self-contained handoff call
`assembleNativeNotices(inputs, output, {bundleInputs: true})`, which stages the verified bodies under
`inputs/`; restore such a bundle at its original canonical path before re-admission. To keep
provenance, set the descriptor's `notices` to `{path, sha256, receipt: {path, sha256}}`; builds and
packaging recheck the inputs and the receipts retain the binding as `noticeAssembly`. Body-only
notice inputs remain supported, and a late replacement with different text does not inherit the
old proof. Source-composed Linux link receipts expose `sdkInputs.noticeSources` for the OpenMP,
zstd and OpenSSL bodies; this is a partial input set, not a complete notice catalog.

Archive packaging is a separate build-only step. Normally the build receipt binds the reviewed
notices beside the executable by absolute path, SHA-256 and byte count:

~~~json
{
  "notices": {
    "path": "/output/THIRD-PARTY-NOTICES.txt",
    "sha256": "<64 lowercase hexadecimal characters>",
    "bytes": 12345
  }
}
~~~

Package the build output with an explicit CMake executable and version:

~~~sh
node scripts/package-native-release.js /output/entrypoint.json /absolute/cmake 0.9.4
~~~

If notice review finishes after the build, package into a new external directory:

~~~sh
node scripts/package-native-release.js /output/entrypoint.json /absolute/cmake 0.9.4 \
  --output /new-distribution --notices /reviewed/THIRD-PARTY-NOTICES.txt
~~~

This does not rebuild or modify the build output or its receipts. The version-2 package receipt
binds the unchanged executable and both the original and selected notice identities (an absent
original is null); version-1 packages remain supported. The archive must contain exactly the
executable and the data-only notices file, and extracted bytes are verified. Packaging records
`publicationQualified: false`; it neither tests nor publishes the release.

### Reproducibility

A pinned recipe is not proof of bit-for-bit reproducibility. Run the focused real-tool checks with
the pinned tools; without these variables the compiler and archive smokes skip while recipe
assertions still run:

~~~sh
MOM_TEST_CMAKE=/absolute/cmake node --test tests/native-release-archive.js
MOM_TEST_SHERMES=/absolute/shermes node --test tests/native-bundle.js tests/native-path.js
MOM_TEST_WINDOWS_CC=/absolute/icx-cl MOM_TEST_WINDOWS_LINKER=/absolute/lld-link node --test tests/native-windows-controller.js
MOM_TEST_WINDOWS_GPU_COMPILER=/absolute/icpx node --test --test-name-pattern='Windows GPU deterministic COFF timestamps' tests/native-windows-gpu-build.js
~~~

On Linux the archive test covers TGZ and ZIP. On Windows select only its ZIP case:

~~~powershell
$env:MOM_TEST_CMAKE = 'C:\absolute\cmake.exe'
node --test --test-name-pattern='Windows ZIP repeat packaging ignores source mtimes and local timezone' tests/native-release-archive.js
~~~

The pinned UPX 5.2.1 recipe ignores ambient `UPX` options and keeps maximum LZMA effort with a fixed
128-MiB dictionary. That control is version-specific: upgrading UPX needs the real packing/restore
checks and a fresh memory measurement. The archive test varies input mtimes and local timezones;
the bundle/AOT test uses fresh output directories and compares bytes. Controller bundles use a
fixed target-side virtual root instead of the checkout path, and minification removes generated
path labels (function names are kept, syntax minification is off).

The end-to-end regression compares two fresh application builds from the same admitted SDK and
Core/GPU components. Normal logic tests run only its small admission fixtures:

~~~sh
MOM_TEST_NATIVE_REPRO=1 \
MOM_TEST_NATIVE_REPRO_CONFIG=/absolute/native-sdk.json \
MOM_TEST_NATIVE_REPRO_UPX=/absolute/pinned/upx \
MOM_TEST_NATIVE_REPRO_UPX_SHA256='<64-lowercase-hex>' \
MOM_TEST_NATIVE_REPRO_CMAKE=/absolute/pinned/cmake \
MOM_TEST_NATIVE_REPRO_ROOT=/absolute/new-pair-directory \
MOM_TEST_NATIVE_REPRO_KEEP=1 \
node --test --test-concurrency=1 tests/native-release-repro.js
~~~

The output root must not exist, its parent must exist, and it must be outside the checkout. Run
under an externally enforced memory cap and deadline (a Node test timeout cannot preempt these
synchronous build APIs). No miner, GPU, VM or pool is started. By default it disables application
object-cache hits, checks stable source/config/tool inputs, and compares raw, packed and archive
hashes, re-reading both outputs after the second build; only declared generated object outputs
are excluded from input equality, and Windows transitive dependency hashes are compared too.
Failures retain their output and a bounded summary; successful output is removed unless
`MOM_TEST_NATIVE_REPRO_KEEP=1`. This proves fixed-input application reproducibility, not a cold
SDK rebuild or reproducibility across relocated checkouts, users or file modes. For an
end-to-end SHA comparison pin the compiler, SDK, Node and other dependencies, keep canonical
source and SDK roots, build into two fresh output directories and compare executable and archive
bytes; receipts and logs record paths and timings, so their JSON bytes are not the target.

For descriptors that defer notices until packaging, also set `MOM_TEST_NATIVE_REPRO_NOTICES`,
`MOM_TEST_NATIVE_REPRO_NOTICES_SHA256`, `MOM_TEST_NATIVE_REPRO_NOTICE_REVIEW` and
`MOM_TEST_NATIVE_REPRO_NOTICE_REVIEW_SHA256` (absolute paths plus explicit hashes); each pass
packages into a fresh sibling directory. For Linux cold/warm cache qualification set
`MOM_TEST_NATIVE_REPRO_CACHE_MODE=warm` and provide a fresh, empty, dedicated cache overlay at
`build/cache` (never clear your development cache): the first pass must miss every application
object and the second must validate and reuse the same cohort.

## Build design notes

- Size reduction uses archive selection, dead-section elimination, stripping and memory-only
  compression, not whole-program LTO: Intel Core bitcode is materialized with its original compiler
  before linking with the GPU LLVM toolchain, whose reader cannot consume that newer IR safely.
  Packed executables decompress in memory at every launch, including replacement workers.
- Linux LLVM consumers use an installed-SDK handoff instead of restoring the compiler build tree:
  the producer fully admits source and build, then exports the installed prefix and unchanged
  receipts with a separately pinned `handoff.json`; consumers verify the installed inventory and
  receipt chain without historical build paths. Missing or mismatched pins fail closed.
- GPU factory transports carry a hash-pinned `handoff.json` binding the unchanged
  factory/configuration/isolation receipts and isolated objects to the original SDK input hashes.
  The producer rechecks the SDK files before and after packaging; final assembly rechecks the
  current source and exact linked object cohort.
- Windows controller support builds I/O/OpenSSL and Hermes in parallel with Intel 2026.2. Intel
  Windows cross-builds pass explicit `/Tc` and `/Tp` source operands (`icx-cl` otherwise reads Linux
  `/workspace/...` filenames as warning options), and the pinned Microsoft SDK selection includes
  static `delayimp.lib` for LLVM's delay-load edges. Transports keep selected static libraries,
  headers and unchanged producer receipts, not executables or build caches.
- Handoffs and receipts are build provenance, not runtime or publication qualification.

## CI layout

- `stability.yml` runs the source gates (`source-gates.yml`: lint, typecheck, logic/host tests on
  Linux, PowerShell syntax and logic on Windows) on pull requests and pushes to master. The complete
  native build runs on `workflow_dispatch` and on every `v*` tag through `release.yml`.
- `build-release-artifacts.yml` holds the whole native graph: source gates, Engineering hashing
  (`test-engineering.yml`: CPU vectors, addon contracts and portable SYCL CPU algorithms under Node.js
  on Linux and Windows), Linux and Windows SDK producers, components, applications, raw and packed
  runtime lanes (`test-native-release.yml`: `cpu-cli`, `portable-cpu` and, on Linux, `glibc-floor`;
  40-minute jobs with `MOM_TEST_NO_POOL_NETWORK=1`), packing (`pack-native-application.yml`, from the
  raw binary and compact build receipts, no SDK or application rebuild, 30 minutes) and packaging
  (`package-native-application.yml`). Packaging waits for every gate; `release.yml` publishes only the
  two archives whose SHA-256 the package jobs reported, and only its tag job has `contents: write`.
  Hosted CPU checks do not replace physical-GPU correctness or performance runs.
- Linux producer jobs use `native-job.yml`. It prepares the glibc sysroot, exports the `/tmp` work
  roots (`scripts/ci/job.js`), the caller's producer pins, `MOM_BUILD_JOBS=4` (public runners have 4
  vCPU/16 GB) and a job deadline `MOM_CI_DEADLINE_EPOCH` five minutes before `timeout-minutes`.
  Downloads land flat in `$RUNNER_TEMP/incoming`; artifacts are named after their files and kept for
  three days. An optional `cache` input restores and saves a named SDK work root with
  `actions/cache` (`scripts/ci/cache.js`; currently `linux-llvm`). The key hashes the recipe and every
  input admission compares, a hit is only a candidate that the producer re-admits, and a release
  never knows how the root was obtained.
- Stage logic lives in `scripts/run-native-ci.js` (stage dispatcher), `scripts/native-ci-*.js` (stage
  implementations, shared job and tar helpers) and `scripts/ci/` (job drivers, sealing, per-platform
  application, JIT, DPC++, host and LLVM steps and PowerShell helpers).
- Sequential chains run as one job of at most 330 minutes: Linux LLVM, Linux source-JIT
  (`scripts/ci/linux-jit.js`), Windows LLVM (`scripts/ci/windows-llvm.js`) and Windows source-JIT
  (`scripts/ci/windows-jit.js`). Chain drivers (`scripts/ci/chain.js`) keep each phase's pinned state
  handoff, fresh-root admission and 55-minute deadline but pass state locally instead of through
  artifacts. Other producers are short jobs with 55-minute internal deadlines.
- Heavy producers run under `scripts/run-bounded-native-ci.sh` (a swap-free 12-GiB scope sized for
  four build jobs on a 16 GB runner).
- Work roots and artifact names carry the fixed suffix `MOM_NATIVE_CI_SUFFIX=ci` instead of a run id,
  so "Re-run failed jobs" reuses earlier artifacts and uploads overwrite them. A job refuses to start
  when a work root from an earlier job is still on the runner.
- `tests/workflows.js` checks action pins, permissions, timeouts, artifact producers, job outputs and
  referenced scripts.

## Useful entrypoints

| Task                           | Entry point                                                                |
| ------------------------------ | -------------------------------------------------------------------------- |
| Run MoM in Engineering Mode    | npm run dev -- [MoM arguments]                                             |
| Lint and strict type check     | npm run check                                                              |
| Logic/CPU/GPU tests            | npm test -- logic, cpu, or an explicit gpu-* selector                      |
| Engineering archive build/test | npm run test:deploy                                                        |
| Build standalone Linux release | npm run build:release                                                      |
| Advanced standalone build      | npm run build:release -- --native <sdk.json> <new-output>                  |
| Package a standalone build     | node scripts/package-native-release.js <entrypoint.json> <cmake> <version> |
| Check a standalone build       | node scripts/check-native-release.js <entrypoint.json> <lane>              |
| GPU compiler/backend matrix    | [GPU-CONFIG.md](GPU-CONFIG.md)                                             |

Prefer the maintained entrypoints over ad hoc compiler commands. Preserve fresh receipts and raw
logs for each stage, and report build-only, runtime, packaging and publication evidence as
separate results.
