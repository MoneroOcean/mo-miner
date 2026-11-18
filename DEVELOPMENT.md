# Development and native builds

This guide covers the maintained developer workflows and the current native-build boundary.
It is not a release-readiness claim. GPU routing details live in [GPU-CONFIG.md](GPU-CONFIG.md);
source-level project context and algorithm notes remain in the other repository documents.

## Requirements and setup

Use the Node.js and npm versions declared by package.json (Node.js 24.15 or newer and npm 11.10
or newer). Install the native development prerequisites for the components you intend to build:

Linux:

~~~sh
sudo ./scripts/install-dev.sh --component all --jobs 8
~~~

Windows, from an elevated PowerShell or Command Prompt:

~~~powershell
scripts\install-dev.bat -Component all -Jobs 8
~~~

The installers accept named component selections; see their usage output before choosing a
smaller set. Installing every vendor SDK is optional if you only need to edit JavaScript or run
source-only tests. Keep vendor toolchains and driver installations separate from project outputs.

Windows static OpenSSL source builds additionally need native Perl. The optional
`-Component perl` installs a checksum-pinned Strawberry runtime (also included in `all`),
without its C/C++ compiler toolchain. It is a build prerequisite, not a miner dependency;
`base,node,oneapi` and ordinary Engineering Mode do not require it.

Install the repository's JavaScript dependencies, then run the source checks:

~~~sh
npm ci --ignore-scripts
npm run check
~~~

Skipping install scripts avoids an implicit node-gyp build; the development entrypoint builds
native workers explicitly when needed. The check command runs ESLint and strict JavaScript/JSDoc
type checking. It validates source quality; it does not build or qualify a native release.

## Engineering Mode

Engineering Mode runs the original JavaScript controller under Node.js. It is the normal
edit-run-debug path and continues to use the maintained native workers/addons.
JavaScript-only edits do not require AOT compilation or native-executable relinking. Native
C++/SYCL edits rebuild affected inputs incrementally; release optimization stays a separate task.

~~~sh
npm run dev -- algorithms
npm run dev -- mine config.json
~~~

On Linux, npm run dev uses the development Docker image by default. Docker keeps the toolchain
environment consistent and persists build outputs/cache through the repository's build
directories. To use SDKs installed directly on the Linux host:

~~~sh
MOM_DEV_HOST=1 npm run dev -- algorithms
~~~

Host mode supports Linux x86-64 only and requires the corresponding host SDKs. It uses the same
maintained compiler entrypoint and incremental native build tree.

On Windows x64, the command first runs the native multicompiler builder, then runs MoM (or the
requested Node/npm command) using the Windows build tree. A requested command can be forwarded:

~~~powershell
npm run dev -- npm test -- logic
npm run dev -- --exec where.exe node
~~~

The --exec option is for commands other than MoM, Node, or npm. Do not use it to bypass the
maintained build entrypoint.

### Selecting a GPU backend

MOM_GPU_BACKEND selects intel, intel_igpu, nvidia, amd, opencl, or all. The default is intel.
For example:

~~~sh
MOM_GPU_BACKEND=nvidia npm run dev -- algorithms
~~~

Backend names select the maintained compiler/factory configuration; they do not prove the
selected hardware is available or validated. See [GPU-CONFIG.md](GPU-CONFIG.md) for the
platform/vendor/algorithm routing matrix and runtime requirements.

MOM_REUSE_BUILT_WORKER=1 is a Docker-worker shortcut and is rejected by native Linux/Windows
development. Native development validates and reuses its own incremental build outputs.

### Incremental outputs and caches

Platform outputs are separated: Linux uses build/lin, Windows uses build/win, and reusable
compiler objects use build/cache. node-gyp temporarily creates a top-level build tree; the
maintained entrypoints park and restore the persistent build tree around that operation.

The shared native object cache is build-time state, not a runtime dependency. Its keys include
source and included-header bytes, compiler/tool identity, build environment, and command flags.
Changed inputs rebuild affected objects; final links, audits, and receipts are still produced
for each build. A cache hit is not a runtime or release qualification result.

Native release/component object-cache controls:

- MOM_NATIVE_NINJA=/absolute/path/to/ninja selects a specific Ninja executable.
- MOM_NATIVE_CACHE=0 disables shared object reuse.

Engineering builds also support:

- MOM_FORCE_REBUILD=1 forces their incremental outputs to rebuild.
- MOM_BUILD_JOBS bounds producer parallelism where supported.

Windows Core has a separate validated MSBuild cache at
build/cache/native-windows-core/v1; the application/GPU object batches use the shared object
cache. Incomplete or stale incremental state takes the conservative rebuild path. Do not delete
or modify cache state while a build is using it.

## Tests and runtime checks

Start with the source-only suites:

~~~sh
npm run check
npm test -- logic
~~~

The logic suite does not require a GPU. The Linux host suite checks the source-built host
configuration and requires Linux, Bash, a C++ compiler, and Boost:

~~~sh
npm test -- host
~~~

CPU correctness tests can be run independently:

~~~sh
npm test -- cpu
~~~

For the same early hashing gate used by CI (CPU vectors, addon contracts, and all portable SYCL
CPU algorithms), install the CPU OpenCL runtime, then run the ordinary development entrypoint:

~~~sh
MOM_DEV_HOST=1 MOM_GPU_BACKEND=opencl MOM_LINUX_BUILD_COMPILER=dpcpp-opencl MOM_OPENCL_DEVICE_TYPE=cpu ONEAPI_DEVICE_SELECTOR=opencl:cpu MOM_REQUIRE_PORTABLE_CPU_TESTS=1 npm run dev -- node tests/run-hash.js engineering
~~~

On a Linux host, `bash scripts/install-opencl-cpu.sh` installs the pinned CPU runtime; Windows
development's `oneapi` component already installs it. No standalone executable is built by this gate.

Hardware test selectors are explicit:

~~~sh
npm test -- gpu-discrete
npm test -- gpu-integrated
npm test -- gpu-multi
npm test -- gpu-portable-cpu
~~~

Run hardware suites only when the corresponding devices and drivers are available and owned by
your test. They are not substitutes for one another: portable CPU coverage is not GPU coverage,
and a skipped or missing device is not a pass. The tests do not require connecting to a mining
pool.

For bounded integrated-GPU stability checks, set `MOM_INTEGRATED_PORTABLE_TEST=1` with
`gpu-integrated`. This opt-in reuses compact correctness fixtures where supported; mining,
benchmarks, and the default integrated-GPU consensus fixtures remain unchanged. Some fixtures
exercise generation or recovery substeps rather than full solvers. These results complement,
not replace, discrete-GPU consensus tests. Set `MOM_REQUIRE_INTEGRATED_GPU_TESTS=1` to fail if
no integrated GPU is available; inspect algorithm skips separately.

To run a test under the selected Engineering build, invoke npm through the development entrypoint:

~~~sh
npm run dev -- npm test -- logic
~~~

npm run test:perf -- <algorithm> measures performance on the available hardware; it is not a
correctness or release gate. npm run test:deploy builds and tests the legacy Engineering
archives, including the configured correctness and performance checks. Confirm device ownership
and stop unrelated mining workloads before running hardware benchmarks.

The standalone-native candidate checker is separate from the Engineering test runner. It accepts
an admitted build receipt and can run CPU, CLI, or an explicit GPU lane:

~~~sh
node scripts/check-native-release.js /candidate/entrypoint.json cpu
node scripts/check-native-release.js /candidate/entrypoint.json cli
node scripts/check-native-release.js /candidate/entrypoint.json gpu --vendor intel --algo pearlhash
~~~

Custom GPU harnesses must unset `MOM_SKIP_SYCL_ALGO_PARAMS`, not set it to `0`: this CPU-probe
flag checks presence. The native checker already removes it for its GPU lanes.

Native Linux builds do not load the MSR module with `modprobe`: they use an already-enabled
interface (`allow_writes=on` or `default`), or an openable MSR device when the parameter cannot
be read. Otherwise MSR tuning is skipped. Engineering builds retain their existing fallback.

The CPU/CLI runtime workflow is not provenance qualification. The GPU lane requires the named
hardware, fails rather than silently accepting missing devices, and does not build the candidate.
Run native builds and candidate checks in a resource-limited environment with swap disabled
wherever practical; command timeouts and job-count settings are not memory limits.

## Native SDK inputs and builds

Native release builds require an explicit SDK descriptor and a new output directory. The
descriptor selects the application source, compiler/toolchain, runtime libraries, GPU component
receipts or source builds, and any optional distribution data. The producer validates those
inputs and records their hashes. It does not automatically provision a complete SDK.

### Pinned source preparation

SDK acquisition and compilation are separate stages. fetch-native-sdk-sources.js can acquire
named pinned Git cohorts or source archives only when the caller opts into networking. Use a new
output directory:

~~~sh
node scripts/fetch-native-sdk-sources.js compiler /absolute/path/to/git /new/compiler-sources --allow-network
node scripts/fetch-native-sdk-sources.js build /new/build-sources --allow-network
~~~

Archive source cohorts have offline verifiers; for example:

~~~sh
node scripts/fetch-native-sdk-sources.js verify-build /absolute/path/to/build-sources
node scripts/fetch-native-sdk-sources.js verify-io /absolute/path/to/io-sources
node scripts/fetch-native-sdk-sources.js verify-zstd /absolute/path/to/zstd-sources
~~~

For SDK Git exports, prepare-native-sdk-sources.js materializes the selected pinned roots and
maintained patches from an explicit roots descriptor into a new directory:

~~~sh
node scripts/prepare-native-sdk-sources.js compiler /absolute/source-roots.json /new/compiler-sources
~~~

Available preparation targets include linux, windows, compiler, and acpp; other components have
their own maintained producers and input descriptors. Keep the original upstream revisions,
patch set, compiler/tool hashes, and generated receipts together. Do not edit vendor checkouts in
place, copy a prebuilt library into a receipt, or rewrite a receipt to make stale bytes appear
current.

Fetch is the explicitly networked stage. After acquisition, verify receipts and run build stages
offline where their toolchains and inputs permit. Use a fresh, host-enforced memory cap for
compilers and SDK builders; use only the required read-only source/SDK mounts and a distinct
writable output/cache directory. SDK receipts are evidence for their named component, not for
the complete application.

### Building a standalone native candidate

On Linux x64, the no-argument entrypoint prepares pinned source-built SDKs, reuses admitted
incremental outputs, then builds, packs, audits and packages the executable:

~~~sh
npm run build:release
~~~

It uses `~/cache/mom/native-sdk`, an exclusive build lock, and an enforced aggregate
4-GiB memory limit with swap disabled. The final archive and SHA256 sidecar are copied to the
checkout only after successful audits. This is a build command, not a runtime test or publication
approval. Successful private application outputs are removed. Failed outputs occupy one
`release-work` slot; inspect that diagnostic directory before clearing it and retrying.
This path does not install system packages: pinned Node, Intel 2026.2, CUDA, and OS development
prerequisites must already be installed. Missing prerequisites fail before SDK acquisition.

On Windows x64, the same no-argument command uses native Windows support producers together with
an already-installed WSL2 distribution for Linux source preparation and the remaining SDK and
application stages. Select the distribution explicitly when more than one is installed:

~~~powershell
$env:MOM_WSL_DISTRO = "<installed WSL2 distribution>"
npm run build:release
~~~

The Windows side needs Node.js 24.15.0, Intel oneAPI 2026.2, Visual Studio C++ Build Tools,
Python, and the pinned native Perl prerequisite. The selected WSL2 distribution needs systemd
user scopes and the pinned Linux Node.js 24.15.0, Intel toolchain, and build tools. The command
does not install prerequisites or change Windows/WSL configuration. It keeps Linux dependencies
separate from Windows `node_modules`.

The WSL ext4 cache at `~/cache/mom/native-sdk` holds large Linux SDKs and incremental build data;
native Windows support and export caches are under
`%LOCALAPPDATA%\MoM\native-wsl-windows\<checkout-hash>\{support,exports,bridge}`. One lock spans
the WSL build graph. Its 4-GiB memory limit and disabled swap apply to the Linux process tree,
not native Windows interop processes; native Windows stages run sequentially with two build jobs.
The archive and checksum sidecar are copied into the checkout only after the shared audits pass.
Failed application output occupies one `windows-release-work` slot, and failed bridge state is
preserved and blocks retry; inspect and resolve diagnostics before retrying. The Windows/WSL
route has not yet received end-to-end runtime qualification on a Windows host.

For advanced builds, use a reviewed descriptor and a new absolute output directory:

~~~sh
npm run build:release -- --native /absolute/sdk.json /new/absolute/output
~~~

The equivalent environment-variable selection is MOM_NATIVE_SDK plus MOM_NATIVE_OUTPUT.
Optional UPX packing requires both MOM_NATIVE_UPX and MOM_NATIVE_UPX_SHA256, or the explicit
--pack /absolute/upx <sha256> arguments. Partial native settings fail closed.

Without UPX, the selected output directory contains the raw executable and an entrypoint.json build
receipt. With UPX, that directory contains the packed executable and receipt, while the raw
executable and its producer receipt remain in the sibling `<output>.unpacked` directory. The
entrypoint marks the result buildOnly: true and publicationQualified: false: successful compilation,
linking, and static audits do not mean the executable has been run or approved for distribution.

On Windows, the descriptor may select maintained Core and GPU source producers or already
admitted component receipts. On Linux, the native SDK composer accepts explicit component source
roots or receipts for the selected Hermes, AdaptiveCpp, ROCm/COMGR, OpenMP, zstd, and CUDA runtime
inputs. These are distinct build stages; a helper's successful component receipt does not imply
that the full application or all platforms are ready.

Windows static links still need MSVC's `oldnames.lib` for upstream POSIX-style CRT names such
as `open` and `read`. Its aliases may not appear in public-symbol map rows: verify a complete
link before removing a library, rather than inferring non-use from the map alone. Keep the
standard compatibility library instead of maintaining replacement CRT aliases.

### CUDA images and in-process JIT

CUDA has two explicit native SDK modes: static NVRTC libraries, or a generated CUDA image bundle.
The image producer, scripts/build-native-cuda-images.js, takes a pinned configuration and emits
the generated image include plus its receipt/sidecar. It queries the selected NVRTC compiler
through its public API and compiles maintained kernel literals without requiring a GPU.

When a descriptor selects the image mode, it names the admitted cudaImages bundle and both CUDA
factories must use that same bundle. The complete link rejects mixing embedded images with the
static-NVRTC libraries. This mode embeds selected compiler-produced device images while
algorithm JIT remains in process; it is not a claim that every descriptor or platform uses this
mode. Engineering Mode's separate NVRTC-based worker is unchanged.

The v0.9.4 standalone candidates do not require Node/V8, `.node` addons, a separate extracted
application-runtime directory, or a host compiler installation to run. Compiler SDKs and tools are
build-time inputs, not launch prerequisites. Ordinary operating-system libraries and system GPU
drivers remain external. Do not copy application DLLs or shared libraries out of the SDK into the
runtime directory.

### Packaging and notices

Release CI packages the checked-in `THIRD-PARTY-NOTICES.txt`. Before packaging, run with the
pinned build Node version:

~~~sh
node scripts/check-native-notices.js
~~~

This checks the reviewed notice bytes and semantic dependency pins in
`runtime/native-notice-review.json`. When dependencies change, review their actual license
texts, update the notice body if needed, then update the snapshot with `dependencyPins()` from
the checker. Do not refresh the snapshot merely to bypass a mismatch. The check detects drift;
it does not certify license completeness or admit arbitrary SDKs.

Assemble complete license texts from admitted SDK sources with an ordered JSON array of
`{component, path, sha256}` inputs:

~~~sh
node scripts/prepare-native-notices.js /absolute/notice-inputs.json /new/notices
~~~

The assembler emits `THIRD-PARTY-NOTICES.txt` and `native-notices.json`, binding the source
hashes without embedding host paths or timestamps in the shipped text. Include each component's
full license file, not just its project's generic license: OpenMP adds runtime-specific terms.
Assembly verifies the selected bytes, not whether the selection fulfills every license obligation.

For a self-contained build-time notice handoff, call
`assembleNativeNotices(inputs, output, {bundleInputs: true})`. This stages the verified literal
bodies under `inputs/` and binds those copies in the receipt. Restore such a bundle at its original
canonical path before re-admission; absolute receipt paths are not rewritten during transport.
The default two-argument API and CLI keep referencing the original source bodies.

To retain assembly provenance, set the native build descriptor's `notices` to
`{path, sha256, receipt: {path, sha256}}`, selecting the assembled text and its
`native-notices.json` receipt. Build and packaging recheck the original license inputs;
the candidate and package receipts retain the binding as `noticeAssembly`. Clean-host
qualification checks that binding without requiring the build sources. Body-only notice
inputs remain supported. A late replacement with different text does not inherit the old proof.

Source-composed Linux link receipts expose `sdkInputs.noticeSources` for the admitted OpenMP,
zstd, and OpenSSL bodies. These rows are bound to the selected SDK input hashes; they are a
partial input set, not a complete release-notice catalog.

Native archive packaging is a separate build-only step. Normally, the candidate receipt binds a
reviewed THIRD-PARTY-NOTICES.txt beside the executable by absolute path, SHA-256 and byte count:

~~~json
{
  "notices": {
    "path": "/candidate/THIRD-PARTY-NOTICES.txt",
    "sha256": "<64 lowercase hexadecimal characters>",
    "bytes": 12345
  }
}
~~~

Then package the admitted candidate with an explicit CMake executable and version:

~~~sh
node scripts/package-native-release.js /candidate/entrypoint.json /absolute/cmake 0.9.4
~~~

If notice review finishes after the executable build, package into a new external directory:

~~~sh
node scripts/package-native-release.js /candidate/entrypoint.json /absolute/cmake 0.9.4 \
  --output /new-distribution --notices /reviewed/THIRD-PARTY-NOTICES.txt
~~~

This does not rebuild or modify the candidate, its receipts, or its original notices. The new
version-2 package receipt binds the unchanged executable and both the original and selected
notice identities; an absent original notice is recorded as null. Existing version-1 packages
remain supported. Use actual notice hashes and byte counts, not the illustrative values above.

The archive is checked to contain exactly the executable and the data-only notices file, and
the extracted bytes are verified. Notices are not runtime dependencies and their presence does
not by itself establish that redistribution obligations are complete. Packaging records
publicationQualified: false; it does not test or publish the release.

### Reproducibility

A pinned recipe and repeatable automation are not, by themselves, proof of bit-for-bit output
reproducibility. Run the focused real-tool checks with the selected pinned tools:

~~~sh
MOM_TEST_CMAKE=/absolute/cmake node --test tests/native-release-archive.js
MOM_TEST_SHERMES=/absolute/shermes node --test tests/native-bundle.js tests/native-path.js
MOM_TEST_WINDOWS_CC=/absolute/icx-cl MOM_TEST_WINDOWS_LINKER=/absolute/lld-link node --test tests/native-windows-controller.js
MOM_TEST_WINDOWS_GPU_COMPILER=/absolute/icpx node --test --test-name-pattern='Windows GPU deterministic COFF timestamps' tests/native-windows-gpu-build.js
~~~

The pinned UPX 5.2.1 recipe ignores ambient `UPX` options and keeps maximum LZMA search
effort with a fixed 128-MiB dictionary. Its dictionary control is version-specific; upgrading
UPX requires the real packing/restore checks and a fresh memory measurement, not just unit tests.

On Linux the archive command tests both TGZ and ZIP. On Windows, select only its ZIP regression:

~~~powershell
$env:MOM_TEST_CMAKE = 'C:\absolute\cmake.exe'
node --test --test-name-pattern='Windows ZIP repeat packaging ignores source mtimes and local timezone' tests/native-release-archive.js
~~~

Real-tool checks are opt-in: without the corresponding environment variables, the compiler and
archive smokes skip while recipe assertions still run. The archive regression varies input
mtimes and local timezones; the bundle/AOT regression uses fresh output directories and compares
generated bytes. Identifier/whitespace minification removes generated path labels; function names
are retained and syntax minification stays disabled. Fixed-input application pairs do not
qualify a cold SDK chain or comparisons across users, file modes, or relocated checkouts.
For an end-to-end SHA comparison, pin the compiler, SDK, Node and other
dependencies, keep canonical source and SDK roots, build into two fresh output directories, and
compare executable and archive bytes. Receipts and logs intentionally record paths and timings,
so their JSON bytes are not the reproducibility target.

A local Linux A/B experiment reproduced the raw executable, packed executable and archive
byte-for-byte with application-object caching disabled (13 misses per pass). Both builds used
the same admitted inputs, pinned SDKs and canonical logical source paths. This does not establish
independent checkouts, SDK rebuilds or reproducibility under arbitrary logical source/SDK paths.
Native controller bundles and embedded-policy lookups now share a fixed target-side virtual
root instead of recording the checkout directory. This removes one known relocation-dependent
input; it does not establish whole-executable reproducibility across relocated checkouts.
A prior Windows A/B run also reproduced raw executables, packed executables and ZIP archives
byte-for-byte, with all 14 application objects rebuilt in each pass (zero cache hits). Both
passes matched that experiment's hardware-tested candidate. An older Windows pair's archive mismatch remains
recorded; the complete pair, not only recipe-level tests, establishes that fixed-input result.
Application changes after either experiment require fresh candidate qualification. Neither
historical pair proves bit-for-bit reproduction of the final application build; the updated
candidates' separate runtime results are recorded below.
The final Linux candidate has now also passed the maintained fresh A/B regression: both passes
rebuilt all 13 application objects with zero cache hits, and their raw executable, packed
executable and archive matched each other and the tested candidate byte-for-byte. The 21-test
run completed in about 7 minutes 41 seconds under a 5-GiB/no-swap host cap, with no OOM.
This used one fixed admitted SDK and canonical source roots, not a cold SDK rebuild or a
cross-checkout comparison. The final Windows fixed-SDK pair also passed all 21 maintained
regression tests in about 7 minutes 14 seconds: both passes rebuilt all 14 application objects,
and their raw, packed and ZIP bytes matched each other and the tested final candidate.
The private outer launcher failed overly strict memory-peak and immediate Docker-removal checks;
an independent post-check rehashed all six outputs, verified the test summary and confirmed clean
removal with no OOM. The failed outer-unit record is retained, not relabelled as a successful run.

The maintained end-to-end regression uses those same public build, packing and archive APIs.
Normal logic tests run its small admission fixtures and skip the expensive pair. To compare
two fresh application builds from the same admitted SDK and Core/GPU components:

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

The output root must not exist, its parent must exist, and it must be outside the checkout.
Run under an externally enforced memory cap and deadline: a Node test timeout cannot preempt
these synchronous build APIs. No miner, GPU, VM or pool is started by this test. By default it disables
application object-cache hits, checks stable source/config/tool inputs, and compares raw,
packed and archive hashes, re-reading both outputs after the second build. Only declared
generated object outputs are omitted from input equality; other pass-local inputs remain
hash-bound with their output-root prefix normalized. Windows transitive dependency hashes are
required and compared as well as application sources. Failures retain their output and bounded summary;
successful output is removed unless `KEEP=1`. This is fixed-input application reproducibility,
not a cold SDK rebuild, checkout-relocation or runtime/publication qualification.

For descriptors that defer notices until packaging, additionally provide
`MOM_TEST_NATIVE_REPRO_NOTICES`, `MOM_TEST_NATIVE_REPRO_NOTICES_SHA256`,
`MOM_TEST_NATIVE_REPRO_NOTICE_REVIEW` and `MOM_TEST_NATIVE_REPRO_NOTICE_REVIEW_SHA256`.
These absolute paths and explicit hashes bind the reviewed notice body and dependency manifest
through the same checks as the packaging workflow; neither descriptor nor producer receipts are
rewritten. Each pass packages into a fresh sibling output directory.

For Linux cold/warm cache qualification, add `MOM_TEST_NATIVE_REPRO_CACHE_MODE=warm` and provide
a fresh, empty, dedicated cache overlay at `build/cache`; never clear your development cache.
The first pass must miss every application object and the second must validate and reuse the
same complete cohort, while retaining all input and executable/archive byte comparisons.

## Current release and CI status

Release and Stability use the native release-artifact orchestrator described below. Legacy
Engineering archives remain available through `--engineering-archive --package-only` and
`npm run test:deploy`; they are not the standalone release pipeline. An independent Engineering
hashing gate builds development addons with the ordinary development scripts, then runs CPU
vectors, addon contracts and all 22 portable SYCL CPU algorithm checks under Node.js on Linux
and Windows. It starts alongside source checks, without waiting for AOT or static SDK builds;
packaging requires both Engineering and standalone qualification. Hosted CPU checks do not
replace physical-GPU correctness or performance qualification. The native prerequisite
workflow hands each platform's same-run raw executable and exact hash to the runtime-only
workflow. Independent packing jobs consume only that raw
binary and compact original build receipts through the existing release packing API; they do
not rebuild SDKs or the application. Raw checks and packing may run in parallel. Separate
runtime jobs then check the exact packed executable rather than assuming the raw result applies.
Both variants use `cpu-cli` and `portable-cpu` lanes with a 40-minute job guard and
`MOM_TEST_NO_POOL_NETWORK=1`; packing has a 30-minute cap and no persistent Actions cache.
These workflows are not publication qualification and have not yet been executed on GitHub.
Local candidate stability includes CPU/CLI and portable SYCL CPU checks on both operating systems,
and all 22 GPU algorithms on Intel, NVIDIA and AMD dGPUs on both systems. Generic OpenCL on B580
also passed on both systems; Linux UHD 750 passed its 21 supported algorithms (Hoohash is unavailable).
The iGPU C29, ZelHash, Equihash192/7 and ZHash cases are bounded generation checks, not full mining.
Windows physical-iGPU coverage is unavailable in the current VM. The installed Linux NVIDIA/AMD
OpenCL drivers lack the required SPIR-V ingestion; Windows generic OpenCL discovery was also
unavailable for these vendors. Unavailable paths are not passes.
These results do not establish hosted end-to-end build/provenance qualification. All 132 dGPU
performance cases (22 algorithms across three vendors and both operating systems) have completed.
All 132 cases are qualified, including a targeted exact-final Windows AMD C29 check described
below. With user approval, the allocation workaround and repeated proof-producing passes close
the C29 performance gate; reopen it if a new performance drop is observed. This acceptance does
not establish the cause of every historical anomaly. The Windows AMD Octopus baseline is
24.89 MH/s: a matched published 0.9.3 control also missed the previous 26.49 MH/s baseline at
24.45 MH/s. Original gate misses remain recorded. Windows AMD C29's allocation workaround has
subsequently passed the 95% README threshold with predominantly dedicated GPU memory; an earlier
anomalous no-proof run remains unexplained. It is retained as historical evidence, not an active
release blocker or a current confirmed sub-threshold result from the final executable.
A matched packed/raw cold/warm capture also found an anomalously high first-process work rate,
despite byte-identical shared JIT kernels. Launch/work accounting still needs investigation;
that rate is not a performance gain or qualification evidence.
After the Windows HIP allocator refresh, eight unprofiled raw/packed C29 comparisons met the
existing 5% gate at 5.67–5.75 graphs/s and produced proofs. Twelve further fresh-process,
cold-cache runs passed the maintained full proofsize-32 vector. These targeted checks did not
reproduce the anomalous no-proof rate; its cause remains unresolved and its result stays excluded.
Four further unprofiled comparisons using the default Windows AdaptiveCpp cache passed at
5.76–5.83 graphs/s with proofs. Their seven JIT binaries match the isolated-cache captures;
redirecting the cache is not an established fix for the historical anomaly.
A later targeted pair measured the exact final packed baseline at 5.70 graphs/s and a private
host-instrumented diagnostic at 5.82 graphs/s; both met the gate and each reported three proof
returns. No sub-20-ms graph was captured, so this run is inconclusive about the historical
fast/no-proof anomaly. The diagnostic patch is not part of the release sources; it adds no
device readbacks or profiling queue properties. Its generated device bitcode differs in type
names, while independent rehashing of all 14 retained JIT files confirms identical kernel
bytes between the two runs; this is not
evidence that the historical defect was fixed or that instrumentation improved performance.
Before the virtual-root update, the refreshed Linux and Windows raw and packed candidates each passed 32 CPU vectors, two
complete CLI comparisons and 67 portable-SYCL CPU tests. Focused Linux syscall checks covered
CLI startup and a CPU worker, permitting only the candidate's own process launches and system libraries.
The updated Linux and Windows raw and packed candidates have since passed those same six
runtime lanes per OS; packed sizes are 81,305,752 and 74,373,632 bytes respectively. Actual Linux
CPU archive and all five final GPU components, plus all 143 Windows GPU objects, match the
preceding candidates byte-for-byte. Windows ran in a GPU-free disposable VM with the actual
QEMU process tree verified under a 10-GiB/no-swap host cap before guest resume; teardown passed.
These CPU/portable checks are not fresh physical-GPU performance measurements.
The subsequent native Windows worker-PATH guard removes development-only search-directory
injection and preserves Engineering behavior. The rebuilt Linux and Windows raw/packed pairs passed
the targeted reconnect/donation/return and worker-loss fixtures. The earlier full-vector
results precede that final controller change; unchanged Core/GPU evidence is retained.
Final packed sizes are 81,346,560 bytes Linux and 74,353,152 bytes Windows. The final Linux
pair also passed complete CLI comparisons and focused candidate-only process/dependency traces.
The final Linux raw and packed executables subsequently each passed all 32 CPU vectors, both
CLI comparisons and all 67 portable-SYCL CPU tests, without skips. The final Windows raw
executable passed the same three lanes. The final packed Windows executable also passed complete
CLI parity (2/2) and all 32 CPU vectors in two disjoint 16-vector batches, with zero skips and the
same executable hash. Its portable-SYCL CPU checks subsequently passed in two disjoint batches:
34 + 33 checks covering all 22 maintained algorithms and the shared image guard exactly once,
with zero skips. Together these close all six raw/packed runtime lanes per OS. The earlier
disk-guard interruptions remain incomplete attempts; they were not relabelled as passes.
Both final archives passed exact-member and extracted-byte audits: each contains only its tested
executable and `THIRD-PARTY-NOTICES.txt`. Independent repacking reproduced the tested packed
executables byte-for-byte. This verifies packaging, not a second application build or publication.
With a two-CPU container limit, CLI startup measured about 50 ms raw versus 4.27 s packed;
this includes in-memory decompression and initialization, not just decoder time. Compression
also adds this cost when launching replacement workers; Engineering Mode does not use it.
The refreshed packed Linux candidate also completed bounded CPU-only `rx/0` checks against
MoneroOcean over plaintext and TLS: one accepted share per run, no rejects, and cooperative exits.
These checks do not cover donation,
reconnect, GPU mining or sustained operation.
Separately, the exact Linux raw and packed candidates passed a network-isolated loopback
mining fixture: primary disconnect/reconnect, TLS donation activation with GhostRider work,
donation expiry, resumed primary CryptoNight work, and cooperative shutdown. Both used one CPU
worker at a time; acknowledgements were synthetic protocol checks, not real-pool accepted shares.
Both Linux variants also passed a separate unexpected-worker-exit fixture: after a synthetic
share acknowledgement, a verified owned worker received SIGTERM; the controller reported the
fatal loss, exited 1, and left neither process running. This verifies fail-and-clean-up behavior,
not automatic worker restart. The exact Windows raw and packed candidates also passed the
loopback reconnect/donation/return fixture in a GPU-free disposable VM, with its temporary
hosts mapping restored. Both Windows variants also passed unexpected-worker-loss cleanup:
only the verified direct mining worker was terminated, the controller reported the fatal loss
and exited 1, and neither process remained. This checks cleanup, not automatic restart;
an already-in-flight share acknowledgement may still arrive during shutdown.
After the virtual-root change, both Linux variants repeated the reconnect/donation/return
and worker-loss fixtures successfully. The final Windows worker-PATH update also repeated
both fixtures successfully for raw/packed executables, with original hosts bytes restored.
Real-pool evidence above still belongs to preceding binaries; it is not a new live-pool check.
Final live-pool, notices/package review and publication approval remain separate gates.

Native size reduction currently uses archive selection, dead-section elimination, stripping,
and memory-only compression. It is not whole-program LTO across all compiler families:
Intel Core bitcode is materialized with its original compiler before linking with the GPU
LLVM toolchain, whose reader cannot consume that newer IR safely. This preserves target
attributes and COMDAT groups; cross-toolchain LTO and its size benefit remain unqualified.

Linux LLVM consumers use an installed-SDK handoff rather than restoring its compiler build tree.
The producer first performs full source/build admission, then exports the installed prefix and
unchanged receipts with a separately pinned `handoff.json`. Consumers verify the complete installed
inventory and receipt chain without requiring historical build paths. COMGR receives the pinned
LLVM sources separately; the full SDK and thin-tool artifacts remain available for consumers that
still need them. Missing or mismatched handoff pins fail closed; they do not fall back to legacy
admission. This transport check is build provenance, not runtime qualification.

Windows controller support uses parallel I/O/OpenSSL and Hermes producer jobs with Intel 2026.2.
The shared Intel Windows cross-build rules use explicit `/Tc` and `/Tp` source operands:
`icx-cl` can otherwise interpret Linux `/workspace/...` filenames as warning options.
The pinned Microsoft SDK selection includes the static `delayimp.lib` helper used by LLVM's
Windows delay-load edges; this adds no external runtime dependency.
They retain selected static libraries, headers and unchanged producer receipts, not executable
tools or full build caches. Hermes consumes the same-run pinned source archive fetched on Linux;
the shared HTTP downloader acquires the smaller I/O/OpenSSL archives directly on Windows.
These newly wired stages still need a complete fresh-run qualification and downstream integration.

GPU factory transports include a separately hash-pinned `handoff.json` that binds the unchanged
factory/configuration/isolation receipts and isolated objects to the original SDK input hashes.
The producer rechecks those SDK files before and after packaging; they are not copied into the
compact transport. This is build provenance, not runtime or publication qualification. Final
assembly must also recheck the current source and exact linked object cohort.

### CI layout

- `stability.yml` runs only the source gates (`source-gates.yml`: lint, typecheck, logic/host tests on
  Linux, PowerShell syntax and logic on Windows) on pull requests and pushes. The complete native build
  runs only on `workflow_dispatch` and on every `v*` tag through `release.yml`.
- `build-release-artifacts.yml` holds the whole native graph: source gates, Engineering hashing, Linux and
  Windows SDK producers, components, applications, raw and packed runtime lanes, packing and packaging.
  Packaging waits for every gate; `release.yml` publishes only the two archives whose SHA-256 the package
  jobs reported, and only its tag job has `contents: write`.
- Linux producer jobs use `native-job.yml`. It exports the `/tmp` work roots (`scripts/ci/job.js`),
  the caller's producer pins, `MOM_BUILD_JOBS=4` (public runners have 4 vCPU/16 GB) and a job deadline
  `MOM_CI_DEADLINE_EPOCH` five minutes before `timeout-minutes`. Downloads land flat in
  `$RUNNER_TEMP/incoming`; artifacts are named after their files and kept for three days.
- Sequential chains run as one job of at most 330 minutes: Linux LLVM (`run-native-ci.js stage
  linux-llvm`), Linux source-JIT (`scripts/ci/linux-jit.js`), Windows LLVM (`scripts/ci/windows-llvm.js`)
  and Windows source-JIT (`scripts/ci/windows-jit.js`). The chain drivers keep each phase's own pinned
  state handoff, fresh-root admission and 55-minute deadline, but pass state locally instead of through
  artifacts. Other producers keep 60-minute jobs with 55-minute internal deadlines.
- Heavy producers run under `scripts/run-bounded-native-ci.sh` (a swap-free 12 GiB scope, sized for four
  build jobs on a 16 GB runner).
- Work roots and artifact names carry the fixed suffix `MOM_NATIVE_CI_SUFFIX=ci` instead of the run id or
  attempt, so "Re-run failed jobs" reuses the artifacts of earlier attempts and uploads overwrite them. A
  job refuses to start when a work root from an earlier job is still present on the runner.
- `tests/workflows.js` checks action pins, permissions, timeouts, artifact producers, job outputs and
  referenced scripts. Workflow fixtures do not replace a real hosted run, which is still outstanding.

## Useful entrypoints

| Task                              | Entry point                                                                |
| --------------------------------- | -------------------------------------------------------------------------- |
| Run MoM in Engineering Mode       | npm run dev -- [MoM arguments]                                             |
| Lint and strict type check        | npm run check                                                              |
| Logic/CPU/GPU tests               | npm test -- logic, cpu, or an explicit gpu-* selector                      |
| Engineering archive build/test    | npm run test:deploy                                                        |
| Build standalone Linux release    | npm run build:release                                                      |
| Advanced native candidate         | npm run build:release -- --native <sdk.json> <new-output>                  |
| Package native candidate          | node scripts/package-native-release.js <entrypoint.json> <cmake> <version> |
| GPU compiler/backend matrix       | [GPU-CONFIG.md](GPU-CONFIG.md)                                             |

Prefer the maintained entrypoints over ad hoc compiler commands. Preserve fresh receipts and
raw logs for each stage, and report build-only, runtime, packaging, and publication evidence as
separate results.
