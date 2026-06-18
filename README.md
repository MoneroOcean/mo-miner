## mom

`mom` is the short name for the **mo-miner** project (the executable, release archives and
config keys all use `mom`); the GitHub repository remains
[MoneroOcean/mo-miner](https://github.com/MoneroOcean/mo-miner).

# About

mom is open-source cryptocurrency miner that is built upon high performance xmrig CPU miner
sources with front-end and network backend rewritten in Node.js to significantly simplify its code.
GPU mining sources are also simplified and rewritten in SYCL from OpenCL/CUDA.
The main goal of this project is to make simple, easy to extend open-source miner with native
miner performance.

# Limitations

The primary development and test platform is Linux with a single-socket x86-64 CPU and an Intel Arc
GPU. NVIDIA GPUs are also supported via the DPC++ CUDA backend (see NVIDIA GPU performance below), and
Windows is supported (see the runtime/install notes below). There is also a generic **SYCL OpenCL
backend** (see "OpenCL backend / AMD GPUs" below): on a box with no Intel-Level-Zero and no NVIDIA-CUDA
GPU it is auto-selected, so an **AMD GPU** (or any OpenCL GPU) runs the same binary via its OpenCL
driver. The OpenCL path is exercised in CI-equivalent testing on the Intel B580 (as an OpenCL/AMD
stand-in), but **no real AMD GPU has been tested yet** — AMD support is best-effort/WIP (a real AMD
card also needs sub-group-size 32/64 handling). 2+ socket CPUs and ARM CPUs are likewise untested.

# Supported algos

* CPU: All xmrig miner CPU supported algos with similar performance
* GPU/SYCL: cn/gpu, c29, kawpow, etchash, autolykos2, pearl

The GPU/SYCL implementations also run on SYCL CPU devices for hash-vector coverage and fallback
testing, though the CPU mining path remains the preferred CPU path for CPU-focused algos.

Miner supports algo switching if you connect it to algo switching pool like
gulf.moneroocean.stream that auto switches to the most profitable algo.
KawPow, Etchash, and Autolykos2 can also be used on non-MoneroOcean stratum pools that serve those
algos directly. Pearl (`pearl`, the PRL NoisyGEMM proof-of-useful-work coin) is **only** mined on
third-party pools (HeroMiners, LuckyPool) — the MoneroOcean pool does not serve it — so see the Pearl
mining section below. Its hashrate is reported as GEMM multiply-accumulate throughput in TH/s (the
same unit ARC-miner uses), not cryptographic H/s, so it is not directly comparable to the other algos.

By default, startup algo-parameter benchmarking only benchmarks active MoneroOcean coin algos
implemented by mom plus `rx/2`: `autolykos2`, `c29`, `cn/gpu`, `etchash`, `ghostrider`,
`kawpow`, `panthera`, `rx/0`, `rx/2`, and `rx/arq`. Use `--bench_algo_params 2` to benchmark every algo
supported locally. `pearl` is **not** in the default set (MoneroOcean can't assign it); mine it with
`--bench_algo_params 0` plus an explicit `--job.dev`, or bench it with `--bench_algo_params 2`.

# Mining Pearl (PRL)

Pearl uses a fixed network NoisyGEMM shape (m=n=131072, k=4096, noise_rank=256) — the defaults — so a
capable GPU (~1.2 GB VRAM for that shape) just needs the pool, wallet and GPU device:

```
# HeroMiners (TLS, direct):
./r.sh node mom.js mine pearl.herominers.com:1200tls <prl1-wallet> --job.algo pearl --job.dev gpu1*131072 --bench_algo_params 0
# LuckyPool (TLS, var-diff):
./r.sh node mom.js mine pearl-eu1.luckypool.io:3360tls <prl1-wallet> --job.algo pearl --job.dev gpu1*131072 --bench_algo_params 0
```

These pools use the `mining.subscribe`+`mining.authorize` dialect (the default for `pearl`). The older
pearlpool.cloud uses the single-`login` dialect and a smaller low-mem shape; select it with env vars:
`MOM_PEARL_LOGIN=1 MOM_PEARL_K=1024 MOM_PEARL_RANK=64` and `--job.dev gpu1*16384`.

Pearl env knobs (all optional; defaults = network-standard shape): `MOM_PEARL_INTENSITY` (m=n,
default 131072), `MOM_PEARL_K` (default 4096), `MOM_PEARL_RANK` (default 256),
`MOM_PEARL_LOGIN` (force the login dialect). In a saved config these can instead live under
`algo_params.pearl.{k,rank}` (the dev `*<batch>` sets the intensity) so no env vars are needed.

# NVIDIA GPU performance

mom also runs on NVIDIA GPUs from the same SYCL kernels via the DPC++ CUDA backend (ahead-of-time
compiled to PTX). There is a single Linux release artifact, `mom-v<version>-lin.tgz`, whose one
`mom.node` runs on both Intel and NVIDIA GPUs (it auto-detects the device). The **Windows** release is
now also a unified Intel+NVIDIA `mom.node` (spir64 + nvptx in one binary): CI builds it by restoring a
prebuilt from-source DPC++ CUDA toolchain (cached as a release asset -- the ~1.5 h LLVM build can't run
in a hosted job; recipe in [scripts/build-windows-nvidia.md](scripts/build-windows-nvidia.md)) plus the
CUDA toolkit for libdevice. GPU algos are verified on real hardware (7/7 hash vectors, kawpow runtime-JIT
included, on an L4 / Windows Server 2022); the hosted CI runners have no GPU, so they build the package
and run the CPU + SYCL-CPU suites. Hashrates measured on an NVIDIA L4 (Ada, sm_89), each compared
against the best closed-source miner benchmarked on the same card:

| Algo | mom (L4) | SOTA reference (same L4) | % of SOTA |
| --- | --- | --- | --- |
| cn/gpu | 2.65 KH/s | SRBMiner-MULTI 3.04 KH/s | 87% |
| c29 (Cuckaroo29) | 4.83 g/s | lolMiner ~5.3 g/s | 91% |
| kawpow | 13.79 MH/s | Rigel 15.56 MH/s | 89% |
| firopow | 13.44 MH/s | ProgPoW family -- cf. kawpow (Rigel 15.56) | ~86% |
| evrprogpow | 13.41 MH/s | ProgPoW family -- cf. kawpow (Rigel 15.56) | ~86% |
| etchash | 28.9 MH/s | ~28.8 MH/s (memory-bandwidth bound) | ~parity |
| autolykos2 | 76.5 MH/s | lolMiner 98.19 MH/s | 78% |
| pearl | 33.6 TH/s | no NVIDIA SOTA (ARC-miner is Intel-only) | — |

SOTA references benchmarked on the same L4: lolMiner (`--benchmark AUTOLYKOS2` / `CR29`), Rigel
(`-a kawpow`), SRBMiner-MULTI (`--algorithm cryptonight_gpu`). firopow/evrprogpow run the same kawpow
kernel family (the kawpow SOTA class applies; they were not separately SOTA-benched). kawpow/firopow/
evrprogpow use the runtime SYCL-source JIT, which needs **two** host pieces (see NVIDIA GPU install
below): a CUDA toolkit (libdevice **and** its headers under `/usr/local/cuda`) *and* a host C++
toolchain (`g++`/libstdc++). Without either, the ProgPoW-family algos fall back to a correct AOT kernel
at ~4 MH/s; the JIT-free algos (etchash/autolykos2/c29/cn-gpu/pearl) are unaffected.

NVIDIA + OpenCL: **not available** — NVIDIA's OpenCL driver doesn't ingest SPIR-V (`cl_khr_il_program`),
so the SYCL OpenCL adapter can't load mom's `spir64` image. CUDA is the only NVIDIA path. Verified on
this L4: `clinfo` reports the NVIDIA OpenCL device as `IL version (n/a)`, and the SYCL OpenCL adapter
drops it entirely, so `gpu1o` does not enumerate (the `opencl` test suite simply skips on NVIDIA). The
CUDA `gpu` suite passes all 7 vectors on the same binary.

# OpenCL backend / AMD GPUs

Besides Level-Zero (Intel) and CUDA (NVIDIA), mom runs the GPU algos through the generic SYCL **OpenCL**
backend — the *same* `spir64` device image, JIT-compiled by the GPU's own OpenCL driver. This is how an
**AMD GPU** (or any OpenCL GPU) runs the miner: on a machine with no Intel-Level-Zero and no NVIDIA-CUDA
GPU, the OpenCL GPU is auto-selected as `gpu1` — **no flag or config needed**, just install an OpenCL ICD
(see Linux install below) and run. On an Intel box the default `gpu1` stays Level-Zero; the same GPU's
OpenCL device is also exposed explicitly as `gpu1o` (and Level-Zero as `gpu1z`).

The OpenCL path is verified on the Intel B580 against the same hash vectors as Level-Zero
(`npm run test:opencl`, which runs them via `gpu1o`), and benchmarks at parity with Level-Zero on every
algo there *except pearl* (below) — same `spir64` kernels (cn/gpu is even marginally faster on OpenCL,
which is why it already defaults to it).

**pearl is the exception.** Its hot loop is an int8 matrix-multiply; the fast paths use dedicated matrix
hardware — Intel **XMX** (the ESIMD kernel, on Level-Zero) or NVIDIA **tensor cores** (on CUDA) — which
are *not reachable* through the OpenCL backend (Intel exposes XMX to OpenCL only via the ESIMD extension,
and AMD/NVIDIA expose their matrix units through HIP/Vulkan, not OpenCL's SPIR-V). So on **any** OpenCL
device — an AMD GPU, the SYCL **CPU** device, or Intel's own OpenCL (`gpu1o`) — pearl runs a **portable
dp4a int8 GEMM** instead (`sycl/pearl.cpp` `search()`): the same int8 A′·B′, but on the regular vector
ALU's 4-wide integer dot product (dp4a; plain scalar MACs on the CPU). It is **bit-identical** to the
XMX/tensor-core paths (int8→int32 is exact; cross-checked tile-for-tile via `MOM_PEARL_CHK`) and
therefore correct everywhere, but far slower — ~**2.2 TH/s** on the B580's OpenCL vs ESIMD's ~30 TH/s at
the same shape, the inherent ALU-vs-systolic-array gap. Level-Zero (`gpu1`/`gpu1z`) keeps the fast ESIMD
path on Intel; only the OpenCL backend takes dp4a. (On an Intel-OpenCL GPU, `MOM_PEARL_ESIMD=1` opts
`gpu1o` back into ESIMD; the dp4a micro-tile is `-DPEARL_ER`/`-DPEARL_EC`, defaulting to the 4×4 B580
sweet spot.)

Caveats: **no real AMD GPU has been tested** (no hardware). AMD wavefronts are 32/64 vs Intel's 16, so
some cooperative kernels may need sub-group-size handling (pearl's dp4a kernel deliberately uses none),
and pearl's dp4a micro-tile may want re-tuning on a real AMD register file. The B580's OpenCL driver is
the closest available stand-in.

# Donation

By default, miner donates 1% of hashrate (can be disabled in config).

# Distribution

Miner mom.node dynamic library can be compiled and run from sources using `./r.sh` script that
will build a Docker container with the SYCL compilers and produce the one unified mom.node that runs
on both Intel and NVIDIA GPUs (a dual-compiler build: oneAPI icx for host objects, the intel/llvm
nightly clang for the SYCL device images + link). Docker buildx is required so builds use BuildKit
instead of Docker's deprecated legacy builder:

```
git clone https://github.com/MoneroOcean/mo-miner.git
cd mo-miner
./r.sh
```

Install buildx from Docker packages when available:

```
sudo apt-get update
sudo apt-get install docker-buildx-plugin
docker buildx version
```

If your distribution does not package it, install the Docker CLI plugin manually from the
[Docker buildx releases](https://github.com/docker/buildx/releases):

```
mkdir -p ~/.docker/cli-plugins
curl -fsSL https://github.com/docker/buildx/releases/download/v0.33.0/buildx-v0.33.0.linux-amd64 \
  -o ~/.docker/cli-plugins/docker-buildx
chmod +x ~/.docker/cli-plugins/docker-buildx
docker buildx version
```

Tagged GitHub releases build Linux x86-64 `.tgz` and Windows x86-64 `.zip` archives with a
`mom` executable, precompiled `mom.node`, and required SYCL runtime libraries. The release
executable embeds the Node.js control plane, so Docker and system Node.js are not required to run
the release artifact.

# Host GPU runtime (Linux)

The release archive bundles mom's SYCL runtime user-space in `libs/` (libsycl, the Unified Runtime
adapters for both Level-Zero and CUDA, libhwloc, the Intel OpenCL CPU runtime, and the kawpow
kernel-compiler JIT). You only need the host GPU driver/runtime that exposes the GPU. On Ubuntu 24.04 / 26.04 (aim for 26.04) these are plain apt packages from the distribution
itself -- **no extra apt repositories and no downloads**. Run the bundled installer, or apt directly.

## Intel GPU

```
sudo ./install.sh    # bundled in the release archive
# or straight from this repo, without the archive:
curl -fsSL https://raw.githubusercontent.com/MoneroOcean/mo-miner/master/scripts/install.sh | sudo bash
# or apt directly:
sudo apt install intel-opencl-icd libze-intel-gpu1 libze1 ocl-icd-libopencl1
```

`intel-opencl-icd` is the Intel OpenCL GPU driver (cn/gpu and OpenCL GPU devices), `libze-intel-gpu1`
the Level-Zero GPU driver (`intel-level-zero-gpu` on older Ubuntu), and `libze1` / `ocl-icd-libopencl1`
the Level-Zero and OpenCL loaders. Verify with `./mom algo_params` -- a `gpu1` device should appear.

## NVIDIA GPU

```
sudo ./install-nvidia.sh    # bundled in the release archive; then reboot
# or straight from this repo, without the archive:
curl -fsSL https://raw.githubusercontent.com/MoneroOcean/mo-miner/master/scripts/install-nvidia.sh | sudo bash
# or apt directly (server = headless/datacenter, -open = Turing+ open kernel modules):
sudo apt install nvidia-driver-580-server-open   # or newer; reboot afterwards
```

Only the proprietary NVIDIA driver is needed (>= 560 for the bundled CUDA 12.6 runtime) -- **not** the
CUDA toolkit. `install-nvidia.sh` picks a driver via `ubuntu-drivers`/apt and is a no-op if one is
already present. After the reboot, `./mom algo_params` should list a `gpu1` device. When running mom
inside Docker, add `--gpus all` (needs `nvidia-container-toolkit`).

One exception: the full-speed **ProgPoW family** (**kawpow**, **firopow**, **evrprogpow**) recompiles
its per-period kernel at runtime (a SYCL-source JIT that folds the ProgPoW program to constants). That
JIT compiles SYCL *source* on the host, so it needs **two** things the driver alone does not provide:
(1) a CUDA **toolkit** -- `libdevice` *and* the CUDA headers under `/usr/local/cuda` (not just the
driver), and (2) a host **C++ toolchain** (`g++`/libstdc++ -- the JIT `#include`s `<type_traits>` etc.).
If either is missing, kawpow/firopow/evrprogpow automatically fall back to a correct ahead-of-time
kernel at roughly a third of the JIT speed (~4 vs ~13 MH/s on an L4); every other algo is unaffected.
For full ProgPoW-family throughput on a driver-only host, install both, e.g. `sudo apt install
nvidia-cuda-toolkit g++` (or NVIDIA's `cuda-toolkit-12-6` for a `/usr/local/cuda` with both
`nvvm/libdevice/libdevice.10.bc` and the CUDA headers, plus `g++`). Verified on a fresh L4: with only
the driver, kawpow JIT errored through three successive stages (missing libdevice -> missing
`<type_traits>` -> missing PTX target) and ran at 4.2 MH/s; with a full toolkit + `g++` it recovered to
13.8 MH/s.

## AMD GPU (OpenCL)

```
sudo ./install.sh    # auto-detects an AMD-only box and installs Mesa OpenCL (rusticl)
# or straight from this repo, without the archive:
curl -fsSL https://raw.githubusercontent.com/MoneroOcean/mo-miner/master/scripts/install.sh | sudo bash
# or apt directly:
sudo apt install mesa-opencl-icd ocl-icd-libopencl1 clinfo
```

An AMD GPU runs through the generic SYCL **OpenCL** backend (the same `spir64` image, JIT-compiled by
the GPU's OpenCL driver). On an AMD-only box mom auto-selects the OpenCL GPU as `gpu1` -- no flag or
config needed. `install.sh` detects AMD via `lspci` and installs Mesa's `mesa-opencl-icd` (rusticl) from
plain apt; rusticl is opt-in per driver, so you may need `RUSTICL_ENABLE=radeonsi ./mom algo_params`.
For wider algo coverage, AMD's own **ROCm** OpenCL runtime (`rocm-opencl-runtime`, from AMD's apt repo)
is the alternative. **AMD is untested** (no hardware) -- see "OpenCL backend / AMD GPUs" above for the
caveats (sub-group size, pearl dp4a micro-tile tuning). Verify with `./mom algo_params`.

# Host GPU runtime (Windows)

Windows release archives include `install.bat` and `install.ps1`. From an **elevated** (Administrator)
Command Prompt, run `install.bat` to install the Intel OpenCL CPU runtime silently:

```
install.bat
```

Or straight from this repo, without the archive -- a single Command Prompt line (it downloads the
script with the built-in `curl` and hands it to PowerShell for you, so no PowerShell knowledge is
needed; still run it from an elevated Command Prompt):

```
curl -fsSL -o "%TEMP%\mom-install.ps1" https://raw.githubusercontent.com/MoneroOcean/mo-miner/master/scripts/install.ps1 && powershell -NoProfile -ExecutionPolicy Bypass -File "%TEMP%\mom-install.ps1"
```

Windows GPU Level Zero support is supplied by the Intel Graphics Driver, not by a small standalone
`libze` package. Keep the Intel Graphics Driver current if `gpuN` Level Zero devices are missing or
slow on Windows. The Windows release package already bundles the oneAPI SYCL and Level Zero loader
DLLs needed by mom. To also let the installer attempt a silent Intel Graphics Driver update
through `winget`, run:

```
install.bat -InstallIntelGraphicsDriver
```

# Usage example

On Linux if you run miner like that for the first time it will benchmark MoneroOcean pool algos
supported by mom plus `rx/2`, then start mining. Use `--bench_algo_params 2` to benchmark
every algo supported locally before mining instead. This is full local benchmark example output for
Intel i7-11700K CPU and Intel Arc B580 GPU:

```
$ ./mom mine gulf.moneroocean.stream:20001tls 89TxfrUmqJJcb1V124WsUzA78Xa3UYHt7Bg8RGMhXVeZYPN8cE5CZEk58Y1m23ZMLHN7wYeJ9da5n5MXharEjrm41hSnWHL --save_config config.json
cpu1: Intel(R) OpenCL
gpu1: Intel(R) oneAPI Unified Runtime over Level-Zero V2
gpu1o: Intel(R) OpenCL Graphics
gpu1z: Intel(R) oneAPI Unified Runtime over Level-Zero V2
2026-06-15 13:10:29 Doing algo benchmarks...
2026-06-15 13:12:04 Algo autolykos2 (gpu1*8388608) hashrate: 37.55 MH/s (37.55 MH/s)
2026-06-15 13:13:10 Algo c29 (gpu1*1) hashrate: 2.73 H/s (2.73 H/s)
2026-06-15 13:14:38 Algo cn/gpu (gpu1o*1280) hashrate: 2.94 KH/s (2.94 KH/s)
2026-06-15 13:16:07 Algo etchash (gpu1*33554432) hashrate: 21.10 MH/s (21.10 MH/s)
2026-06-15 13:17:08 Algo ghostrider (cpu*8^8) hashrate: 1.66 KH/s (211.34 H/s, 200.52 H/s, 212.52 H/s, 204.79 H/s, 200.47 H/s, 204.75 H/s, 211.35 H/s, 212.58 H/s)
2026-06-15 13:19:04 Algo kawpow (gpu1*37282560) hashrate: 20.92 MH/s (20.92 MH/s)
2026-06-15 13:20:13 Algo panthera (cpu*4^16) hashrate: 4.21 KH/s (255.77 H/s, 263.38 H/s, 266.65 H/s, 259.05 H/s, 265.44 H/s, 260.45 H/s, 259.93 H/s, 270.73 H/s, 285.56 H/s, 263.45 H/s, 252.52 H/s, 262.90 H/s, 253.04 H/s, 269.66 H/s, 268.26 H/s, 255.31 H/s)
2026-06-15 13:21:37 Algo pearl (gpu1*131072) hashrate: 52.11 TH/s (52.11 TH/s)
2026-06-15 13:22:39 Algo rx/0 (cpu*8) hashrate: 6.16 KH/s (6.16 KH/s)
2026-06-15 13:23:42 Algo rx/2 (cpu*8) hashrate: 5.10 KH/s (5.10 KH/s)
2026-06-15 13:24:44 Algo rx/arq (cpu*16) hashrate: 39.42 KH/s (39.42 KH/s)
2026-06-15 13:24:44 Connecting to primary gulf.moneroocean.stream:20001tls pool
2026-06-15 13:24:44 Got new cn/gpu algo job with 269.06 KH/share target and 2095451 height
2026-06-15 13:25:42 Got new cn/gpu algo job with 506.00 KH/share target and 2095451 height
2026-06-15 13:25:49 Algo cn/gpu (gpu1o*1280) hashrate: 2.95 KH/s (2.95 KH/s)
2026-06-15 13:26:28 Got new cn/gpu algo job with 515.42 KH/share target and 2095452 height
2026-06-15 13:26:36 Share accepted by the pool (1/0)
...
```

Next time you can reuse saved config.json file to avoid running benchmarks again before mining:

```
$ ./mom mine ./config.json
2026-06-15 13:30:01 Loading config file ./config.json
2026-06-15 13:30:01 Connecting to primary gulf.moneroocean.stream:20001tls pool
2026-06-15 13:30:01 Got new cn/gpu algo job with 341.49 KH/share target and 2095460 height
2026-06-15 13:30:09 Share accepted by the pool (1/0)
...
```

Saved `algo_params.*.perf` values are local hashrates in H/s. mom advertises KawPow to
MoneroOcean as `kawpow1` with raw H/s while continuing to mine pool jobs named `kawpow`. Cycle
algorithms whose protocol units are solutions per second, currently `c29`, are converted
automatically when sending `algo-perf`. `pearl` reports GEMM multiply-accumulate throughput (TH/s),
not H/s; MoneroOcean does not switch to it, so its perf is informational only.

Without parameters miner will show help:

```
$ ./mom

# Node.js/SYCL based CPU/GPU miner v0.7.1
$ ./mom <directive> <parameter>+ [<option>+]

Directives:
  mine  (<pool_address:port[tls]> <login> [<pass>]|<config.json>)
  test  <algo> <result_hash_hex_str>
  bench <algo>
  algo_params

Options:
--job '{...}':                      JSON string of the default job params (mostly used in test/bench mode)
  --job.dev:                        device config line "[<dev>[*B][^P],]+", dev = {cpu, gpu<N>, cpu<N>}, N = device number, B = hash batch size, P = number of parallel processes ("cpu" by default)
  --job.blob_hex:                   hexadecimal string of input blob ("0305A0DBD6BF05CF16E503F3A66F78007CBF34144332ECBFC22ED95C8700383B309ACE1923A0964B00000008BA939A62724C0D7581FCE5761E9D8A0E6A1C3F924FDD8493D1115649C05EB601" by default)
  --job.seed_hex:                   hexadecimal string of seed hash blob (used for rx algos) ("3132333435363738393031323334353637383930313233343536373839303132" by default)
  --job.height:                     Block height used by some algos (0 by default)

--pool_time '{...}':                JSON string of pool related timings (in seconds)
  --pool_time.stats:                time to show pool mining stats (600 by default)
  --pool_time.connect_throttle:     time between pool connection attempts (60 by default)
  --pool_time.primary_reconnect:    time to try to use primary pool if currently on backup pool (90 by default)
  --pool_time.first_job_wait:       consider pool bad if no first job after connection (15 by default)
  --pool_time.close_wait:           keep pool socket to submit delayed jobs (10 by default)
  --pool_time.donate_interval:      time before donation pool is activated (6000 by default)
  --pool_time.donate_length:        donation pool work time (60 by default)
  --pool_time.keepalive:            interval to send keepalive messages (300 by default)

--add.pool '{["<key>": <value>,]+}': add backup pool, defined by the following keys:
  url:                              pool DNS or IP address
  port:                             pool port
  is_tls:                           is pool port is encrypted using TLS/SSL (false by default)
  tls_verify:                       verify pool TLS/SSL certificate (false by default)
  is_nicehash:                      nicehash nonce mining mode support (false by default)
  is_keepalive:                     sends keepalive messages to the pool to avoid disconnect (true by default)
  use_subscribe:                    pearl pools: use mining.subscribe+authorize handshake; set false for the login-dialect pearl pool (pearlpool.cloud) and the MoneroOcean donate pool (true by default)
  worker:                           pearl subscribe-dialect worker name (mining.authorize) ("mom" by default)
  login:                            pool login data
  pass:                             pool password ("" by default)

--new.default_msr.<name> '{["<key>": <value>,]+}': stores default MSR register values to restore them without reboot, keys should be hex strings with 0x prefix
  value:                            MSR register value in hex string with 0x prefix format
  mask:                             MSR register mask in hex string with 0x prefix format ("0xFFFFFFFFFFFFFFFF" by default)

--new.algo_param.<name> '{["<key>": <value>,]+}': new algo params, defined by the following keys:
  dev:                              device config line "[<dev>[*B][^P],]+", dev = {cpu, gpu<N>, cpu<N>}, N = device number, B = hash batch size, P = number of parallel processes ("cpu" by default)

--log_level:                        log level: 0=minimal, 1=verbose, 2=network debug, 3=compute core debug (0 by default)
--bench_algo_params:                benchmark algo params before mining: 0=skip, 1=active MoneroOcean coin algos plus rx/2, 2=all supported algos (1 by default)
--save_config:                      file name to save config in JSON format (only for mine directive) ("" by default)
2026-06-15 13:58:03 ERROR: No directive specified
```

You can run test and benchmark separately for algo you need like this:

```
./r.sh node mom.js test cn/gpu e55cb23e51649a59b127b96b515f2bf7bfea199741a0216cf838ded06eff82df --job '{"algo":"cn/gpu","dev":"gpu1*8"}'
./r.sh node mom.js bench cn/gpu --job '{"algo":"cn/gpu","dev":"gpu1*1280"}'
./r.sh node mom.js bench etchash --job '{"algo":"etchash","dev":"gpu1*256"}'
./r.sh node mom.js bench autolykos2 --job '{"algo":"autolykos2","dev":"gpu1*1"}'
./r.sh node mom.js bench pearl --job '{"algo":"pearl","dev":"gpu1*131072"}'
```

Project test suites are npm entry points:

```
npm test
npm run test:perf
npm run test:perf:ghostrider
```

`npm test` runs the hash-vector suite under Docker and skips GPU cases when no GPU device is
available. `npm run test:perf` benchmarks every supported algo with mom's detected mining
device config and prints each hashrate in the same test reporter output. Individual benchmark entry
points are available as `npm run test:perf:<algo>` for named scripts in `package.json`, for example
`npm run test:perf:rx/0`, `npm run test:perf:cn-heavy/tube`, or `npm run test:perf:c29`. Any
supported algo can also be selected by passing it to the generic perf runner, for example
`npm run test:perf -- etchash` or `npm run test:perf -- autolykos2`. Set
`MOM_PERF_SAMPLES=N` to collect `N` hashrate reports and use the median for perf tests:

```
MOM_PERF_SAMPLES=3 ./r.sh npm run test:perf:etchash
```

Enable huge pages for better performance (check [Huge Pages](https://xmrig.com/docs/miner/hugepages)):

```
sudo sysctl -w vm.nr_hugepages=1280
sudo bash -c "echo vm.nr_hugepages=1280 >> /etc/sysctl.conf"
```

For repeatable RandomX performance tests, make sure other services are not consuming huge pages or
CPU. A local `monerod` can reserve hugetlb pages and lower `rx/0` hashrate; stop it before perf
tests, or increase `vm.nr_hugepages` enough for both processes. On systems that run it as
`xmr.service`:

```
sudo systemctl stop xmr.service
npm run test:perf:rx/0
sudo systemctl start xmr.service
```

# License

mom is licensed under [GPL-3.0-or-later](LICENSE).
