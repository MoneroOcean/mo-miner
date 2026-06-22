# About

mom (short name for the **MO-Miner**) is an open-source cryptocurrency miner
that is built upon high performance xmrig CPU miner sources with front-end and network backend
rewritten in Node.js to significantly simplify its code. GPU mining sources are also simplified
and rewritten in SYCL from OpenCL/CUDA. The main goal of this project is to make simple,
easy to extend open-source miner with native miner performance.

The executable, release archives, and config keys use `mom`; the GitHub repository remains
[MoneroOcean/mo-miner](https://github.com/MoneroOcean/mo-miner).

Miner supports algo switching on pools such as `gulf.moneroocean.stream`. By default, startup
algo-parameter benchmarking covers active MoneroOcean coin algos implemented by mom plus `rx/2`
and `pearl`.

## Supported algos

* CPU: All xmrig miner CPU supported algos with similar performance.
* GPU/SYCL: Check table below.

Missing measurements use `-`; known measurements without a same-platform reusable SOTA run use `(%?)`.
Percentages compare mom against the miner(s) named in `SOTA ref(s)`. A bare miner name means it is
used as the reference for Intel, NVIDIA, and AMD; tagged entries limit the miner to the listed GPU
platform(s).
`memory-bandwidth bound` describes algorithms such as `etchash` whose speed is mainly limited by
GPU memory bandwidth; it is not a miner name.

Support: `mo` = MoneroOcean can serve it; `mom` = implemented by mom; `-` = backlog/not implemented.
`Top coin` is the highest-market-cap known mineable coin for the exact algo/variant.

| Algo                  | Support | Top coin | Intel B580           | OpenCL              | NVIDIA L4          | Windows L4        | SOTA ref(s)                                                                  |
| --------------------- | ------- | -------- | -------------------- | ------------------- | ------------------ | ----------------- | ---------------------------------------------------------------------------- |
| **Cuckoo Cycle**      |         |          |                      |                     |                    |                   |                                                                              |
| `c29`                 | mo,mom  | TARI     | 2.73 g/s (%?)        | 2.50 g/s (%?)       | 4.93 g/s (93%)     | 4.86 g/s          | lolMiner (NVIDIA/AMD; no B580 ref)                                           |
| `cuckaroo30`          | -       | CTXC     | -                    | -                   | -                  | -                 | lolMiner                                                                     |
| **ProgPoW family**    |         |          |                      |                     |                    |                   |                                                                              |
| `kawpow`              | mo,mom  | RVN      | 20.92 MH/s (110%)    | 18.63 MH/s (98%)    | 13.17 MH/s (84%)   | 13.28 MH/s        | SRBMiner (Intel), GMiner/Rigel (NVIDIA)                                      |
| `evrprogpow`          | mom     | EVR      | 21.05 MH/s (110%)    | 18.69 MH/s (97%)    | 12.83 MH/s (82%)   | 12.95 MH/s        | SRBMiner                                                                     |
| `firopow`             | mom     | FIRO     | 20.95 MH/s (112%)    | 18.58 MH/s (99%)    | 12.82 MH/s (82%)   | 12.95 MH/s        | SRBMiner                                                                     |
| `meowpow`             | mom     | MEWC     | 19.08 MH/s (101%)    | 18.55 MH/s (99%)    | 14.97 MH/s (96%)   | 14.74 MH/s        | SRBMiner                                                                     |
| `progpowz`            | -       | ZANO     | -                    | -                   | -                  | -                 | SRBMiner                                                                     |
| **FishHash family**   |         |          |                      |                     |                    |                   |                                                                              |
| `fishhash`            | mom     | IRON     | 11.82 MH/s (92%)     | 11.81 MH/s (92%)    | 7.40 MH/s (38%)    | 7.33 MH/s         | SRBMiner (Intel), Rigel (NVIDIA)                                             |
| `karlsenhashv2`       | mom     | KLS      | 10.40 MH/s (83%)     | 10.37 MH/s (83%)    | 10.54 MH/s (55%)   | 10.06 MH/s        | SRBMiner (Intel), Rigel (NVIDIA)                                             |
| **Equihash**          |         |          |                      |                     |                    |                   |                                                                              |
| `equihash125_4`       | mom     | FLUX     | 18.91 Sol/s (%?)     | 19.07 Sol/s (%?)    | 15.74 Sol/s (31%)  | 11.86 Sol/s       | lolMiner (Intel Arc A-series/NVIDIA; no B580 ref)                            |
| `equihash144_5`       | -       | BTG      | -                    | -                   | -                  | -                 | lolMiner (Intel, AMD), miniZ (NVIDIA)                                        |
| `equihash192_7`       | -       | ZCL      | -                    | -                   | -                  | -                 | miniZ (NVIDIA), lolMiner (Intel, AMD)                                        |
| **Misc**              |         |          |                      |                     |                    |                   |                                                                              |
| `autolykos2`          | mo,mom  | ERG      | 37.55 MH/s (112%)    | 36.03 MH/s (108%)   | 77.02 MH/s (78%)   | 69.50 MH/s        | SRBMiner (Intel), GMiner/Rigel (NVIDIA)                                      |
| `cn/gpu`              | mo,mom  | RYO      | 2.94 KH/s (107%)     | 2.94 KH/s (107%)    | 2.68 KH/s (88%)    | 1.87 KH/s         | SRBMiner                                                                     |
| `beamhash3`           | mom     | BEAM     | 7.68 Sol/s (%?)      | 7.66 Sol/s (%?)     | 8.20 Sol/s (32%)   | 6.04 Sol/s        | lolMiner (Intel Arc A-series/NVIDIA), GMiner (NVIDIA); no B580 ref           |
| `pearl`               | mom     | PRL      | 52.11 TH/s (158%)    | ~2.2 TH/s (7%)      | 34.58 TH/s (178%)  | 34.73 TH/s        | ARC-miner (Intel), BzMiner (NVIDIA)                                          |
| `pyrinhashv2`         | mom     | PYI      | 385.07 MH/s (%?)     | 387.80 MH/s (%?)    | 258.08 MH/s (5%)   | 216.24 MH/s       | SRBMiner 2.6.8 (Intel pre-B580), lolMiner (NVIDIA); no B580 ref              |
| `hoohashv2`           | -       | HTN      | -                    | -                   | -                  | -                 | SRBMiner                                                                     |
| `nxlhash`             | -       | NXL      | -                    | -                   | -                  | -                 | SRBMiner                                                                     |
| `octopus`             | -       | CFX      | -                    | -                   | -                  | -                 | lolMiner (Intel, AMD), GMiner (NVIDIA)                                       |
| `verthash`            | -       | VTC      | -                    | -                   | -                  | -                 | SRBMiner, VerthashMiner (AMD/NVIDIA)                                         |
| `walahash`            | -       | WALA     | -                    | -                   | -                  | -                 | SRBMiner (Intel), Rigel (NVIDIA)                                             |
| **ASIC-exposed**      |         |          |                      |                     |                    |                   |                                                                              |
| `etchash`             | mo,mom  | ETC      | 21.10 MH/s (100%)    | 20.00 MH/s (100%)   | 28.84 MH/s (99%)   | 27.53 MH/s        | GMiner/Rigel (NVIDIA)                                                        |
| `kheavyhash`          | mom     | KAS      | 160.89 MH/s (%?)     | 162.86 MH/s (%?)    | 303.52 MH/s (42%)  | 288.31 MH/s       | BzMiner/GMiner (NVIDIA), BzMiner/SRBMiner (Intel; no stable B580 ref)        |

Platform notes:

| Platform | OS / kernel | GPU / backend | Driver / runtime | Toolchain | Power / clocks / VRAM | Notes |
| -------- | ----------- | ------------- | ---------------- | --------- | --------------------- | ----- |
| Intel B580 | Ubuntu 24.04.4 LTS, Linux `6.17.0-35-generic` | Intel B580, Level Zero | `xe`, `intel-opencl-icd`/`libze-intel-gpu1` `26.22.38646.4-0`, `libze1` `1.28.6` | - | - | GuC/HuC versions not recorded. |
| OpenCL | Ubuntu 24.04.4 LTS, Linux `6.17.0-35-generic` | Intel B580, SYCL OpenCL | `intel-opencl-icd` `26.22.38646.4-0` | - | - | `(%?)` cells have no stable same-B580 SOTA reference. |
| NVIDIA L4 | Ubuntu 26.04 LTS, Linux `7.0.0-14-generic` | NVIDIA L4 Ada (`sm_89`), DPC++ CUDA | NVIDIA `595.71.05` open kernel module, driver CUDA `13.2` | CUDA toolkit `12.4.131`, Node.js `22.22.1`, `g++` `15.2.0` | 72 W, 2040 MHz core / 6251 MHz memory, 23034 MiB | Linux SOTA reference platform. |
| Windows L4 | Windows Server 2022 Standard `10.0.20348` | NVIDIA L4, DPC++ CUDA | NVIDIA `596.36` | CUDA toolkit `12.6` (`nvcc`, `cudart`, `nvrtc`, `nvrtc_dev`) | 72 W, 2040 MHz core / 6251 MHz memory, 23034 MiB | Release hash-vector suite and ProgPoW source-JIT validated. |

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
itself -- **no extra apt repositories and no downloads**. The bundled `install.sh` auto-detects the
GPU(s) present (Intel / AMD / NVIDIA) and installs each one's runtime; or apt directly.

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
sudo ./install.sh    # bundled in the release archive; auto-detects NVIDIA; then reboot
# or straight from this repo, without the archive:
curl -fsSL https://raw.githubusercontent.com/MoneroOcean/mo-miner/master/scripts/install.sh | sudo bash
# or apt directly (server = headless/datacenter, -open = Turing+ open kernel modules):
sudo apt install nvidia-driver-580-server-open nvidia-cuda-toolkit g++   # or newer; reboot afterwards
```

The unified `install.sh` detects the NVIDIA GPU and installs both (a) the proprietary driver (>= 560 for
the bundled CUDA 12.6 runtime; via `ubuntu-drivers`/apt, a no-op if one is already present) and (b) the
CUDA toolkit + `g++` that the **ProgPoW family** needs for full speed. After the reboot, `./mom
algo_params` should list a `gpu1` device. When running mom inside Docker, add `--gpus all` (needs
`nvidia-container-toolkit`).

Why the toolkit: the full-speed **ProgPoW family** (`kawpow`, `firopow`, `evrprogpow`, `meowpow`)
recompiles its per-period kernel at runtime (a SYCL-source JIT that folds the ProgPoW program to constants). That
JIT compiles SYCL *source* on the host, so beyond the driver it needs a CUDA **toolkit** (`libdevice` +
`ptxas`) and a host **C++ toolchain** (`g++`/libstdc++ -- it `#include`s `<type_traits>` etc.).
`install.sh` installs both (`nvidia-cuda-toolkit` + `g++`) and the JIT auto-detects the toolkit. **If you
set up only the driver manually** (skipping `install.sh`/the toolkit),
kawpow/firopow/evrprogpow/meowpow fall back to a correct ahead-of-time kernel at roughly a third of
the JIT speed (~4 vs ~13.8 MH/s on an L4; verified end-to-end); every other algo is unaffected.

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
is the alternative. **AMD is untested** (no hardware) -- see "Backend notes" above for the caveats
(sub-group size, pearl dp4a micro-tile tuning). Verify with `./mom algo_params`.

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

On NVIDIA Windows hosts, the installer can also install the current data-center driver when
`nvidia-smi` is missing and install the CUDA toolkit pieces used by runtime SYCL-source JIT paths
(`nvcc`, `cudart`, `nvrtc`, `nvrtc_dev`):

```
install.bat -InstallNVIDIADriver -InstallCudaToolkit
```

Reboot after driver installation, then verify `nvidia-smi`, `nvcc -V`, and `mom.cmd algo_params`.

# Usage example

On Linux if you run miner like that for the first time it will benchmark MoneroOcean pool algos
supported by mom plus `rx/2`, then start mining. Use `--bench_algo_params 2` to benchmark
every algo supported locally before mining instead. This is full local benchmark example output for
Intel i7-11700K CPU and Intel Arc B580 GPU. Run mom as **root** (or grant it the MSR + huge-page caps)
so RandomX (`rx/*`) can apply the MSR mod and 2 MiB/1 GiB huge pages -- without that the CPU RandomX
rates are ~40% lower (e.g. `rx/0` ~3.7 instead of 6.16 KH/s; GPU algos are unaffected):

```
$ ./mom mine gulf.moneroocean.stream:20001tls 89TxfrUmqJJcb1V124WsUzA78Xa3UYHt7Bg8RGMhXVeZYPN8cE5CZEk58Y1m23ZMLHN7wYeJ9da5n5MXharEjrm41hSnWHL --save_config config.json
cpu1: Intel(R) OpenCL
gpu1: Intel(R) oneAPI Unified Runtime over Level-Zero V2
gpu1o: Intel(R) OpenCL Graphics
gpu1z: Intel(R) oneAPI Unified Runtime over Level-Zero V2
2026-06-22 15:21:15 Doing algo benchmarks...
2026-06-22 15:23:09 Algo autolykos2 (gpu1*8388608) hashrate: 37.66 MH/s (37.66 MH/s)
2026-06-22 15:24:14 Algo c29 (gpu1*1) hashrate: 2.78 H/s (2.78 H/s)
2026-06-22 15:25:58 Algo cn/gpu (gpu1o*1280) hashrate: 2.94 KH/s (2.94 KH/s)
2026-06-22 15:27:29 Algo etchash (gpu1*33554432) hashrate: 21.10 MH/s (21.10 MH/s)
2026-06-22 15:28:30 Algo ghostrider (cpu*8^8) hashrate: 1.65 KH/s (210.85 H/s, 210.76 H/s, 199.94 H/s, 204.12 H/s, 211.90 H/s, 211.79 H/s, 204.13 H/s, 199.92 H/s)
2026-06-22 15:30:30 Algo kawpow (gpu1*37282560) hashrate: 20.95 MH/s (20.95 MH/s)
2026-06-22 15:31:40 Algo panthera (cpu*4^16) hashrate: 4.20 KH/s (261.93 H/s, 264.43 H/s, 273.02 H/s, 267.38 H/s, 263.41 H/s, 284.90 H/s, 254.31 H/s, 268.43 H/s, 262.42 H/s, 248.54 H/s, 256.79 H/s, 250.11 H/s, 259.60 H/s, 255.85 H/s, 258.94 H/s, 271.94 H/s)
2026-06-22 15:33:05 Algo pearl (gpu1*131072) hashrate: 51.58 TH/s (51.58 TH/s)
2026-06-22 15:34:07 Algo rx/0 (cpu*8) hashrate: 6.11 KH/s (6.11 KH/s)
2026-06-22 15:35:10 Algo rx/2 (cpu*8) hashrate: 5.03 KH/s (5.03 KH/s)
2026-06-22 15:36:12 Algo rx/arq (cpu*16) hashrate: 38.95 KH/s (38.95 KH/s)
2026-06-22 15:36:12 Connecting to primary gulf.moneroocean.stream:20001tls pool
2026-06-22 15:36:12 Got new cn/gpu algo job with 206.29 KH/share target and 2100441 height
2026-06-22 15:37:11 Share accepted by the pool (1/0)
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

## Direct pool examples

Use this template for non-MoneroOcean pools. mom infers the stratum dialect from `--job.algo`.

```
./r.sh node mom.js mine <endpoint> <wallet.worker> <dev / command suffix> --bench_algo_params 0
```

| Coin | Algo          | Endpoint                           | Donation address (owner)                                                                                         | Dev / command suffix                                                   |
| ---- | ------------- | ---------------------------------- | ---------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| RVN  | kawpow        | stratum.ravenminer.com:13838tls    | `RSJZNSvzt3PJdGVKahSczRrhinc24KA6wU` (hans-schmidt, Ravencoin/Evrmore maintainer)                                | --job.algo kawpow --job.dev gpu1*37282560                              |
| FIRO | firopow       | pool.woolypooly.com:3104tls        | `a4vQ7zr5CEBDEdNQBFVvHcM1BRVYKEnuEv` (Firo Core Team funding proposal)                                           | --job.algo firopow --job.dev gpu1*37282560                             |
| EVR  | evrprogpow    | us-east.mining4people.com:24173tls | `EaBGnWtDiAseYZiyvNT1u3WTjAeYtAR7MV` (hans-schmidt, Evrmore maintainer)                                          | --job.algo evrprogpow --job.dev gpu1*37282560                          |
| MEWC | meowpow       | stratum-eu.rplant.xyz:17120tls     | `MPyNGZSSZ4rbjkVJRLn3v64pMcktpEYJnU` (MeowCoin donation address)                                                 | --job.algo meowpow --job.dev gpu1*37282560                             |
| PRL  | pearl         | pearl.herominers.com:1200tls       | `prl1p79wzxcvatcsmnzp9xp0ep0rvfe9ans05mjtxnt4d9x0qqej0mtdqfrezc0` (ARC-miner PRL donation address)               | --job.algo pearl --job.dev gpu1*131072                                 |
| IRON | fishhash      | ironfish.herominers.com:1145tls    | `66e044578b31c6c4c05810b0e5281bdf36138ad41bf6844ba317dc7c506bf9ac` (GMiner/Rigel bundled sample)                 | --job.algo fishhash --job.dev gpu1*33554432                            |
| KAS  | kheavyhash    | kaspa.herominers.com:1206tls       | `precqv0krj3r6uyyfa36ga7s0u9jct0v4wg8ctsfde2gkrsgwgw8jgxfzfc98` (Kaspa Devfund)                                  | --job.algo kheavyhash --job.dev gpu1*47934720                          |
| KLS  | karlsenhashv2 | pool.woolypooly.com:3132           | `qzrq7v5jhsc5znvtfdg6vxg7dz5x8dqe4wrh90jkdnwehp6vr8uj7csdss2l7` (Karlsen Devfund)                                | --job.algo karlsenhashv2 --job.dev gpu1*33554432                       |
| PYI  | pyrinhashv2   | ca.pyrin.herominers.com:1177       | `qq92h3nryfwq0gkh73cwvjh9hhqlq2mank9sfxtgc99hqwn2ec6u2gszphr0u` (lolMiner bundled sample)                        | --job.algo pyrinhashv2 --job.dev gpu1*47934720                         |
| FLUX | equihash125_4 | flux.herominers.com:1200tls        | `t1Mzja9iJcEYeW5B4m4s1tJG8M42odFZ16A` (Flux development address)                                                 | --job.algo equihash125_4 --job.dev gpu1*1                              |
| BEAM | beamhash3     | beam.2miners.com:5252tls           | `2346a827cb56ca74e34680593e50d7b1fa4a169332415a1d5984c6f874395c3684b` (Wilke Trei, Beam)                         | --job.algo beamhash3 --job.dev gpu1*1                                  |

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
  --job.algo:                       algo name of the job (only used with "mine" directive) (null by default)
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
  protocol:                         pool protocol override: login, raven, eth, ethproxy, erg, pearl, equihash, kaspa, beam, or ironfish (null by default)
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
