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
Windows is supported (see the runtime/install notes below). 2+ socket CPUs, ARM CPUs, and AMD GPUs are
not yet tested — support is possible and WIP.

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

mom also builds for NVIDIA GPUs from the same SYCL kernels using the DPC++ CUDA backend
(ahead-of-time compiled to PTX/SASS); the NVIDIA Linux release artifact is
`mom-v<version>-lin-nvidia.tgz`. Hashrates measured on an NVIDIA L4 (Ada, sm_89), each
compared against the best closed-source miner benchmarked on the same card:

| Algo | mom (L4) | SOTA reference (same L4) | % of SOTA |
| --- | --- | --- | --- |
| cn/gpu | 2.65 KH/s | SRBMiner-MULTI 3.04 KH/s | 87% |
| c29 (Cuckaroo29) | 4.83 g/s | lolMiner ~5.3 g/s | 91% |
| kawpow | 13.2 MH/s | Rigel 15.56 MH/s | 85% |
| etchash | 28.9 MH/s | ~28.8 MH/s (memory-bandwidth bound) | ~parity |
| autolykos2 | 76.5 MH/s | lolMiner 98.19 MH/s | 78% |
| pearl | 33.6 TH/s | no NVIDIA SOTA (ARC-miner is Intel-only) | — |

SOTA references benchmarked on the same L4: lolMiner (`--benchmark AUTOLYKOS2` / `CR29`), Rigel
(`-a kawpow`), SRBMiner-MULTI (`--algorithm cryptonight_gpu`).

# Donation

By default, miner donates 1% of hashrate (can be disabled in config).

# Distribution

Miner mom.node dynamic library can be compiled and run from sources using `./r.sh` script that
will build Docker container with SYCL compiler. Docker buildx is required so builds use BuildKit
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

# Host Intel GPU Runtime

Linux release archives include `install.sh`. Run it as root on Ubuntu 24.04 or 26.04 if GPU
performance is lower outside Docker, or if Level Zero/OpenCL GPU detection is missing:

```
sudo ./install.sh
```

The script does not add an apt repository. It discovers the latest upstream GitHub releases at
runtime, downloads the matching Ubuntu `.deb` packages, verifies GitHub-provided SHA-256 digests,
and installs a coherent GPU compute runtime set:

* Intel compute-runtime: `libze-intel-gpu1` or `intel-level-zero-gpu`, `intel-opencl-icd`,
  `intel-ocloc`, and `libigdgmm*`
* Intel Graphics Compiler: `intel-igc-core-2` and `intel-igc-opencl-2`
* oneAPI Level Zero loader: `libze1` or the older `level-zero` package name

Set `MOM_COMPUTE_RUNTIME_RELEASE`, `MOM_IGC_RELEASE`, or `MOM_LEVEL_ZERO_RELEASE`
to a GitHub release tag to pin a specific version. The installer keeps mom's bundled SYCL
runtime libraries in place. Those bundled libraries include the oneAPI SYCL and Unified Runtime
user-space pieces mom loads directly; `install.sh` only installs the host GPU driver/runtime
side that exposes Level Zero/OpenCL devices to them. Installing the full oneAPI runtime
system-wide would be much larger than the release-bundled runtime closure.

Windows release archives include `install.bat` and `install.ps1`. Run `install.bat` from an
elevated Administrator command prompt to install the Intel OpenCL CPU runtime silently:

```
install.bat
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

# Node.js/SYCL based CPU/GPU miner v0.7.0
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
