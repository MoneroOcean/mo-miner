## mo-miner

# About

mo-miner is open-source cryptocurrency miner that is built upon high performance xmrig CPU miner
sources with front-end and network backend rewritten in Node.js to significantly simplify its code.
GPU mining sources are also simplified and rewritten in SYCL from OpenCL/CUDA.
The main goal of this project is to make simple, easy to extend open-source miner with native
miner performance.

# Limitations

The only platform tested at this moment is Linux with one socket x86-64 CPU and Intel Arc GPU
(my current development system). 2+ socket CPU, Windows, ARM CPU, nVidia/AMD GPU support
is possible and WIP.

# Supported algos

* CPU: All xmrig miner CPU supported algos with similar performance
* GPU/SYCL: cn/gpu, c29, kawpow, etchash, autolykos2

The GPU/SYCL implementations also run on SYCL CPU devices for hash-vector coverage and fallback
testing, though the CPU mining path remains the preferred CPU path for CPU-focused algos.

Miner supports algo switching if you connect it to algo switching pool like
gulf.moneroocean.stream that auto switches to the most profitable algo.
KawPow, Etchash, and Autolykos2 can also be used on non-MoneroOcean stratum pools that serve those
algos directly.

# Donation

By default, miner donates 1% of hashrate (can be disabled in config).

# Distribution

Miner mo-miner.node dynamic library can be compiled and run from sources using `./r.sh` script that
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
`mo-miner` executable, precompiled `mo-miner.node`, and required SYCL runtime libraries. The release
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

Set `MOMINER_COMPUTE_RUNTIME_RELEASE`, `MOMINER_IGC_RELEASE`, or `MOMINER_LEVEL_ZERO_RELEASE`
to a GitHub release tag to pin a specific version. The installer keeps mo-miner's bundled SYCL
runtime libraries in place. Those bundled libraries include the oneAPI SYCL and Unified Runtime
user-space pieces mo-miner loads directly; `install.sh` only installs the host GPU driver/runtime
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
DLLs needed by mo-miner. To also let the installer attempt a silent Intel Graphics Driver update
through `winget`, run:

```
install.bat -InstallIntelGraphicsDriver
```

# Usage example

On Linux if you run miner like that for the first time it will benchmark all supported algos and
will start mining (this is perf numbers for Intel i7-11700K CPU and Intel Arc B580 GPU):

```
$ ./mo-miner mine gulf.moneroocean.stream:20001tls 89TxfrUmqJJcb1V124WsUzA78Xa3UYHt7Bg8RGMhXVeZYPN8cE5CZEk58Y1m23ZMLHN7wYeJ9da5n5MXharEjrm41hSnWHL --save_config config.json
cpu1: Intel(R) OpenCL
gpu1: Intel(R) oneAPI Unified Runtime over Level-Zero V2
gpu1o: Intel(R) OpenCL Graphics
gpu1z: Intel(R) oneAPI Unified Runtime over Level-Zero V2
2026-06-04 04:38:04 Doing algo benchmarks...
2026-06-04 04:39:04 Algo argon2/chukwa (cpu^16) hashrate: 49.56 KH/s (3.10 KH/s, 3.10 KH/s, 3.10 KH/s, 3.10 KH/s, 3.07 KH/s, 3.07 KH/s, 3.10 KH/s, 3.10 KH/s, 3.10 KH/s, 3.10 KH/s, 3.10 KH/s, 3.10 KH/s, 3.10 KH/s, 3.10 KH/s, 3.10 KH/s, 3.10 KH/s)
2026-06-04 04:40:05 Algo argon2/chukwav2 (cpu^16) hashrate: 16.70 KH/s (1.04 KH/s, 1.05 KH/s, 1.04 KH/s, 1.05 KH/s, 1.05 KH/s, 1.04 KH/s, 1.04 KH/s, 1.04 KH/s, 1.05 KH/s, 1.05 KH/s, 1.05 KH/s, 1.05 KH/s, 1.04 KH/s, 1.05 KH/s, 1.04 KH/s, 1.04 KH/s)
2026-06-04 04:41:05 Algo argon2/wrkz (cpu^16) hashrate: 76.16 KH/s (4.77 KH/s, 4.77 KH/s, 4.77 KH/s, 4.69 KH/s, 4.77 KH/s, 4.76 KH/s, 4.76 KH/s, 4.77 KH/s, 4.69 KH/s, 4.76 KH/s, 4.78 KH/s, 4.78 KH/s, 4.77 KH/s, 4.76 KH/s, 4.78 KH/s, 4.77 KH/s)
2026-06-04 04:42:20 Algo autolykos2 (gpu1*8388608) hashrate: 38.06 MH/s (38.06 MH/s)
2026-06-04 04:43:25 Algo c29 (gpu1*1) hashrate: 2.79 H/s (2.79 H/s)
2026-06-04 04:44:25 Algo cn-heavy/0 (cpu^4) hashrate: 339.46 H/s (86.38 H/s, 82.30 H/s, 85.66 H/s, 85.12 H/s)
2026-06-04 04:45:26 Algo cn-heavy/tube (cpu^4) hashrate: 295.55 H/s (75.03 H/s, 75.68 H/s, 72.18 H/s, 72.66 H/s)
2026-06-04 04:46:26 Algo cn-heavy/xhv (cpu^4) hashrate: 338.58 H/s (81.38 H/s, 86.20 H/s, 84.79 H/s, 86.21 H/s)
2026-06-04 04:47:27 Algo cn-lite/0 (cpu^16) hashrate: 2.34 KH/s (149.60 H/s, 144.47 H/s, 144.58 H/s, 141.51 H/s, 150.22 H/s, 141.63 H/s, 141.26 H/s, 149.49 H/s, 150.69 H/s, 144.82 H/s, 149.63 H/s, 146.83 H/s, 147.32 H/s, 141.45 H/s, 150.85 H/s, 150.41 H/s)
2026-06-04 04:48:27 Algo cn-lite/1 (cpu^16) hashrate: 2.30 KH/s (147.57 H/s, 141.59 H/s, 138.91 H/s, 147.56 H/s, 138.74 H/s, 146.43 H/s, 138.75 H/s, 142.07 H/s, 146.44 H/s, 141.99 H/s, 146.51 H/s, 147.50 H/s, 147.55 H/s, 141.66 H/s, 146.13 H/s, 138.93 H/s)
2026-06-04 04:49:28 Algo cn-pico/0 (cpu*4^16) hashrate: 14.70 KH/s (913.45 H/s, 926.11 H/s, 915.18 H/s, 927.85 H/s, 908.70 H/s, 914.65 H/s, 927.26 H/s, 908.58 H/s, 925.60 H/s, 925.89 H/s, 909.00 H/s, 924.78 H/s, 927.40 H/s, 913.97 H/s, 907.79 H/s, 927.27 H/s)
2026-06-04 04:50:28 Algo cn-pico/tlo (cpu*4^16) hashrate: 13.08 KH/s (824.47 H/s, 824.45 H/s, 823.95 H/s, 824.14 H/s, 824.82 H/s, 809.09 H/s, 823.20 H/s, 813.29 H/s, 807.95 H/s, 814.35 H/s, 813.91 H/s, 824.28 H/s, 824.39 H/s, 809.24 H/s, 814.14 H/s, 808.46 H/s)
2026-06-04 04:51:29 Algo cn/0 (cpu^8) hashrate: 621.13 H/s (79.38 H/s, 76.61 H/s, 79.95 H/s, 80.04 H/s, 76.53 H/s, 74.67 H/s, 74.54 H/s, 79.41 H/s)
2026-06-04 04:52:29 Algo cn/1 (cpu^8) hashrate: 614.90 H/s (75.86 H/s, 79.16 H/s, 78.57 H/s, 73.78 H/s, 73.89 H/s, 78.62 H/s, 75.80 H/s, 79.20 H/s)
2026-06-04 04:53:30 Algo cn/2 (cpu^8) hashrate: 617.00 H/s (79.04 H/s, 76.44 H/s, 77.16 H/s, 76.67 H/s, 78.57 H/s, 74.91 H/s, 75.20 H/s, 79.02 H/s)
2026-06-04 04:54:30 Algo cn/ccx (cpu^8) hashrate: 1.15 KH/s (138.63 H/s, 148.76 H/s, 141.70 H/s, 137.70 H/s, 137.83 H/s, 147.43 H/s, 148.80 H/s, 149.40 H/s)
2026-06-04 04:55:31 Algo cn/double (cpu^8) hashrate: 302.34 H/s (37.23 H/s, 36.10 H/s, 39.11 H/s, 38.51 H/s, 36.93 H/s, 39.05 H/s, 38.47 H/s, 36.94 H/s)
2026-06-04 04:56:31 Algo cn/fast (cpu^8) hashrate: 1.13 KH/s (141.42 H/s, 136.35 H/s, 147.58 H/s, 148.18 H/s, 142.58 H/s, 143.62 H/s, 135.72 H/s, 136.79 H/s)
2026-06-04 04:57:43 Algo cn/gpu (gpu1*1280) hashrate: 2.32 KH/s (2.32 KH/s)
2026-06-04 04:58:43 Algo cn/half (cpu^8) hashrate: 706.89 H/s (80.12 H/s, 89.32 H/s, 97.75 H/s, 83.97 H/s, 94.64 H/s, 83.93 H/s, 80.54 H/s, 96.60 H/s)
2026-06-04 04:59:44 Algo cn/r (cpu^8) hashrate: 548.44 H/s (71.34 H/s, 68.92 H/s, 65.42 H/s, 71.94 H/s, 68.35 H/s, 67.66 H/s, 67.59 H/s, 67.22 H/s)
2026-06-04 05:00:44 Algo cn/rto (cpu^8) hashrate: 537.74 H/s (69.70 H/s, 69.24 H/s, 67.37 H/s, 62.35 H/s, 65.18 H/s, 66.65 H/s, 67.17 H/s, 70.08 H/s)
2026-06-04 05:01:45 Algo cn/rwz (cpu^8) hashrate: 749.60 H/s (90.10 H/s, 95.26 H/s, 92.89 H/s, 97.09 H/s, 91.12 H/s, 91.30 H/s, 95.68 H/s, 96.16 H/s)
2026-06-04 05:02:45 Algo cn/upx2 (cpu*5^16) hashrate: 52.80 KH/s (3.37 KH/s, 3.22 KH/s, 3.25 KH/s, 3.29 KH/s, 3.32 KH/s, 3.33 KH/s, 3.27 KH/s, 3.36 KH/s, 3.29 KH/s, 3.29 KH/s, 3.30 KH/s, 3.30 KH/s, 3.27 KH/s, 3.27 KH/s, 3.33 KH/s, 3.33 KH/s)
2026-06-04 05:03:46 Algo cn/xao (cpu^8) hashrate: 276.95 H/s (35.23 H/s, 30.06 H/s, 34.62 H/s, 34.54 H/s, 33.32 H/s, 35.44 H/s, 36.76 H/s, 36.98 H/s)
2026-06-04 05:04:47 Algo cn/zls (cpu^8) hashrate: 753.77 H/s (91.98 H/s, 97.04 H/s, 95.18 H/s, 98.66 H/s, 99.06 H/s, 94.85 H/s, 89.33 H/s, 87.67 H/s)
2026-06-04 05:06:03 Algo etchash (gpu1*4660320) hashrate: 15.00 MH/s (15.00 MH/s)
2026-06-04 05:07:04 Algo ghostrider (cpu*8^8) hashrate: 1.48 KH/s (188.17 H/s, 180.98 H/s, 185.91 H/s, 190.11 H/s, 192.20 H/s, 193.60 H/s, 176.50 H/s, 174.47 H/s)
2026-06-04 05:08:35 Algo kawpow (gpu1*37282560) hashrate: 20.87 MH/s (20.87 MH/s)
2026-06-04 05:09:37 Algo rx/0 (cpu*8) hashrate: 6.03 KH/s (6.03 KH/s)
2026-06-04 05:10:39 Algo rx/2 (cpu*8) hashrate: 4.97 KH/s (4.97 KH/s)
2026-06-04 05:11:41 Algo rx/arq (cpu*16) hashrate: 39.02 KH/s (39.02 KH/s)
2026-06-04 05:12:43 Algo rx/graft (cpu*8) hashrate: 5.54 KH/s (5.54 KH/s)
2026-06-04 05:13:46 Algo rx/sfx (cpu*8) hashrate: 5.73 KH/s (5.73 KH/s)
2026-06-04 05:14:48 Algo rx/wow (cpu*16) hashrate: 7.88 KH/s (7.88 KH/s)
2026-06-04 05:15:50 Algo rx/yada (cpu*8) hashrate: 5.92 KH/s (5.92 KH/s)
2026-06-04 05:15:50 Connecting to primary gulf.moneroocean.stream:20001tls pool
2026-06-04 05:15:51 Login to the pool succeeded
2026-06-04 05:15:52 Got new kawpow algo job with 65.41 MH/share target and 4395729 height
KawPow DAG for epoch 586 calculated (26.60 s)
2026-06-04 05:16:27 Share accepted by the pool (1/0)
...
```

Next time you can reuse saved config.json file to avoid running benchmarks again before mining:

```
$ ./mo-miner mine ./config.json
2023-02-24 05:55:59 Loading config file ./config.json
2023-02-24 05:55:59 Connecting to primary gulf.moneroocean.stream:20064tls pool
2023-02-24 05:55:59 Got new cn/gpu algo job with 12004 diff
2023-02-24 05:56:24 Share accepted by the pool (1/0)
...
```

Without parameters miner will show help:

```
$ ./mo-miner

# Node.js/SYCL based CPU/GPU miner v0.1
$ ./mo-miner <directive> <parameter>+ [<option>+]

Directives:
  mine  (<pool_address:port[tls]> <login> [<pass>]|<config.json>)
  test  <algo> <result_hash_hex_str>
  bench <algo>

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
  is_nicehash:                      nicehash nonce mining mode support (false by default)
  is_keepalive:                     sends keepalive messages to the pool to avoid disconnect (true by default)
  login:                            pool login data
  pass:                             pool password ("" by default)

--new.default_msr.<name> '{["<key>": <value>,]+}': stores default MSR register values to restore them without reboot, keys should be hex strings with 0x prefix
  value:                            MSR register value in hex string with 0x prefix format
  mask:                             MSR register mask in hex string with 0x prefix format ("0xFFFFFFFFFFFFFFFF" by default)

--new.algo_param.<name> '{["<key>": <value>,]+}': new algo params, defined by the following keys:
  dev:                              device config line "[<dev>[*B][^T],]+", dev = {cpu, gpu<N>, cpu<N>}, N = device number, B = hash batch size, T = number of parallel threads ("cpu" by default)

--log_level:                        log level: 0=minimal, 1=verbose, 2=network debug, 3=compute core debug (0 by default)
--bench_algo_params:                benchmark algo params before mining; set 0 to skip hashrate benchmarking (1 by default)
--save_config:                      file name to save config in JSON format (only for mine directive) ("" by default)
2023-02-24 05:58:24 ERROR: No directive specified
```

You can run test and benchmark separately for algo you need like this:

```
./r.sh node mo-miner.js test cn/gpu e55cb23e51649a59b127b96b515f2bf7bfea199741a0216cf838ded06eff82df --job '{"algo":"cn/gpu","dev":"gpu1*8"}'
./r.sh node mo-miner.js bench cn/gpu --job '{"algo":"cn/gpu","dev":"gpu1*1280"}'
./r.sh node mo-miner.js bench etchash --job '{"algo":"etchash","dev":"gpu1*256"}'
./r.sh node mo-miner.js bench autolykos2 --job '{"algo":"autolykos2","dev":"gpu1*1"}'
```

Project test suites are npm entry points:

```
npm test
npm run test:perf
npm run test:perf:ghostrider
```

`npm test` runs the hash-vector suite under Docker and skips GPU cases when no GPU device is
available. `npm run test:perf` benchmarks every supported algo with mo-miner's detected mining
device config and prints each hashrate in the same test reporter output. Individual benchmark entry
points are available as `npm run test:perf:<algo>` for named scripts in `package.json`, for example
`npm run test:perf:rx/0`, `npm run test:perf:cn-heavy/tube`, or `npm run test:perf:c29`. Any
supported algo can also be selected by passing it to the generic perf runner, for example
`npm run test:perf -- etchash` or `npm run test:perf -- autolykos2`.

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

mo-miner is licensed under [GPL-3.0-or-later](LICENSE).
