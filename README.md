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
* GPU: cn/gpu, c29

Miner supports algo switching if you connect it to algo switching pool like
gulf.moneroocean.stream that auto switches to the most profitable algo.

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

Linux release archives include `install.sh`. Run it as root on Ubuntu 24.04 or 26.04 if `cn/gpu`
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
2026-06-02 03:02:02 Doing algo benchmarks...
2026-06-02 03:03:02 Algo argon2/chukwa (cpu^16) hashrate: 49.43 KH/s (3.07 KH/s, 3.09 KH/s, 3.10 KH/s, 3.09 KH/s, 3.10 KH/s, 3.08 KH/s, 3.09 KH/s, 3.08 KH/s, 3.09 KH/s, 3.09 KH/s, 3.10 KH/s, 3.09 KH/s, 3.09 KH/s, 3.09 KH/s, 3.09 KH/s, 3.09 KH/s)
2026-06-02 03:04:03 Algo argon2/chukwav2 (cpu^16) hashrate: 16.68 KH/s (1.04 KH/s, 1.04 KH/s, 1.04 KH/s, 1.05 KH/s, 1.04 KH/s, 1.04 KH/s, 1.04 KH/s, 1.04 KH/s, 1.04 KH/s, 1.04 KH/s, 1.04 KH/s, 1.04 KH/s, 1.05 KH/s, 1.04 KH/s, 1.04 KH/s, 1.04 KH/s)
2026-06-02 03:05:03 Algo argon2/wrkz (cpu^16) hashrate: 76.38 KH/s (4.80 KH/s, 4.80 KH/s, 4.71 KH/s, 4.77 KH/s, 4.78 KH/s, 4.77 KH/s, 4.78 KH/s, 4.70 KH/s, 4.78 KH/s, 4.78 KH/s, 4.80 KH/s, 4.77 KH/s, 4.78 KH/s, 4.79 KH/s, 4.78 KH/s, 4.80 KH/s)
2026-06-02 03:06:09 Algo c29 (gpu1*1) hashrate: 2.79 H/s (2.79 H/s)
2026-06-02 03:07:10 Algo cn-heavy/0 (cpu^4) hashrate: 347.62 H/s (88.26 H/s, 87.94 H/s, 88.38 H/s, 83.05 H/s)
2026-06-02 03:08:10 Algo cn-heavy/tube (cpu^4) hashrate: 303.17 H/s (76.16 H/s, 73.02 H/s, 77.02 H/s, 76.97 H/s)
2026-06-02 03:09:11 Algo cn-heavy/xhv (cpu^4) hashrate: 336.77 H/s (87.05 H/s, 86.39 H/s, 80.51 H/s, 82.81 H/s)
2026-06-02 03:10:11 Algo cn-lite/0 (cpu^16) hashrate: 2.35 KH/s (149.49 H/s, 150.94 H/s, 144.75 H/s, 141.63 H/s, 141.43 H/s, 150.68 H/s, 144.89 H/s, 150.70 H/s, 144.83 H/s, 149.65 H/s, 144.94 H/s, 141.58 H/s, 141.72 H/s, 149.62 H/s, 149.44 H/s, 150.82 H/s)
2026-06-02 03:11:12 Algo cn-lite/1 (cpu^16) hashrate: 2.29 KH/s (145.96 H/s, 145.92 H/s, 141.32 H/s, 137.77 H/s, 137.90 H/s, 141.28 H/s, 146.92 H/s, 146.18 H/s, 145.96 H/s, 146.95 H/s, 138.10 H/s, 147.00 H/s, 141.57 H/s, 146.94 H/s, 141.39 H/s, 137.95 H/s)
2026-06-02 03:12:12 Algo cn-pico/0 (cpu*4^16) hashrate: 14.64 KH/s (923.83 H/s, 923.60 H/s, 922.66 H/s, 911.29 H/s, 920.41 H/s, 905.62 H/s, 923.99 H/s, 911.32 H/s, 923.69 H/s, 920.44 H/s, 911.00 H/s, 903.58 H/s, 922.16 H/s, 911.38 H/s, 905.08 H/s, 903.05 H/s)
2026-06-02 03:13:13 Algo cn-pico/tlo (cpu*4^16) hashrate: 13.05 KH/s (821.34 H/s, 805.48 H/s, 823.56 H/s, 823.55 H/s, 819.80 H/s, 807.61 H/s, 807.17 H/s, 804.53 H/s, 811.61 H/s, 823.62 H/s, 820.63 H/s, 811.88 H/s, 823.92 H/s, 821.28 H/s, 813.96 H/s, 813.14 H/s)
2026-06-02 03:14:13 Algo cn/0 (cpu^8) hashrate: 622.33 H/s (76.72 H/s, 80.17 H/s, 80.11 H/s, 74.74 H/s, 79.49 H/s, 74.84 H/s, 79.52 H/s, 76.74 H/s)
2026-06-02 03:15:14 Algo cn/1 (cpu^8) hashrate: 615.89 H/s (78.68 H/s, 78.58 H/s, 76.00 H/s, 75.96 H/s, 74.03 H/s, 79.29 H/s, 79.25 H/s, 74.09 H/s)
2026-06-02 03:16:14 Algo cn/2 (cpu^8) hashrate: 621.71 H/s (76.89 H/s, 76.89 H/s, 75.45 H/s, 79.39 H/s, 79.19 H/s, 75.33 H/s, 79.34 H/s, 79.22 H/s)
2026-06-02 03:17:15 Algo cn/ccx (cpu^8) hashrate: 1.19 KH/s (143.83 H/s, 152.37 H/s, 147.33 H/s, 143.76 H/s, 152.61 H/s, 147.43 H/s, 153.76 H/s, 153.68 H/s)
2026-06-02 03:18:15 Algo cn/double (cpu^8) hashrate: 314.19 H/s (38.12 H/s, 40.05 H/s, 40.15 H/s, 38.81 H/s, 40.05 H/s, 40.09 H/s, 38.09 H/s, 38.84 H/s)
2026-06-02 03:19:16 Algo cn/fast (cpu^8) hashrate: 1.19 KH/s (152.26 H/s, 152.21 H/s, 153.44 H/s, 153.25 H/s, 143.15 H/s, 143.20 H/s, 146.55 H/s, 143.05 H/s)
2026-06-02 03:20:24 Algo cn/gpu (gpu1*1280) hashrate: 2.42 KH/s (2.42 KH/s)
2026-06-02 03:21:25 Algo cn/half (cpu^8) hashrate: 801.36 H/s (103.19 H/s, 104.10 H/s, 101.71 H/s, 101.01 H/s, 104.54 H/s, 97.55 H/s, 94.83 H/s, 94.42 H/s)
2026-06-02 03:22:25 Algo cn/r (cpu^8) hashrate: 594.05 H/s (75.58 H/s, 75.80 H/s, 72.12 H/s, 73.52 H/s, 75.63 H/s, 73.52 H/s, 72.16 H/s, 75.71 H/s)
2026-06-02 03:23:26 Algo cn/rto (cpu^8) hashrate: 614.81 H/s (73.94 H/s, 79.17 H/s, 73.86 H/s, 78.56 H/s, 75.81 H/s, 79.17 H/s, 75.88 H/s, 78.43 H/s)
2026-06-02 03:24:26 Algo cn/rwz (cpu^8) hashrate: 819.35 H/s (104.65 H/s, 104.59 H/s, 104.33 H/s, 99.28 H/s, 101.33 H/s, 101.34 H/s, 104.37 H/s, 99.48 H/s)
2026-06-02 03:25:27 Algo cn/upx2 (cpu*5^16) hashrate: 54.29 KH/s (3.36 KH/s, 3.42 KH/s, 3.38 KH/s, 3.38 KH/s, 3.38 KH/s, 3.42 KH/s, 3.41 KH/s, 3.38 KH/s, 3.36 KH/s, 3.37 KH/s, 3.41 KH/s, 3.41 KH/s, 3.37 KH/s, 3.41 KH/s, 3.42 KH/s, 3.42 KH/s)
2026-06-02 03:26:27 Algo cn/xao (cpu^8) hashrate: 315.35 H/s (40.30 H/s, 37.91 H/s, 40.26 H/s, 40.60 H/s, 38.89 H/s, 38.87 H/s, 40.63 H/s, 37.89 H/s)
2026-06-02 03:27:28 Algo cn/zls (cpu^8) hashrate: 818.64 H/s (104.59 H/s, 99.32 H/s, 101.24 H/s, 104.32 H/s, 101.21 H/s, 104.55 H/s, 104.11 H/s, 99.30 H/s)
2026-06-02 03:28:29 Algo ghostrider (cpu*8^8) hashrate: 1.60 KH/s (197.65 H/s, 205.02 H/s, 203.96 H/s, 197.59 H/s, 205.01 H/s, 193.62 H/s, 193.58 H/s, 203.97 H/s)
2026-06-02 03:29:59 Algo kawpow (gpu1*37282560) hashrate: 20.88 MH/s (20.88 MH/s)
2026-06-02 03:31:03 Algo rx/0 (cpu*8) hashrate: 6.10 KH/s (6.10 KH/s)
2026-06-02 03:32:06 Algo rx/2 (cpu*8) hashrate: 5.07 KH/s (5.07 KH/s)
2026-06-02 03:33:09 Algo rx/arq (cpu*16) hashrate: 39.10 KH/s (39.10 KH/s)
2026-06-02 03:34:12 Algo rx/graft (cpu*8) hashrate: 5.72 KH/s (5.72 KH/s)
2026-06-02 03:35:15 Algo rx/sfx (cpu*8) hashrate: 5.92 KH/s (5.92 KH/s)
2026-06-02 03:36:19 Algo rx/wow (cpu*16) hashrate: 7.89 KH/s (7.89 KH/s)
2026-06-02 03:37:22 Algo rx/yada (cpu*8) hashrate: 5.91 KH/s (5.91 KH/s)
2026-06-02 03:37:22 Connecting to primary gulf.moneroocean.stream:20001tls pool
2026-06-02 03:37:23 Login to the pool succeeded
2026-06-02 03:37:23 Got new kawpow algo job with 56546580 diff and 1631411 height
KawPow DAG for epoch 217 calculated (8.62 s)
2026-06-02 03:37:34 Share accepted by the pool (1/0)
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
device config and prints each hashrate in the same test reporter output. Individual benchmark entry points are available as
`npm run test:perf:<algo>`, for example `npm run test:perf:rx/0`,
`npm run test:perf:cn-heavy/tube`, or `npm run test:perf:c29`.

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
