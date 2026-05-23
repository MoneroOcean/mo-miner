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
2026-05-23 00:33:49 Doing algo benchmarks...
2026-05-23 00:34:49 Algo argon2/chukwa (cpu^16) hashrate: 50657.82 H/s (3172.17, 3160.39, 3169.17, 3178.28, 3177.11, 3171.34, 3168.06, 3168.73, 3141.46, 3174.23, 3176.01, 3134.18, 3159.56, 3175.45, 3166.45, 3165.23)
2026-05-23 00:35:50 Algo argon2/chukwav2 (cpu^16) hashrate: 16885.64 H/s (1054.66, 1051.13, 1051.26, 1057.89, 1055.56, 1051.86, 1050.69, 1057.75, 1057.26, 1057.08, 1056.73, 1054.80, 1056.25, 1059.43, 1055.46, 1057.82)
2026-05-23 00:36:50 Algo argon2/wrkz (cpu^16) hashrate: 76960.75 H/s (4829.84, 4808.01, 4816.01, 4809.01, 4812.42, 4820.75, 4837.01, 4818.17, 4807.84, 4837.75, 4816.59, 4812.59, 4754.75, 4817.17, 4814.01, 4748.84)
2026-05-23 00:37:56 Algo c29 (gpu1*1) hashrate: 2.79 H/s (2.79)
2026-05-23 00:38:56 Algo cn-heavy/0 (cpu^4) hashrate: 341.99 H/s (85.69, 81.91, 86.19, 88.20)
2026-05-23 00:39:57 Algo cn-heavy/tube (cpu^4) hashrate: 312.04 H/s (80.15, 77.11, 75.33, 79.46)
2026-05-23 00:40:57 Algo cn-heavy/xhv (cpu^4) hashrate: 341.04 H/s (86.66, 80.73, 86.49, 87.16)
2026-05-23 00:41:58 Algo cn-lite/0 (cpu^16) hashrate: 2347.42 H/s (141.58, 144.93, 141.64, 144.88, 149.74, 141.70, 150.68, 149.82, 150.45, 149.88, 150.22, 150.75, 144.88, 144.91, 149.65, 141.72)
2026-05-23 00:42:58 Algo cn-lite/1 (cpu^16) hashrate: 2278.44 H/s (138.97, 144.82, 145.16, 137.73, 145.58, 144.00, 140.65, 145.83, 140.81, 145.90, 137.70, 140.77, 138.23, 145.18, 146.42, 140.70)
2026-05-23 00:43:59 Algo cn-pico/0 (cpu*4^16) hashrate: 14722.82 H/s (909.14, 916.51, 930.39, 928.85, 915.92, 909.45, 924.16, 915.92, 909.73, 926.95, 930.00, 927.01, 909.77, 929.64, 915.65, 923.72)
2026-05-23 00:44:59 Algo cn-pico/tlo (cpu*4^16) hashrate: 13128.46 H/s (826.91, 827.00, 811.40, 811.01, 826.12, 811.38, 810.71, 826.78, 817.50, 816.94, 826.85, 826.89, 826.40, 826.91, 818.63, 817.03)
2026-05-23 00:46:00 Algo cn/0 (cpu^8) hashrate: 622.43 H/s (79.58, 76.71, 80.15, 74.72, 74.82, 76.78, 80.11, 79.55)
2026-05-23 00:47:00 Algo cn/1 (cpu^8) hashrate: 615.83 H/s (78.72, 74.00, 75.94, 75.95, 74.04, 78.67, 79.21, 79.30)
2026-05-23 00:48:01 Algo cn/2 (cpu^8) hashrate: 621.40 H/s (79.25, 75.69, 79.50, 77.03, 77.65, 75.56, 79.66, 77.05)
2026-05-23 00:49:01 Algo cn/ccx (cpu^8) hashrate: 1198.48 H/s (154.17, 147.92, 153.02, 144.23, 144.09, 154.19, 147.82, 153.04)
2026-05-23 00:50:02 Algo cn/double (cpu^8) hashrate: 315.71 H/s (38.27, 39.01, 39.03, 38.29, 40.25, 40.34, 40.31, 40.22)
2026-05-23 00:51:02 Algo cn/fast (cpu^8) hashrate: 1198.14 H/s (153.01, 144.10, 144.16, 154.16, 147.82, 153.03, 154.09, 147.78)
2026-05-23 00:52:10 Algo cn/gpu (gpu1o*1280) hashrate: 2685.82 H/s (2685.82)
2026-05-23 00:53:11 Algo cn/half (cpu^8) hashrate: 773.13 H/s (88.22, 103.22, 98.46, 100.66, 93.55, 88.48, 100.38, 100.15)
2026-05-23 00:54:11 Algo cn/r (cpu^8) hashrate: 595.96 H/s (75.86, 73.80, 76.04, 76.08, 72.32, 72.36, 75.84, 73.66)
2026-05-23 00:55:12 Algo cn/rto (cpu^8) hashrate: 614.68 H/s (78.48, 78.53, 75.79, 73.88, 79.10, 75.86, 73.92, 79.13)
2026-05-23 00:56:12 Algo cn/rwz (cpu^8) hashrate: 822.62 H/s (105.07, 104.78, 99.81, 105.09, 99.80, 101.72, 101.68, 104.66)
2026-05-23 00:57:13 Algo cn/upx2 (cpu*5^16) hashrate: 54562.86 H/s (3381.50, 3419.89, 3403.83, 3380.61, 3381.33, 3422.16, 3435.49, 3380.10, 3434.60, 3436.09, 3422.71, 3404.60, 3418.22, 3403.71, 3434.54, 3403.49)
2026-05-23 00:58:14 Algo cn/xao (cpu^8) hashrate: 315.57 H/s (37.93, 37.88, 40.67, 38.92, 40.66, 40.34, 38.90, 40.27)
2026-05-23 00:59:14 Algo cn/zls (cpu^8) hashrate: 822.00 H/s (101.63, 104.97, 99.64, 104.75, 104.66, 101.58, 104.99, 99.77)
2026-05-23 01:00:15 Algo ghostrider (cpu*8^8) hashrate: 1604.25 H/s (205.54, 194.07, 194.02, 204.48, 205.51, 198.13, 198.03, 204.47)
2026-05-23 01:01:18 Algo rx/0 (cpu*8) hashrate: 5928.99 H/s (5928.99)
2026-05-23 01:02:21 Algo rx/2 (cpu*8) hashrate: 5077.44 H/s (5077.44)
2026-05-23 01:03:24 Algo rx/arq (cpu*16) hashrate: 39243.27 H/s (39243.27)
2026-05-23 01:04:28 Algo rx/graft (cpu*8) hashrate: 5735.79 H/s (5735.79)
2026-05-23 01:05:31 Algo rx/sfx (cpu*8) hashrate: 5931.82 H/s (5931.82)
2026-05-23 01:06:34 Algo rx/wow (cpu*16) hashrate: 7854.22 H/s (7854.22)
2026-05-23 01:07:37 Algo rx/yada (cpu*8) hashrate: 5911.47 H/s (5911.47)
2026-05-23 01:07:37 Connecting to primary gulf.moneroocean.stream:20001tls pool
2026-05-23 01:07:38 Got new c29 algo job with 1 diff and 269917 height
2026-05-23 01:07:51 Share accepted by the pool (1/0)
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
--save_config:                      file name to save config in JSON format (only for mine directive) ("" by default)
2023-02-24 05:58:24 ERROR: No directive specified
```

You can run test and benchmark separately for algo you need like this:

```
./r.sh node mo-miner.js test cn/gpu e55cb23e51649a59b127b96b515f2bf7bfea199741a0216cf838ded06eff82df --job '{"algo":"cn/gpu","dev":"gpu1*8"}'
./r.sh node mo-miner.js bench cn/gpu --job '{"algo":"cn/gpu","dev":"gpu1z*960"}'
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
