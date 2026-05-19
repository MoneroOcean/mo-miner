## MoMiner

# About

MoMiner is open-source cryptocurrency miner that is built upon high performance xmrig CPU miner
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

Miner mominer.node dynamic library can be compiled and run from sources using `./r.sh` script that
will build Docker container with SYCL compiler. Docker buildx is required so builds use BuildKit
instead of Docker's deprecated legacy builder:

```
git clone https://github.com/MoneroOcean/mominer.git
cd mominer
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
`mominer` executable, precompiled `mominer.node`, and required SYCL runtime libraries. The release
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
to a GitHub release tag to pin a specific version. The installer keeps mominer's bundled SYCL
runtime libraries in place. Those bundled libraries include the oneAPI SYCL and Unified Runtime
user-space pieces mominer loads directly; `install.sh` only installs the host GPU driver/runtime
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
DLLs needed by mominer. To also let the installer attempt a silent Intel Graphics Driver update
through `winget`, run:

```
install.bat -InstallIntelGraphicsDriver
```

# Usage example

On Linux if you run miner like that for the first time it will benchmark all supported algos and
will start mining (this is perf numbers for Intel i7-11700K CPU and Intel Arc B580 GPU):

```
$ ./mominer mine gulf.moneroocean.stream:20001tls 89TxfrUmqJJcb1V124WsUzA78Xa3UYHt7Bg8RGMhXVeZYPN8cE5CZEk58Y1m23ZMLHN7wYeJ9da5n5MXharEjrm41hSnWHL --save_config config.json
cpu1: Intel(R) OpenCL
gpu1: Intel(R) oneAPI Unified Runtime over Level-Zero V2
gpu1o: Intel(R) OpenCL Graphics
gpu1z: Intel(R) oneAPI Unified Runtime over Level-Zero V2
2026-05-12 04:16:50 Doing algo benchmarks...
2026-05-12 04:17:50 Algo argon2/chukwa (cpu^16) hashrate: 47960.00 H/s (2997.35, 2999.07, 2991.80, 3009.85, 2998.35, 3003.47, 3001.68, 2996.45, 2979.62, 2995.57, 2998.68, 2984.07, 2993.40, 3002.73, 3007.73, 3000.18)
2026-05-12 04:18:51 Algo argon2/chukwav2 (cpu^16) hashrate: 16271.11 H/s (1012.25, 1015.25, 1020.21, 1017.82, 1016.15, 1019.62, 1018.27, 1014.48, 1020.58, 1017.62, 1017.03, 1012.36, 1022.13, 1017.30, 1019.18, 1010.87)
2026-05-12 04:19:51 Algo argon2/wrkz (cpu^16) hashrate: 73792.52 H/s (4616.94, 4580.44, 4611.59, 4622.26, 4601.76, 4556.35, 4619.76, 4629.59, 4625.01, 4616.42, 4624.85, 4615.35, 4610.01, 4630.44, 4607.09, 4624.68)
2026-05-12 04:21:08 Algo c29 (gpu1*1) hashrate: 0.80 H/s (0.80)
2026-05-12 04:22:08 Algo cn-heavy/0 (cpu^4) hashrate: 336.89 H/s (80.88, 86.73, 85.95, 83.33)
2026-05-12 04:23:08 Algo cn-heavy/tube (cpu^4) hashrate: 301.43 H/s (76.81, 76.09, 74.91, 73.61)
2026-05-12 04:24:09 Algo cn-heavy/xhv (cpu^4) hashrate: 336.23 H/s (83.89, 86.04, 80.59, 85.72)
2026-05-12 04:25:09 Algo cn-lite/0 (cpu^16) hashrate: 2322.05 H/s (143.41, 149.18, 148.45, 143.87, 139.97, 140.06, 148.20, 147.55, 148.87, 143.71, 143.08, 147.06, 139.58, 149.86, 148.90, 140.31)
2026-05-12 04:26:10 Algo cn-lite/1 (cpu^16) hashrate: 2280.48 H/s (138.28, 140.86, 145.32, 146.85, 146.16, 145.67, 139.92, 139.79, 141.73, 137.56, 137.63, 147.07, 137.27, 144.45, 144.81, 147.11)
2026-05-12 04:27:10 Algo cn-pico/0 (cpu*4^16) hashrate: 14680.83 H/s (908.23, 921.88, 927.52, 923.03, 924.59, 906.61, 926.23, 915.00, 915.45, 924.31, 924.28, 913.57, 921.24, 904.46, 908.33, 916.10)
2026-05-12 04:28:11 Algo cn-pico/tlo (cpu*4^16) hashrate: 13055.17 H/s (821.38, 823.56, 819.41, 812.72, 809.49, 808.41, 811.84, 809.00, 824.41, 808.18, 826.56, 815.80, 822.05, 812.68, 809.10, 820.58)
2026-05-12 04:29:11 Algo cn/0 (cpu^8) hashrate: 616.41 H/s (73.97, 76.07, 79.56, 78.85, 79.41, 78.69, 76.08, 73.79)
2026-05-12 04:30:11 Algo cn/1 (cpu^8) hashrate: 610.54 H/s (78.08, 75.38, 78.66, 78.67, 73.11, 73.31, 78.06, 75.27)
2026-05-12 04:31:12 Algo cn/2 (cpu^8) hashrate: 622.00 H/s (79.34, 79.55, 79.51, 79.16, 76.94, 75.24, 75.26, 77.00)
2026-05-12 04:32:12 Algo cn/ccx (cpu^8) hashrate: 1186.61 H/s (151.92, 142.14, 142.51, 152.81, 146.36, 152.67, 151.67, 146.53)
2026-05-12 04:33:13 Algo cn/double (cpu^8) hashrate: 314.07 H/s (40.10, 40.17, 40.05, 38.10, 38.04, 38.82, 38.75, 40.05)
2026-05-12 04:34:13 Algo cn/fast (cpu^8) hashrate: 1188.67 H/s (151.76, 143.05, 146.11, 152.00, 153.24, 146.49, 153.13, 142.89)
2026-05-12 04:35:19 Algo cn/gpu (gpu1*960) hashrate: 2222.08 H/s (2222.08)
2026-05-12 04:36:20 Algo cn/half (cpu^8) hashrate: 1218.76 H/s (156.33, 156.00, 147.92, 148.27, 155.73, 151.58, 147.65, 155.27)
2026-05-12 04:37:20 Algo cn/r (cpu^8) hashrate: 586.12 H/s (71.14, 74.72, 74.72, 72.61, 74.51, 72.59, 71.15, 74.67)
2026-05-12 04:38:21 Algo cn/rto (cpu^8) hashrate: 604.99 H/s (78.07, 77.49, 74.56, 77.83, 74.62, 77.31, 72.49, 72.61)
2026-05-12 04:39:21 Algo cn/rwz (cpu^8) hashrate: 816.42 H/s (102.32, 104.54, 104.35, 98.83, 102.55, 100.99, 98.94, 103.88)
2026-05-12 04:40:21 Algo cn/upx2 (cpu*5^16) hashrate: 54449.18 H/s (3431.70, 3380.27, 3434.05, 3377.27, 3377.71, 3424.66, 3420.21, 3383.05, 3373.22, 3432.21, 3367.60, 3426.87, 3371.94, 3422.21, 3397.65, 3428.54)
2026-05-12 04:41:22 Algo cn/xao (cpu^8) hashrate: 308.58 H/s (39.61, 38.10, 39.52, 37.00, 39.54, 36.90, 37.99, 39.93)
2026-05-12 04:42:22 Algo cn/zls (cpu^8) hashrate: 810.57 H/s (99.94, 103.09, 103.00, 103.25, 98.20, 100.43, 98.73, 103.93)
2026-05-12 04:43:23 Algo ghostrider (cpu*8^8) hashrate: 1574.35 H/s (194.52, 190.18, 200.79, 194.52, 202.21, 190.17, 200.06, 201.91)
2026-05-12 04:44:27 Algo rx/0 (cpu*8) hashrate: 5855.01 H/s (5855.01)
2026-05-12 04:45:30 Algo rx/2 (cpu*8) hashrate: 5050.21 H/s (5050.21)
2026-05-12 04:46:33 Algo rx/arq (cpu*16) hashrate: 39159.68 H/s (39159.68)
2026-05-12 04:47:36 Algo rx/graft (cpu*8) hashrate: 5702.72 H/s (5702.72)
2026-05-12 04:48:39 Algo rx/sfx (cpu*8) hashrate: 5902.84 H/s (5902.84)
2026-05-12 04:49:43 Algo rx/wow (cpu*16) hashrate: 7854.73 H/s (7854.73)
2026-05-12 04:50:46 Algo rx/yada (cpu*8) hashrate: 5863.44 H/s (5863.44)
2026-05-12 04:50:46 Connecting to primary gulf.moneroocean.stream:20001tls pool
2026-05-12 04:50:46 Got new c29 algo job with 1 diff and 262457 height
2026-05-12 04:52:00 Share accepted by the pool (1/0)
2026-05-12 04:52:01 Algo c29 (gpu1*1) hashrate: 0.81 H/s (0.81)
```

Next time you can reuse saved config.json file to avoid running benchmarks again before mining:

```
$ ./mominer mine ./config.json
2023-02-24 05:55:59 Loading config file ./config.json
2023-02-24 05:55:59 Connecting to primary gulf.moneroocean.stream:20064tls pool
2023-02-24 05:55:59 Got new cn/gpu algo job with 12004 diff
2023-02-24 05:56:24 Share accepted by the pool (1/0)
...
```

Without parameters miner will show help:

```
$ ./mominer

# Node.js/SYCL based CPU/GPU miner v0.1
$ ./mominer <directive> <parameter>+ [<option>+]

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
./r.sh node mominer.js test cn/gpu e55cb23e51649a59b127b96b515f2bf7bfea199741a0216cf838ded06eff82df --job '{"algo":"cn/gpu","dev":"gpu1*8"}'
./r.sh node mominer.js bench cn/gpu --job '{"algo":"cn/gpu","dev":"gpu1z*960"}'
```

Project test suites are npm entry points:

```
npm test
npm run test:perf
npm run test:perf:ghostrider
```

`npm test` runs the hash-vector suite under Docker and skips GPU cases when no GPU device is
available. `npm run test:perf` benchmarks every supported algo with mominer's detected mining
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

MoMiner is licensed under [GPL-3.0-or-later](LICENSE).
