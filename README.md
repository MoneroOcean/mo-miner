# About

mom (short name for the **MO-Miner**) is an open-source cryptocurrency miner
that is built upon high performance xmrig CPU miner sources with front-end and network backend
rewritten in Node.js to significantly simplify its code. GPU implementations share SYCL sources
across vendors, with narrowly scoped source-JIT overrides where they provide a material gain. The
main goal of this project is to make a simple, easy-to-extend open-source miner with native miner
performance.

Miner supports algo switching on pools such as `gulf.moneroocean.stream`. By default, startup
algo-parameter benchmarking covers active MoneroOcean coin algos implemented by mom plus `rx/2`
and `pearlhash`.

## Supported algos

* CPU: All xmrig miner CPU supported algos with similar performance.
* GPU/SYCL: Check table below.

Missing measurements use `-`; `timeout` means the implementation exists but did not reach a
steady production rate. Percentages compare mom with the best same-GPU reference named in `SOTA`.
`A770*` in the SOTA column means that a same-algorithm A770 mom/SOTA ratio is applied to B580; `%?`
means no defensible same-vendor reference was available. Hashrate.no entries are public A770 results
whose miner is not listed and which we could not reproduce locally. Windows percentages use a
fresh same-GPU Windows reference where the reference miner ran successfully; otherwise they use
the same-GPU Linux reference. Cross-algorithm scaling is not used.
`lolMiner B580` is lolMiner 1.98a's unchanged required Intel Flux kernels rebuilt by the current
Intel driver after removing unused legacy entry points that crash IGC; they measured 30.9 Sol/s
directly on the B580.

Support: `mo` = MoneroOcean can serve it; `mom` = implemented by mom; `-` = backlog/not implemented.
`Coin` is the highest-market-cap known mineable coin for the exact algo/variant.

Linux and Windows columns were measured on the local hardware listed below, one GPU selected at a time.

| Algo                | Sup.   | Coin | Intel B580 Lin     | Intel B580 Win     | RTX 5060 Ti Lin   | RTX 5060 Ti Win   | RX 9060 XT Lin     | RX 9060 XT Win    | SOTA source: Intel / NVIDIA / AMD    |
| ------------------- | ------ | ---- | ------------------ | ------------------ | ----------------- | ----------------- | ------------------ | ----------------- | ------------------------------------ |
| **Cuckoo Cycle**    |        |      |                    |                    |                   |                   |                    |                   |                                      |
| `c29`               | mo,mom | TARI | 2.82 g/s (%?)      | 2.79 g/s (%?)      | 6.32 g/s (85%)    | 6.28 g/s (85%)    | 6.15 g/s (106%)    | 5.95 g/s (103%)   | - / lolMiner / lolMiner              |
| `cuckaroo30`        | -      | CTXC | -                  | -                  | -                 | -                 | -                  | -                 | - / lolMiner / lolMiner              |
| **ProgPoW family**  |        |      |                    |                    |                   |                   |                    |                   |                                      |
| `kawpow`            | mo,mom | RVN  | 20.97 MH/s (110%)  | 20.93 MH/s (110%)  | 21.98 MH/s (85%)  | 21.75 MH/s (82%)  | 19.45 MH/s (99%)   | 16.74 MH/s (84%)  | SRBMiner / Rigel / SRBMiner          |
| `evrprogpow`        | mom    | EVR  | 21.03 MH/s (110%)  | 21.02 MH/s (110%)  | 21.08 MH/s (80%)  | 21.12 MH/s (81%)  | 19.49 MH/s (99%)   | 17.35 MH/s (88%)  | SRBMiner                             |
| `firopow`           | mom    | FIRO | 20.96 MH/s (112%)  | 20.97 MH/s (112%)  | 21.81 MH/s (83%)  | 21.13 MH/s (80%)  | 19.46 MH/s (99%)   | 16.91 MH/s (85%)  | SRBMiner                             |
| `meowpow`           | mom    | MEWC | 21.09 MH/s (112%)  | 21.09 MH/s (112%)  | 25.09 MH/s (99%)  | 25.04 MH/s (94%)  | 19.56 MH/s (100%)  | 17.29 MH/s (86%)  | SRBMiner                             |
| `progpowz`          | -      | ZANO | -                  | -                  | -                 | -                 | -                  | -                 | SRBMiner                             |
| **FishHash family** |        |      |                    |                    |                   |                   |                    |                   |                                      |
| `fishhash`          | mom    | IRON | 13.98 MH/s (109%)  | 14.03 MH/s (109%)  | 33.29 MH/s (94%)  | 33.48 MH/s (97%)  | 13.07 MH/s (100%)  | 11.57 MH/s (100%) | SRBMiner / miniZ / lolMiner          |
| `karlsenhashv2`     | mom    | KLS  | 13.35 MH/s (107%)  | 12.44 MH/s (100%)  | 31.52 MH/s (89%)  | 32.98 MH/s (95%)  | 13.09 MH/s (100%)  | 11.58 MH/s (100%) | SRBMiner / miniZ / SRBMiner          |
| **Equihash**        |        |      |                    |                    |                   |                   |                    |                   |                                      |
| `beamhash3`         | mom    | BEAM | 14.74 Sol/s (104%) | 16.13 Sol/s (113%) | 32.92 Sol/s (80%) | 32.99 Sol/s (80%) | 21.19 Sol/s (109%) | 19.58 Sol/s (99%) | Hashrate.no A770* / miniZ / lolMiner |
| `zelhash`           | mom    | FLUX | 45.68 Sol/s (148%) | 45.59 Sol/s (148%) | 63.68 Sol/s (83%) | 61.70 Sol/s (81%) | 42.01 Sol/s (90%)  | 41.88 Sol/s (91%) | lolMiner B580 / lolMiner / lolMiner  |
| `equihash192_7`     | -      | ZCL  | -                  | -                  | -                 | -                 | -                  | -                 | lolMiner / miniZ / lolMiner          |
| `zhash`             | -      | BTG  | -                  | -                  | -                 | -                 | -                  | -                 | lolMiner / miniZ / lolMiner          |
| **Misc**            |        |      |                    |                    |                   |                   |                    |                   |                                      |
| `autolykos2`        | mo,mom | ERG  | 37.72 MH/s (112%)  | 38.84 MH/s (116%)  | 104.30 MH/s (88%) | 101.85 MH/s (86%) | 37.44 MH/s (100%)  | 31.44 MH/s (103%) | SRBMiner / Rigel / lolMiner          |
| `cn/gpu`            | mo,mom | RYO  | 2.86 KH/s (104%)   | 2.89 KH/s (105%)   | 3.39 KH/s (88%)   | 3.48 KH/s (90%)   | 2.55 KH/s (95%)    | 2.70 KH/s (104%)  | SRBMiner / XMR-Stak / SRBMiner       |
| `pearlhash`         | mom    | PRL  | 51.97 TH/s (149%)  | 49.96 TH/s (143%)  | 71.50 TH/s (82%)  | 71.32 TH/s (81%)  | 41.77 TH/s (98%)   | 42.94 TH/s (99%)  | ARC-miner / WildRig / KRig-Miner     |
| `hoohashv2`         | -      | HTN  | -                  | -                  | -                 | -                 | -                  | -                 | SRBMiner                             |
| `nxlhash`           | -      | NXL  | -                  | -                  | -                 | -                 | -                  | -                 | SRBMiner                             |
| `octopus`           | -      | CFX  | -                  | -                  | -                 | -                 | -                  | -                 | lolMiner / GMiner / lolMiner         |
| `verthash`          | -      | VTC  | -                  | -                  | -                 | -                 | -                  | -                 | VerthashMiner                        |
| `walahash`          | -      | WALA | -                  | -                  | -                 | -                 | -                  | -                 | SRBMiner / Rigel / -                 |
| **ASIC-exposed**    |        |      |                    |                    |                   |                   |                    |                   |                                      |
| `etchash`           | mo,mom | ETC  | 21.13 MH/s (100%)  | 21.13 MH/s (100%)  | 51.99 MH/s (100%) | 51.73 MH/s (100%) | 19.60 MH/s (100%)  | 17.39 MH/s (89%)  | lolMiner / Rigel / SRBMiner          |

Platform notes:

| Platform        | OS / kernel                       | GPU / backend                | Driver / runtime                                         | Toolchain                         |
| --------------- | --------------------------------- | ---------------------------- | -------------------------------------------------------- | --------------------------------- |
| Intel B580 Lin  | Linux `7.0.0-27-generic`          | Arc B580 / Level Zero        | `xe`; Intel GPU `26.05.37020.3-1`; Level Zero `1.28.2-2` | oneAPI 2026                       |
| Intel B580 Win  | Windows 11 Pro 24H2 (build 26100) | Arc B580 / Level Zero        | Intel Graphics `32.0.101.8860`                           | oneAPI 2026.1                     |
| RTX 5060 Ti Lin | Linux `7.0.0-27-generic`          | RTX 5060 Ti Blackwell / CUDA | NVIDIA `595.71.05`; CUDA `13.2`                          | DPC++ + AdaptiveCpp overrides     |
| RTX 5060 Ti Win | Windows 11 Pro 24H2 (build 26100) | RTX 5060 Ti Blackwell / CUDA | NVIDIA `610.62`; CUDA UMD `13.3`                         | DPC++ + AdaptiveCpp overrides     |
| RX 9060 XT Lin  | Linux `7.0.0-27-generic`          | RX 9060 XT / HIP             | `amdgpu`; ROCm/HIP `7.1`; gfx1200                        | AdaptiveCpp SSCP + HIP source-JIT |
| RX 9060 XT Win  | Windows 11 Pro 24H2 (build 26100) | RX 9060 XT / HIP             | AMD Graphics `32.0.22042.14002`; HIP SDK `7.1.1`         | AdaptiveCpp SSCP + HIP source-JIT |

Benchmark conditions:

- Linux measurements used Ubuntu 26.04 LTS and the unified multicompiler container. Intel used
  `intel-opencl-icd`/`libze-intel-gpu1` and `libze1` packages.
- Intel B580 Linux used `ONEAPI_DEVICE_SELECTOR=level_zero:gpu` and `ZE_AFFINITY_MASK=0`.
- RTX 5060 Ti Linux used `ONEAPI_DEVICE_SELECTOR=cuda:*`, a 150 W limit, 3090 MHz core,
  14001 MHz memory, and 16311 MiB VRAM. Windows used a 150 W limit and 16311 MiB VRAM.
- RX 9060 XT Linux used a 145 W test cap; Windows used the driver's `-15%` setting (about 145 W
  from its 170 W default). The Linux driver exposed up to 2700 MHz core, 1258 MHz memory, and
  16304 MiB VRAM.
- Windows measurements used the High-performance power plan.

# Install

## Linux

The release archive bundles its compiler and SYCL user-space. Run the bundled installer once to
auto-detect every GPU and install the required host support:

```
sudo ./install.sh
```

Or run the same installer directly from this repository:

```
curl -fsSL https://raw.githubusercontent.com/MoneroOcean/mo-miner/master/scripts/install.sh | sudo bash
```

For Intel it installs the Level Zero and OpenCL runtime, for NVIDIA the driver/runtime and source-JIT
tools needed by full-speed ProgPoW and PearlHash, and for AMD the HIP runtime. Other vendors use the OpenCL ICD
from their display driver. It preserves a suitable existing driver and does nothing on GPU-less
systems. Reboot if requested, then run `./mom algo_params`; each
detected GPU should appear as a `gpuN` device. When running mom inside Docker on NVIDIA, add
`--gpus all` and install the NVIDIA container toolkit on the host.
For Intel Arc, enable **Above 4G Decoding** and **Resizable BAR** in UEFI; mom enables Level Zero's
large-allocation support automatically.

## Windows

Install the current display driver for each GPU, then run the bundled installer from an
Administrator Command Prompt:

```
install.bat
```

Or run the same installer directly from this repository:

```
curl -fsSL -o "%TEMP%\mom-install.ps1" https://raw.githubusercontent.com/MoneroOcean/mo-miner/master/scripts/install.ps1 && powershell -NoProfile -ExecutionPolicy Bypass -File "%TEMP%\mom-install.ps1"
```

The installer verifies the bundled runtimes and automatically adds missing NVIDIA source-JIT support
needed for full-speed ProgPoW and PearlHash. Intel, AMD, and generic OpenCL GPUs use their current display drivers.
Run `mom.cmd algo_params` afterward; each detected GPU should appear as a `gpuN` device.

# GPU selection

The release launcher auto-detects the backend on a single-vendor system. On a mixed-vendor system,
select which GPU runtime mom should use for that process:

```
MOM_GPU_BACKEND=intel ./mom algo_params       # Linux
set MOM_GPU_BACKEND=intel && mom.cmd algo_params  # Windows Command Prompt
```

Valid values are `intel`, `nvidia`, `amd`, and `opencl`. The first three select a vendor device
group; `opencl` is the generic path for another GPU vendor. Run separate mom processes with
different values to use multiple GPU vendors concurrently. To expose just one GPU from the selected
vendor, add the zero-based `MOM_GPU_INDEX`:

```
MOM_GPU_BACKEND=intel MOM_GPU_INDEX=1 ./mom algo_params
set MOM_GPU_BACKEND=intel && set MOM_GPU_INDEX=1 && mom.cmd algo_params
```

First run `algo_params` without the index to see each numbered GPU and its full hardware name.
`MOM_GPU_INDEX` is zero-based within the selected vendor. Only that physical GPU is benchmarked;
CUDA/HIP may renumber the isolated device to `gpu1`, so use the name printed by the selected run in
explicit `--job.dev` settings.

# Per-algorithm GPU backend

Each `algo_params.<algo>.backend` value selects the implementation independently of the physical
`gpuN` device name:

| Value         | Meaning                                                            |
| ------------- | ------------------------------------------------------------------ |
| `auto`        | Use the measured `GPU-COMPILERS.md` choice (default).              |
| `sycl`        | Generic SYCL implementation.                                       |
| `sycl-opencl` | Generic SYCL through OpenCL.                                       |
| `sycl-l0`     | Generic SYCL through Intel Level Zero.                             |
| `sycl-native` | SYCL using vendor-specific extensions or target-specific tuning.   |
| `native`      | Source-JIT HIP/CUDA override when available, with `sycl` fallback. |

For example, setting `"backend": "sycl"` in the `pearlhash` entry of `config.json` disables its
policy-selected native or tuned hot kernel without changing its API-neutral `dev` device selection.
Device identifiers never encode an execution API; switch implementations with `backend` and keep
`dev` API-neutral. Status output makes the choice visible: `auto[sycl-native]` means `auto` resolved
to `sycl-native`, while an explicit selection is printed directly.

The number after `gpuN*` is the algorithm's batch/intensity control. One value is sufficient for
the current GPU algorithms because their other launch geometry is derived from the selected device
and backend. PearlHash additionally accepts named `m`, `n`, `k`, and `rank` fields in its
`algo_params.pearlhash` entry; named algorithm-specific fields will be used if another algorithm gains
a second independently useful tuning control.

# Usage example

On Linux if you run miner like that for the first time it will benchmark MoneroOcean pool algos
supported by mom plus `rx/2`, then start mining. Use `--bench_algo_params 2` to benchmark
every algo supported locally before mining instead. See [CPU performance setup](#cpu-performance-setup)
before benchmarking CPU algorithms. This example explicitly selects the Intel GPU backend; see
[GPU selection](#gpu-selection) for other GPUs and mixed-vendor systems.

```
$ MOM_GPU_BACKEND=intel MOM_GPU_INDEX=0 ./mom mine gulf.moneroocean.stream:20001tls 89TxfrUmqJJcb1V124WsUzA78Xa3UYHt7Bg8RGMhXVeZYPN8cE5CZEk58Y1m23ZMLHN7wYeJ9da5n5MXharEjrm41hSnWHL --save_config config.json
gpu1: Intel(R) Arc(TM) B580 Graphics via Intel(R) oneAPI Unified Runtime over Level-Zero V2
2026-07-19 15:25:33 Doing algo benchmarks...
2026-07-19 15:27:30 Algo autolykos2 (gpu1*8388608) hashrate: 37.72 MH/s (37.72 MH/s)
2026-07-19 15:28:35 Algo c29 (gpu1*1) hashrate: 2.78 H/s (2.78 H/s)
2026-07-19 15:30:25 Algo cn/gpu (gpu1*1280) hashrate: 2.88 KH/s (2.88 KH/s)
2026-07-19 15:31:56 Algo etchash (gpu1*33554432) hashrate: 21.13 MH/s (21.13 MH/s)
2026-07-19 15:32:57 Algo ghostrider (cpu*8^8) hashrate: 1.64 KH/s (209.21 H/s, 210.00 H/s, 202.30 H/s, 201.68 H/s, 210.02 H/s, 198.13 H/s, 197.82 H/s, 207.49 H/s)
2026-07-19 15:34:58 Algo kawpow (gpu1*37282560) hashrate: 20.95 MH/s (20.95 MH/s)
2026-07-19 15:36:07 Algo panthera (cpu*4^16) hashrate: 5.16 KH/s (326.78 H/s, 314.66 H/s, 324.24 H/s, 319.90 H/s, 323.86 H/s, 324.56 H/s, 331.25 H/s, 319.02 H/s, 317.66 H/s, 323.39 H/s, 322.89 H/s, 328.28 H/s, 317.52 H/s, 326.35 H/s, 322.83 H/s, 313.86 H/s)
2026-07-19 15:37:12 Algo pearlhash (gpu1*131072) hashrate: 52.00 TH/s (52.00 TH/s)
2026-07-19 15:38:14 Algo rx/0 (cpu*8) hashrate: 5.92 KH/s (5.92 KH/s)
2026-07-19 15:39:16 Algo rx/2 (cpu*8) hashrate: 5.25 KH/s (5.25 KH/s)
2026-07-19 15:40:18 Algo rx/arq (cpu*16) hashrate: 41.49 KH/s (41.49 KH/s)
2026-07-19 15:40:18 Saving config file to config.json
2026-07-19 15:40:18 Connecting to primary gulf.moneroocean.stream:20001tls pool
2026-07-19 15:40:18 Got new c29 algo job with 1 H/share target and 306965 height
2026-07-19 15:40:25 Share accepted by the pool (1/0)
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
automatically when sending `algo-perf`. `pearlhash` reports GEMM multiply-accumulate throughput (TH/s),
not H/s; MoneroOcean does not switch to it, so its perf is informational only.

## Mining on other pools

Use this template for non-MoneroOcean pools. mom infers the stratum dialect from `--job.algo`.

```
./mom mine <endpoint> <wallet.worker> <dev / command suffix> --bench_algo_params 0
```

| Coin | Algo          | Endpoint                           | Donation address (owner)                                                                           | Dev / command suffix                             |
| ---- | ------------- | ---------------------------------- | -------------------------------------------------------------------------------------------------- | ------------------------------------------------ |
| RVN  | kawpow        | stratum.ravenminer.com:13838tls    | `RSJZNSvzt3PJdGVKahSczRrhinc24KA6wU` (hans-schmidt, Ravencoin/Evrmore maintainer)                  | --job.algo kawpow --job.dev gpu1*37282560        |
| FIRO | firopow       | pool.woolypooly.com:3104tls        | `a4vQ7zr5CEBDEdNQBFVvHcM1BRVYKEnuEv` (Firo Core Team funding proposal)                             | --job.algo firopow --job.dev gpu1*37282560       |
| EVR  | evrprogpow    | us-east.mining4people.com:24173tls | `EaBGnWtDiAseYZiyvNT1u3WTjAeYtAR7MV` (hans-schmidt, Evrmore maintainer)                            | --job.algo evrprogpow --job.dev gpu1*37282560    |
| MEWC | meowpow       | stratum-eu.rplant.xyz:17120tls     | `MPyNGZSSZ4rbjkVJRLn3v64pMcktpEYJnU` (MeowCoin donation address)                                   | --job.algo meowpow --job.dev gpu1*37282560       |
| PRL  | pearlhash     | pearl.herominers.com:1200tls       | `prl1p79wzxcvatcsmnzp9xp0ep0rvfe9ans05mjtxnt4d9x0qqej0mtdqfrezc0` (ARC-miner PRL donation address) | --job.algo pearlhash --job.dev gpu1*131072       |
| IRON | fishhash      | ironfish.herominers.com:1145tls    | `66e044578b31c6c4c05810b0e5281bdf36138ad41bf6844ba317dc7c506bf9ac` (GMiner/Rigel bundled sample)   | --job.algo fishhash --job.dev gpu1*33554432      |
| KLS  | karlsenhashv2 | pool.woolypooly.com:3132           | `qzrq7v5jhsc5znvtfdg6vxg7dz5x8dqe4wrh90jkdnwehp6vr8uj7csdss2l7` (Karlsen Devfund)                  | --job.algo karlsenhashv2 --job.dev gpu1*33554432 |
| FLUX | zelhash       | flux.herominers.com:1200tls        | `t1Mzja9iJcEYeW5B4m4s1tJG8M42odFZ16A` (Flux development address)                                   | --job.algo zelhash --job.dev gpu1*1              |
| BEAM | beamhash3     | beam.2miners.com:5252tls           | `2346a827cb56ca74e34680593e50d7b1fa4a169332415a1d5984c6f874395c3684b` (Wilke Trei, Beam)           | --job.algo beamhash3 --job.dev gpu1*1            |

Without parameters miner will show help:

```
$ ./mom

# Node.js/SYCL based CPU/GPU miner v0.8.0
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
  --job.backend:                    GPU implementation (see README backend table) ("auto" by default)

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
  protocol:                         pool protocol override: login, raven, eth, ethproxy, erg, pearlhash, zelhash, kaspa, beam, or ironfish (null by default)
  tls_verify:                       verify pool TLS/SSL certificate (false by default)
  is_nicehash:                      nicehash nonce mining mode support (false by default)
  is_keepalive:                     sends keepalive messages to the pool to avoid disconnect (true by default)
  use_subscribe:                    PearlHash pools: use mining.subscribe+authorize handshake; set false for pearlpool.cloud's login dialect and the MoneroOcean donate pool (true by default)
  worker:                           PearlHash subscribe-dialect worker name (mining.authorize) ("mom" by default)
  login:                            pool login data
  pass:                             pool password ("" by default)

--new.default_msr.<name> '{["<key>": <value>,]+}': stores default MSR register values to restore them without reboot, keys should be hex strings with 0x prefix
  value:                            MSR register value in hex string with 0x prefix format
  mask:                             MSR register mask in hex string with 0x prefix format ("0xFFFFFFFFFFFFFFFF" by default)

--new.algo_param.<name> '{["<key>": <value>,]+}': new algo params, defined by the following keys:
  dev:                              device config line "[<dev>[*B][^P],]+", dev = {cpu, gpu<N>, cpu<N>}, N = device number, B = hash batch size, P = number of parallel processes ("cpu" by default)
  backend:                          GPU implementation (see README backend table) ("auto" by default)

--log_level:                        log level: 0=minimal, 1=verbose, 2=network debug, 3=compute core debug (0 by default)
--bench_algo_params:                benchmark algo params before mining: 0=skip, 1=active MoneroOcean coin algos plus rx/2, 2=all supported algos (1 by default)
--save_config:                      file name to save config in JSON format (only for mine directive) ("" by default)
2026-06-15 13:58:03 ERROR: No directive specified
```

You can run test and benchmark separately for algo you need like this:

```
./mom test cn/gpu e55cb23e51649a59b127b96b515f2bf7bfea199741a0216cf838ded06eff82df --job '{"algo":"cn/gpu","dev":"gpu1*8"}'
./mom bench cn/gpu --job '{"algo":"cn/gpu","dev":"gpu1*1280"}'
./mom bench etchash --job '{"algo":"etchash","dev":"gpu1*256"}'
./mom bench autolykos2 --job '{"algo":"autolykos2","dev":"gpu1*1"}'
./mom bench pearlhash --job '{"algo":"pearlhash","dev":"gpu1*131072"}'
```

## CPU performance setup

Run mom as root, or grant it MSR and huge-page capabilities, so RandomX can apply its MSR tuning and
use large pages. Without them, CPU RandomX performance can be about 40% lower; GPU performance is
unaffected.

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
sudo ./mom bench rx/0
sudo systemctl start xmr.service
```

# Development and releases

Compiler selection, development setup, and release packaging are documented in
[DEVELOPMENT.md](DEVELOPMENT.md). Release notes are attached to versioned GitHub releases.

# License

mom is licensed under [GPL-3.0-or-later](LICENSE).
