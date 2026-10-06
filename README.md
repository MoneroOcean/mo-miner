# MO-Miner

`mom` is an open-source cryptocurrency miner for CPU and GPU
algorithms. It supports pool algorithm switching and a simple command-line configuration.

mom supports algorithm switching through `mom.moneroocean.stream`.

## Supported algorithms

- CPU: the algorithms reported by `mom algorithms`, using the bundled XMRig engine.
- GPU: the algorithms in the table below when supported by the selected device.

NVIDIA compatibility targets Maxwell (GeForce GTX 750 Ti) and newer, subject to the memory
requirements below. Older-card validation is ongoing.

Percentages compare mom with the named reference. A plain miner name is a controlled same-GPU,
same-OS comparison. A `(Linux)` or `(Windows)` suffix identifies the reference OS. An asterisk marks
a cell whose comparison is approximate: a limited measured average, another GPU or OS, a published
result, a scaled estimate, or the best result found when no controlled same-GPU reference was available.
`best peer*` is the fastest credible result found on a comparable GPU when no controlled same-model
reference exists; it is context, not a same-hardware target.
`TBD` means the reference comparison is being rechecked.

Every listed algorithm is implemented by mom; `/ MO` marks algorithms served by MoneroOcean.
Preferred Cuckaroo names are `c29` and `c30`; accepted aliases are `cuckaroo`/`cuckaroo29`/`c29xtm` and
`cuckaroo30`/`c30ctx`, respectively.

GPU cells use `A770` and `B580` for Intel Arc A770 and B580, `5060 Ti` for NVIDIA RTX 5060 Ti,
and `9060 XT` for AMD RX 9060 XT. Measurements ran one selected GPU at a time. Each algorithm groups
the GPUs on adjacent rows; the last column names the comparison source.

| Algo / coin / pool      | GPU     | Linux               | Windows             | Reference / evidence           |
| ----------------------- | ------- | ------------------- | ------------------- | ------------------------------ |
| `c29` / TARI / MO       | B580    | 5.94 g/s (80%*)     | 6.13 g/s (83%*)     | best peer*                     |
|                         | A770    | 4.13 g/s (56%*)     | -                   | best peer*                     |
|                         | 5060 Ti | 6.32 g/s (82%)      | 6.28 g/s (101%)     | BZMiner                        |
|                         | 9060 XT | 6.15 g/s (93%)      | 5.95 g/s (90%)      | BZMiner                        |
| `c30` / CTXC            | B580    | 1.41 g/s (124%*)    | 1.49 g/s (135%*)    | MoM 12-GiB-limited peer*       |
|                         | A770    | 1.41 g/s (124%*)    | -                   | MoM 12-GiB-limited peer*       |
|                         | 5060 Ti | 2.89 g/s (89%*)     | 2.83 g/s (87%*)     | Hashrate.no/cross-GPU*         |
|                         | 9060 XT | 1.64 g/s (103%*)    | 1.35 g/s (84%*)     | Hashrate.no/cross-GPU*         |
| `kawpow` / RVN / MO     | B580    | 20.97 MH/s (110%)   | 20.93 MH/s (110%)   | SRBMiner                       |
|                         | A770    | 13.96 MH/s (148%*)  | -                   | WildRig*                       |
|                         | 5060 Ti | 21.98 MH/s (85%)    | 21.75 MH/s (82%)    | Rigel                          |
|                         | 9060 XT | 19.45 MH/s (99%)    | 16.74 MH/s (84%)    | SRBMiner                       |
| `evrprogpow` / EVR      | B580    | 21.03 MH/s (110%)   | 21.02 MH/s (110%)   | SRBMiner                       |
|                         | A770    | 21.67 MH/s (122%)   | -                   | WildRig                        |
|                         | 5060 Ti | 21.08 MH/s (80%)    | 21.12 MH/s (81%)    | SRBMiner                       |
|                         | 9060 XT | 19.49 MH/s (99%)    | 17.35 MH/s (88%)    | SRBMiner                       |
| `firopow` / FIRO        | B580    | 20.96 MH/s (112%)   | 20.97 MH/s (112%)   | SRBMiner                       |
|                         | A770    | 15.33 MH/s (165%)   | -                   | WildRig                        |
|                         | 5060 Ti | 21.13 MH/s (80%)    | 21.13 MH/s (80%)    | SRBMiner                       |
|                         | 9060 XT | 19.46 MH/s (99%)    | 16.91 MH/s (85%)    | SRBMiner                       |
| `meowpow` / MEWC        | B580    | 21.09 MH/s (112%)   | 21.09 MH/s (112%)   | SRBMiner                       |
|                         | A770    | 17.79 MH/s (99%)    | -                   | WildRig                        |
|                         | 5060 Ti | 25.09 MH/s (99%)    | 25.04 MH/s (94%)    | SRBMiner                       |
|                         | 9060 XT | 19.56 MH/s (100%)   | 17.29 MH/s (86%)    | SRBMiner                       |
| `fishhash` / IRON       | B580    | 12.60 MH/s (98%)    | 13.07 MH/s (102%)   | SRBMiner                       |
|                         | A770    | 12.90 MH/s (81%*)   | -                   | SRBMiner*                      |
|                         | 5060 Ti | 31.20 MH/s (88%)    | 31.32 MH/s (91%)    | miniZ                          |
|                         | 9060 XT | 13.07 MH/s (100%)   | 11.57 MH/s (100%)   | lolMiner                       |
| `karlsenhashv2` / KLS   | B580    | 12.75 MH/s (102%)   | 12.44 MH/s (100%)   | SRBMiner                       |
|                         | A770    | 14.69 MH/s (97%)    | -                   | SRBMiner                       |
|                         | 5060 Ti | 31.52 MH/s (89%)    | 32.98 MH/s (95%)    | miniZ                          |
|                         | 9060 XT | 13.09 MH/s (100%)   | 11.58 MH/s (100%)   | SRBMiner                       |
| `beamhash3` / BEAM      | B580    | 24.72 Sol/s (60%*)  | 23.78 Sol/s (58%*)  | best peer*                     |
|                         | A770    | 24.16 Sol/s (105%*) | -                   | lolMiner*                      |
|                         | 5060 Ti | 32.42 Sol/s (78%)   | 32.99 Sol/s (80%)   | miniZ                          |
|                         | 9060 XT | 22.05 Sol/s (113%)  | 21.81 Sol/s (110%)  | lolMiner                       |
| `zelhash` / CS          | B580    | 47.12 Sol/s (153%)  | 45.59 Sol/s (148%)  | lolMiner                       |
|                         | A770    | 44.40 Sol/s (118%*) | -                   | lolMiner*                      |
|                         | 5060 Ti | 63.68 Sol/s (83%)   | 61.70 Sol/s (81%)   | lolMiner                       |
|                         | 9060 XT | 43.41 Sol/s (93%)   | 43.29 Sol/s (94%)   | lolMiner                       |
| `equihash192_7` / ZCL   | B580    | 15.64 I/s (51%*)    | 15.93 I/s (52%*)    | best peer*                     |
|                         | A770    | 17.16 I/s (55%*)    | -                   | best peer*                     |
|                         | 5060 Ti | 26.85 I/s (87%)     | 24.70 I/s (80%)     | miniZ                          |
|                         | 9060 XT | 20.00 I/s (99%)     | 19.31 I/s (96%)     | lolMiner                       |
| `zhash` / BTG           | B580    | 60.77 Sol/s (56%*)  | 61.70 Sol/s (58%*)  | best peer*                     |
|                         | A770    | 57.10 Sol/s (86%*)  | -                   | lolMiner*                      |
|                         | 5060 Ti | 87.63 Sol/s (81%)   | 86.32 Sol/s (80%)   | miniZ                          |
|                         | 9060 XT | 74.92 Sol/s (90%)   | 71.16 Sol/s (86%)   | lolMiner                       |
| `autolykos2` / ERG / MO | B580    | 39.02 MH/s (96%)    | 38.84 MH/s (102%)   | BZMiner                        |
|                         | A770    | 27.09 MH/s (100%*)  | -                   | SRBMiner*                      |
|                         | 5060 Ti | 104.30 MH/s (87%)   | 101.85 MH/s (85%)   | BZMiner                        |
|                         | 9060 XT | 37.44 MH/s (100%)   | 31.44 MH/s (86%)    | lolMiner / BZMiner             |
| `cn/gpu` / RYO / MO     | B580    | 2.94 KH/s (107%)    | 2.89 KH/s (105%)    | SRBMiner                       |
|                         | A770    | 2.41 KH/s (106%*)   | -                   | SRBMiner*                      |
|                         | 5060 Ti | 3.39 KH/s (88%)     | 3.48 KH/s (90%)     | XMR-Stak                       |
|                         | 9060 XT | 3.17 KH/s (118%)    | 2.81 KH/s (107%)    | SRBMiner / BZMiner             |
| `pearlhash` / PRL       | B580    | 45.43 TH/s (122%)   | 41.74 TH/s (111%)   | BZMiner                        |
|                         | A770    | 46.07 TH/s (123%*)  | -                   | best peer*                     |
|                         | 5060 Ti | 80.48 TH/s (88%)    | 80.34 TH/s (88%)    | BZMiner                        |
|                         | 9060 XT | 42.80 TH/s (92%)    | 42.78 TH/s (92%*)   | KRig (Linux)                   |
| `hoohash` / HTN         | B580    | 3.03 MH/s (174%*)   | 2.99 MH/s (180%*)   | best peer*                     |
|                         | A770    | Unsupported         | -                   | No FP64                        |
|                         | 5060 Ti | 1.44 MH/s (83%)     | 1.42 MH/s (81%*)    | hoo_gpu (Linux)                |
|                         | 9060 XT | 1.71 MH/s (115%)    | 1.66 MH/s (111%*)   | hoo_gpu_amd (Linux)            |
| `octopus` / CFX         | B580    | 31.27 MH/s (61%*)   | 31.24 MH/s (61%*)   | best peer*                     |
|                         | A770    | 16.60 MH/s (32%*)   | -                   | best peer*                     |
|                         | 5060 Ti | 45.62 MH/s (88%)    | 45.44 MH/s (89%)    | lolMiner                       |
|                         | 9060 XT | 32.80 MH/s (86%)    | 26.49 MH/s (82%)    | lolMiner                       |
| `xelishashv3` / XEL     | B580    | 2.15 KH/s (31%*)    | 2.16 KH/s (31%*)    | best peer*                     |
|                         | A770    | 1.91 KH/s (32%)     | -                   | SRBMiner                       |
|                         | 5060 Ti | 5.88 KH/s (85%)     | 5.90 KH/s (85%)     | Rigel                          |
|                         | 9060 XT | 2.62 KH/s (99%)     | 2.61 KH/s (110%)    | SRBMiner / BZMiner             |
| `verthash` / VTC        | B580    | 299.83 KH/s (95%)   | 298.59 KH/s (94%)   | SRBMiner                       |
|                         | A770    | 607.68 KH/s (62%)   | -                   | SRBMiner                       |
|                         | 5060 Ti | 908.06 KH/s (98%)   | 873.84 KH/s (94%)   | TBMiner                        |
|                         | 9060 XT | 293.48 KH/s (101%)  | 292.17 KH/s (100%)  | TBMiner                        |
| `walahash` / WALA       | B580    | 623.38 MH/s (172%)  | 566.97 MH/s (140%)  | SRBMiner                       |
|                         | A770    | 686.08 MH/s (106%*) | -                   | SRBMiner*                      |
|                         | 5060 Ti | 873.49 MH/s (85%)   | 892.07 MH/s (87%)   | SRBMiner                       |
|                         | 9060 XT | 679.50 MH/s (87%)   | 667.68 MH/s (85%)   | SRBMiner                       |
| `etchash` / ETC / MO    | B580    | 21.13 MH/s (100%)   | 21.13 MH/s (100%)   | lolMiner                       |
|                         | A770    | 20.43 MH/s (76%*)   | -                   | Nanominer*                     |
|                         | 5060 Ti | 51.99 MH/s (100%)   | 51.73 MH/s (100%)   | Rigel                          |
|                         | 9060 XT | 19.60 MH/s (100%)   | 17.39 MH/s (89%)    | SRBMiner                       |
| `nexapow` / NEXA        | B580    | 28.06 MH/s (37%*)   | 27.01 MH/s (35%*)   | best peer*                     |
|                         | A770    | 25.04 MH/s (86%*)   | -                   | WildRig*                       |
|                         | 5060 Ti | 76.10 MH/s (80%)    | 77.97 MH/s (82%)    | lolMiner                       |
|                         | 9060 XT | 28.21 MH/s (87%)    | 26.79 MH/s (86%)    | WildRig                        |

Rates are steady-state measurements from the listed local GPUs on Ubuntu 26.04 LTS and Windows 11
24H2, one GPU per process. The RTX 5060 Ti and RX 9060 XT used 150 W and 145 W limits; the B580 was
not power-limited. Actual speed varies with drivers, clocks, power limits, and pool workload.

A770 measurements are Linux-only, with an observed 190 W cap. Starred miner names use measured
averages where available; thermal throttling, short benchmarks or different pool workloads make
these approximate comparisons. When a matched A770 reference is unavailable, `best peer*` uses a
measured reference from another GPU. Some A770 rates use earlier MoM builds:
FishHash, KarlsenHashV2, BeamHash3, Equihash192_7, Verthash, Walahash and Etchash.

## GPU memory requirements

These are practical card-memory classes for the current automatic/default settings. Leave some
headroom for the OS and display use; DAG- and table-based networks can grow over time.

- **1 GiB minimum:** `hoohash`, `cn/gpu`, `xelishashv3`
- **1 GiB minimum; 5 GiB recommended:** `nexapow`
- **2 GiB minimum:** `verthash`
- **2 GiB minimum; 8 GiB recommended:** `pearlhash`
- **2 GiB minimum; 8 GiB recommended:** `walahash`
- **3 GiB minimum; 8 GiB recommended:** `autolykos2`
- **4 GiB minimum:** `zhash`
- **5 GiB minimum:** `etchash`
- **6 GiB minimum:** `kawpow`, `firopow`, `evrprogpow`, `meowpow`, `octopus`, `fishhash`,
  `karlsenhashv2`
- **8 GiB minimum:** `c29`, `zelhash`, `equihash192_7`, `beamhash3`
- **12 GiB minimum; 16 GiB recommended:** `c30`

`hoohash` is unavailable on Intel Arc A770 and Intel UHD 750 because these devices
lack double-precision (FP64) support. On UHD 750, `c29`, `zelhash`, `equihash192_7`,
and `zhash` are also unavailable: their solvers require 1,024 threads per GPU group,
while this integrated GPU supports 512.

Use `./mom algorithms` to list supported algorithms and their automatic CPU/GPU settings.

Optional controls are listed under [GPU tuning](#gpu-tuning).

# Install

The commands below assume the release archive has been extracted and the current directory is its
top-level folder. Examples use `./mom` on Linux; use `mom.cmd` on Windows.

## Linux

The release archive includes the application libraries. On Ubuntu 24.04 or 26.04, run the bundled
installer once to detect supported GPU vendors and install missing host runtime prerequisites:

```
sudo ./install.sh
```

Or download the v0.9.0 installer from this repository, inspect it, and then run it:

```
curl --fail --location --proto '=https' --tlsv1.2 \
  --output mom-install.sh \
  https://raw.githubusercontent.com/MoneroOcean/mo-miner/v0.9.0/scripts/install.sh
sudo bash mom-install.sh
```

The installer preserves suitable existing drivers, adds the invoking user to existing `render` and
`video` groups when needed, and makes no GPU changes on CPU-only systems. Reboot or sign out and
back in if requested, then run `./mom algorithms`; supported GPUs for the selected backend appear
as `gpuN` devices. When running mom inside Docker on NVIDIA, add
`--gpus all` and install the NVIDIA container toolkit on the host.
For Intel Arc, [Intel recommends](https://www.intel.com/content/www/us/en/support/articles/000090831/graphics.html)
enabling **Above 4G Decoding** and **Resizable BAR** in UEFI.

## Windows

Install the current display driver for each GPU, then run the bundled installer from an
Administrator Command Prompt:

```
install.bat
```

Or download the v0.9.0 installer from this repository, inspect it, and then run it:

```bat
curl --fail --location ^
  --output "%TEMP%\mom-install.ps1" ^
  https://raw.githubusercontent.com/MoneroOcean/mo-miner/v0.9.0/scripts/install.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File "%TEMP%\mom-install.ps1"
```

The release includes its application GPU runtimes. Current vendor display drivers remain a
prerequisite for Intel, AMD, and NVIDIA devices. The generic OpenCL path additionally requires a
driver that accepts SPIR-V programs.
On NVIDIA systems, the installer also adds missing CUDA/C++ components used by runtime-compiled
kernels; pass `-SkipCudaToolkit` only when those components are managed separately.
Run `mom.cmd algorithms` afterward; supported GPUs for the selected backend appear as `gpuN`
devices.

Some GPU algorithms have a slow first launch while the driver compiles device code. Let that launch
finish and keep the miner's cache directory; later launches normally reuse the compiled code.

# GPU selection

The release launcher auto-detects the backend on a single-vendor system. On a mixed-vendor system,
select which GPU runtime mom should use for that process:

```
MOM_GPU_BACKEND=intel ./mom algorithms       # Linux
set "MOM_GPU_BACKEND=intel" && mom.cmd algorithms  # Windows Command Prompt
```

Valid values are `intel`, `nvidia`, `amd`, and `opencl`. The first three select a vendor device
group; `opencl` is the generic path for another GPU vendor and defaults to GPU devices. Set
`MOM_OPENCL_DEVICE_TYPE=cpu` to use its portable CPU path. Run separate mom processes with different
values to use multiple GPU vendors concurrently. To select one GPU, add the zero-based
`MOM_GPU_INDEX`:

```
MOM_GPU_BACKEND=intel MOM_GPU_INDEX=0 ./mom algorithms
set "MOM_GPU_BACKEND=intel" && set "MOM_GPU_INDEX=0" && mom.cmd algorithms
```

First run `algorithms` without the index to see each numbered GPU and its full hardware name.
Intel and OpenCL indices follow the displayed devices; NVIDIA and AMD indices use their vendor
runtime's logical ordering, which need not match displayed `gpuN` names or `nvidia-smi` rows.
CUDA/HIP renumber an isolated device to `gpu1`; use the selected run's reported name in explicit
`--job.dev` settings. With an inherited CUDA/HIP visibility mask, use a single-device mask and
`MOM_GPU_INDEX=0`, or omit the index and select a reported `--job.dev`. Indexed inherited lists
are rejected rather than widening visibility. The portable OpenCL CPU path lists `cpuN` devices.

# Per-algorithm GPU backend

Each `algo_params.<algo>.backend` value selects an implementation; `auto` uses
[GPU-CONFIG.md](GPU-CONFIG.md):

| Value         | Meaning                                                          |
| ------------- | ---------------------------------------------------------------- |
| `auto`        | Use the configured choice for the selected GPU (default).        |
| `sycl`        | Use the portable GPU path.                                       |
| `sycl-opencl` | Use the OpenCL driver path.                                      |
| `sycl-l0`     | Use the Intel Level Zero path.                                   |
| `sycl-native` | Use the vendor-tuned GPU path.                                   |
| `native`      | Use the vendor-native path when available; otherwise use `sycl`. |

Status output shows the resolved backend, for example `auto[sycl-native]`; an explicit selection is
printed directly.

## GPU tuning

Defaults normally need no manual tuning. The short `gpuN*VALUE` form sets an algorithm's primary
control; its expanded form is
`gpuN*[name=value;name=value]`. For example, `gpu1*4194304` and
`gpu1*[intensity=4194304]` are equivalent for KawPow, while
`gpu1*[workgroup=128]` overrides only its workgroup and leaves intensity at its configured or
automatic value. A bare `gpuN` adds no inline tuning overrides; brackets are accepted only in the
starred `*[...]` form.

| Algorithms                                   | `*VALUE` means   | Other fields / limits                                          |
| -------------------------------------------- | ---------------- | -------------------------------------------------------------- |
| `cn/gpu`, `octopus`, `walahash`              | `intensity`      | —                                                              |
| `c30`                                        | `intensity`      | fixed at `1`                                                   |
| `xelishashv3`, `nexapow`, `verthash`         | `intensity`      | —                                                              |
| `c29`                                        | `seed_workgroup` | `seed_blocks`                                                  |
| `kawpow`, `firopow`, `evrprogpow`, `meowpow` | `intensity`      | `workgroup`, `dag_workgroup`, `dag_chunk`                      |
| `etchash`                                    | `intensity`      | `dag_workgroup`, `dag_chunk`                                   |
| `autolykos2`                                 | `intensity`      | `workgroup`, `prehash_workgroup`, `table_chunk`, `search_mode` |
| `fishhash`, `karlsenhashv2`                  | `intensity`      | `workgroup`, `search_mode`                                     |
| `hoohash`                                    | `intensity`      | `workgroup`                                                    |
| `pearlhash`                                  | `m`              | `n`, `k`, `rank`, `cache_block`, `tile`                        |
| `equihash192_7`, `zhash`                     | `intensity`      | fixed at `1`                                                   |
| `zelhash`                                    | `slots`          | —                                                              |
| `beamhash3`                                  | `workgroup`      | `compact_workgroup`, `scatter_workgroup`, `layout`             |

Fields may be supplied inline or in the optional `algo_params.<algo>.tuning` object; inline values
take precedence. `search_mode` accepts
`auto`, `scalar`, or `cooperative`; `layout` accepts `auto`, `compact`, or `full`. Pearl's `tile`
accepts `auto`, `1x1`, `2x2`, `2x4`, `4x2`, `4x4`, or `8x2`; the setting applies to AMD GPUs.
Other secondary fields affect only implementations that expose the corresponding launch control.

For an optional empirical pass, add `--gpu_tune 1` to the first `mine` command together with
`--save_config config.json`. It can take hours, especially for algorithms with large datasets.

The object form is useful for a saved configuration, while the inline form can override one worker:

```
{
  "algo_params": {"kawpow": {"dev": "gpu1", "backend": "auto", "tuning": {"workgroup": 256}}}
}
```

Repeated entries run independent workers on one GPU:

```
gpu1*[intensity=2097152],gpu1*[intensity=4194304]
gpu1*[workgroup=256]^2
```

# Mining

Donation mining defaults to a 60-second window every 100 minutes through
`mom.moneroocean.stream`. It reuses the current algorithm when supported, including fixed
PearlHash, or uses measured compatible MoneroOcean algorithms on the same selected devices.
Donation mining is disabled when no compatible algorithm is available.

On its first run, this Linux example benchmarks MoneroOcean algorithms supported by mom plus `rx/2`
and `pearlhash`, then starts mining. Use `--bench_algo_params 2` to benchmark every locally supported
algorithm instead. The example selects Intel; see [GPU selection](#gpu-selection) for other GPUs.

The example uses MoneroOcean's donation address. Replace it with your own wallet address to receive
payouts.

```
MOM_GPU_BACKEND=intel MOM_GPU_INDEX=0 ./mom mine \
  mom.moneroocean.stream:20001tls \
  89TxfrUmqJJcb1V124WsUzA78Xa3UYHt7Bg8RGMhXVeZYPN8cE5CZEk58Y1m23ZMLHN7wYeJ9da5n5MXharEjrm41hSnWHL \
  --save_config config.json
```

Next time you can reuse saved config.json file to avoid running benchmarks again before mining:

```
MOM_GPU_BACKEND=intel MOM_GPU_INDEX=0 ./mom mine ./config.json
```

Keep the same backend and device-selection environment controls when reusing a config; they are
not stored in the JSON file.

## Mining on other pools

Use this template for other pools, choosing the algorithm with `--job.algo`.

```
./mom mine <endpoint> <address.worker> --job.algo <algo> --bench_algo_params 0
```

Options follow the two positional arguments. Add `--job.dev gpuN` only when a specific device must
be pinned; normal single-GPU discovery does not require it.

For Pearl pools using `mining.subscribe` that send final jackpot targets, set
`pearlhash_target_format` to `"jackpot"` in the pool configuration; the default is `"base"`.

| Coin | Algo          | Endpoint                            | Address                                                                  | Owner / purpose                                                |
| ---- | ------------- | ----------------------------------- | ------------------------------------------------------------------------ | -------------------------------------------------------------- |
| RVN  | kawpow        | stratum.ravenminer.com:13838tls     | `RSJZNSvzt3PJdGVKahSczRrhinc24KA6wU`                                     | hans-schmidt, Ravencoin/Evrmore maintainer                     |
| FIRO | firopow       | firo.cedric-crispin.com:4064        | `a4vQ7zr5CEBDEdNQBFVvHcM1BRVYKEnuEv`                                     | Firo Core Team funding proposal                                |
| EVR  | evrprogpow    | eu.evrpool.org:1111                 | `EaBGnWtDiAseYZiyvNT1u3WTjAeYtAR7MV`                                     | hans-schmidt, Evrmore maintainer                               |
| MEWC | meowpow       | stratum-eu.rplant.xyz:17120tls      | `MPyNGZSSZ4rbjkVJRLn3v64pMcktpEYJnU`                                     | MeowCoin project donation                                      |
| PRL  | pearlhash     | pearl.herominers.com:1200tls        | `prl1pfu7yr6u6mfkku3mh2deyuwegcnpaunjz4vlsvaj2shg2qjkaux2q76uyud`        | Pearl-Miner development (PrimeAI Foundation)                   |
| IRON | fishhash      | de.ironfish.herominers.com:1145tls  | `0a88574634e8eba1bc90e6b4e6c381b65de2660d72e1dfe0c5ed6c2aba3826d6`       | Nanopool test address; fallback                                |
| KLS  | karlsenhashv2 | pool.tr.woolypooly.com:3132         | `karlsen:qzrq7v5jhsc5znvtfdg6vxg7dz5x8dqe4wrh90jkdnwehp6vr8uj7csdss2l7`  | Karlsen Dev Fund                                               |
| HTN  | hoohash       | zenithpool.net:42494                | `hoosat:qp4ad2eh72xc8dtjjyz4llxzq9utn6k26uyl644xxw70wskdfl85zsqj9k4vz`   | Hoosat development                                             |
| CFX  | octopus       | cfx-eu.kryptex.network:7027         | `cfx:aatnhz2z908skxhxa0shn15tg64c2ejbeajs3s48dh`                         | Public test recipient; ownership unverified                    |
| NEXA | nexapow       | nexa.kryptex.network:7026           | `nexa:nqtsq5g5utr3g40hpkul9gf07j707epzqaeana3tl3p68dmw`                  | Nexa community funding                                         |
| CTXC | c30           | pool.tr.woolypooly.com:40000tls     | `0x90dd8bf208c9c29f8033935ee82742ad5424c679`                             | Cortex Labs/Foundation holdings                                |
| ZCL  | equihash192_7 | equihash192.eu.mine.zpool.ca:2192   | `t1f1xRt73TWgVWJQEFAW6DLwKtwmQvFahrN`                                    | hellcatz, LuckPool development and maintenance                 |
| WALA | walahash      | mine.xdag1usa.com:3094              | `waglayla:qqnz8s8xcrvjykdq326umlaz0xnp49wf3gxnun5rcp2xjzfux9p6sg2acf7jd` | WagLayla Dev/Ops Fund                                          |
| XEL  | xelishashv3   | de.xelis.herominers.com:1225        | `xel:vs3mfyywt0fjys0rgslue7mm4wr23xdgejsjk0ld7f2kxng4d4nqqnkdufz`        | Xelis development (daemon developer fee)                       |
| CS   | zelhash       | nl.rabbitminer.cc:1103              | `t1Lz6NHYByyDVWhRFn32hwd42nUtDdBpLc1`                                    | Cloud Service Dev Fund (source-defined; controller unverified) |
| BTG  | zhash         | btg.2miners.com:14040tls            | `GQb77ZuMCyJGZFyxpzqNfm7GB1rQreP4n6`                                     | Miningcore development                                         |
| BEAM | beamhash3     | beam.2miners.com:5252tls            | `e17cc06481d9ae88e1e0181efee407fa8c36a861b9df723845eddc8fb1ba552048`     | Beam Light Wallet development (@vsnation)                      |
| VTC  | verthash      | vertcoin.cedric-crispin.com:3334    | `VertionJAZJ7ZMauEdXaagRb4XP7cw6FXV`                                     | Electrum-VTC development                                       |

These are public example recipients; verified ownership or purpose is shown where known. Use your
own wallet address to receive payouts. Pool minimum payouts still apply.
For CFX on Kryptex, append `.worker` to the address when passing it to `mom mine`.

On first Verthash use, mom generates and verifies the 1.20-GiB `verthash.dat` file in the current
directory. A message is printed before this one-time operation, which usually takes a few minutes.
Use a trusted writable directory and keep the file for later runs; `MOM_VERTHASH_DATA` remains
available as an optional path override.

You can benchmark an algorithm directly:

```
./mom bench cn/gpu --job.dev "gpu1*[intensity=1280]"
./mom bench etchash --job.dev "gpu1*[intensity=256]"
./mom bench autolykos2 --job.dev "gpu1*[intensity=1]"
./mom bench pearlhash --job.dev "gpu1*[m=131072]"
```

## CPU performance setup

RandomX performs best with MSR tuning and large pages. Run mom as root or configure the required
permissions and [huge pages](https://xmrig.com/docs/miner/hugepages). These settings affect CPU
mining only.

# License

mom is licensed under [GPL-3.0-or-later](LICENSE).
