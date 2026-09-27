# Intel Arc Alchemist PearlHash validation

The Alchemist path uses SIMD8 DPAS operands, 32x8 VNNI-packed B fragments,
and paired DPASW half-A operands. The default is two row tiles, one column tile,
64x64 cache blocking, early loads of both current B fragments, and a specialized
rank256 loop with unroll8. Odd column grids use ordinary DPAS. Incomplete row
pairs use the portable search. The existing Xe2 kernel is unchanged.

A770 16GB is hardware-tested. A750/A580 use the same architecture routing but
have not been tested on physical cards. Select `sycl-native` (or the policy's
native choice) and the Level Zero runtime to exercise the tuned path; explicit
`sycl` selects the portable implementation. Keep `dev` API-neutral.

## Windows with Intel C++

Install Intel oneAPI DPC++/C++ and the Windows C++ SDK/linker tools. The helper
uses Intel's compiler; Windows still requires SDK headers and link libraries.
It imports the environment, uses a unique temporary directory, and locates
installed TCM/UMF dependencies. CMPLR_ROOT can select another Intel installation.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/build-pearlhash-alchemist.ps1 -Run
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/build-pearlhash-alchemist.ps1 -RunOnly -Benchmark -Samples 12 -MatrixM 65536 -MatrixN 65536
```

The helper defaults to `ONEAPI_DEVICE_SELECTOR=level_zero:gpu` and enables large
USM allocations. `-ValidateOnly` runs only correctness checks; `-RunOnly` reuses
the existing binary. When benchmarking changed source, omit `-RunOnly` to rebuild.
Do not profile PowerShell or cmd: point VTune directly at the native executable.

## Linux build recipe (not validated on this Windows machine)

After loading the oneAPI environment:

```sh
mkdir -p build/alchemist
icpx -fsycl -fsycl-device-code-split=per_kernel -std=c++20 -O2 tests/pearlhash_alchemist.cpp -o build/alchemist/pearlhash-alchemist-test
ONEAPI_DEVICE_SELECTOR=level_zero:gpu UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1 build/alchemist/pearlhash-alchemist-test
```

## What is checked

Twenty comparisons cover ten shapes and two seeds, zero/all-one targets,
whole-search checksums, winner status, and valid winning coordinates. Shapes
exercise rank256 and runtime-rank paths, blocked traversal, odd-column DPAS,
partial loop-unroll groups, and portable fallback for incomplete row pairs.
The checks compare against the existing portable algorithm, not a pool verifier.

The optional benchmark uses five warmups and the requested measured samples for
search-only, full attempt, and preparation phases. It prints median and p10/p90
milliseconds and TMAC/s (`m*n*k/time`), with a zero target that excludes proof and
submission work. Do not present this as accepted-share throughput. Benchmark
matrices at 65536-square are not independently reference-validated in full.

## Measurement context

Development used an A770 16GB, Intel compiler 2026.1.1, and driver 32.0.101.9030.
Before cleanup, the winning configuration reported 44.15-44.76 TH/s in instrumented
live runs and 46.83 TH/s in one completed window of a 90-second uninstrumented run.
These short miner-reported work rates do not establish accepted shares, sustained
performance, or power efficiency. Pool verification remains a separate issue.

Pre-wait sleeping inflated the live wall-time EMA and stretched attempts while
search execution stayed near 300ms. The production fix waits immediately on Intel
SYCL GPUs and retains the existing pacing for other backends. Host CPU cost of
that change was not quantified.

Register double buffering, next-B preloads, SLM A sharing, split ownership, and
alternate transcript/scheduling variants did not beat the selected full pipeline.
They are excluded from production. A separate synthetic constant-operand test
reached 83.5% DPASW XMX activity, which rules out a universal 50% activity ceiling
but does not represent the changing operands and transcript work of mining.
