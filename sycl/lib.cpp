// Copyright GNU GPLv3 (c) 2023-2026 MoneroOcean <support@moneroocean.stream>

#include "lib-internal.h"
#include "cn-gpu/launch.h"
#include "../native/cpu-scheduling.h"
#include <algorithm>
#include <cctype>
#include <cstring>
#include <iostream>
#include <limits>
#include <sstream>
#include <vector>

// Shared runtime/discovery, launch heuristics, then algorithm admission. Algorithm entrypoints
// themselves live in their named .cpp files; these fragments share this registry's lifetime.
#include "runtime.inc"
#include "intensity.inc"
using mom::cpu::add_result_dev;
using mom::cpu::cpu_algo_devs;
using mom::cpu::CpuTopology;
using mom::cpu::normalize_cpu_topology;
#include "algo-params.inc"
