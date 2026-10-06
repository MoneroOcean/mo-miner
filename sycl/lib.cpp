// Copyright GNU GPLv3 (c) 2023-2025 MoneroOcean <support@moneroocean.stream>

#include "lib-internal.h"
#include "../native/cpu-scheduling.h"
#include <algorithm>
#include <cctype>
#include <cstring>
#include <iostream>
#include <limits>
#include <sstream>
#include <vector>

#include "runtime.inc"
#include "intensity.inc"
using mom::cpu::add_result_dev;
using mom::cpu::append_grouped_cpu_devs;
using mom::cpu::cpu_thread_batches;
#include "algo_params.inc"
