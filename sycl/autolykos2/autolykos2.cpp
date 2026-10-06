// Copyright GNU GPLv3 (c) 2026 MoneroOcean <support@moneroocean.stream>

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <vector>

#include "../lib-internal.h"
#include "../../native/consts.h"

#include "blake2b.inc"
#include "prehash.inc"
inline uint32_t calc_n(const uint32_t height) {
  if (height < INCREASE_START)
    return INIT_N_LEN;
  if (height >= INCREASE_END)
    return MAX_N_LEN;

  uint32_t n = INIT_N_LEN;
  const uint32_t iters = (height - INCREASE_START) / INCREASE_PERIOD + 1;
  for (uint32_t i = 0; i < iters; ++i)
    n = n / 100U * 105U;
  return n;
}

static uint64_t now_ms() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

static uint64_t now_us() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

static void format_duration_ms(char* out, size_t out_size, uint64_t ms) {
  if (ms < 1000) {
    std::snprintf(out, out_size, "%" PRIu64 "ms", ms);
  } else if (ms < 60000) {
    std::snprintf(out, out_size, "%.2f s", static_cast<double>(ms) / 1000.0);
  } else {
    std::snprintf(out, out_size, "%.2f min", static_cast<double>(ms) / 60000.0);
  }
}

// Parse env var `name` as a u32, leaving the caller's fallback intact on invalid input.
static bool env_u32(const char* name, uint32_t& out) {
  unsigned long parsed = 0;
  if (!mom_parse_env_ulong(name, parsed) || parsed > std::numeric_limits<uint32_t>::max())
    return false;
  out = static_cast<uint32_t>(parsed);
  return true;
}

#include "state.inc"
#include "entry.inc"
