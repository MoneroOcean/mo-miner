// Copyright GNU GPLv3 (c) 2026 MoneroOcean <support@moneroocean.stream>
#pragma once

#include <bit>
#include <cstdint>

inline int32_t cn_gpu_reduce_int(const int32_t* const raw, const unsigned lane) {
  const unsigned b = (lane / 4U) * 16U + lane % 4U;
  return raw[b] ^ raw[b + 4] ^ raw[b + 8] ^ raw[b + 12];
}

inline float cn_gpu_reduce_float(const float* const raw, const unsigned lane) {
  const unsigned b = (lane / 4U) * 16U + lane % 4U;
  return (raw[b] + raw[b + 4]) + (raw[b + 8] + raw[b + 12]);
}

// First-stage reductions replace only the first 16 scalars. Keep those values in the completed
// seed buffer: ints at [0,16), float bits at [16,32). Later reads of the untouched tail stay in raw.
// Writing raw[0,16) during the second stage is then safe even across independently scheduled lanes.
inline int32_t cn_gpu_reduce_staged_int(const int32_t* const raw,
                                       const int32_t* const partials, const unsigned lane) {
  const auto load = [&](const unsigned index) { return index < 16 ? partials[index] : raw[index]; };
  return partials[lane] ^ load(lane + 4) ^ load(lane + 8) ^ load(lane + 12);
}

inline float cn_gpu_reduce_staged_float(const float* const raw,
                                       const int32_t* const partials, const unsigned lane) {
  const unsigned b = (lane / 4U) * 16U + lane % 4U;
  const auto load = [&](const unsigned index) {
    return index < 16 ? std::bit_cast<float>(partials[16 + index]) : raw[index];
  };
  return (load(b) + load(b + 4)) + (load(b + 8) + load(b + 12));
}
