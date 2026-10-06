// Copyright GNU GPLv3 (c) 2026 MoneroOcean <support@moneroocean.stream>

#pragma once

#include <cstdint>
#include "nvidia-features.h"

#if defined(MOM_SYCL_ADAPTIVECPP_CUDA)
#include <hipSYCL/sycl/jit.hpp>
#endif

namespace mom::nvidia {

// Accumulators are bit patterns: both signed and unsigned DP4A wrap modulo 2^32.
// The generic SSCP frontend must retain the optional instruction until target reflection.
// __CUDA_ARCH__ would instead erase it before the selected device's JIT runs.
template <bool Signed, bool Native>
inline uint32_t dot4(const uint32_t a, const uint32_t b, const uint32_t accumulator) {
#if defined(MOM_SYCL_ADAPTIVECPP_CUDA)
  __acpp_if_target_sscp(
    namespace jit = sycl::AdaptiveCpp_jit;
    namespace query = jit::reflection_query;
    if constexpr (Native) {
      if (jit::knows<query::compiler_backend>() &&
          jit::reflect<query::compiler_backend>() == jit::compiler_backend::ptx &&
          jit::knows<query::target_vendor_id>() &&
          jit::reflect<query::target_vendor_id>() == jit::vendor_id::nvidia &&
          jit::knows<query::target_arch>() &&
          has_dp4a(jit::reflect<query::target_arch>())) {
        uint32_t result;
        if constexpr (Signed)
          asm volatile("dp4a.s32.s32 %0, %1, %2, %3;"
                       : "=r"(result) : "r"(a), "r"(b), "r"(accumulator));
        else
          asm volatile("dp4a.u32.u32 %0, %1, %2, %3;"
                       : "=r"(result) : "r"(a), "r"(b), "r"(accumulator));
        return result;
      }
    }
  )
#endif
  int product = 0;
  for (unsigned shift = 0; shift < 32; shift += 8) {
    int x = static_cast<int>((a >> shift) & 255);
    int y = static_cast<int>((b >> shift) & 255);
    if constexpr (Signed) {
      x -= (x & 128) ? 256 : 0;
      y -= (y & 128) ? 256 : 0;
    }
    product += x * y;
  }
  return accumulator + static_cast<uint32_t>(product);
}

template <bool Native>
inline uint32_t dot4_s8(const uint32_t a, const uint32_t b, const uint32_t accumulator) {
  return dot4<true, Native>(a, b, accumulator);
}

template <bool Native>
inline uint32_t dot4_u8(const uint32_t a, const uint32_t b, const uint32_t accumulator) {
  return dot4<false, Native>(a, b, accumulator);
}

} // namespace mom::nvidia
