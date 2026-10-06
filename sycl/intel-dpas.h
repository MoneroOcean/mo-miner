// Copyright GNU GPLv3 (c) 2026 MoneroOcean <support@moneroocean.stream>

#pragma once

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>

#include <cstdint>
#include <string>

namespace mom::intel_matrix {

namespace esimd = sycl::ext::intel::esimd;

inline unsigned dpas_width(const sycl::device& device) {
  namespace matrix = sycl::ext::oneapi::experimental::matrix;
  try {
    for (const auto& shape : device.get_info<
             sycl::ext::oneapi::experimental::info::device::matrix_combinations>()) {
      if (shape.atype == matrix::matrix_type::sint8 &&
          shape.btype == matrix::matrix_type::sint8 &&
          shape.ctype == matrix::matrix_type::sint32 &&
          shape.dtype == matrix::matrix_type::sint32 && shape.msize == 8 &&
          shape.ksize == 32 && (shape.nsize == 8 || shape.nsize == 16))
        return static_cast<unsigned>(shape.nsize);
    }
  } catch (const sycl::exception&) {
    // Some Intel runtimes omit joint_matrix shapes even though ESIMD/XMX is available.
  }
  // On Intel XMX generations, native EU width distinguishes the 256-bit Xe-HPG
  // register layout from 512-bit Xe-HPC/Xe2. Never infer it from the CU count.
  if (device.has(sycl::aspect::ext_intel_gpu_eu_simd_width)) {
    const unsigned width = device.get_info<sycl::ext::intel::info::device::gpu_eu_simd_width>();
    if (width == 8 || width == 16)
      return width;
  }
  throw std::string("Cannot determine Intel DPAS execution width");
}

// Preserve the shared 8x16 accumulator and 32x16 VNNI input layout. Xe-HPG has
// eight DPAS columns, whereas Xe-HPC/Xe2 has sixteen; a wider simd argument does
// not make the hardware execute a wider instruction.
template <unsigned Width>
SYCL_ESIMD_FUNCTION inline esimd::simd<int32_t, 128> dpas_8x16x32(
    esimd::simd<int32_t, 128> accumulator, esimd::simd<int8_t, 512> b,
    esimd::simd<int8_t, 256> a) {
  static_assert(Width == 8 || Width == 16);
  if constexpr (Width == 16) {
    return esimd::xmx::dpas<8, 8, int32_t, int32_t, int8_t, int8_t>(accumulator, b, a);
  } else {
#pragma unroll
    for (unsigned column = 0; column < 16; column += 8) {
      auto packed_b = b.template bit_cast_view<int32_t, 8, 16>()
                          .template select<8, 1, 8, 1>(0, column).read();
      esimd::simd<int8_t, 256> half_b = packed_b.template bit_cast_view<int8_t>().read();
      auto output = accumulator.template bit_cast_view<int32_t, 8, 16>()
                        .template select<8, 1, 8, 1>(0, column);
      esimd::simd<int32_t, 64> half_accumulator = output.read();
      output = esimd::xmx::dpas<8, 8, int32_t, int32_t, int8_t, int8_t>(
          half_accumulator, half_b, a);
    }
    return accumulator;
  }
}

} // namespace mom::intel_matrix
