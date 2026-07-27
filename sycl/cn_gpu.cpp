// Copyright GNU GPLv3 (c) 2023-2025 MoneroOcean <support@moneroocean.stream>

// SYCL cn/gpu implementation based on the public CryptoNight-GPU specification.
// OpenCL mining code by wolf9466, fireice_uk and psychocrypt
#include <sycl/sycl.hpp>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#if defined(MOM_SYCL_HAS_CUDA) && !defined(__SYCL_DEVICE_ONLY__)
#include <cuda.h>
#include <nvrtc.h>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#endif

#include "lib-internal.h"
#include "../native/consts.h"

const unsigned HASH_DATA_AREA = 136;
const unsigned CN_MEMORY      = 2 * 1024 * 1024;
const unsigned CN_GPU_ITER    = 0xC000;
const uint32_t CN_GPU_MASK    = 0x1FFFC0;

const unsigned CN_MEMORY4  = CN_MEMORY / sizeof(uint32_t);
const unsigned CN_MEMORY8  = CN_MEMORY / sizeof(uint64_t);
const unsigned CN_MEMORY16 = CN_MEMORY / sizeof(sycl::uint4);

const unsigned WORKGROUP_SIZE = 16; // 16 threads/hash; two hashes fill an Arc 32-wide sub-group

#if !defined(MOM_SYCL_PORTABLE_OPENCL)
alignas(64) const constexpr uint32_t AES[256] = {
  0xA56363C6, 0x847C7CF8, 0x997777EE, 0x8D7B7BF6, 0x0DF2F2FF, 0xBD6B6BD6, 0xB16F6FDE, 0x54C5C591,
  0x50303060, 0x03010102, 0xA96767CE, 0x7D2B2B56, 0x19FEFEE7, 0x62D7D7B5, 0xE6ABAB4D, 0x9A7676EC,
  0x45CACA8F, 0x9D82821F, 0x40C9C989, 0x877D7DFA, 0x15FAFAEF, 0xEB5959B2, 0xC947478E, 0x0BF0F0FB,
  0xECADAD41, 0x67D4D4B3, 0xFDA2A25F, 0xEAAFAF45, 0xBF9C9C23, 0xF7A4A453, 0x967272E4, 0x5BC0C09B,
  0xC2B7B775, 0x1CFDFDE1, 0xAE93933D, 0x6A26264C, 0x5A36366C, 0x413F3F7E, 0x02F7F7F5, 0x4FCCCC83,
  0x5C343468, 0xF4A5A551, 0x34E5E5D1, 0x08F1F1F9, 0x937171E2, 0x73D8D8AB, 0x53313162, 0x3F15152A,
  0x0C040408, 0x52C7C795, 0x65232346, 0x5EC3C39D, 0x28181830, 0xA1969637, 0x0F05050A, 0xB59A9A2F,
  0x0907070E, 0x36121224, 0x9B80801B, 0x3DE2E2DF, 0x26EBEBCD, 0x6927274E, 0xCDB2B27F, 0x9F7575EA,
  0x1B090912, 0x9E83831D, 0x742C2C58, 0x2E1A1A34, 0x2D1B1B36, 0xB26E6EDC, 0xEE5A5AB4, 0xFBA0A05B,
  0xF65252A4, 0x4D3B3B76, 0x61D6D6B7, 0xCEB3B37D, 0x7B292952, 0x3EE3E3DD, 0x712F2F5E, 0x97848413,
  0xF55353A6, 0x68D1D1B9, 0x00000000, 0x2CEDEDC1, 0x60202040, 0x1FFCFCE3, 0xC8B1B179, 0xED5B5BB6,
  0xBE6A6AD4, 0x46CBCB8D, 0xD9BEBE67, 0x4B393972, 0xDE4A4A94, 0xD44C4C98, 0xE85858B0, 0x4ACFCF85,
  0x6BD0D0BB, 0x2AEFEFC5, 0xE5AAAA4F, 0x16FBFBED, 0xC5434386, 0xD74D4D9A, 0x55333366, 0x94858511,
  0xCF45458A, 0x10F9F9E9, 0x06020204, 0x817F7FFE, 0xF05050A0, 0x443C3C78, 0xBA9F9F25, 0xE3A8A84B,
  0xF35151A2, 0xFEA3A35D, 0xC0404080, 0x8A8F8F05, 0xAD92923F, 0xBC9D9D21, 0x48383870, 0x04F5F5F1,
  0xDFBCBC63, 0xC1B6B677, 0x75DADAAF, 0x63212142, 0x30101020, 0x1AFFFFE5, 0x0EF3F3FD, 0x6DD2D2BF,
  0x4CCDCD81, 0x140C0C18, 0x35131326, 0x2FECECC3, 0xE15F5FBE, 0xA2979735, 0xCC444488, 0x3917172E,
  0x57C4C493, 0xF2A7A755, 0x827E7EFC, 0x473D3D7A, 0xAC6464C8, 0xE75D5DBA, 0x2B191932, 0x957373E6,
  0xA06060C0, 0x98818119, 0xD14F4F9E, 0x7FDCDCA3, 0x66222244, 0x7E2A2A54, 0xAB90903B, 0x8388880B,
  0xCA46468C, 0x29EEEEC7, 0xD3B8B86B, 0x3C141428, 0x79DEDEA7, 0xE25E5EBC, 0x1D0B0B16, 0x76DBDBAD,
  0x3BE0E0DB, 0x56323264, 0x4E3A3A74, 0x1E0A0A14, 0xDB494992, 0x0A06060C, 0x6C242448, 0xE45C5CB8,
  0x5DC2C29F, 0x6ED3D3BD, 0xEFACAC43, 0xA66262C4, 0xA8919139, 0xA4959531, 0x37E4E4D3, 0x8B7979F2,
  0x32E7E7D5, 0x43C8C88B, 0x5937376E, 0xB76D6DDA, 0x8C8D8D01, 0x64D5D5B1, 0xD24E4E9C, 0xE0A9A949,
  0xB46C6CD8, 0xFA5656AC, 0x07F4F4F3, 0x25EAEACF, 0xAF6565CA, 0x8E7A7AF4, 0xE9AEAE47, 0x18080810,
  0xD5BABA6F, 0x887878F0, 0x6F25254A, 0x722E2E5C, 0x241C1C38, 0xF1A6A657, 0xC7B4B473, 0x51C6C697,
  0x23E8E8CB, 0x7CDDDDA1, 0x9C7474E8, 0x211F1F3E, 0xDD4B4B96, 0xDCBDBD61, 0x868B8B0D, 0x858A8A0F,
  0x907070E0, 0x423E3E7C, 0xC4B5B571, 0xAA6666CC, 0xD8484890, 0x05030306, 0x01F6F6F7, 0x120E0E1C,
  0xA36161C2, 0x5F35356A, 0xF95757AE, 0xD0B9B969, 0x91868617, 0x58C1C199, 0x271D1D3A, 0xB99E9E27,
  0x38E1E1D9, 0x13F8F8EB, 0xB398982B, 0x33111122, 0xBB6969D2, 0x70D9D9A9, 0x898E8E07, 0xA7949433,
  0xB69B9B2D, 0x221E1E3C, 0x92878715, 0x20E9E9C9, 0x49CECE87, 0xFF5555AA, 0x78282850, 0x7ADFDFA5,
  0x8F8C8C03, 0xF8A1A159, 0x80898909, 0x170D0D1A, 0xDABFBF65, 0x31E6E6D7, 0xC6424284, 0xB86868D0,
  0xC3414182, 0xB0999929, 0x772D2D5A, 0x110F0F1E, 0xCBB0B07B, 0xFC5454A8, 0xD6BBBB6D, 0x3A16162C
};
#endif

// The arithmetic form is the standards-only fallback. Keeping these helpers available in every
// build lets the algorithm select its compile-time policy in C++ while the initialized native
// table above remains the sole preprocessor/code-generation boundary.
inline uint8_t cn_aes_xtime(const uint8_t value) {
  return static_cast<uint8_t>((value << 1) ^ ((value & 0x80U) ? 0x1bU : 0U));
}

inline uint8_t cn_aes_mul(uint8_t a, uint8_t b) {
  uint8_t result = 0;
  for (unsigned bit = 0; bit < 8; ++bit) {
    if (b & 1U) result ^= a;
    a = cn_aes_xtime(a);
    b >>= 1;
  }
  return result;
}

inline uint8_t cn_aes_sbox(const uint8_t value) {
  // x^254 is the multiplicative inverse in GF(2^8); zero maps to zero. The affine transform then
  // produces the standard AES S-box without a program-scope initialized table.
  uint8_t inverse = value ? 1U : 0U;
  uint8_t power = value;
  unsigned exponent = value ? 254U : 0U;
  while (exponent) {
    if (exponent & 1U) inverse = cn_aes_mul(inverse, power);
    power = cn_aes_mul(power, power);
    exponent >>= 1;
  }
  const auto rotl8 = [](const uint8_t x, const unsigned n) {
    return static_cast<uint8_t>((x << n) | (x >> (8U - n)));
  };
  return static_cast<uint8_t>(inverse ^ rotl8(inverse, 1) ^ rotl8(inverse, 2) ^
                              rotl8(inverse, 3) ^ rotl8(inverse, 4) ^ 0x63U);
}

inline uint32_t cn_aes_table(const uint8_t index) {
  const uint32_t s = cn_aes_sbox(index);
  const uint32_t s2 = cn_aes_xtime(static_cast<uint8_t>(s));
  return s2 | (s << 8) | (s << 16) | ((s2 ^ s) << 24);
}

inline uint32_t cn_keccak_round_constant(const unsigned round, const bool high) {
  switch (round) {
    case 0:  return high ? 0x00000000U : 0x00000001U;
    case 1:  return high ? 0x00000000U : 0x00008082U;
    case 2:  return high ? 0x80000000U : 0x0000808aU;
    case 3:  return high ? 0x80000000U : 0x80008000U;
    case 4:  return high ? 0x00000000U : 0x0000808bU;
    case 5:  return high ? 0x00000000U : 0x80000001U;
    case 6:  return high ? 0x80000000U : 0x80008081U;
    case 7:  return high ? 0x80000000U : 0x00008009U;
    case 8:  return high ? 0x00000000U : 0x0000008aU;
    case 9:  return high ? 0x00000000U : 0x00000088U;
    case 10: return high ? 0x00000000U : 0x80008009U;
    case 11: return high ? 0x00000000U : 0x8000000aU;
    case 12: return high ? 0x00000000U : 0x8000808bU;
    case 13: return high ? 0x80000000U : 0x0000008bU;
    case 14: return high ? 0x80000000U : 0x00008089U;
    case 15: return high ? 0x80000000U : 0x00008003U;
    case 16: return high ? 0x80000000U : 0x00008002U;
    case 17: return high ? 0x80000000U : 0x00000080U;
    case 18: return high ? 0x00000000U : 0x0000800aU;
    case 19: return high ? 0x80000000U : 0x8000000aU;
    case 20: return high ? 0x80000000U : 0x80008081U;
    case 21: return high ? 0x80000000U : 0x00008080U;
    case 22: return high ? 0x00000000U : 0x80000001U;
    default: return high ? 0x80000000U : 0x80008008U;
  }
}

// NVIDIA (sm_89) has a native 64-bit integer datapath, so the plain uint64 Keccak below is fastest;
// the 32-bit-pair variant (#else) exists for Xe2 (Intel Arc), which lacks a native 64-bit ALU. The
// nvptx device pass (__NVPTX__) gets the 64-bit Keccak, the spir64 pass the 32-bit-pair one, so the
// combined build is right on both. Define MOM_CNGPU_K32 to force the 32-bit-pair path on nvptx for
// A/B measurement (it loses on NVIDIA).
#if defined(__NVPTX__) && !defined(MOM_CNGPU_K32)
void keccak(uint64_t* const s) { // Hot device-side Keccak; keep explicit steps for backend codegen.
  static const uint32_t rotc[24] = {
    1,  3,  6,  10, 15, 21, 28, 36, 45, 55, 2,  14,
    27, 41, 56, 8,  25, 43, 62, 18, 39, 61, 20, 44
  }, piln[24] = {
    10, 7,  11, 17, 18, 3, 5,  16, 8,  21, 24, 4,
    15, 23, 19, 13, 12, 2, 20, 14, 22, 9,  6,  1
  };
  static const uint64_t rndc[24] = {
    0x0000000000000001, 0x0000000000008082, 0x800000000000808a,
    0x8000000080008000, 0x000000000000808b, 0x0000000080000001,
    0x8000000080008081, 0x8000000000008009, 0x000000000000008a,
    0x0000000000000088, 0x0000000080008009, 0x000000008000000a,
    0x000000008000808b, 0x800000000000008b, 0x8000000000008089,
    0x8000000000008003, 0x8000000000008002, 0x8000000000000080,
    0x000000000000800a, 0x800000008000000a, 0x8000000080008081,
    0x8000000000008080, 0x0000000080000001, 0x8000000080008008
  };
  for (unsigned round = 0; round < 24; ++ round) {
    uint64_t bc[5];
    // Theta step.
    bc[0] = s[0] ^ s[5] ^ s[10] ^ s[15] ^ s[20] ^
            mo_rotate(s[2] ^ s[7] ^ s[12] ^ s[17] ^ s[22], uint64_t{1});
    bc[1] = s[1] ^ s[6] ^ s[11] ^ s[16] ^ s[21] ^
            mo_rotate(s[3] ^ s[8] ^ s[13] ^ s[18] ^ s[23], uint64_t{1});
    bc[2] = s[2] ^ s[7] ^ s[12] ^ s[17] ^ s[22] ^
            mo_rotate(s[4] ^ s[9] ^ s[14] ^ s[19] ^ s[24], uint64_t{1});
    bc[3] = s[3] ^ s[8] ^ s[13] ^ s[18] ^ s[23] ^
            mo_rotate(s[0] ^ s[5] ^ s[10] ^ s[15] ^ s[20], uint64_t{1});
    bc[4] = s[4] ^ s[9] ^ s[14] ^ s[19] ^ s[24] ^
            mo_rotate(s[1] ^ s[6] ^ s[11] ^ s[16] ^ s[21], uint64_t{1});
    s[0] ^= bc[4]; s[5] ^= bc[4]; s[10] ^= bc[4]; s[15] ^= bc[4]; s[20] ^= bc[4];
    s[1] ^= bc[0]; s[6] ^= bc[0]; s[11] ^= bc[0]; s[16] ^= bc[0]; s[21] ^= bc[0];
    s[2] ^= bc[1]; s[7] ^= bc[1]; s[12] ^= bc[1]; s[17] ^= bc[1]; s[22] ^= bc[1];
    s[3] ^= bc[2]; s[8] ^= bc[2]; s[13] ^= bc[2]; s[18] ^= bc[2]; s[23] ^= bc[2];
    s[4] ^= bc[3]; s[9] ^= bc[3]; s[14] ^= bc[3]; s[19] ^= bc[3]; s[24] ^= bc[3];
    // Rho and Pi steps.
    uint64_t t = s[1];
    for (unsigned i = 0; i < 24; ++ i) {
      bc[0] = s[piln[i]];
      s[piln[i]] = mo_rotate(t, static_cast<uint64_t>(rotc[i]));
      t = bc[0];
    }
    // Chi step.
    for (unsigned i = 0; i < 25; i += 5) {
      const uint64_t tmp1 = s[i], tmp2 = s[i + 1];
      s[i    ] = mo_bitselect(s[i    ] ^ s[i + 2], s[i    ], s[i + 1]);
      s[i + 1] = mo_bitselect(s[i + 1] ^ s[i + 3], s[i + 1], s[i + 2]);
      s[i + 2] = mo_bitselect(s[i + 2] ^ s[i + 4], s[i + 2], s[i + 3]);
      s[i + 3] = mo_bitselect(s[i + 3] ^ tmp1,     s[i + 3], s[i + 4]);
      s[i + 4] = mo_bitselect(s[i + 4] ^ tmp2,     s[i + 4], tmp1);
    }
    s[0] ^= rndc[round];
  }
}
#else
// 32-bit-pair Keccak-f[1600] for Xe2 (Intel Arc B580), which has no native
// 64-bit integer ALU. Each 64-bit lane is held as two uint32 (lo/hi). Bit-exact
// with the uint64 body above. rho/pi rotation amounts are compile-time constants
// (the loop is fully unrolled), so each rotl64 reduces to shift+or on uint32.

// rotl64(value held as lo/hi) by a compile-time amount R, 1 <= R <= 63.
template <unsigned R>
static inline void mo_rotl64_32(uint32_t& lo, uint32_t& hi) {
  if constexpr (R == 32) {
    const uint32_t t = lo; lo = hi; hi = t;
  } else if constexpr (R < 32) {
    const uint32_t nlo = (lo << R) | (hi >> (32 - R));
    const uint32_t nhi = (hi << R) | (lo >> (32 - R));
    lo = nlo; hi = nhi;
  } else { // R > 32
    const unsigned s = R - 32;
    const uint32_t nlo = (hi << s) | (lo >> (32 - s));
    const uint32_t nhi = (lo << s) | (hi >> (32 - s));
    lo = nlo; hi = nhi;
  }
}

template <unsigned Position, unsigned Rotation>
static inline void mo_rho_pi_32(uint32_t* const lo, uint32_t* const hi,
                                uint32_t& tlo, uint32_t& thi) {
  const uint32_t next_lo = lo[Position], next_hi = hi[Position];
  uint32_t rotated_lo = tlo, rotated_hi = thi;
  mo_rotl64_32<Rotation>(rotated_lo, rotated_hi);
  lo[Position] = rotated_lo;
  hi[Position] = rotated_hi;
  tlo = next_lo;
  thi = next_hi;
}

void keccak(uint64_t* const s) { // Hot device-side Keccak; keep explicit steps for backend codegen.
#if !defined(MOM_SYCL_PORTABLE_OPENCL)
  static const uint32_t rndc_lo[24] = {
    0x00000001, 0x00008082, 0x0000808a, 0x80008000, 0x0000808b, 0x80000001,
    0x80008081, 0x00008009, 0x0000008a, 0x00000088, 0x80008009, 0x8000000a,
    0x8000808b, 0x0000008b, 0x00008089, 0x00008003, 0x00008002, 0x00000080,
    0x0000800a, 0x8000000a, 0x80008081, 0x00008080, 0x80000001, 0x80008008
  };
  static const uint32_t rndc_hi[24] = {
    0x00000000, 0x00000000, 0x80000000, 0x80000000, 0x00000000, 0x00000000,
    0x80000000, 0x80000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000,
    0x00000000, 0x80000000, 0x80000000, 0x80000000, 0x00000000, 0x80000000
  };
#endif
  // Split state into lo/hi halves.
  uint32_t lo[25], hi[25];
  for (unsigned i = 0; i < 25; ++ i) {
    lo[i] = static_cast<uint32_t>(s[i]);
    hi[i] = static_cast<uint32_t>(s[i] >> 32);
  }
  for (unsigned round = 0; round < 24; ++ round) {
    uint32_t bclo[5], bchi[5];
    // Theta step. The inner rotate is rotl64 by 1 (R<32 path, R=1).
    // bc[k] = column[k] ^ rotl64(column[k+2 mod 5], 1)
    {
      uint32_t col0lo = lo[0]^lo[5]^lo[10]^lo[15]^lo[20];
      uint32_t col0hi = hi[0]^hi[5]^hi[10]^hi[15]^hi[20];
      uint32_t col1lo = lo[1]^lo[6]^lo[11]^lo[16]^lo[21];
      uint32_t col1hi = hi[1]^hi[6]^hi[11]^hi[16]^hi[21];
      uint32_t col2lo = lo[2]^lo[7]^lo[12]^lo[17]^lo[22];
      uint32_t col2hi = hi[2]^hi[7]^hi[12]^hi[17]^hi[22];
      uint32_t col3lo = lo[3]^lo[8]^lo[13]^lo[18]^lo[23];
      uint32_t col3hi = hi[3]^hi[8]^hi[13]^hi[18]^hi[23];
      uint32_t col4lo = lo[4]^lo[9]^lo[14]^lo[19]^lo[24];
      uint32_t col4hi = hi[4]^hi[9]^hi[14]^hi[19]^hi[24];
      bclo[0] = col0lo ^ ((col2lo << 1) | (col2hi >> 31));
      bchi[0] = col0hi ^ ((col2hi << 1) | (col2lo >> 31));
      bclo[1] = col1lo ^ ((col3lo << 1) | (col3hi >> 31));
      bchi[1] = col1hi ^ ((col3hi << 1) | (col3lo >> 31));
      bclo[2] = col2lo ^ ((col4lo << 1) | (col4hi >> 31));
      bchi[2] = col2hi ^ ((col4hi << 1) | (col4lo >> 31));
      bclo[3] = col3lo ^ ((col0lo << 1) | (col0hi >> 31));
      bchi[3] = col3hi ^ ((col0hi << 1) | (col0lo >> 31));
      bclo[4] = col4lo ^ ((col1lo << 1) | (col1hi >> 31));
      bchi[4] = col4hi ^ ((col1hi << 1) | (col1lo >> 31));
    }
    for (unsigned k = 0; k < 5; ++ k) {
      const uint32_t dlo = bclo[(k + 4) % 5], dhi = bchi[(k + 4) % 5];
      lo[k     ] ^= dlo; hi[k     ] ^= dhi;
      lo[k +  5] ^= dlo; hi[k +  5] ^= dhi;
      lo[k + 10] ^= dlo; hi[k + 10] ^= dhi;
      lo[k + 15] ^= dlo; hi[k + 15] ^= dhi;
      lo[k + 20] ^= dlo; hi[k + 20] ^= dhi;
    }
    // Rho and Pi steps - fully unrolled so each rotation amount is constant.
    // Chain: t = s[1]; for each i: tmp=s[piln[i]]; s[piln[i]]=rotl(t,rotc[i]); t=tmp;
    // piln = {10,7,11,17,18,3,5,16,8,21,24,4,15,23,19,13,12,2,20,14,22,9,6,1}
    // rotc = {1,3,6,10,15,21,28,36,45,55,2,14,27,41,56,8,25,43,62,18,39,61,20,44}
    uint32_t tlo = lo[1], thi = hi[1];
    mo_rho_pi_32<10, 1>(lo, hi, tlo, thi);
    mo_rho_pi_32<7,  3>(lo, hi, tlo, thi);
    mo_rho_pi_32<11, 6>(lo, hi, tlo, thi);
    mo_rho_pi_32<17, 10>(lo, hi, tlo, thi);
    mo_rho_pi_32<18, 15>(lo, hi, tlo, thi);
    mo_rho_pi_32<3,  21>(lo, hi, tlo, thi);
    mo_rho_pi_32<5,  28>(lo, hi, tlo, thi);
    mo_rho_pi_32<16, 36>(lo, hi, tlo, thi);
    mo_rho_pi_32<8,  45>(lo, hi, tlo, thi);
    mo_rho_pi_32<21, 55>(lo, hi, tlo, thi);
    mo_rho_pi_32<24, 2>(lo, hi, tlo, thi);
    mo_rho_pi_32<4,  14>(lo, hi, tlo, thi);
    mo_rho_pi_32<15, 27>(lo, hi, tlo, thi);
    mo_rho_pi_32<23, 41>(lo, hi, tlo, thi);
    mo_rho_pi_32<19, 56>(lo, hi, tlo, thi);
    mo_rho_pi_32<13, 8>(lo, hi, tlo, thi);
    mo_rho_pi_32<12, 25>(lo, hi, tlo, thi);
    mo_rho_pi_32<2,  43>(lo, hi, tlo, thi);
    mo_rho_pi_32<20, 62>(lo, hi, tlo, thi);
    mo_rho_pi_32<14, 18>(lo, hi, tlo, thi);
    mo_rho_pi_32<22, 39>(lo, hi, tlo, thi);
    mo_rho_pi_32<9,  61>(lo, hi, tlo, thi);
    mo_rho_pi_32<6,  20>(lo, hi, tlo, thi);
    mo_rho_pi_32<1,  44>(lo, hi, tlo, thi);
    // Chi step. mo_bitselect(a,b,c) = (a & ~c) | (b & c), component-wise.
    for (unsigned i = 0; i < 25; i += 5) {
      const uint32_t t1lo = lo[i], t1hi = hi[i];
      const uint32_t t2lo = lo[i + 1], t2hi = hi[i + 1];
      uint32_t n0lo = ((lo[i    ] ^ lo[i + 2]) & ~lo[i + 1]) | (lo[i    ] & lo[i + 1]);
      uint32_t n0hi = ((hi[i    ] ^ hi[i + 2]) & ~hi[i + 1]) | (hi[i    ] & hi[i + 1]);
      uint32_t n1lo = ((lo[i + 1] ^ lo[i + 3]) & ~lo[i + 2]) | (lo[i + 1] & lo[i + 2]);
      uint32_t n1hi = ((hi[i + 1] ^ hi[i + 3]) & ~hi[i + 2]) | (hi[i + 1] & hi[i + 2]);
      uint32_t n2lo = ((lo[i + 2] ^ lo[i + 4]) & ~lo[i + 3]) | (lo[i + 2] & lo[i + 3]);
      uint32_t n2hi = ((hi[i + 2] ^ hi[i + 4]) & ~hi[i + 3]) | (hi[i + 2] & hi[i + 3]);
      uint32_t n3lo = ((lo[i + 3] ^ t1lo) & ~lo[i + 4]) | (lo[i + 3] & lo[i + 4]);
      uint32_t n3hi = ((hi[i + 3] ^ t1hi) & ~hi[i + 4]) | (hi[i + 3] & hi[i + 4]);
      uint32_t n4lo = ((lo[i + 4] ^ t2lo) & ~t1lo) | (lo[i + 4] & t1lo);
      uint32_t n4hi = ((hi[i + 4] ^ t2hi) & ~t1hi) | (hi[i + 4] & t1hi);
      lo[i    ] = n0lo; hi[i    ] = n0hi;
      lo[i + 1] = n1lo; hi[i + 1] = n1hi;
      lo[i + 2] = n2lo; hi[i + 2] = n2hi;
      lo[i + 3] = n3lo; hi[i + 3] = n3hi;
      lo[i + 4] = n4lo; hi[i + 4] = n4hi;
    }
#if defined(MOM_SYCL_PORTABLE_OPENCL)
    lo[0] ^= cn_keccak_round_constant(round, false);
    hi[0] ^= cn_keccak_round_constant(round, true);
#else
    lo[0] ^= rndc_lo[round];
    hi[0] ^= rndc_hi[round];
#endif
  }
  // Recombine lo/hi into the 64-bit state.
  for (unsigned i = 0; i < 25; ++ i) {
    s[i] = static_cast<uint64_t>(lo[i]) | (static_cast<uint64_t>(hi[i]) << 32);
  }
}
#endif

void generate512(const unsigned idx, const uint64_t* const in, uint64_t* out) { // Expands one scratchpad stripe.
  uint64_t hash[25];
  hash[0] = in[0] ^ idx;
  for (unsigned i = 1; i < 25; ++ i) hash[i] = in[i];

  // Three Keccak rounds emit one 512-byte chunk (20 + 22 + 22 = 64 uint64 words).
  for (unsigned a = 0; a < 3; ++ a) {
    keccak(hash);
    const unsigned skip = a == 0 ? 20U : 22U;
    for (unsigned i = 0; i < skip; ++ i) out[i] = hash[i];
    out += skip;
  }
}

inline int32_t* lpad_ptr(const unsigned idx, const unsigned n, int32_t* const lpad) {
  return reinterpret_cast<int32_t*>(
    reinterpret_cast<uint8_t*>(lpad) + (idx & CN_GPU_MASK) + n * 16
  );
}

// Correctly-rounded f32 division. The cn/gpu reference (Intel/Level-Zero with
// -cl-fp32-correctly-rounded-divide-sqrt) computes the recurrence's one division
// with an IEEE correctly-rounded result. DPC++'s CUDA backend emits a division that is 1 ULP low
// for some operands (not correctly rounded), and a single 1-ULP error desyncs the chaotic
// recurrence into a wrong hash. Force the PTX correctly-rounded div.rn.f32 per component in the
// nvptx device pass (__NVPTX__); everywhere else use the plain operator/ (unchanged Intel codegen).
inline sycl::float4 mo_div_rn(const sycl::float4 a, const sycl::float4 b) {
#if defined(__NVPTX__) && (!defined(MOM_SYCL_ADAPTIVECPP) || defined(__CUDA_ARCH__))
  sycl::float4 r;
  #pragma unroll
  for (int k = 0; k < 4; ++k) {
    float q;
    asm("div.rn.f32 %0, %1, %2;" : "=f"(q) : "f"(a[k]), "f"(b[k]));
    r[k] = q;
  }
  return r;
#else
  return a / b;
#endif
}

inline sycl::float4 my_and_or_ps(const sycl::float4 x, const uint32_t _and, const uint32_t _or) { // Float bit shaping used by cn/gpu recurrence.
  const sycl::uint4 i = (sycl::bit_cast<sycl::uint4>(x) & _and) | _or;
  return sycl::bit_cast<sycl::float4>(i);
}

inline sycl::float4 fma_break(const sycl::float4 x) { // Breaks the FMA dependency chain.
  return my_and_or_ps(x, 0xFEFFFFFF, 0x00800000);
}

inline const char* cn_gpu_fp_compile_options(const sycl::device& dev) {
  constexpr const char* optimized =
    "-cl-fp32-correctly-rounded-divide-sqrt -cl-mad-enable -cl-fast-relaxed-math -cl-no-signed-zeros";
  constexpr const char* stable =
    "-cl-fp32-correctly-rounded-divide-sqrt -cl-mad-enable -cl-fast-relaxed-math -cl-no-signed-zeros -cl-opt-disable";

  if constexpr (mom_sycl_portable_opencl) {
    (void)dev;
    (void)optimized;
    return stable;
  } else {
    if (sycl_is_level_zero_gpu(dev)) return "-cl-fp32-correctly-rounded-divide-sqrt";
    return dev.is_cpu() ? stable : optimized; // CPU backends otherwise reassociate the fp recurrence.
  }
}

inline void sub_round(
  const sycl::float4 n0, const sycl::float4 n1, const sycl::float4 n2, const sycl::float4 n3,
  const sycl::float4 rnd_c, sycl::float4* const pn, sycl::float4* const pd, sycl::float4* const pc
) {
  const sycl::float4 nn2 = n0 * *pc;
  const sycl::float4 nn  = fma_break((n1 + *pc) * (nn2 * nn2));
  *pn += nn;

  const sycl::float4 dd2 = n2 * *pc;
  const sycl::float4 dd  = fma_break((n3 - *pc) * (dd2 * dd2));
  *pd += dd;
  *pc = ((*pc + rnd_c) + sycl::float4(0.734375f)) + my_and_or_ps(nn + dd, 0x807FFFFF, 0x40000000); // order matters
}

inline void round_compute( // The divisor mask also prevents divide-by-near-zero overflow.
  const sycl::float4 n0, const sycl::float4 n1, const sycl::float4 n2, const sycl::float4 n3,
  const sycl::float4 rnd_c, sycl::float4* const pc, sycl::float4* const pr
) {
  sycl::float4 n = sycl::float4(0.0f);
  sycl::float4 d = sycl::float4(0.0f);

  sub_round(n0, n1, n2, n3, rnd_c, &n, &d, pc);
  sub_round(n1, n2, n3, n0, rnd_c, &n, &d, pc);
  sub_round(n2, n3, n0, n1, rnd_c, &n, &d, pc);
  sub_round(n3, n0, n1, n2, rnd_c, &n, &d, pc);
  sub_round(n3, n2, n1, n0, rnd_c, &n, &d, pc);
  sub_round(n2, n1, n0, n3, rnd_c, &n, &d, pc);
  sub_round(n1, n0, n3, n2, rnd_c, &n, &d, pc);
  sub_round(n0, n3, n2, n1, rnd_c, &n, &d, pc);
  *pr += mo_div_rn(n, my_and_or_ps(d, 0xFF7FFFFF, 0x40000000));
}

inline sycl::int4 single_comupte(
  const sycl::float4 n0, const sycl::float4 n1, const sycl::float4 n2, const sycl::float4 n3,
  const float cnt, const sycl::float4 rnd_c, sycl::float4* const psum
) {
  sycl::float4 c = sycl::float4(cnt);
  sycl::float4 r = sycl::float4(0.0f);

  // Partial unroll improves pipeline utilization.
  for (int i = 0; i < 4; ++ i) round_compute(n0, n1, n2, n3, rnd_c, &c, &r);

  const sycl::float4 r2 = my_and_or_ps(r, 0x807FFFFF, 0x40000000);
  *psum = r2;
  return (r2 * sycl::float4(536870880.0f)).convert<int32_t, sycl::rounding_mode::rte>();
}

inline sycl::int4 my_alignr_epi8(const sycl::int4 a, const unsigned rot) {
  const unsigned right = 8 * rot;
  const unsigned left  = 32 - right;
  // Reinterpret int->uint (bit-preserving) by explicit construction.
  const sycl::uint4 u(static_cast<uint32_t>(a[0]), static_cast<uint32_t>(a[1]),
                      static_cast<uint32_t>(a[2]), static_cast<uint32_t>(a[3]));

  return sycl::int4(
    (u[0] >> right) | (u[1] << left),
    (u[1] >> right) | (u[2] << left),
    (u[2] >> right) | (u[3] << left),
    (u[3] >> right) | (u[0] << left)
  );
}

inline void single_comupte_wrap(
  const sycl::int4 v0, const sycl::int4 v1, const sycl::int4 v2, const sycl::int4 v3,
  const unsigned rot, const float cnt, const sycl::float4 rnd_c,
  sycl::float4* const psum, sycl::int4* const pout
) {
  const sycl::float4 n0 = v0.convert<float, sycl::rounding_mode::rte>(),
                     n1 = v1.convert<float, sycl::rounding_mode::rte>(),
                     n2 = v2.convert<float, sycl::rounding_mode::rte>(),
                     n3 = v3.convert<float, sycl::rounding_mode::rte>();
  const sycl::int4 r = single_comupte(n0, n1, n2, n3, cnt, rnd_c, psum);

  *pout = rot == 0 ? r : my_alignr_epi8(r, rot);
}

template <bool NarrowSupported = !mom_sycl_adaptivecpp>
inline void cn_recurrence_barrier(const sycl::nd_item<1>& item, const sycl::sub_group& subgroup,
                                  const bool use_narrow) {
  if constexpr (NarrowSupported) {
    if (use_narrow) {
      sycl::group_barrier(subgroup);
      return;
    }
  }
  item.barrier(sycl::access::fence_space::local_space);
}

template <bool Portable = mom_sycl_portable_opencl>
inline unsigned cn_lane_select(const unsigned lane, const unsigned column) {
  if constexpr (Portable) {
    uint32_t packed;
    switch (lane) {
      case 0: packed = 0xe4; break; case 1: packed = 0x78; break;
      case 2: packed = 0x9c; break; case 3: packed = 0x6c; break;
      case 4: packed = 0xe1; break; case 5: packed = 0x39; break;
      case 6: packed = 0x8d; break; case 7: packed = 0x2d; break;
      case 8: packed = 0xc6; break; case 9: packed = 0x72; break;
      case 10: packed = 0x1e; break; case 11: packed = 0x4e; break;
      case 12: packed = 0x27; break; case 13: packed = 0x4b; break;
      case 14: packed = 0x93; break; default: packed = 0x63; break;
    }
    return (packed >> (2U * column)) & 3U;
  } else {
    const unsigned lanes[16][4] = {
      {0, 1, 2, 3}, {0, 2, 3, 1}, {0, 3, 1, 2}, {0, 3, 2, 1},
      {1, 0, 2, 3}, {1, 2, 3, 0}, {1, 3, 0, 2}, {1, 3, 2, 0},
      {2, 1, 0, 3}, {2, 0, 3, 1}, {2, 3, 1, 0}, {2, 3, 0, 1},
      {3, 1, 2, 0}, {3, 2, 0, 1}, {3, 0, 1, 2}, {3, 0, 2, 1}
    };
    return lanes[lane][column];
  }
}

template <bool Portable = mom_sycl_portable_opencl>
inline float cn_lane_count(const unsigned lane) {
  if constexpr (Portable) {
    unsigned numerator;
    switch (lane) {
      case 0: numerator = 172; break; case 1: numerator = 164; break;
      case 2: numerator = 174; break; case 3: numerator = 175; break;
      case 4: numerator = 183; break; case 5: numerator = 179; break;
      case 6: numerator = 177; break; case 7: numerator = 167; break;
      case 8: numerator = 181; break; case 9: numerator = 163; break;
      case 10: numerator = 161; break; case 11: numerator = 165; break;
      case 12: numerator = 169; break; case 13: numerator = 173; break;
      case 14: numerator = 171; break; default: numerator = 187; break;
    }
    return static_cast<float>(numerator) * (1.0f / 128.0f);
  } else {
    const float counts[16] = {
      1.34375f,   1.28125f,   1.359375f,  1.3671875f,
      1.4296875f, 1.3984375f, 1.3828125f, 1.3046875f,
      1.4140625f, 1.2734375f, 1.2578125f, 1.2890625f,
      1.3203125f, 1.3515625f, 1.3359375f, 1.4609375f
    };
    return counts[lane];
  }
}

inline uint32_t sw(const uint32_t inw) { // AES S-box for key expansion.
#if !defined(MOM_SYCL_PORTABLE_OPENCL)
  static const uint8_t sbox[256] = {
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
  };
#endif
  union { uint8_t b[4]; uint32_t u; };
  u = inw;
#if defined(MOM_SYCL_PORTABLE_OPENCL)
  b[0] = cn_aes_sbox(b[0]); b[1] = cn_aes_sbox(b[1]);
  b[2] = cn_aes_sbox(b[2]); b[3] = cn_aes_sbox(b[3]);
#else
  b[0] = sbox[b[0]]; b[1] = sbox[b[1]]; b[2] = sbox[b[2]]; b[3] = sbox[b[3]];
#endif
  return u;
}

void aes_expend_key(uint32_t* const keybuf) { // Unrolled AES-256 key expansion.
#if !defined(MOM_SYCL_PORTABLE_OPENCL)
  static const uint32_t rcon[8] = { 0x8d, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40 };
#endif

  for (unsigned c = 8, i = 1; c < 40; ++ c) {
    const uint32_t t = ((!(c & 7)) || ((c & 7) == 4)) ? sw(keybuf[c - 1]) : keybuf[c - 1];
    if (!(c & 7)) {
#if defined(MOM_SYCL_PORTABLE_OPENCL)
      uint32_t round_constant = 1;
      for (unsigned step = 1; step < i; ++step)
        round_constant = cn_aes_xtime(static_cast<uint8_t>(round_constant));
      ++i;
#else
      const uint32_t round_constant = rcon[i++];
#endif
      keybuf[c] = keybuf[c - 8] ^ mo_rotate(t, 24U) ^ round_constant;
    } else {
      keybuf[c] = keybuf[c - 8] ^ t;
    }
  }
}

// Local-memory T-table: AES base table rotated by 0/8/16/24 bytes.
using AesTable = sycl::local_accessor<uint32_t, 1>;

// Cooperatively fill the four rotated T-tables; caller must barrier before first use.
inline void aes_fill_tables(const AesTable& aes0, const AesTable& aes1, const AesTable& aes2,
                            const AesTable& aes3, const sycl::nd_item<2>& nd) {
  const unsigned linear = nd.get_local_id(0) * nd.get_local_range(1) + nd.get_local_id(1);
  const unsigned threads = nd.get_local_range(0) * nd.get_local_range(1);
  for (unsigned i = linear; i < 256; i += threads) {
#if defined(MOM_SYCL_PORTABLE_OPENCL)
    const uint32_t a = cn_aes_table(static_cast<uint8_t>(i));
#else
    const uint32_t a = AES[i];
#endif
    aes0[i] = a; aes1[i] = mo_rotate(a, 8U); aes2[i] = mo_rotate(a, 16U); aes3[i] = mo_rotate(a, 24U);
  }
}

inline void aes_round(sycl::uint4* const px, const sycl::uint4 key, const AesTable& aes0,
                      const AesTable& aes1, const AesTable& aes2, const AesTable& aes3) {
  union { uint8_t b[4]; uint32_t u; } u0, u1, u2, u3;
  u0.u = px->x(); u1.u = px->y(); u2.u = px->z(); u3.u = px->w();

  px->x() = key[0] ^ aes0[u0.b[0]] ^ aes1[u1.b[1]] ^ aes2[u2.b[2]] ^ aes3[u3.b[3]];
  px->y() = key[1] ^ aes0[u1.b[0]] ^ aes1[u2.b[1]] ^ aes2[u3.b[2]] ^ aes3[u0.b[3]];
  px->z() = key[2] ^ aes0[u2.b[0]] ^ aes1[u3.b[1]] ^ aes2[u0.b[2]] ^ aes3[u1.b[3]];
  px->w() = key[3] ^ aes0[u3.b[0]] ^ aes1[u0.b[1]] ^ aes2[u1.b[2]] ^ aes3[u2.b[3]];
}

union AesKey {
  uint32_t u[40]; sycl::uint4 u4[10]; sycl::uint8 u8[5]; // aligned at use sites for vector loads
  AesKey() {} // explicit default ctor: sycl::vec has a non-trivial default ctor, so the union needs one to stay default-constructible (uninitialized).
};

inline sycl::async_handler cn_gpu_exception_handler() {
  return [] (sycl::exception_list exceptions) {
    for (std::exception_ptr const& e : exceptions) {
      try {
        std::rethrow_exception(e);
      } catch(sycl::exception const& e) {
        printf("Caught asynchronous SYCL exception:\n%s\n", e.what()); throw;
      }
    }
  };
}

#if defined(MOM_SYCL_HAS_CUDA) && !defined(__SYCL_DEVICE_ONLY__)
struct CnGpuCudaDriverApi {
#if defined(_WIN32)
  HMODULE library = nullptr;
#else
  void* library = nullptr;
#endif
  decltype(&cuInit) init = nullptr;
  decltype(&cuDeviceGetAttribute) device_attribute = nullptr;
  decltype(&cuDevicePrimaryCtxRetain) retain_primary = nullptr;
  decltype(&cuDevicePrimaryCtxRelease) release_primary = nullptr;
  decltype(&cuCtxSetCurrent) set_current = nullptr;
  decltype(&cuModuleLoadData) module_load = nullptr;
  decltype(&cuModuleUnload) module_unload = nullptr;
  decltype(&cuModuleGetFunction) module_function = nullptr;
  decltype(&cuLaunchKernel) launch = nullptr;
  decltype(&cuEventCreate) event_create = nullptr;
  decltype(&cuEventDestroy) event_destroy = nullptr;
  decltype(&cuEventRecord) event_record = nullptr;
  decltype(&cuEventQuery) event_query = nullptr;
  decltype(&cuGetErrorString) error_string = nullptr;
  std::string error;

  template <typename T> bool symbol(T& output, const char* const name) {
#if defined(_WIN32)
    output = reinterpret_cast<T>(GetProcAddress(library, name));
#else
    output = reinterpret_cast<T>(dlsym(library, name));
#endif
    if (output) return true;
    error = std::string("missing CUDA driver symbol ") + name;
    return false;
  }

  CnGpuCudaDriverApi() {
#if defined(_WIN32)
    library = LoadLibraryA("nvcuda.dll");
#else
    library = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
#endif
    if (!library) {
      error = "the CUDA driver library is not installed";
      return;
    }
    (void)(symbol(init, "cuInit") &&
           symbol(device_attribute, "cuDeviceGetAttribute") &&
           symbol(retain_primary, "cuDevicePrimaryCtxRetain") &&
           symbol(release_primary, "cuDevicePrimaryCtxRelease") &&
           symbol(set_current, "cuCtxSetCurrent") &&
           symbol(module_load, "cuModuleLoadData") &&
           symbol(module_unload, "cuModuleUnload") &&
           symbol(module_function, "cuModuleGetFunction") &&
           symbol(launch, "cuLaunchKernel") &&
           symbol(event_create, "cuEventCreate") &&
           symbol(event_destroy, "cuEventDestroy") &&
           symbol(event_record, "cuEventRecord") &&
           symbol(event_query, "cuEventQuery") &&
           symbol(error_string, "cuGetErrorString"));
  }
  bool available() const {
    return library && init && device_attribute && retain_primary && release_primary &&
           set_current && module_load && module_unload && module_function && launch &&
           event_create && event_destroy && event_record && event_query && error_string;
  }
  static CnGpuCudaDriverApi& instance() {
    static CnGpuCudaDriverApi api;
    return api;
  }
};

struct CnGpuNvrtcApi {
#if defined(_WIN32)
  HMODULE library = nullptr;
#else
  void* library = nullptr;
#endif
  decltype(&nvrtcCreateProgram) create_program = nullptr;
  decltype(&nvrtcCompileProgram) compile_program = nullptr;
  decltype(&nvrtcGetProgramLogSize) get_log_size = nullptr;
  decltype(&nvrtcGetProgramLog) get_log = nullptr;
  decltype(&nvrtcGetPTXSize) get_ptx_size = nullptr;
  decltype(&nvrtcGetPTX) get_ptx = nullptr;
  decltype(&nvrtcDestroyProgram) destroy_program = nullptr;
  decltype(&nvrtcGetErrorString) error_string = nullptr;
  std::string error;

  template <typename T> bool symbol(T& output, const char* const name) {
#if defined(_WIN32)
    output = reinterpret_cast<T>(GetProcAddress(library, name));
#else
    output = reinterpret_cast<T>(dlsym(library, name));
#endif
    if (output) return true;
    error = std::string("missing NVRTC symbol ") + name;
    return false;
  }

  CnGpuNvrtcApi() {
#if defined(_WIN32)
    const char* const names[] = {"nvrtc64_130_0.dll", "nvrtc64_120_0.dll",
                                 "nvrtc64_121_0.dll", "nvrtc64_122_0.dll"};
    for (const char* const name : names)
      if ((library = LoadLibraryA(name))) break;
#else
    const char* const names[] = {"libnvrtc.so", "libnvrtc.so.13", "libnvrtc.so.12"};
    for (const char* const name : names)
      if ((library = dlopen(name, RTLD_NOW | RTLD_LOCAL))) break;
#endif
    if (!library) {
      error = "NVRTC is not installed";
      return;
    }
    (void)(symbol(create_program, "nvrtcCreateProgram") &&
           symbol(compile_program, "nvrtcCompileProgram") &&
           symbol(get_log_size, "nvrtcGetProgramLogSize") &&
           symbol(get_log, "nvrtcGetProgramLog") &&
           symbol(get_ptx_size, "nvrtcGetPTXSize") &&
           symbol(get_ptx, "nvrtcGetPTX") &&
           symbol(destroy_program, "nvrtcDestroyProgram") &&
           symbol(error_string, "nvrtcGetErrorString"));
  }
  bool available() const {
    return library && create_program && compile_program && get_log_size && get_log &&
           get_ptx_size && get_ptx && destroy_program && error_string;
  }
  static CnGpuNvrtcApi& instance() {
    static CnGpuNvrtcApi api;
    return api;
  }
};

struct CnGpuCudaExpand {
  CUmodule module = nullptr;
  CUfunction kernel = nullptr;
  CUevent done = nullptr;
  CUcontext context = nullptr;
  CUdevice device = 0;
  bool retained_primary = false;
  bool checked = false;
  bool enabled = false;

  static void cuda_check(CnGpuCudaDriverApi& api, const CUresult status,
                         const char* const operation) {
    if (status == CUDA_SUCCESS) return;
    const char* detail = nullptr;
    if (api.error_string) (void)api.error_string(status, &detail);
    throw std::string(operation) + ": " + (detail ? detail : "CUDA driver error");
  }
  static std::string rtc_error(CnGpuNvrtcApi& api, const nvrtcResult status,
                               nvrtcProgram program, const char* const operation) {
    size_t size = 0;
    std::string log;
    if (program && api.get_log_size(program, &size) == NVRTC_SUCCESS && size) {
      log.resize(size);
      (void)api.get_log(program, log.data());
    }
    return std::string(operation) + ": " + api.error_string(status) +
           (log.empty() ? "" : "\n" + log);
  }
  static CUdevice native_device(const sycl::device& device) {
#if defined(MOM_SYCL_ADAPTIVECPP)
    (void)device;
    return 0;
#else
    return sycl::get_native<sycl::backend::ext_oneapi_cuda>(device);
#endif
  }
  static uint64_t hash_text(const std::string& value) {
    uint64_t hash = 1469598103934665603ULL;
    for (const unsigned char byte : value) {
      hash ^= byte;
      hash *= 1099511628211ULL;
    }
    return hash;
  }
  static std::filesystem::path cache_dir() {
    if (const char* configured = std::getenv("MOM_JIT_CACHE_DIR"); configured && *configured)
      return configured;
#if defined(_WIN32)
    if (const char* base = std::getenv("LOCALAPPDATA"); base && *base)
      return std::filesystem::path(base) / "mom" / "jit";
#else
    if (const char* base = std::getenv("XDG_CACHE_HOME"); base && *base)
      return std::filesystem::path(base) / "mom" / "jit";
    if (const char* home = std::getenv("HOME"); home && *home)
      return std::filesystem::path(home) / ".cache" / "mom" / "jit";
#endif
    return {};
  }
  static std::vector<char> read_cache(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) return {};
    const std::streamsize size = stream.tellg();
    if (size <= 0 || size > 64 * 1024 * 1024) return {};
    std::vector<char> code(static_cast<size_t>(size));
    stream.seekg(0);
    if (!stream.read(code.data(), size)) return {};
    return code;
  }
  static void write_cache(const std::filesystem::path& path,
                          const std::vector<char>& code) noexcept {
    try {
      if (path.empty() || code.empty()) return;
      std::filesystem::create_directories(path.parent_path());
      const std::filesystem::path temporary = path.string() + ".tmp";
      {
        std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
        if (!stream.write(code.data(), static_cast<std::streamsize>(code.size()))) return;
      }
      std::error_code error;
      std::filesystem::rename(temporary, path, error);
      if (error) {
        std::filesystem::remove(path, error);
        error.clear();
        std::filesystem::rename(temporary, path, error);
        if (error) std::filesystem::remove(temporary, error);
      }
    } catch (...) {
      // A read-only cache is not a mining failure; the in-process module remains usable.
    }
  }

  void release() noexcept {
    CnGpuCudaDriverApi& api = CnGpuCudaDriverApi::instance();
    if (api.available() && context) (void)api.set_current(context);
    if (done && api.event_destroy) (void)api.event_destroy(done);
    if (module && api.module_unload) (void)api.module_unload(module);
    if (retained_primary && api.release_primary) (void)api.release_primary(device);
    done = nullptr;
    module = nullptr;
    kernel = nullptr;
    context = nullptr;
    retained_primary = false;
    enabled = false;
  }

  bool ensure(sycl::queue& queue, std::string* const reason = nullptr) {
    const auto unavailable = [&](const std::string& message) {
      if (reason) *reason = message;
      return false;
    };
    if (!mom_is_cuda(queue.get_device()))
      return unavailable("the selected SYCL device is not CUDA");
    if (checked) return enabled;
    checked = true;
    try {
      CnGpuCudaDriverApi& driver = CnGpuCudaDriverApi::instance();
      CnGpuNvrtcApi& rtc = CnGpuNvrtcApi::instance();
      if (!driver.available()) throw driver.error;
      if (!rtc.available()) throw rtc.error;
      cuda_check(driver, driver.init(0), "cuInit(cn/gpu)");
      device = native_device(queue.get_device());
      int major = 0;
      cuda_check(driver, driver.device_attribute(
        &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device),
        "cuDeviceGetAttribute(cn/gpu major)");
      cuda_check(driver, driver.retain_primary(&context, device),
                 "cuDevicePrimaryCtxRetain(cn/gpu)");
      retained_primary = true;
      cuda_check(driver, driver.set_current(context), "cuCtxSetCurrent(cn/gpu)");

      const std::string source(
#include "cuda/cn_gpu_expand_kernel.inc"
      );
      // Keep the cached program forward-compatible with GPUs newer than the installed NVRTC.
      // CUDA 12.x cannot name Blackwell's compute_120 target, but its compute_80 PTX is accepted
      // and optimized for the actual device by current drivers. Older GPUs retain the SYCL path.
      if (major < 8) throw std::string("the native cn/gpu expansion requires compute capability 8.0+");
      const std::string architecture = "--gpu-architecture=compute_80";
      const std::vector<std::string> option_storage{
        "--std=c++17", architecture, "--device-as-default-execution-space"
      };
      std::vector<const char*> options;
      for (const auto& option : option_storage) options.push_back(option.c_str());

      std::ostringstream key;
      key << source << '\0' << architecture << "\0cn-gpu-expand-cuda-v1";
      std::ostringstream filename;
      filename << "cn-gpu-expand-compute80-" << std::hex
               << std::setw(16) << std::setfill('0') << hash_text(key.str()) << ".ptx";
      const std::filesystem::path root = cache_dir();
      const std::filesystem::path path =
        root.empty() ? std::filesystem::path{} : root / filename.str();
      std::vector<char> code = path.empty() ? std::vector<char>{} : read_cache(path);
      if (!code.empty() && driver.module_load(&module, code.data()) != CUDA_SUCCESS) {
        std::error_code error;
        std::filesystem::remove(path, error);
        code.clear();
      }
      if (code.empty()) {
        nvrtcProgram program = nullptr;
        nvrtcResult status = rtc.create_program(
          &program, source.c_str(), "cn-gpu-expand.cu", 0, nullptr, nullptr);
        if (status != NVRTC_SUCCESS)
          throw rtc_error(rtc, status, program, "nvrtcCreateProgram(cn/gpu)");
        status = rtc.compile_program(program, static_cast<int>(options.size()), options.data());
        if (status != NVRTC_SUCCESS) {
          const std::string message =
            rtc_error(rtc, status, program, "nvrtcCompileProgram(cn/gpu)");
          (void)rtc.destroy_program(&program);
          throw message;
        }
        size_t size = 0;
        status = rtc.get_ptx_size(program, &size);
        if (status != NVRTC_SUCCESS) {
          const std::string message =
            rtc_error(rtc, status, program, "nvrtcGetPTXSize(cn/gpu)");
          (void)rtc.destroy_program(&program);
          throw message;
        }
        code.resize(size);
        status = rtc.get_ptx(program, code.data());
        if (status != NVRTC_SUCCESS) {
          const std::string message =
            rtc_error(rtc, status, program, "nvrtcGetPTX(cn/gpu)");
          (void)rtc.destroy_program(&program);
          throw message;
        }
        (void)rtc.destroy_program(&program);
        cuda_check(driver, driver.module_load(&module, code.data()),
                   "cuModuleLoadData(cn/gpu)");
        write_cache(path, code);
      }
      cuda_check(driver, driver.module_function(&kernel, module, "cn_gpu_expand_cuda"),
                 "cuModuleGetFunction(cn/gpu)");
      cuda_check(driver, driver.event_create(&done, CU_EVENT_DEFAULT),
                 "cuEventCreate(cn/gpu)");
      enabled = true;
    } catch (const std::string& error) {
      release();
      if (reason) *reason = error;
    } catch (const std::exception& error) {
      release();
      if (reason) *reason = error.what();
    }
    return enabled;
  }

  void launch(uint64_t* states, uint64_t* scratchpads, const unsigned batch) {
    CnGpuCudaDriverApi& api = CnGpuCudaDriverApi::instance();
    cuda_check(api, api.set_current(context), "cuCtxSetCurrent(cn/gpu launch)");
    void* state_arg = states;
    void* scratchpad_arg = scratchpads;
    unsigned batch_arg = batch;
    void* args[] = {&state_arg, &scratchpad_arg, &batch_arg};
    constexpr unsigned threads = 32;
    const uint64_t total = static_cast<uint64_t>(batch) * 4096U;
    cuda_check(api, api.launch(
      kernel, static_cast<unsigned>((total + threads - 1) / threads), 1, 1,
      threads, 1, 1, 0, nullptr, args, nullptr), "cuLaunchKernel(cn/gpu)");
    cuda_check(api, api.event_record(done, nullptr), "cuEventRecord(cn/gpu)");
  }

  void wait() {
    CnGpuCudaDriverApi& api = CnGpuCudaDriverApi::instance();
    for (;;) {
      const CUresult status = api.event_query(done);
      if (status == CUDA_SUCCESS) return;
      if (status != CUDA_ERROR_NOT_READY)
        cuda_check(api, status, "cuEventQuery(cn/gpu)");
#if defined(_WIN32)
      mom_sycl_poll_pause();
#else
      std::this_thread::sleep_for(std::chrono::microseconds(100));
#endif
    }
  }
  void fail(const std::string& error) noexcept {
    release();
    checked = true;
    std::fprintf(stderr, "cn/gpu CUDA source-JIT failed, using SYCL: %s\n", error.c_str());
  }
  ~CnGpuCudaExpand() { release(); }
};
#else
struct CnGpuCudaExpand {
  bool ensure(sycl::queue&, std::string* const reason = nullptr) {
    if (reason) *reason = "CUDA source-JIT support is not part of this worker";
    return false;
  }
  void launch(uint64_t*, uint64_t*, unsigned) {}
  void wait() {}
  void fail(const std::string&) noexcept {}
};
#endif

struct CnGpuState { // Per-device queue and persistent allocations; avoids buffer churn in the hot path.
  sycl::device device;
  sycl::queue queue;
  std::unique_ptr<MomKernelBundle> bundle;
  bool shared_scratch; // CPU or backends without device allocation need shared scratch/IO.
  bool shared_io;
  bool native_notice = false;
  uint8_t* inputs = nullptr, *outputs = nullptr;
  uint64_t* spads = nullptr, *lpads = nullptr;
  MomBufferAllocation<uint8_t> buffered_inputs, buffered_outputs;
  MomBufferAllocation<uint64_t> buffered_spads, buffered_lpads;
  CnGpuCudaExpand cuda_expand;
  unsigned input_cap = 0, batch_cap = 0;
  double wait_ema_us = 0.0; // EMA of recent batch wall times (us); paces the pre-read sleep below
  std::mutex mutex;
  CnGpuState(const std::string& dev_str)
    : device(get_dev(dev_str)),
      queue(device, cn_gpu_exception_handler(), sycl::property_list{sycl::property::queue::in_order{}}),
      shared_scratch(device.is_cpu() || !mom_has_usm_device(device)),
      shared_io(device.is_cpu() || sycl_is_level_zero_gpu(device) ||
                !mom_has_usm_device(device))
  {
    if constexpr (!mom_sycl_portable_opencl) {
      if (!mom_has_usm_shared(device) || (!device.is_cpu() && !mom_has_usm_device(device))) {
        throw std::string("cn/gpu SYCL device does not support required allocations");
      }
    }

    if (!mom_is_cuda(device)) {
      // The OpenCL "-cl-*" build options are an OpenCL/Level-Zero JIT mechanism for the Intel build.
      // On the CUDA backend, setting SYCL_PROGRAM_COMPILE_OPTIONS to them makes the AOT nvptx module
      // fail at runtime with UR_RESULT_ERROR_UNKNOWN, so skip it there; CUDA FP determinism comes
      // instead from -ffp-contract=off in the build flags (binding.gyp).
      set_sycl_env("SYCL_PROGRAM_COMPILE_OPTIONS", cn_gpu_fp_compile_options(device));
    }
    bundle = std::make_unique<MomKernelBundle>(
      mom_get_exec_bundle(queue.get_context())
    );
  }

  ~CnGpuState() { sycl_cleanup_noexcept("cn/gpu", [&] { release(); }); }

  void release() {
    if constexpr (mom_sycl_portable_opencl) {
      buffered_inputs.release(); buffered_spads.release();
      buffered_lpads.release(); buffered_outputs.release();
    } else {
      auto free_ptr = [&](auto*& ptr) { if (ptr) sycl::free(ptr, queue); ptr = nullptr; };
      free_ptr(inputs); free_ptr(spads); free_ptr(lpads); free_ptr(outputs);
    }
    input_cap = batch_cap = 0;
  }

  template <typename T> T* allocate(const size_t count, const bool shared) {
    return shared ? sycl::malloc_shared<T>(count, queue) : sycl::malloc_device<T>(count, queue);
  }

  void ensure_capacity(const unsigned batch, const unsigned input_size) {
    const unsigned input_bytes = input_size * batch;
    if (batch <= batch_cap && input_bytes <= input_cap) return;

    // Resize lazily; steady-state jobs reuse allocations.
    queue.wait_and_throw();
    release();

    input_cap = input_bytes; batch_cap = batch;
    if constexpr (mom_sycl_portable_opencl) {
      buffered_inputs.allocate(input_cap);
      buffered_spads.allocate(25 * batch_cap);
      buffered_lpads.allocate(CN_MEMORY8 * batch_cap);
      buffered_outputs.allocate(HASH_LEN * batch_cap);
    } else {
      inputs = allocate<uint8_t>(input_cap, shared_io);
      spads = allocate<uint64_t>(25 * batch_cap, shared_scratch);
      lpads = allocate<uint64_t>(CN_MEMORY8 * batch_cap, shared_scratch);
      outputs = allocate<uint8_t>(HASH_LEN * batch_cap, shared_io);
      if (!inputs || !spads || !lpads || !outputs) {
        release();
        throw std::string("Can't allocate cn/gpu SYCL buffers");
      }
    }
  }
};

using CnGpuStateMap = std::map<std::string, std::unique_ptr<CnGpuState>>;

static CnGpuStateMap& cn_gpu_states() {
  static CnGpuStateMap* const states = new CnGpuStateMap;
  return *states;
}

static std::mutex& cn_gpu_states_mutex() {
  static std::mutex* const mutex = new std::mutex;
  return *mutex;
}

static CnGpuState& cn_gpu_state(const std::string& dev_str) {
  std::lock_guard<std::mutex> lock(cn_gpu_states_mutex());
  auto& state = cn_gpu_states()[dev_str];
  if (!state) state = std::make_unique<CnGpuState>(dev_str);
  return *state;
}

void cn_gpu_cleanup_states() noexcept {
  try {
    std::lock_guard<std::mutex> lock(cn_gpu_states_mutex());
    cn_gpu_states().clear();
  } catch (...) {
    std::fprintf(stderr, "cn/gpu: ordered SYCL cleanup failed\n");
  }
}

struct CnGpuDevicePointers {
  uint8_t* inputs;
  uint64_t* spads;
  uint64_t* lpads;
  uint8_t* outputs;
  uint32_t* spads4;
  int32_t* lpads4;
  sycl::uint4* lpads16;
};

template <bool Buffered>
class CnGpuViews;

template <>
class CnGpuViews<false> {
  CnGpuDevicePointers pointers_;

public:
  CnGpuViews(CnGpuState& state, sycl::handler& handler)
    : pointers_{state.inputs, state.spads, state.lpads, state.outputs,
                reinterpret_cast<uint32_t*>(state.spads),
                reinterpret_cast<int32_t*>(state.lpads),
                reinterpret_cast<sycl::uint4*>(state.lpads)} {
    (void)handler;
  }

  CnGpuDevicePointers pointers() const { return pointers_; }
};

template <>
class CnGpuViews<true> {
  using ByteView = sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                                  sycl::target::device>;
  using WordView = sycl::accessor<uint64_t, 1, sycl::access_mode::read_write,
                                  sycl::target::device>;
  ByteView inputs_, outputs_;
  WordView spads_, lpads_;

public:
  CnGpuViews(CnGpuState& state, sycl::handler& handler)
    : inputs_(state.buffered_inputs.device_view<sycl::access_mode::read_write>(handler)),
      outputs_(state.buffered_outputs.device_view<sycl::access_mode::read_write>(handler)),
      spads_(state.buffered_spads.device_view<sycl::access_mode::read_write>(handler)),
      lpads_(state.buffered_lpads.device_view<sycl::access_mode::read_write>(handler)) {}

  CnGpuDevicePointers pointers() const {
    auto* const inputs = inputs_.template get_multi_ptr<sycl::access::decorated::no>().get();
    auto* const spads = spads_.template get_multi_ptr<sycl::access::decorated::no>().get();
    auto* const lpads = lpads_.template get_multi_ptr<sycl::access::decorated::no>().get();
    auto* const outputs = outputs_.template get_multi_ptr<sycl::access::decorated::no>().get();
    return {inputs, spads, lpads, outputs, reinterpret_cast<uint32_t*>(spads),
            reinterpret_cast<int32_t*>(lpads), reinterpret_cast<sycl::uint4*>(lpads)};
  }
};

using CnGpuActiveViews = CnGpuViews<mom_sycl_portable_opencl>;

// cn_gpu owns the full submit chain so the hot path is visible in one function.
void cn_gpu(
  const uint8_t* const inputs, const unsigned input_size, uint8_t* const output,
  void* const, const unsigned batch, const std::string& dev_str, const std::string& backend
) {
  CnGpuState& state = cn_gpu_state(dev_str);
  std::lock_guard<std::mutex> state_lock(state.mutex);
  // The FP recurrence packs two 16-thread hashes into one 32-wide group, so round
  // the in-flight batch up to even; the extra hash is computed but never read.
  const unsigned batch_eff = (batch + 1u) & ~1u;
  state.ensure_capacity(batch_eff, input_size);
  sycl::queue& q = state.queue;
  auto& kb = *state.bundle;
  uint8_t* const d_inputs = state.inputs;
  uint8_t* const d_outputs = state.outputs;
  const unsigned input_bytes = input_size * batch;
  const unsigned output_bytes = HASH_LEN * batch;

  const auto batch_start = std::chrono::steady_clock::now();  // for the pre-read sleep EMA below

  if constexpr (mom_sycl_portable_opencl) {
    state.buffered_inputs.write(q, inputs, input_bytes);
  } else {
    if (state.shared_io) {
      std::memcpy(d_inputs, inputs, input_bytes);
    } else {
      q.memcpy(d_inputs, inputs, input_bytes);
    }
  }

  // Initial Keccak pads each input into its 200-byte state.
  q.submit([&](sycl::handler& h) {
    CnGpuActiveViews cn_views(state, h);
    mom_use_bundle(h, kb);
    h.parallel_for(sycl::nd_range(sycl::range((batch + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE * WORKGROUP_SIZE),
                                 sycl::range(WORKGROUP_SIZE)),
                   [=](sycl::nd_item<1> nd) {
      const auto [d_inputs, d_spads, d_lpads, d_outputs,
                  d_spads4, d_lpads4, d_lpads16] = cn_views.pointers();
      const auto t = nd.get_global_id(0);
      if (t >= batch) return;

      uint8_t spad[200];
      std::memcpy(spad, &d_inputs[input_size * t], input_size);
      spad[input_size] = 1;
      std::memset(spad + input_size + 1, 0, 200 - input_size - 1);
      spad[HASH_DATA_AREA - 1] |= 0x80;
      keccak(reinterpret_cast<uint64_t*>(spad));
      std::memcpy(&d_spads[25 * t], spad, 200);
    });
  });

  // Expand Keccak state into each 2 MiB scratchpad (one thread per 512-byte stripe). Windows
  // NVIDIA uses the source-JIT override only when the policy/user selected native; every other
  // device and an unavailable/failed NVRTC path executes this complete SYCL implementation.
  bool native_expansion = false;
  if constexpr (!mom_sycl_portable_opencl) {
    const bool native_requested = backend == "auto" || backend == "native";
    if (native_requested && mom_is_cuda(state.device)) {
      std::string reason;
      if (state.cuda_expand.ensure(q, &reason)) {
        try {
          q.wait_and_throw();
          state.cuda_expand.launch(state.spads, state.lpads, batch);
          state.cuda_expand.wait();
          native_expansion = true;
        } catch (const std::string& error) {
          state.cuda_expand.fail(error);
        } catch (const std::exception& error) {
          state.cuda_expand.fail(error.what());
        }
      } else if (!state.native_notice) {
        state.native_notice = true;
        std::fprintf(stderr, "cn/gpu native expansion unavailable, using SYCL: %s\n",
                     reason.c_str());
      }
    }
  }
  if (!native_expansion) {
    q.submit([&](sycl::handler& h) {
      CnGpuActiveViews cn_views(state, h);
      const unsigned stripes = CN_MEMORY8 / 64; // 512-byte stripes per scratchpad
      mom_use_bundle(h, kb);
      const unsigned total_threads = batch * stripes;
      unsigned expand_wg = WORKGROUP_SIZE;
#if defined(_WIN32)
      // NVIDIA's Windows CUDA driver runs this register-heavy Keccak expansion 7.9% faster with
      // eight-thread blocks; Linux, Intel, and AMD retain the measured 16-thread default.
      if (mom_is_cuda(state.device)) expand_wg = 8;
#endif
      const unsigned wg_size = std::min(expand_wg, stripes);
      h.parallel_for(sycl::nd_range(
                       sycl::range((total_threads + wg_size - 1) / wg_size * wg_size),
                       sycl::range(wg_size)),
                     [=](sycl::nd_item<1> nd) {
        const auto [d_inputs, d_spads, d_lpads, d_outputs,
                    d_spads4, d_lpads4, d_lpads16] = cn_views.pointers();
        const auto t = nd.get_global_id(0);
        if (t >= total_threads) return;

        const unsigned stripe = t % stripes;
        const unsigned hash_idx = t / stripes;
        const uint64_t* const spad = &d_spads[hash_idx * 25];
        uint64_t* const lpad = &d_lpads[hash_idx * CN_MEMORY8];
        generate512(stripe, spad, &lpad[stripe * 64]);
      });
    });
  }

  // Keep the long floating-point recurrence in the common SYCL pipeline on every backend.
  // The Intel OpenCL CPU runtime accepts the requested 16-wide subgroup but does not provide the
  // local-memory ordering this recurrence needs from subgroup-only barriers. Keep the measured
  // narrow barrier path on GPUs; the archive/generic-OpenCL CPU validation path uses the original
  // workgroup barrier. This is a uniform host-captured branch and does not alter GPU arithmetic.
  const bool use_narrow_recurrence_barriers =
    !mom_sycl_adaptivecpp && !mom_sycl_portable_opencl && state.device.is_gpu();
  q.submit([&](sycl::handler& h) {
    CnGpuActiveViews cn_views(state, h);
    // The recurrence uses 16 threads/hash. Pack TWO independent hashes per workgroup so a native
    // NVIDIA/AMD 32-lane warp/wave is full; Intel requests one 16-lane subgroup per hash. Every
    // local-memory dependency stays inside its 16-lane slice. DPC++ therefore uses subgroup
    // barriers instead of paying four workgroup barriers in each of 49,152 rounds (+2.9% NVIDIA,
    // +5.1% AMD and +3.2% Intel in counterbalanced tests). AdaptiveCpp deliberately retains the
    // workgroup barriers: its generic/SSCP lowering was neutral on Windows AMD and 0.6-0.9% slower
    // on NVIDIA, so keeping the proven fallback avoids a compiler-specific regression.
    const auto vi0 = sycl::local_accessor<sycl::int4, 1>(sycl::range(2 * WORKGROUP_SIZE), h);
    const auto vi1 = sycl::local_accessor<sycl::int4, 1>(sycl::range(2 * WORKGROUP_SIZE), h);
    const auto vf0 = sycl::local_accessor<sycl::float4, 1>(sycl::range(2 * WORKGROUP_SIZE), h);
    mom_use_bundle(h, kb);
    h.parallel_for(sycl::nd_range(sycl::range(batch_eff * WORKGROUP_SIZE),
                                  sycl::range(2 * WORKGROUP_SIZE)),
                   [=](sycl::nd_item<1> nd) MOM_REQD_SG_16 {
      const auto [d_inputs, d_spads, d_lpads, d_outputs,
                  d_spads4, d_lpads4, d_lpads16] = cn_views.pointers();
      const unsigned lid = nd.get_local_id();
      const unsigned sub = lid >> 4;        // which of the two warp-packed hashes
      const unsigned l = lid & 15U;         // 0..15 lane role within the hash
      const unsigned base16 = sub * 16U;    // int4 slice base for this hash
      const unsigned base32 = sub * 64U;    // int32/float slice base for this hash
      const unsigned ld = l / 4;
      const unsigned lm = l % 4;
      const unsigned b = ld * 16 + lm;
      const unsigned group_hash = nd.get_group().get_group_id() * 2U + sub;
      const uint32_t* const spad = &d_spads4[group_hash * (25 * 2)];
      int32_t* const lpad = &d_lpads4[group_hash * CN_MEMORY4];
      uint32_t s = *spad >> 8;
      sycl::float4 sf(0.0f);
      sycl::int4* const vi = &vi0[base16];
      int32_t* const vi4 = reinterpret_cast<int32_t*>(&vi0[0]) + base32;
      sycl::int4* const vo = &vi1[base16];
      int32_t* const vo4 = reinterpret_cast<int32_t*>(&vi1[0]) + base32;
      sycl::float4* const vf = &vf0[base16];
      float* const vf4 = reinterpret_cast<float*>(&vf0[0]) + base32;
      const auto sg = nd.get_sub_group();

      for (unsigned i = 0; i < CN_GPU_ITER; ++ i) {
        const int32_t xi = lpad_ptr(s, ld, lpad)[lm];
        vi4[l] = xi;

        cn_recurrence_barrier(nd, sg, use_narrow_recurrence_barriers);

        single_comupte_wrap(
          vi[cn_lane_select(l, 0)], vi[cn_lane_select(l, 1)],
          vi[cn_lane_select(l, 2)], vi[cn_lane_select(l, 3)],
          lm, cn_lane_count(l), sf, vf + l, vo + l
        );

        cn_recurrence_barrier(nd, sg, use_narrow_recurrence_barriers);

        { int32_t xo = vo4[b];
          for (unsigned dd = b + 4; dd < (ld + 1) * 16; dd += 4) xo ^= vo4[dd];
          lpad_ptr(s, ld, lpad)[lm] = xo ^ xi;
          vo4[l] = xo;
        }

        vf4[l] = (vf4[b] + vf4[b + 4]) + (vf4[b + 8] + vf4[b + 12]);

        cn_recurrence_barrier(nd, sg, use_narrow_recurrence_barriers);

        const float xf = sycl::fabs((vf4[b] + vf4[b + 4]) + (vf4[b + 8] + vf4[b + 12]));
        vo4[l] ^= vo4[l + 4] ^ vo4[l + 8] ^ vo4[l + 12] ^
                  static_cast<int32_t>(xf * 16777216.0f);
        vf4[l] = xf * 0.015625f;

        cn_recurrence_barrier(nd, sg, use_narrow_recurrence_barriers);

        sf = vf[0];
        s = vo[0][0] ^ vo[0][1] ^ vo[0][2] ^ vo[0][3];
      }
    });
    });

  const unsigned aes_wg = WORKGROUP_SIZE / 4; // hashes per workgroup row; dim1 (8) splits each hash
  const bool use_gpu_aes = !mom_sycl_portable_opencl && state.device.is_gpu();
  if (use_gpu_aes) { // GPU path replaces the AES local-memory ring with subgroup shuffles.
    q.submit([&](sycl::handler& h) {
      CnGpuActiveViews cn_views(state, h);
      const AesTable aes0(sycl::range(256), h), aes1(sycl::range(256), h),
                     aes2(sycl::range(256), h), aes3(sycl::range(256), h);
      mom_use_bundle(h, kb);
      h.parallel_for(sycl::nd_range(sycl::range(batch, 8), sycl::range(aes_wg, 8)),
                     [=](sycl::nd_item<2> nd) MOM_REQD_SG_16 {
        const auto [d_inputs, d_spads, d_lpads, d_outputs,
                    d_spads4, d_lpads4, d_lpads16] = cn_views.pointers();
        const unsigned l1 = nd.get_local_id(1);
        aes_fill_tables(aes0, aes1, aes2, aes3, nd);

        nd.barrier(sycl::access::fence_space::local_space);

        const auto spad = d_spads4 + (nd.get_global_id(0) * (25 * 2));
        const sycl::uint4* const lpad = &d_lpads16[nd.get_global_id(0) * CN_MEMORY16];
        sycl::uint4 x;
        mo_vec_load(x, l1 + 4, spad);
        alignas(32) AesKey key;
        mo_vec_load(key.u8[0], 1, spad);
        aes_expend_key(key.u);
        const auto sg = nd.get_sub_group();
        const unsigned row_lane_base = sg.get_local_linear_id() & ~7u;

        // Subgroup shuffle replaces the old local-memory neighbor exchange on GPU.
        auto next_row_value = [&](const sycl::uint4 value) {
          const sycl::uint4 shifted = mom_shift_group_left(sg, value, 1);
          const sycl::uint4 wrapped = mom_select_from_group(sg, value, row_lane_base);
          return l1 == 7 ? wrapped : shifted;
        };

        sycl::uint4 x2s(0);

        for (unsigned i = 0, i1 = l1; i < CN_MEMORY16/8; ++i, i1 = (i1 + 16) % CN_MEMORY16) {
          x ^= lpad[i1];
          const sycl::uint4 x2l = next_row_value(x2s);
          x ^= x2l;
          for (unsigned j = 0; j < 10; ++ j) aes_round(&x, key.u4[j], aes0, aes1, aes2, aes3);
          const sycl::uint4 x1s = x;
          x ^= lpad[i1 + 8];
          const sycl::uint4 x1l = next_row_value(x1s);
          x ^= x1l;
          for (unsigned j = 0; j < 10; ++ j) aes_round(&x, key.u4[j], aes0, aes1, aes2, aes3);
          x2s = x;
        }

        const sycl::uint4 x2l = next_row_value(x2s);
        x ^= x2l;

        for (unsigned i = 0; i < 16; ++ i) {
          for (unsigned j = 0; j < 10; ++ j) aes_round(&x, key.u4[j], aes0, aes1, aes2, aes3);
          const sycl::uint4 x1s = x;
          const sycl::uint4 x1l = next_row_value(x1s);
          x ^= x1l;
        }
        mo_vec_store(x, l1 + 4, spad);
      });
    });
  } else { // CPU path keeps the barriered local-memory exchange for stable behavior.
    q.submit([&](sycl::handler& h) {
      CnGpuActiveViews cn_views(state, h);
      const AesTable aes0(sycl::range(256), h), aes1(sycl::range(256), h),
                     aes2(sycl::range(256), h), aes3(sycl::range(256), h);
      const auto x1 = sycl::local_accessor<sycl::uint4, 2>(sycl::range(aes_wg, 8), h);
      const auto x2 = sycl::local_accessor<sycl::uint4, 2>(sycl::range(aes_wg, 8), h);
      mom_use_bundle(h, kb);
      h.parallel_for(sycl::nd_range(sycl::range(batch, 8), sycl::range(aes_wg, 8)),
                     [=](sycl::nd_item<2> nd) {
        const auto [d_inputs, d_spads, d_lpads, d_outputs,
                    d_spads4, d_lpads4, d_lpads16] = cn_views.pointers();
        const unsigned l0 = nd.get_local_id(0), l1 = nd.get_local_id(1);
        aes_fill_tables(aes0, aes1, aes2, aes3, nd);

        nd.barrier(sycl::access::fence_space::local_space);

        const auto spad = d_spads4 + (nd.get_global_id(0) * (25 * 2));
        const sycl::uint4* const lpad = &d_lpads16[nd.get_global_id(0) * CN_MEMORY16];
        sycl::uint4 x;
        mo_vec_load(x, l1 + 4, spad);
        alignas(32) AesKey key;
        mo_vec_load(key.u8[0], 1, spad);
        aes_expend_key(key.u);
        sycl::uint4 &x1s = x1[l0][l1], &x1l = x1[l0][(l1 + 1) % 8],
                    &x2s = x2[l0][l1], &x2l = x2[l0][(l1 + 1) % 8];
        x2s = sycl::uint4(0);

        nd.barrier(sycl::access::fence_space::local_space);

        for (unsigned i = 0, i1 = l1; i < CN_MEMORY16/8; ++i, i1 = (i1 + 16) % CN_MEMORY16) {
          x ^= lpad[i1];
          x ^= x2l;
          for (unsigned j = 0; j < 10; ++ j) aes_round(&x, key.u4[j], aes0, aes1, aes2, aes3);
          x1s = x;
          nd.barrier(sycl::access::fence_space::local_space);
          x ^= lpad[i1 + 8];
          x ^= x1l;
          for (unsigned j = 0; j < 10; ++ j) aes_round(&x, key.u4[j], aes0, aes1, aes2, aes3);
          x2s = x;
          nd.barrier(sycl::access::fence_space::local_space);
        }

        x ^= x2l;

        for (unsigned i = 0; i < 16; ++ i) {
          for (unsigned j = 0; j < 10; ++ j) aes_round(&x, key.u4[j], aes0, aes1, aes2, aes3);
          nd.barrier(sycl::access::fence_space::local_space);
          x1s = x;
          nd.barrier(sycl::access::fence_space::local_space);
          x ^= x1l;
        }
        mo_vec_store(x, l1 + 4, spad);
      });
    });
  }

  sycl::event final_event = q.submit([&](sycl::handler& h) { // Final Keccak stays on device; the host scratchpad argument is unused.
    CnGpuActiveViews cn_views(state, h);
    mom_use_bundle(h, kb);
    h.parallel_for(sycl::nd_range(sycl::range((batch + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE * WORKGROUP_SIZE),
                                 sycl::range(WORKGROUP_SIZE)),
                   [=](sycl::nd_item<1> nd) {
      const auto [d_inputs, d_spads, d_lpads, d_outputs,
                  d_spads4, d_lpads4, d_lpads16] = cn_views.pointers();
      const auto t = nd.get_global_id(0);
      if (t >= batch) return;

      uint64_t* const spad = &d_spads[25 * t];
      keccak(spad);

      uint64_t* const out = reinterpret_cast<uint64_t*>(d_outputs + HASH_LEN * t);
      out[0] = spad[0]; out[1] = spad[1]; out[2] = spad[2]; out[3] = spad[3];
    });
  });

  // shared_io is false only for the device-memory GPUs that take the blocking D2H output read
  // below: Intel OpenCL (cn/gpu's policy backend) and the CUDA backend. On both, that read pins a
  // host core for the whole batch -- their queue waits return before the kernels finish, and the
  // CUDA D2H copy to pageable host memory is synchronous, so the real GPU sync happens inside the
  // read. (CPU and Level-Zero take the shared_io path instead: a poll-freed wait then a host-side
  // copy of already-ready data.) To stop pinning the core, sleep through most of the batch before
  // the read -- sized from an EMA of recent batch wall times -- leaving only a short spinning tail.
  // cn/gpu batches are long (hundreds of ms) and very stable, so a 90% sleep adds no latency (it
  // stays under the batch) while cutting steady-state host usage to ~10% of a core.
  const bool sleep_before_read = !state.shared_io;
  if (sleep_before_read && state.wait_ema_us > 2000.0) {
    std::this_thread::sleep_for(std::chrono::microseconds(static_cast<long>(state.wait_ema_us * 0.90)));
  }

  if constexpr (mom_sycl_portable_opencl) {
    sycl_wait_and_throw(final_event, state.device);
    state.buffered_outputs.read(output, output_bytes);
  } else {
    if (state.shared_io) {
      sycl_wait_and_throw(final_event, state.device);
      std::memcpy(output, d_outputs, output_bytes);
    } else {
      sycl_wait_and_throw(q.memcpy(output, d_outputs, output_bytes), state.device);
    }
  }

  if (sleep_before_read) {
    // Track the full batch wall time (submit + sleep + read). With the sleep sized below the
    // batch this equals the true GPU time, so the EMA self-corrects: oversleep -> measured time
    // drops -> next sleep shrinks; undersleep -> read spins longer -> EMA rises toward the truth.
    const double batch_us = static_cast<double>(
      std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - batch_start).count());
    state.wait_ema_us = state.wait_ema_us == 0.0 ? batch_us : state.wait_ema_us * 0.8 + batch_us * 0.2;
  }
}
