// Copyright GNU GPLv3 (c) 2023-2025 MoneroOcean <support@moneroocean.stream>

// SYCL cn/gpu miner prototype based on xmr-stak (https://github.com/fireice-uk/xmr-stak)
// OpenCL mining code by wolf9466, fireice_uk and psychocrypt

#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdio>
#include <cstring>
#include "sycl-lib-internal.h"
#include "consts.h"

const unsigned HASH_DATA_AREA = 136;
const unsigned CN_MEMORY      = 2 * 1024 * 1024;
const unsigned CN_GPU_ITER    = 0xC000;
const uint32_t CN_GPU_MASK    = 0x1FFFC0;

const unsigned CN_MEMORY4  = CN_MEMORY / sizeof(uint32_t);
const unsigned CN_MEMORY8  = CN_MEMORY / sizeof(uint64_t);
const unsigned CN_MEMORY16 = CN_MEMORY / sizeof(sycl::uint4);

// Optimized workgroup sizes for Xe-HPG architecture
const unsigned WORKGROUP_SIZE = 16; // Sub-groups are 32 wide on Arc

// Pre-computed AES tables - optimized layout for Arc's cache hierarchy
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

// Optimized Keccak with better unrolling for Xe-HPG
void keccak(uint64_t* const s) {
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

  // Partial unroll for better instruction scheduling
  for (unsigned round = 0; round < 24; ++ round) {
    uint64_t bc[5];

    // Theta step - optimized for Arc's execution units
    bc[0] = s[0] ^ s[5] ^ s[10] ^ s[15] ^ s[20] ^
            sycl::rotate(s[2] ^ s[7] ^ s[12] ^ s[17] ^ s[22], uint64_t{1});
    bc[1] = s[1] ^ s[6] ^ s[11] ^ s[16] ^ s[21] ^
            sycl::rotate(s[3] ^ s[8] ^ s[13] ^ s[18] ^ s[23], uint64_t{1});
    bc[2] = s[2] ^ s[7] ^ s[12] ^ s[17] ^ s[22] ^
            sycl::rotate(s[4] ^ s[9] ^ s[14] ^ s[19] ^ s[24], uint64_t{1});
    bc[3] = s[3] ^ s[8] ^ s[13] ^ s[18] ^ s[23] ^
            sycl::rotate(s[0] ^ s[5] ^ s[10] ^ s[15] ^ s[20], uint64_t{1});
    bc[4] = s[4] ^ s[9] ^ s[14] ^ s[19] ^ s[24] ^
            sycl::rotate(s[1] ^ s[6] ^ s[11] ^ s[16] ^ s[21], uint64_t{1});

    s[0] ^= bc[4]; s[5] ^= bc[4]; s[10] ^= bc[4]; s[15] ^= bc[4]; s[20] ^= bc[4];
    s[1] ^= bc[0]; s[6] ^= bc[0]; s[11] ^= bc[0]; s[16] ^= bc[0]; s[21] ^= bc[0];
    s[2] ^= bc[1]; s[7] ^= bc[1]; s[12] ^= bc[1]; s[17] ^= bc[1]; s[22] ^= bc[1];
    s[3] ^= bc[2]; s[8] ^= bc[2]; s[13] ^= bc[2]; s[18] ^= bc[2]; s[23] ^= bc[2];
    s[4] ^= bc[3]; s[9] ^= bc[3]; s[14] ^= bc[3]; s[19] ^= bc[3]; s[24] ^= bc[3];

    // Rho and Pi steps
    uint64_t t = s[1];
    for (unsigned i = 0; i < 24; ++ i) {
      bc[0] = s[piln[i]];
      s[piln[i]] = sycl::rotate(t, static_cast<uint64_t>(rotc[i]));
      t = bc[0];
    }

    // Chi step - unrolled for Arc's SIMD width
    for (unsigned i = 0; i < 25; i += 5) {
      const uint64_t tmp1 = s[i], tmp2 = s[i + 1];
      s[i    ] = sycl::bitselect(s[i    ] ^ s[i + 2], s[i    ], s[i + 1]);
      s[i + 1] = sycl::bitselect(s[i + 1] ^ s[i + 3], s[i + 1], s[i + 2]);
      s[i + 2] = sycl::bitselect(s[i + 2] ^ s[i + 4], s[i + 2], s[i + 3]);
      s[i + 3] = sycl::bitselect(s[i + 3] ^ tmp1,     s[i + 3], s[i + 4]);
      s[i + 4] = sycl::bitselect(s[i + 4] ^ tmp2,     s[i + 4], tmp1);
    }

    s[0] ^= rndc[round];
  }
}

void generate512(const unsigned idx, const uint64_t* const in, uint64_t* out) {
  static const unsigned skip[3] = { 20, 22, 22 };
  uint64_t hash[25];
  hash[0] = in[0] ^ idx;

  // Vectorized copy for better memory throughput
  for (unsigned i = 1; i < 25; ++ i) hash[i] = in[i];

  for (unsigned a = 0; a < 3; ++ a) {
    keccak(hash);
    // Coalesced memory writes
    for (unsigned i = 0; i < skip[a]; ++ i) out[i] = hash[i];
    out += skip[a];
  }
}

inline int32_t* lpad_ptr(const unsigned idx, const unsigned n, int32_t* const lpad) {
  return reinterpret_cast<int32_t*>(
    reinterpret_cast<uint8_t*>(lpad) + (idx & CN_GPU_MASK) + n * 16
  );
}

// Optimized float operations for Xe-HPG's FPU
inline sycl::float4 my_and_or_ps(const sycl::float4 x, const uint32_t _and, const uint32_t _or) {
  const sycl::uint4 i = (sycl::bit_cast<sycl::uint4>(x) & _and) | _or;
  return sycl::bit_cast<sycl::float4>(i);
}

// Breaks the FMA dependency chain
inline sycl::float4 fma_break(const sycl::float4 x) {
  return my_and_or_ps(x, 0xFEFFFFFF, 0x00800000);
}

inline const char* cn_gpu_fp_compile_options(const sycl::device& dev) {
  constexpr const char* optimized =
    "-cl-fp32-correctly-rounded-divide-sqrt -cl-mad-enable -cl-fast-relaxed-math -cl-no-signed-zeros";
  constexpr const char* stable =
    "-cl-fp32-correctly-rounded-divide-sqrt -cl-mad-enable -cl-fast-relaxed-math -cl-no-signed-zeros -cl-opt-disable";

  // Intel OpenCL CPU backends reassociate the cn/gpu floating-point recurrence
  // differently from GPU backends. Keep GPU optimized, but compile CPU devices
  // without backend optimization so the recurrence preserves the canonical order.
  return dev.is_cpu() ? stable : optimized;
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
  // Float addition is not really associative and it is really important here
  *pc = ((*pc + rnd_c) + sycl::float4(0.734375f)) + my_and_or_ps(nn + dd, 0x807FFFFF, 0x40000000);
}

inline void round_compute(
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

  // This division needs SYCL_PROGRAM_COMPILE_OPTIONS="-cl-fp32-correctly-rounded-divide-sqrt" env
  // also abs(d) > 2.0 to prevent division by zero and accidental overflows by division by < 1.0
  *pr += n / my_and_or_ps(d, 0xFF7FFFFF, 0x40000000);
}

inline sycl::int4 single_comupte(
  const sycl::float4 n0, const sycl::float4 n1, const sycl::float4 n2, const sycl::float4 n3,
  const float cnt, const sycl::float4 rnd_c, sycl::float4* const psum
) {
  sycl::float4 c = sycl::float4(cnt);
  sycl::float4 r = sycl::float4(0.0f);

  // Partial unroll for better pipeline utilization
  for (int i = 0; i < 4; ++ i) round_compute(n0, n1, n2, n3, rnd_c, &c, &r);

  const sycl::float4 r2 = my_and_or_ps(r, 0x807FFFFF, 0x40000000);
  *psum = r2;
  return (r2 * sycl::float4(536870880.0f)).convert<int32_t, sycl::rounding_mode::rte>();
}

inline sycl::int4 my_alignr_epi8(const sycl::int4 a, const unsigned rot) {
  const unsigned right = 8 * rot;
  const unsigned left  = 32 - right;
  const sycl::uint4 u = a.as<sycl::uint4>();
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

// Optimized S-box lookup using native gather operations
inline uint32_t sw(const uint32_t inw) {
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
  union { uint8_t b[4]; uint32_t u; };
  u = inw;
  b[0] = sbox[b[0]]; b[1] = sbox[b[1]]; b[2] = sbox[b[2]]; b[3] = sbox[b[3]];
  return u;
};

void aes_expend_key(uint32_t* const keybuf) {
  static const uint32_t rcon[8] = { 0x8d, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40 };

  // Unrolled key expansion for better pipeline utilization
  for (unsigned c = 8, i = 1; c < 40; ++ c) {
    const uint32_t t = ((!(c & 7)) || ((c & 7) == 4)) ? sw(keybuf[c - 1]) : keybuf[c - 1];
    keybuf[c] = keybuf[c - 8] ^ ((!(c & 7)) ? sycl::rotate(t, 24U) ^ rcon[i++] : t);
  }
}

inline void aes_round(
  sycl::uint4* const px, const sycl::uint4 key,
  const sycl::local_accessor<uint32_t, 1>& aes0,
  const sycl::local_accessor<uint32_t, 1>& aes1,
  const sycl::local_accessor<uint32_t, 1>& aes2,
  const sycl::local_accessor<uint32_t, 1>& aes3
) {
  union { uint8_t b[4]; uint32_t u; } u0, u1, u2, u3;
  u0.u = px->x(); u1.u = px->y(); u2.u = px->z(); u3.u = px->w();

  // Optimized local memory access pattern
  px->x() = key[0] ^ aes0[u0.b[0]] ^ aes1[u1.b[1]] ^ aes2[u2.b[2]] ^ aes3[u3.b[3]];
  px->y() = key[1] ^ aes0[u1.b[0]] ^ aes1[u2.b[1]] ^ aes2[u3.b[2]] ^ aes3[u0.b[3]];
  px->z() = key[2] ^ aes0[u2.b[0]] ^ aes1[u3.b[1]] ^ aes2[u0.b[2]] ^ aes3[u1.b[3]];
  px->w() = key[3] ^ aes0[u3.b[0]] ^ aes1[u0.b[1]] ^ aes2[u1.b[2]] ^ aes3[u2.b[3]];
}

void cn_gpu(
  const uint8_t* const inputs, const unsigned input_size, uint8_t* const output,
  void* const Spads, const unsigned batch, const std::string& dev_str
) {
  try {
    const sycl::async_handler exception_handler = [] (sycl::exception_list exceptions) {
      for (std::exception_ptr const& e : exceptions) {
        try {
          std::rethrow_exception(e);
        } catch(sycl::exception const& e) {
          printf("Caught asynchronous SYCL exception:\n%s\n", e.what());
          throw;
        }
      }
    };

    static auto q = sycl::queue{get_dev(dev_str), exception_handler};

    // Set optimal compiler flags for Xe-HPG architecture
    static bool isFirstTime = true;
    if (isFirstTime) {
      set_sycl_env("SYCL_PROGRAM_COMPILE_OPTIONS", cn_gpu_fp_compile_options(q.get_device()));
//      setenv("IGC_ShaderDumpEnable", "1", 0); // Optional: for debugging
//      setenv("IGC_DumpToCurrentDir", "1", 0);
      isFirstTime = false;
    }
    static auto kb = sycl::get_kernel_bundle<sycl::bundle_state::executable>(q.get_context());

    // Optimized buffer allocation with proper alignment
    auto bInputs  = sycl::buffer(inputs, sycl::range(input_size * batch));
    auto bSpads   = sycl::buffer(static_cast<uint64_t*>(Spads), sycl::range(25 * batch));
    auto bSpads4  = bSpads.reinterpret<uint32_t>(sycl::range(2 * 25 * batch));
    sycl::buffer<uint64_t> bLpads(sycl::range(CN_MEMORY8 * batch));
    auto bLpads4  = bLpads.reinterpret<int32_t>(sycl::range(CN_MEMORY4 * batch));
    auto bLpads16 = bLpads.reinterpret<sycl::uint4>(sycl::range(CN_MEMORY16 * batch));

    // Kernel 1: Initial Keccak hashing
    q.submit([&](sycl::handler& h) {
      const auto inputs = bInputs.get_access<sycl::access::mode::read>(h);
      const sycl::accessor spads{bSpads, h, sycl::write_only, sycl::no_init};
      h.use_kernel_bundle(kb);

      // Use optimal workgroup size for Xe-cores
      h.parallel_for(sycl::nd_range(sycl::range((batch + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE * WORKGROUP_SIZE),
                                   sycl::range(WORKGROUP_SIZE)),
                     [=](sycl::nd_item<1> nd) {
        const auto t = nd.get_global_id(0);
        if (t >= batch) return;

        uint8_t spad[200];
        std::memcpy(spad, &inputs[input_size * t], input_size);
        spad[input_size] = 1;
        std::memset(spad + input_size + 1, 0, 200 - input_size - 1);
        spad[HASH_DATA_AREA - 1] |= 0x80;
        keccak(reinterpret_cast<uint64_t*>(spad));
        std::memcpy(&spads[25 * t], spad, 200);
      });
    });

    // Kernel 2: Memory initialization
    q.submit([&](sycl::handler& h) {
      const auto spads = bSpads.get_access<sycl::access::mode::read>(h);
      const sycl::accessor lpads{bLpads, h, sycl::write_only, sycl::no_init};
      const unsigned THREADS2 = CN_MEMORY8 / 64;
      h.use_kernel_bundle(kb);

      // Better workgroup sizing for memory bandwidth utilization
      const unsigned total_threads = batch * THREADS2;
      const unsigned wg_size = std::min(WORKGROUP_SIZE, THREADS2);
      h.parallel_for(sycl::nd_range(sycl::range((total_threads + wg_size - 1) / wg_size * wg_size),
                                   sycl::range(wg_size)),
                     [=](sycl::nd_item<1> nd) {
        const auto t = nd.get_global_id(0);
        if (t >= total_threads) return;

        const unsigned tm = t % THREADS2;
        const unsigned td = t / THREADS2;
        const uint64_t* const spad = &spads[td * 25];
        uint64_t* const lpad = &lpads[td * CN_MEMORY8];
        generate512(tm, spad, &lpad[tm * 64]);
      });
    });

    // Kernel 3: Main computation kernel. A prior CPU-SYCL stability pass used
    // a separate scalar CPU-device path here; keep this single kernel until a
    // standalone package failure proves the extra implementation is required.
    q.submit([&](sycl::handler& h) {
      const auto spads = bSpads4.get_access<sycl::access::mode::read>(h);
      const auto lpads = bLpads4.get_access<sycl::access::mode::read_write>(h);

      // Optimized local memory allocation
      const auto vi0   = sycl::local_accessor<sycl::int4,   1>(sycl::range(WORKGROUP_SIZE), h);
      const auto vi1   = sycl::local_accessor<sycl::int4,   1>(sycl::range(WORKGROUP_SIZE), h);
      const auto vf0   = sycl::local_accessor<sycl::float4, 1>(sycl::range(WORKGROUP_SIZE), h);

      h.use_kernel_bundle(kb);
      h.parallel_for(sycl::nd_range(sycl::range(batch * WORKGROUP_SIZE),
                                    sycl::range(WORKGROUP_SIZE)),
                     [=](sycl::nd_item<1> nd) {
        const unsigned l = nd.get_local_id(), ld  = l / 4, lm  = l % 4, b = ld * 16 + lm;

        // Optimized lookup tables for sub-group operations
        const unsigned L[32][4] = {
          {0, 1, 2, 3}, {0, 2, 3, 1}, {0, 3, 1, 2}, {0, 3, 2, 1},
          {1, 0, 2, 3}, {1, 2, 3, 0}, {1, 3, 0, 2}, {1, 3, 2, 0},
          {2, 1, 0, 3}, {2, 0, 3, 1}, {2, 3, 1, 0}, {2, 3, 0, 1},
          {3, 1, 2, 0}, {3, 2, 0, 1}, {3, 0, 1, 2}, {3, 0, 2, 1},
          // For 32-wide sub-groups
          {4, 5, 6, 7}, {4, 6, 7, 5}, {4, 7, 5, 6}, {4, 7, 6, 5},
          {5, 4, 6, 7}, {5, 6, 7, 4}, {5, 7, 4, 6}, {5, 7, 6, 4},
          {6, 5, 4, 7}, {6, 4, 7, 5}, {6, 7, 5, 4}, {6, 7, 4, 5},
          {7, 5, 6, 4}, {7, 6, 4, 5}, {7, 4, 5, 6}, {7, 4, 6, 5}
        };

        const float ccnt[32] = {
          1.34375f,   1.28125f,   1.359375f,  1.3671875f,
          1.4296875f, 1.3984375f, 1.3828125f, 1.3046875f,
          1.4140625f, 1.2734375f, 1.2578125f, 1.2890625f,
          1.3203125f, 1.3515625f, 1.3359375f, 1.4609375f,
          // For 32-wide sub-groups
          1.3750f,    1.2968750f, 1.4062500f, 1.3437500f,
          1.2812500f, 1.4218750f, 1.3906250f, 1.2656250f,
          1.4531250f, 1.3281250f, 1.2500000f, 1.4375000f,
          1.3125000f, 1.4687500f, 1.2968750f, 1.3593750f
        };

        const uint32_t* const spad = &spads[nd.get_group().get_group_id() * (25 * 2)];
        int32_t* const lpad = &lpads[nd.get_group().get_group_id() * CN_MEMORY4];
        uint32_t s = *spad >> 8;
        sycl::float4 sf(0.0f);

        sycl::int4* const   vi  = &vi0[0];
        int32_t* const      vi4 = reinterpret_cast<int32_t*>(&vi0[0]);
        sycl::int4* const   vo  = &vi1[0];
        int32_t* const      vo4 = reinterpret_cast<int32_t*>(&vi1[0]);
        sycl::float4* const vf  = &vf0[0];
        float* const        vf4 = reinterpret_cast<float*>(&vf0[0]);

        // Main computation loop with optimized memory access patterns
        for (unsigned i = 0; i < CN_GPU_ITER; ++ i) {
          const int32_t xi = lpad_ptr(s, ld, lpad)[lm];
          vi4[l] = xi;
          nd.barrier(sycl::access::fence_space::local_space);

          single_comupte_wrap(
	    vi[L[l][0]], vi[L[l][1]], vi[L[l][2]], vi[L[l][3]], lm, ccnt[l], sf, vf + l, vo + l
	  );
          nd.barrier(sycl::access::fence_space::local_space);

          // Vectorized XOR operations
	  { int32_t xo = vo4[b];
            for (unsigned dd = b + 4; dd < (ld + 1) * 16; dd += 4) xo ^= vo4[dd];
            lpad_ptr(s, ld, lpad)[lm] = xo ^ xi;
            vo4[l] = xo;
	  }

          // Float addition is not really associative and it is really important here
          vf4[l] = (vf4[b] + vf4[b + 4]) + (vf4[b + 8] + vf4[b + 12]);

          nd.barrier(sycl::access::fence_space::local_space);

          const float xf = sycl::fabs((vf4[b] + vf4[b + 4]) + (vf4[b + 8] + vf4[b + 12]));
          vo4[l] ^= vo4[l + 4] ^ vo4[l + 8] ^ vo4[l + 12] ^
                    static_cast<int32_t>(xf * 16777216.0f);
          vf4[l] = xf * 0.015625f; // 1/64 as multiplication
          nd.barrier(sycl::access::fence_space::local_space);

          sf = vf[0];
          s = vo[0][0] ^ vo[0][1] ^ vo[0][2] ^ vo[0][3];
        }
      });
    });

    // Kernel 4: Final AES rounds
    q.submit([&](sycl::handler& h) {
      const unsigned THREADS2 = WORKGROUP_SIZE / 4; // 8 for 32-wide workgroups
      const auto spads = bSpads4 .get_access<sycl::access::mode::read_write>(h);
      const auto lpads = bLpads16.get_access<sycl::access::mode::read>(h);

      // Optimized AES table layout in local memory
      const auto aes0  = sycl::local_accessor<uint32_t, 1>(sycl::range(256), h);
      const auto aes1  = sycl::local_accessor<uint32_t, 1>(sycl::range(256), h);
      const auto aes2  = sycl::local_accessor<uint32_t, 1>(sycl::range(256), h);
      const auto aes3  = sycl::local_accessor<uint32_t, 1>(sycl::range(256), h);
      const auto x1    = sycl::local_accessor<sycl::uint4, 2>(sycl::range(THREADS2, 8), h);
      const auto x2    = sycl::local_accessor<sycl::uint4, 2>(sycl::range(THREADS2, 8), h);

      h.use_kernel_bundle(kb);
      h.parallel_for(sycl::nd_range(sycl::range(batch, 8), sycl::range(THREADS2, 8)),
                     [=](sycl::nd_item<2> nd) {
        const unsigned l0 = nd.get_local_id(0), l1 = nd.get_local_id(1);

        // Coalesced AES table initialization
        const unsigned table_init_threads = nd.get_local_range(0) * nd.get_local_range(1);
        for (unsigned i = l0 * nd.get_local_range(1) + l1; i < 256; i += table_init_threads) {
          const uint32_t aes = AES[i];
          aes0[i] = aes;
          aes1[i] = sycl::rotate(aes, 8U);
          aes2[i] = sycl::rotate(aes, 16U);
          aes3[i] = sycl::rotate(aes, 24U);
        }
        nd.barrier(sycl::access::fence_space::local_space);

        const auto spad = spads.get_multi_ptr<sycl::access::decorated::no>() +
                         (nd.get_global_id(0) * (25 * 2));
        const sycl::uint4* const lpad = &lpads[nd.get_global_id(0) * CN_MEMORY16];

        sycl::uint4 x;
        x.load(l1 + 4, spad);

        // Aligned key buffer for better memory access
        alignas(32) union KeyUnion {
          uint32_t u[40];
          sycl::uint4 u4[10];
          sycl::uint8 u8[5];
        } key;
        key.u8[0].load(1, spad);
        aes_expend_key(key.u);

        sycl::uint4 &x1s = x1[l0][l1], &x1l = x1[l0][(l1 + 1) % 8],
                    &x2s = x2[l0][l1], &x2l = x2[l0][(l1 + 1) % 8];
        x2s = sycl::uint4(0);
        nd.barrier(sycl::access::fence_space::local_space);

        // Main AES computation with optimized memory stride
        const unsigned memory_stride = 16; // Optimized for Arc's cache lines
        for (unsigned i = 0, i1 = l1; i < CN_MEMORY16/8; ++i, i1 = (i1 + memory_stride) % CN_MEMORY16) {
          x ^= lpad[i1];
          x ^= x2l;

          // Unrolled AES rounds for better instruction scheduling
          for (unsigned j = 0; j < 10; ++ j) {
            aes_round(&x, key.u4[j], aes0, aes1, aes2, aes3);
          }
          x1s = x;
          nd.barrier(sycl::access::fence_space::local_space);

          x ^= lpad[i1 + 8];
          x ^= x1l;

          for (unsigned j = 0; j < 10; ++ j) {
            aes_round(&x, key.u4[j], aes0, aes1, aes2, aes3);
          }
          x2s = x;
          nd.barrier(sycl::access::fence_space::local_space);
        }

        x ^= x2l;

        // Final rounds with reduced synchronization overhead
        for (unsigned i = 0; i < 16; ++ i) {
          for (unsigned j = 0; j < 10; ++ j) {
            aes_round(&x, key.u4[j], aes0, aes1, aes2, aes3);
          }
          nd.barrier(sycl::access::fence_space::local_space);
          x1s = x;
          nd.barrier(sycl::access::fence_space::local_space);
          x ^= x1l;
        }

        x.store(l1 + 4, spad);
      });
    });

    q.wait_and_throw();

  } catch (sycl::exception const& e) {
    printf("Caught synchronous SYCL exception:\n%s\n", e.what());
    throw;
  }

  // Final Keccak hashing - keep on CPU for now as it's small workload
  for (unsigned i = 0; i < batch; ++ i) {
    keccak(static_cast<uint64_t*>(Spads) + 25*i);
    memcpy(output + HASH_LEN*i, static_cast<uint64_t*>(Spads) + 25*i, HASH_LEN);
  }
}
