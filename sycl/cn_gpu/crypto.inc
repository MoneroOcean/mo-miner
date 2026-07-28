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
