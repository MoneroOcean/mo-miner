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

template <bool NarrowSupported = !mom_sycl_adaptivecpp && !mom_sycl_portable_opencl>
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

inline void cn_recurrence_barrier(const sycl::nd_item<1>& item, const bool use_narrow) {
  if constexpr (mom_sycl_portable_opencl) {
    item.barrier(sycl::access::fence_space::local_space);
  } else {
    cn_recurrence_barrier(item, item.get_sub_group(), use_narrow);
  }
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
