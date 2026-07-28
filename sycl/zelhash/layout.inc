namespace mom_zelhash {

// CUDA needs aligned stores on both compilers; HIP is faster with the denser scalar record stream.
#if defined(MOM_SYCL_HAS_CUDA) || defined(MOM_SYCL_ADAPTIVECPP_CUDA)
#define MOM_EQ_ALIGNED_GPU_RECORDS 1
#endif
// DPC++ lowers a coalesced leaf->position map much better than four random reverse-map stores.
// The rare recovery pass resolves final leaves with one linear inversion scan.
#if defined(MOM_SYCL_HAS_CUDA)
#define MOM_EQ_FORWARD_MAP 1
#endif

// ---- 125,4 parameters (fluxd equihash.{h,cpp}) ----
constexpr unsigned N = 125, K = 4;
constexpr unsigned COLLISION_BIT_LENGTH  = N / (K + 1);              // 25
constexpr unsigned COLLISION_BYTE_LENGTH = (COLLISION_BIT_LENGTH + 7) >> 3; // 4
constexpr unsigned HASH_LENGTH           = (K + 1) * COLLISION_BYTE_LENGTH; // 20  (== EQUIHASH_ROW_LEN)
constexpr unsigned INDICES_PER_HASH      = 512 / N;                 // 4
constexpr unsigned HASH_OUTPUT           = INDICES_PER_HASH * ((N + 7) >> 3); // 64
constexpr unsigned SUB_ELEMENT_BYTES     = (N + 7) >> 3;            // 16
constexpr uint64_t HEADER_LEN            = 140;
[[maybe_unused]] constexpr uint64_t NUM_ENTRIES = 1ull << (COLLISION_BIT_LENGTH + 1); // 2^26 (M2 rounds)
static_assert(HASH_LENGTH == EQUIHASH_ROW_LEN, "row length must match the host-side dump buffer width");

constexpr unsigned ceil_div_u64(uint64_t value, unsigned divisor) {
  return static_cast<unsigned>((value + divisor - 1u) / divisor);
}
constexpr unsigned ceil_sqrt(unsigned value) {
  unsigned root = 0;
  while (static_cast<uint64_t>(root) * root < value) ++root;
  return root;
}
constexpr unsigned round_up(unsigned value, unsigned alignment) {
  return (value + alignment - 1u) / alignment * alignment;
}
constexpr unsigned round_down(unsigned value, unsigned alignment) {
  return value / alignment * alignment;
}
constexpr unsigned bucket_capacity(unsigned buckets, unsigned deviations, unsigned alignment,
                                   bool round_capacity_down = false) {
  // Bucket occupancy is binomial. Size the arena from its mean plus a backend-appropriate number
  // of standard deviations, then align the physical stride for vector memory transactions. This
  // derives the old hand-tuned capacities while scaling automatically if the bucket geometry changes.
  const unsigned mean = ceil_div_u64(NUM_ENTRIES, buckets);
  const unsigned limit = mean + deviations * ceil_sqrt(mean);
  return round_capacity_down ? round_down(limit, alignment) : round_up(limit, alignment);
}

// ---- Wagner solver memory plan. Intel oneAPI gains from 16384 super-buckets and 1024-lane groups:
// its ~29-KiB scratch arena amortizes each bucket with fewer register-held pairs per lane. CUDA/HIP
// instead lose materially with that geometry and retain 8192 buckets, 1024 lanes, and a ~53-KiB
// arena. AdaptiveCpp CUDA cannot opt a Windows kernel into >48 KiB dynamic shared memory, so its
// lower-priority fallback uses 32768 smaller buckets and the same common collision code with a
// ~16-KiB scratch arena.
// Four materialized levels hold the gen output and round-0..2 survivors. Round 3 filters its final
// field in-place and emits only zero pairs, so the otherwise 0.27--0.53-GiB level-4 stream is absent.
#if defined(MOM_SYCL_ADAPTIVECPP_CUDA)
constexpr unsigned RESTBITS  = 10;
constexpr unsigned BUCKBITS  = COLLISION_BIT_LENGTH - RESTBITS;     // 15
constexpr unsigned NBUCKETS  = 1u << BUCKBITS;                      // 32768
constexpr unsigned NRESTBINS = 1u << RESTBITS;                      // 1024 exact bins per bucket
constexpr unsigned MAX_SLOTS = bucket_capacity(NBUCKETS, 1u, 64u);
constexpr unsigned ROUND_WG   = 1024;
#elif !defined(MOM_SYCL_HAS_CUDA) && !defined(MOM_SYCL_HAS_HIP) && !defined(MOM_SYCL_ADAPTIVECPP)
constexpr unsigned RESTBITS  = 11;
constexpr unsigned BUCKBITS  = COLLISION_BIT_LENGTH - RESTBITS;     // 14
constexpr unsigned NBUCKETS  = 1u << BUCKBITS;                      // 16384
constexpr unsigned NRESTBINS = 1u << RESTBITS;                      // 2048 exact bins per super-bucket
constexpr unsigned MAX_SLOTS = bucket_capacity(NBUCKETS, 6u, 16u);
constexpr unsigned ROUND_WG  = 1024;
#elif defined(MOM_SYCL_HAS_CUDA)
// Eight fixed records per 1024-lane work-group make the collision input fully unrollable. A mildly
// larger, non-power-of-two bucket count keeps the same total capacity while reducing the per-bucket
// cap from 8704 to exactly 8192. The quotient is the exact-bin id and the remainder selects a bucket.
constexpr unsigned RESTBITS  = 12;                                  // packed metadata width
constexpr unsigned NBUCKETS  = 8759;
constexpr unsigned NRESTBINS = 3831;                                // ceil(2^25 / 8759)
constexpr unsigned MAX_SLOTS = bucket_capacity(NBUCKETS, 6u, 16u);
constexpr unsigned ROUND_WG  = 1024;
#else
constexpr unsigned RESTBITS  = 12;
constexpr unsigned BUCKBITS  = COLLISION_BIT_LENGTH - RESTBITS;     // 13
constexpr unsigned NBUCKETS  = 1u << BUCKBITS;                      // 8192
constexpr unsigned NRESTBINS = 1u << RESTBITS;                      // 4096 exact bins per super-bucket
#if defined(MOM_SYCL_HAS_HIP)
// Three standard deviations preserve virtually every generated row while avoiding the large,
// power-of-two-related strides that concentrate progressive records in too few cache sets.
constexpr unsigned MAX_SLOTS = bucket_capacity(NBUCKETS, 3u, 16u, true);
#else
constexpr unsigned MAX_SLOTS = bucket_capacity(NBUCKETS, 5u, 64u);
#endif
constexpr unsigned ROUND_WG  = 1024;
#endif
#if !defined(MOM_SYCL_HAS_CUDA) && !defined(MOM_SYCL_HAS_HIP) && !defined(MOM_SYCL_ADAPTIVECPP)
// Intel Xe benefits from walking exact-bin chains directly in SLM. Splitting the two early rounds
// keeps their larger follower records below the device's local-memory limit; later rounds fit whole.
constexpr bool ROUND_DIRECT_CHAINS = true;
#else
constexpr bool ROUND_DIRECT_CHAINS = false;
#endif
constexpr unsigned NLEVELS   = K + 1;                              // five progressive field counts
constexpr unsigned NSTORED_LEVELS = K;                             // final zero pairs replace level 4
static_assert(MAX_SLOTS < (1u << 16));
inline uint32_t stored_rest(uint32_t word) { return word & ((1u << RESTBITS) - 1u); }
#if defined(MOM_SYCL_HAS_CUDA)
inline uint32_t field_rest(uint32_t field) { return (field & 0x1ffffffu) / NBUCKETS; }
inline uint32_t field_bucket(uint32_t field) { return (field & 0x1ffffffu) % NBUCKETS; }
inline uint32_t field_from_parts(uint32_t bucket, uint32_t rest) { return rest * NBUCKETS + bucket; }
#else
inline uint32_t field_rest(uint32_t field) { return field & (NRESTBINS - 1u); }
inline uint32_t field_bucket(uint32_t field) { return field >> RESTBITS; }
inline uint32_t field_from_parts(uint32_t bucket, uint32_t rest) { return (bucket << RESTBITS) | rest; }
#endif
template <unsigned R, bool GPU_COMPACT> inline uint32_t collision_rest(const uint32_t* g) {
#if defined(MOM_SYCL_HAS_CUDA)
  if constexpr (GPU_COMPACT && R == 0u)
    return (g[0] >> 25) | ((g[1] >> 25) << 7);
#endif
  return R == 0u ? field_rest(g[0]) : stored_rest(g[0]);
}

// Progressive records carry one fewer 25-bit collision field after each round. The common level 0
// densely packs five fields plus its 26-bit leaf index into five words. CUDA stores the five fields in
// one aligned vector and keeps the recovery-only index in a separate stream. At L>=1 the active field's
// high bits are the physical bucket and need not be stored; w0 combines active_low with the previous
// (parent) bucket, followed by the remaining full-width fields. Parent pair IDs are reconstructed only
// for the handful of final zero nodes, avoiding both the old multi-GiB tree stream and 26 tree bits in
// every survivor.
// Intel and HIP use the natural {5,4,3,2,1} shape. Intel's heavy level-1 scatter must be emitted as
// one explicit 16-byte vector store: four scalar stores at the same stride are pathological on Xe.
// CUDA keeps aligned 16-byte records through level 2 and the natural 8-byte level 3 used by the
// final collision round. Physical 16-slot tiling below retains aligned transactions without the
// cache-alias penalty previously seen with dense late records in bucket-major order.
constexpr unsigned level_fields(unsigned L) { return NLEVELS - L; }            // 5,4,3,2,1
#if !defined(MOM_SYCL_HAS_CUDA) && !defined(MOM_SYCL_HAS_HIP) && !defined(MOM_SYCL_ADAPTIVECPP)
constexpr unsigned LEVEL_U32_TBL[NLEVELS]    = { 5, 4, 3, 2, 1 };
#else
constexpr unsigned LEVEL_U32_TBL[NLEVELS]    = { 5, 5, 3, 2, 1 };
#endif
// The combined DPC++ binary carries both compile-time shapes and selects by the queue's actual backend.
constexpr unsigned level_u32(unsigned L, bool gpu_compact = false) {
#if defined(MOM_EQ_ALIGNED_GPU_RECORDS)
#if defined(MOM_SYCL_HAS_CUDA)
  constexpr unsigned CUDA_U32_TBL[NLEVELS] = { 4, 4, 4, 2, 2 };
#else
  constexpr unsigned CUDA_U32_TBL[NLEVELS] = { 4, 4, 4, 4, 2 };
#endif
  if (gpu_compact) return CUDA_U32_TBL[L];
#endif
  return gpu_compact && L == 1 ? 4u : LEVEL_U32_TBL[L];
}
#if defined(MOM_SYCL_HAS_CUDA)
// CUDA record levels use 16-slot tiles rather than storing one complete bucket at a time:
//
//   [slot / 16][bucket][slot % 16]
//
// Adjacent lanes still issue two coalesced 16-record segments, while different warps spread their
// random bucket stores across fewer cache sectors. Logical bucket/slot IDs remain unchanged, so the
// common collision and recovery algorithms only need this physical-address translation.
inline size_t cuda_record_pos(uint32_t bucket, uint32_t slot) {
  return (((size_t)(slot >> 4u) * NBUCKETS + bucket) << 4u) + (slot & 15u);
}
#endif
// Common dense level-0 leaf: five 25-bit fields + one 26-bit Equihash index = 151 bits in five words.
// f0 stays in w0[24:0], so bucket/rest-bin extraction remains a single masked load. A 20-byte stride is
// also the measured Intel/HIP-safe scatter shape (unlike CUDA's separate aligned hash/index streams).
inline void dense_l0_store(uint32_t* o, const uint32_t* f, uint32_t eh_index) {
  o[0] = f[0] | ((f[1] & 0x7fu) << 25);
  o[1] = (f[1] >> 7) | ((f[2] & 0x3fffu) << 18);
  o[2] = (f[2] >> 14) | ((f[3] & 0x1fffffu) << 11);
  o[3] = (f[3] >> 21) | ((f[4] & 0x01ffffffu) << 4) | ((eh_index & 0x7u) << 29);
  o[4] = eh_index >> 3;
}
template <unsigned R, bool GPU_COMPACT>
inline uint32_t load_follower(const uint32_t* g, unsigned k) {
  if constexpr (R == 0) {
#if defined(MOM_SYCL_HAS_CUDA)
    if constexpr (GPU_COMPACT) return g[k] & 0x1ffffffu;
#endif
    switch (k) {
      case 0: return ((g[0] >> 25) & 0x7fu) | ((g[1] & 0x3ffffu) << 7);
      case 1: return (g[1] >> 18) | ((g[2] & 0x7ffu) << 14);
      case 2: return (g[2] >> 11) | ((g[3] & 0xfu) << 21);
      default: return (g[3] >> 4) & 0x01ffffffu;
    }
  } else {
    return g[k + 1u];
  }
}
inline uint32_t dense_l0_load_index(const uint32_t* g) {
  return (g[3] >> 29) | ((g[4] & 0x007fffffu) << 3);
}
#if !defined(MOM_SYCL_HAS_CUDA) && !defined(MOM_SYCL_HAS_HIP) && !defined(MOM_SYCL_ADAPTIVECPP)
using IntelU32x4 = uint32_t __attribute__((ext_vector_type(4)));
static_assert(sizeof(IntelU32x4) == 16);
inline void store_intel_u32x4(uint32_t* p, uint32_t x, uint32_t y, uint32_t z, uint32_t w) {
  *reinterpret_cast<IntelU32x4*>(p) = IntelU32x4{x, y, z, w};
}
#else
inline void store_intel_u32x4(uint32_t* p, uint32_t x, uint32_t y, uint32_t z, uint32_t w) {
  p[0] = x; p[1] = y; p[2] = z; p[3] = w;
}
#endif
#if defined(MOM_EQ_ALIGNED_GPU_RECORDS)
using NativeU32x4 = uint32_t __attribute__((ext_vector_type(4)));
static_assert(sizeof(NativeU32x4) == 16);
inline void store_aligned_u32x4(uint32_t* p, uint32_t x, uint32_t y, uint32_t z, uint32_t w) {
  // Both CUDA toolchains scalarize sycl::vec::store here. Preserve the 16-byte alignment in IR so
  // the two traffic-heavy CUDA rounds lower to one vector global store per survivor.
  *reinterpret_cast<NativeU32x4*>(p) = NativeU32x4{x, y, z, w};
}
inline void dense_l0_store_compact(uint32_t* o, const uint32_t* f) {
  // CUDA keeps the 125 hash bits in one aligned vector and writes the leaf mapping to a separate
  // stream. Generation scatters and round-0 reads are 16 rather than 20 bytes per row.
#if defined(MOM_SYCL_HAS_CUDA)
  // The physical bucket already carries most of f0. Keep only its exact-bin quotient in spare high
  // bits and store f1..f4 directly, so collision rounds avoid unpacking cross-word 25-bit fields.
  const uint32_t rest = field_rest(f[0]);
  store_aligned_u32x4(o, f[1] | ((rest & 0x7fu) << 25),
                         f[2] | ((rest >> 7) << 25), f[3], f[4]);
#else
  store_aligned_u32x4(o,
    f[0] | ((f[1] & 0x7fu) << 25),
    (f[1] >> 7) | ((f[2] & 0x3fffu) << 18),
    (f[2] >> 14) | ((f[3] & 0x1fffffu) << 11),
    (f[3] >> 21) | ((f[4] & 0x01ffffffu) << 4));
#endif
}
#endif

// ===========================================================================================
// BLAKE2b-512 with personalization (RFC 7693 + BLAKE2 param block). Self-contained so the personal
// param block matches the Equihash 125,4 reference exactly. Used host-side (base midstate) and device-side
// (per-index finalize). All scalar / IGC-safe.
// ===========================================================================================
static constexpr uint64_t B2B_IV[8] = {
  0x6a09e667f3bcc908ull, 0xbb67ae8584caa73bull, 0x3c6ef372fe94f82bull, 0xa54ff53a5f1d36f1ull,
  0x510e527fade682d1ull, 0x9b05688c2b3e6c1full, 0x1f83d9abfb41bd6bull, 0x5be0cd19137e2179ull,
};
static constexpr uint8_t B2B_SIGMA[12][16] = {
  { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15},
  {14,10, 4, 8, 9,15,13, 6, 1,12, 0, 2,11, 7, 5, 3},
  {11, 8,12, 0, 5, 2,15,13,10,14, 3, 6, 7, 1, 9, 4},
  { 7, 9, 3, 1,13,12,11,14, 2, 6, 5,10, 4, 0,15, 8},
  { 9, 0, 5, 7, 2, 4,10,15,14, 1,11,12, 6, 8, 3,13},
  { 2,12, 6,10, 0,11, 8, 3, 4,13, 7, 5,15,14, 1, 9},
  {12, 5, 1,15,14,13, 4,10, 0, 7, 6, 3, 9, 2, 8,11},
  {13,11, 7,14,12, 1, 3, 9, 5, 0,15, 4, 8, 6, 2,10},
  { 6,15,14, 9,11, 3, 0, 8,12, 2,13, 7, 1, 4,10, 5},
  {10, 2, 8, 4, 7, 6, 1, 5,15,11, 9,14, 3,12,13, 0},
  { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15},
  {14,10, 4, 8, 9,15,13, 6, 1,12, 0, 2,11, 7, 5, 3},
};
