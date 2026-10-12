// Octopus consensus math and lane-level hashing; included inside mom_octopus.
static std::atomic<uint64_t> next_points_version{1};
#ifdef MOM_OCTOPUS_HAS_SYCL_NATIVE
// Native matrix search is opt-in: only a nonempty value other than literal 0 requests it.
static bool sycl_native_requested() {
  const char* configured = std::getenv("MOM_OCTOPUS_SYCL_NATIVE");
  return configured && *configured && std::strcmp(configured, "0") != 0;
}

// Admit only supported Intel/CUDA/HIP matrix hardware; all other devices retain the portable route.
static bool native_matrix_supported(sycl::queue& queue) {
#if defined(OCTOPUS_ESIMD)
  return sycl_is_level_zero_gpu(queue.get_device()) &&
         queue.get_device().has(sycl::aspect::ext_intel_matrix);
#elif defined(MOM_SYCL_HAS_CUDA)
  return mom::nvidia::has_int8_async_matrix(
      mom::nvidia::compute_capability(queue.get_device()));
#elif defined(MOM_SYCL_HAS_HIP)
  hipDevice_t device = -1;
  hipDeviceProp_t properties{};
  if (!mom_is_hip(queue.get_device()))
    return false;
  mom::hip_queue_device(queue, device);
  return hipGetDeviceProperties(&properties, device) == hipSuccess &&
         mom::amd::has_gfx12_int8_wmma(properties.gcnArchName);
#else
  (void)queue;
  return false;
#endif
}
#endif

constexpr uint32_t MOD = 1032193;
constexpr uint32_t CACHE_NODE_WORDS = 16;
constexpr uint32_t DATASET_PARENTS = 256;
constexpr uint32_t PAGE_WORDS = 64;
constexpr uint32_t POLYNOMIAL_SIZE = 1024;
constexpr uint32_t POLYNOMIAL_LANES = 32;
constexpr uint32_t POLYNOMIAL_BATCH = 16;
constexpr uint32_t COOPERATIVE_WARPS = 4;
constexpr uint32_t ACCESSES = 32;
constexpr uint32_t FNV_PRIME = 0x01000193U;
constexpr uint64_t EPOCH_LENGTH = 1ULL << 19;
constexpr unsigned MAX_RESULTS = 15;

struct Result {
  uint32_t count;
  uint64_t nonce[MAX_RESULTS];
  uint8_t hash[MAX_RESULTS][HASH_LEN];
};

inline uint32_t load32(const uint8_t* const p) {
  return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
         (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}

inline uint64_t load64(const uint8_t* const p) {
  return static_cast<uint64_t>(load32(p)) | (static_cast<uint64_t>(load32(p + 4)) << 32);
}

inline void store32(uint8_t* const p, const uint32_t x) {
  p[0] = static_cast<uint8_t>(x);
  p[1] = static_cast<uint8_t>(x >> 8);
  p[2] = static_cast<uint8_t>(x >> 16);
  p[3] = static_cast<uint8_t>(x >> 24);
}

inline uint64_t rotl64(const uint64_t x, const unsigned n) {
  return mo_rotate(x, static_cast<uint64_t>(n));
}

inline uint32_t fnv32(const uint32_t x, const uint32_t y) {
  return x * FNV_PRIME ^ y;
}
inline uint64_t fnv64(const uint64_t x, const uint64_t y) {
  return x * static_cast<uint64_t>(FNV_PRIME) ^ y;
}

// Older Windows DPC++ needs explicit NVPTX unrolling to keep Keccak state out of local memory.
// Restrict it to that device target so portable SPIR-V does not gain vendor loop-control metadata.
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
#define MOM_OCTOPUS_NVPTX_UNROLL _Pragma("unroll")
#else
#define MOM_OCTOPUS_NVPTX_UNROLL
#endif

// Apply one Keccak-f round to the little-endian lane state.
inline void keccak_round(uint64_t state[25], const unsigned round) {
  static constexpr uint64_t round_constants[24] = {
      0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808AULL, 0x8000000080008000ULL,
      0x000000000000808BULL, 0x0000000080000001ULL, 0x8000000080008081ULL, 0x8000000000008009ULL,
      0x000000000000008AULL, 0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000AULL,
      0x000000008000808BULL, 0x800000000000008BULL, 0x8000000000008089ULL, 0x8000000000008003ULL,
      0x8000000000008002ULL, 0x8000000000000080ULL, 0x000000000000800AULL, 0x800000008000000AULL,
      0x8000000080008081ULL, 0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL,
  };
  static constexpr unsigned rotations[24] = {
      1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14, 27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44,
  };
  static constexpr unsigned permutation[24] = {
      10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1,
  };

  uint64_t column[5];
  MOM_OCTOPUS_NVPTX_UNROLL
  for (unsigned x = 0; x < 5; ++x)
    column[x] = state[x] ^ state[x + 5] ^ state[x + 10] ^ state[x + 15] ^ state[x + 20];
  MOM_OCTOPUS_NVPTX_UNROLL
  for (unsigned x = 0; x < 5; ++x) {
    const uint64_t d = column[(x + 4) % 5] ^ rotl64(column[(x + 1) % 5], 1);
    MOM_OCTOPUS_NVPTX_UNROLL
    for (unsigned y = 0; y < 25; y += 5)
      state[x + y] ^= d;
  }

  uint64_t t = state[1];
  MOM_OCTOPUS_NVPTX_UNROLL
  for (unsigned i = 0; i < 24; ++i) {
    const unsigned index = permutation[i];
    const uint64_t previous = state[index];
    state[index] = rotl64(t, rotations[i]);
    t = previous;
  }
  MOM_OCTOPUS_NVPTX_UNROLL
  for (unsigned y = 0; y < 25; y += 5) {
    MOM_OCTOPUS_NVPTX_UNROLL
    for (unsigned x = 0; x < 5; ++x)
      column[x] = state[y + x];
    MOM_OCTOPUS_NVPTX_UNROLL
    for (unsigned x = 0; x < 5; ++x)
      state[y + x] = column[x] ^ ((~column[(x + 1) % 5]) & column[(x + 2) % 5]);
  }
  state[0] ^= round_constants[round];
}

#undef MOM_OCTOPUS_NVPTX_UNROLL

// Run the full Keccak-f permutation used by the cache and proof hash.
inline void keccak(uint64_t state[25]) {
  for (unsigned round = 0; round < 24; ++round)
    keccak_round(state, round);
}

// Hash one cache node in place with Keccak-512 padding, not SHA3 padding.
inline void keccak512_words(uint32_t words[CACHE_NODE_WORDS]) {
  uint64_t state[25]{};
  for (unsigned i = 0; i < 8; ++i)
    state[i] =
        static_cast<uint64_t>(words[i * 2]) | (static_cast<uint64_t>(words[i * 2 + 1]) << 32);
  state[8] = 0x8000000000000001ULL; // Keccak padding for a 64-byte input.
  keccak(state);
  for (unsigned i = 0; i < 8; ++i) {
    words[i * 2] = static_cast<uint32_t>(state[i]);
    words[i * 2 + 1] = static_cast<uint32_t>(state[i] >> 32);
  }
}

// Bind the problem header to the polynomial value before dataset mixing.
inline void keccak512_header_seed(const uint8_t header[HASH_LEN], const uint64_t value,
                                  uint32_t out[CACHE_NODE_WORDS]) {
  uint64_t state[25]{};
  for (unsigned i = 0; i < 4; ++i)
    state[i] = load64(header + i * 8);
  state[4] = value;
  state[5] = 1; // Keccak padding starts immediately after the 40-byte input.
  state[8] = 0x8000000000000000ULL;
  keccak(state);
  for (unsigned i = 0; i < 8; ++i) {
    out[i * 2] = static_cast<uint32_t>(state[i]);
    out[i * 2 + 1] = static_cast<uint32_t>(state[i] >> 32);
  }
}

// Finalize the seed and compressed dataset mix into the pool-facing proof hash.
inline void keccak256_seed_mix(const uint32_t seed[CACHE_NODE_WORDS], const uint32_t mix[8],
                               uint8_t out[HASH_LEN]) {
  uint64_t state[25]{};
  for (unsigned i = 0; i < 8; ++i)
    state[i] = static_cast<uint64_t>(seed[i * 2]) | (static_cast<uint64_t>(seed[i * 2 + 1]) << 32);
  for (unsigned i = 0; i < 4; ++i)
    state[8 + i] =
        static_cast<uint64_t>(mix[i * 2]) | (static_cast<uint64_t>(mix[i * 2 + 1]) << 32);
  state[12] = 1; // Keccak padding after the 96-byte input.
  state[16] = 0x8000000000000000ULL;
  keccak(state);
  for (unsigned i = 0; i < 4; ++i) {
    store32(out + i * 8, static_cast<uint32_t>(state[i]));
    store32(out + i * 8 + 4, static_cast<uint32_t>(state[i] >> 32));
  }
}

struct SipHash {
  uint64_t v0, v1, v2, v3;

  // Apply one SipHash mixing round to the coefficient generator state.
  void round() {
    v0 += v1;
    v2 += v3;
    v1 = rotl64(v1, 13);
    v3 = rotl64(v3, 16);
    v1 ^= v0;
    v3 ^= v2;
    v0 = rotl64(v0, 32);
    v2 += v1;
    v0 += v3;
    v1 = rotl64(v1, 17);
    v3 = rotl64(v3, 21);
    v1 ^= v2;
    v3 ^= v0;
    v2 = rotl64(v2, 32);
  }

  // Bind an aligned nonce into the SipHash-2-4 state used for polynomial coefficients.
  void hash24(const uint64_t nonce) {
    v3 ^= nonce;
    round();
    round();
    v0 ^= nonce;
    v2 ^= 0xFF;
    round();
    round();
    round();
    round();
  }

  uint64_t value() const {
    return v0 ^ v1 ^ v2 ^ v3;
  }
};

inline uint32_t gcd(uint32_t a, uint32_t b) {
  while (b != 0) {
    const uint32_t remainder = a % b;
    a = b;
    b = remainder;
  }
  return a;
}

// Exponentiate in the Octopus prime field; all intermediate products fit uint64_t.
inline uint32_t pow_mod(uint32_t base, uint32_t exponent) {
  uint32_t result = 1;
  while (exponent != 0) {
    if (exponent & 1U)
      result = static_cast<uint32_t>((static_cast<uint64_t>(result) * base) % MOD);
    base = static_cast<uint32_t>((static_cast<uint64_t>(base) * base) % MOD);
    exponent >>= 1;
  }
  return result;
}

// Choose a generator power whose exponent is coprime to MOD-1.
inline uint32_t remap(const uint64_t value) {
  uint32_t exponent = static_cast<uint32_t>(value % (MOD - 2)) + 1;
  uint32_t divisor;
  while ((divisor = gcd(exponent, MOD - 1)) != 1)
    exponent /= divisor;
  return pow_mod(11, exponent);
}

// Reduce the bounded polynomial product exactly; callers keep values below 2*MOD^2+MOD.
inline uint32_t mod_field(const uint64_t value) {
  // All inputs are below 2*MOD^2+MOD < 2^41.  floor(2^41/MOD) gives a
  // quotient at most one low, replacing 64-bit division with one multiply.
  constexpr uint32_t RECIPROCAL = 2130438;
  const uint32_t product_high =
#if defined(MOM_SYCL_ADAPTIVECPP) || defined(MOM_SYCL_PORTABLE_OPENCL)
      // Some OpenCL CPU JITs miscompile the SPIR-V mul_hi builtin. The explicit widening form has
      // identical integer semantics and leaves native DPC++ GPU artifacts on their measured path.
      static_cast<uint32_t>((static_cast<uint64_t>(static_cast<uint32_t>(value)) * RECIPROCAL) >>
                            32);
#else
      sycl::mul_hi(static_cast<uint32_t>(value), RECIPROCAL);
#endif
  const uint32_t quotient = (product_high + static_cast<uint32_t>(value >> 32) * RECIPROCAL) >> 9;
  const uint32_t remainder = static_cast<uint32_t>(value - static_cast<uint64_t>(quotient) * MOD);
  return remainder >= MOD ? remainder - MOD : remainder;
}

// Multiply-add in the prime field; compatibility builds avoid approximate floating division.
inline uint32_t mad_field(const uint32_t a, const uint32_t b, const uint32_t c) {
  if constexpr (mom_sycl_portable_opencl) {
    // OpenCL does not require enough precision from the floating approximation below to keep its
    // quotient within one correction. Reuse the exact reciprocal reduction in the compatibility
    // artifact; native GPU builds retain their measured FMA path.
    return mod_field(static_cast<uint64_t>(a) * b + c);
  }
  constexpr float INVERSE = 1.0f / MOD;
  const uint32_t quotient = static_cast<uint32_t>(
      sycl::fma(static_cast<float>(a), static_cast<float>(b), static_cast<float>(c)) * INVERSE);
  int32_t remainder = static_cast<int32_t>(a * b + c - quotient * MOD);
  if (remainder < 0)
    remainder += MOD;
  if (remainder >= static_cast<int32_t>(MOD))
    remainder -= MOD;
  return static_cast<uint32_t>(remainder);
}

// Compare display-order hash bytes against the inclusive pool target.
inline bool meets_target(const uint8_t hash[HASH_LEN], const uint8_t* const target) {
  for (unsigned i = 0; i < HASH_LEN; ++i) {
    if (hash[i] != target[i])
      return hash[i] < target[i];
  }
  return true;
}

// Claim one winner atomically before publishing its nonce and hash.
inline void save_result(Result* const result, const uint64_t nonce, const uint8_t hash[HASH_LEN]) {
  using Atomic = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                  sycl::access::address_space::global_space>;
  const uint32_t index = Atomic(result->count).fetch_add(1);
  if (index >= MAX_RESULTS)
    return;
  result->nonce[index] = nonce;
  for (unsigned i = 0; i < HASH_LEN; ++i)
    result->hash[index][i] = hash[i];
}

// Regenerate one Ethash-style dataset node from the epoch cache for light-DAG search.
inline void dataset_item(const uint32_t* const cache, const FastModData cache_mod,
                         const uint32_t node_index, uint32_t item[CACHE_NODE_WORDS]) {
  const uint32_t cache_index = fast_mod_dev(node_index, cache_mod);
  for (unsigned word = 0; word < CACHE_NODE_WORDS; ++word)
    item[word] = cache[cache_index * CACHE_NODE_WORDS + word];
  item[0] ^= node_index;
  keccak512_words(item);
  for (uint32_t i = 0; i < DATASET_PARENTS; ++i) {
    const uint32_t parent = fast_mod_dev(fnv32(node_index ^ i, item[i & 15]), cache_mod);
    for (unsigned word = 0; word < CACHE_NODE_WORDS; ++word)
      item[word] = fnv32(item[word], cache[parent * CACHE_NODE_WORDS + word]);
  }
  keccak512_words(item);
}

// Generate one lane of coefficients for a nonce-aligned polynomial group; callers barrier before use.
inline void make_polynomial_coefficients(const uint8_t* const header, const uint64_t nonce,
                                         const uint32_t lane, const uint32_t offset,
                                         sycl::local_accessor<uint32_t, 1> coefficients) {
  const uint64_t v0 = load64(header), v1 = load64(header + 8);
  const uint64_t v2 = load64(header + 16), v3 = load64(header + 24);
  SipHash sip{v0, v1, v2, v3};
  sip.hash24((nonce / POLYNOMIAL_LANES) * POLYNOMIAL_LANES + lane);
  for (uint32_t row = 0; row < POLYNOMIAL_SIZE / POLYNOMIAL_LANES; ++row) {
    sip.round();
    coefficients[offset + row * POLYNOMIAL_LANES + lane] = static_cast<uint32_t>(sip.value()) % MOD;
  }
}

// Evaluate one polynomial point by Horner reduction after coefficients and parameters are visible.
inline void make_polynomial_lane(const uint64_t nonce, const uint32_t lane,
                                 sycl::local_accessor<uint32_t, 1> coefficients,
                                 sycl::local_accessor<uint32_t, 1> polynomial,
                                 sycl::local_accessor<uint32_t, 1> parameters) {
  const uint32_t a = parameters[0];
  const uint32_t b = parameters[1];
  const uint32_t c = parameters[2];
  const uint32_t w = parameters[3];
  const uint32_t w2 = parameters[4];
  const uint32_t remainder = static_cast<uint32_t>(nonce % POLYNOMIAL_LANES);
  const uint32_t exponent = remainder + lane * POLYNOMIAL_LANES;
  const uint32_t wpow = pow_mod(w, exponent);
  const uint32_t w2pow = pow_mod(w2, exponent);
  const uint32_t x =
      mod_field(static_cast<uint64_t>(a) * w2pow + static_cast<uint64_t>(b) * wpow + c);
  uint32_t value = 0;
  for (uint32_t i = POLYNOMIAL_SIZE; i-- != 0;)
    value = mad_field(value, x, coefficients[i]);
  polynomial[lane] = value;
}

// Hash one nonce per workgroup; inactive lanes must still participate in every barrier.
template <bool FULL_DAG>
inline void octopus_hash_scalar(
    sycl::nd_item<1> item, const uint8_t* const header, const uint64_t nonce,
    const uint32_t* const cache, const FastModData cache_mod, const uint32_t* const dag,
    const FastModData page_mod, const bool active, const bool is_test, const uint8_t* const target,
    Result* const result, sycl::local_accessor<uint32_t, 1> coefficients,
    sycl::local_accessor<uint32_t, 1> polynomial, sycl::local_accessor<uint32_t, 1> parameters,
    sycl::local_accessor<uint32_t, 1> seed, sycl::local_accessor<uint32_t, 1> mix,
    sycl::local_accessor<uint32_t, 1> dataset_words, sycl::local_accessor<uint32_t, 1> page,
    sycl::local_accessor<uint32_t, 1> compressed) {
  const uint32_t lane = static_cast<uint32_t>(item.get_local_id(0));

  make_polynomial_coefficients(header, nonce, lane, 0, coefficients);
  sycl::group_barrier(item.get_group());

  if (lane == 0) {
    const uint64_t v0 = load64(header), v1 = load64(header + 8);
    const uint64_t v2 = load64(header + 16), v3 = load64(header + 24);
    const uint32_t a = remap(v0), b = remap(v1);
    uint64_t c_input = v2;
    uint32_t c;
    do {
      c = remap(c_input++);
    } while ((static_cast<uint64_t>(b) * b) % MOD == (static_cast<uint64_t>(4) * a * c) % MOD);
    const uint32_t w = remap(v3);
    parameters[0] = a;
    parameters[1] = b;
    parameters[2] = c;
    parameters[3] = w;
    parameters[4] = mod_field(static_cast<uint64_t>(w) * w);
  }
  sycl::group_barrier(item.get_group());

  make_polynomial_lane(nonce, lane, coefficients, polynomial, parameters);
  sycl::group_barrier(item.get_group());
  if (lane == 0) {
    uint64_t polynomial_hash = 0;
    for (uint32_t i = 0; i < POLYNOMIAL_LANES; ++i)
      polynomial_hash = fnv64(polynomial_hash, polynomial[i]);
    uint32_t local_seed[CACHE_NODE_WORDS];
    keccak512_header_seed(header, polynomial_hash, local_seed);
    for (unsigned i = 0; i < CACHE_NODE_WORDS; ++i)
      seed[i] = local_seed[i];
  }
  sycl::group_barrier(item.get_group());

  for (unsigned word = lane; word < PAGE_WORDS; word += POLYNOMIAL_LANES)
    mix[word] = seed[word & (CACHE_NODE_WORDS - 1)];
  sycl::group_barrier(item.get_group());

  for (uint32_t access = 0; access < ACCESSES; ++access) {
    if (lane == 0) {
      page[0] = fast_mod_dev(fnv32(seed[0] ^ access ^ polynomial[access], mix[access]), page_mod);
    }
    sycl::group_barrier(item.get_group());

    if constexpr (!FULL_DAG) {
      if (lane == 0) {
        for (unsigned node = 0; node < 4; ++node) {
          uint32_t item_words[CACHE_NODE_WORDS];
          dataset_item(cache, cache_mod, page[0] * 4 + node, item_words);
          for (unsigned word = 0; word < CACHE_NODE_WORDS; ++word)
            dataset_words[node * CACHE_NODE_WORDS + word] = item_words[word];
        }
      }
      sycl::group_barrier(item.get_group());
    }

    for (unsigned word = lane; word < PAGE_WORDS; word += POLYNOMIAL_LANES) {
      const size_t base =
          (static_cast<size_t>(page[0]) * 4 + word / CACHE_NODE_WORDS) * CACHE_NODE_WORDS;
      const uint32_t value = FULL_DAG ? dag[base + word % CACHE_NODE_WORDS] : dataset_words[word];
      mix[word] = fnv32(mix[word], value);
    }
    // The next page uses mix[access + 1], so all 64 words must be visible before it is selected.
    sycl::group_barrier(item.get_group());
  }

  if (lane < 8) {
    const unsigned left_offset = lane * 4;
    const unsigned right_offset = (8 + lane) * 4;
    uint32_t left = mix[left_offset], right = mix[right_offset];
    for (unsigned j = 1; j < 4; ++j) {
      left = fnv32(left, mix[left_offset + j]);
      right = fnv32(right, mix[right_offset + j]);
    }
    compressed[lane] = fnv32(left, right);
  }
  sycl::group_barrier(item.get_group());

  if (lane == 0 && active) {
    uint32_t local_seed[CACHE_NODE_WORDS];
    uint32_t local_compressed[8];
    for (unsigned i = 0; i < CACHE_NODE_WORDS; ++i)
      local_seed[i] = seed[i];
    for (unsigned i = 0; i < 8; ++i)
      local_compressed[i] = compressed[i];
    uint8_t output[HASH_LEN];
    keccak256_seed_mix(local_seed, local_compressed, output);
    if (is_test || meets_target(output, target))
      save_result(result, nonce, output);
  }
}

template <bool FULL_DAG>
inline void
octopus_hash_batched(sycl::nd_item<1> item, const uint8_t* const header, const uint64_t start,
                     const uint32_t* const cache, const FastModData cache_mod,
                     const uint32_t* const dag, const FastModData page_mod, const uint32_t count,
                     const uint8_t* const target, Result* const result, const bool is_test,
                     sycl::local_accessor<uint32_t, 1> coefficients, const uint32_t* const points) {
  constexpr uint32_t SUBGROUP = POLYNOMIAL_LANES;
  constexpr uint32_t SUBGROUPS = 4;
  constexpr uint32_t WORKGROUP = SUBGROUP * SUBGROUPS;
  const uint32_t local = static_cast<uint32_t>(item.get_local_id(0));
  const uint32_t subgroup = local / SUBGROUP;
  const uint32_t lane = local % SUBGROUP;
  const uint64_t last_nonce = start + count - 1;
  const uint64_t aligned_start = start & ~(static_cast<uint64_t>(SUBGROUP) - 1);
  const uint64_t group_start = aligned_start + static_cast<uint64_t>(item.get_group(0)) * WORKGROUP;
  const uint64_t block_start = group_start + static_cast<uint64_t>(subgroup) * SUBGROUP;
  const uint64_t nonce = block_start + lane;
  const bool block_active = block_start <= last_nonce && block_start + SUBGROUP - 1 >= start;
  const bool active = nonce >= start && nonce <= last_nonce;

  // A subgroup owns one aligned 32-nonce block.  Coefficients are generated once
  // per active block; every active lane then owns exactly one nonce and evaluates
  // all 32 points for it.  Thus count remains a nonce count, not 32x work items;
  // only the aligned padding lanes in the first/last block are inactive.
  if (block_active)
    make_polynomial_coefficients(header, block_start, lane, subgroup * POLYNOMIAL_SIZE,
                                 coefficients);
  sycl::group_barrier(item.get_group());

  if (!active)
    return;

  uint32_t polynomial[POLYNOMIAL_LANES];
  const uint32_t coefficient_offset = subgroup * POLYNOMIAL_SIZE;
  const uint32_t point_offset = lane;
  // Sixteen accumulators keep adjacent point evaluations together without hiding
  // the exact Horner order behind a vendor-specific vector intrinsic.
  for (uint32_t first = 0; first < POLYNOMIAL_LANES; first += POLYNOMIAL_BATCH) {
    uint32_t value[POLYNOMIAL_BATCH] = {};
    uint32_t x[POLYNOMIAL_BATCH];
    for (unsigned packed = 0; packed < POLYNOMIAL_BATCH; ++packed)
      x[packed] = points[point_offset + (first + packed) * SUBGROUP];
    for (uint32_t coefficient = POLYNOMIAL_SIZE; coefficient-- != 0;) {
      const uint32_t term = coefficients[coefficient_offset + coefficient];
      for (unsigned packed = 0; packed < POLYNOMIAL_BATCH; ++packed)
        value[packed] = mad_field(value[packed], x[packed], term);
    }
    for (unsigned packed = 0; packed < POLYNOMIAL_BATCH; ++packed)
      polynomial[first + packed] = value[packed];
  }

  uint64_t polynomial_hash = 0;
  for (uint32_t i = 0; i < POLYNOMIAL_LANES; ++i)
    polynomial_hash = fnv64(polynomial_hash, polynomial[i]);
  uint32_t seed[CACHE_NODE_WORDS];
  keccak512_header_seed(header, polynomial_hash, seed);
  uint32_t mix[PAGE_WORDS];
  for (unsigned word = 0; word < PAGE_WORDS; ++word)
    mix[word] = seed[word & (CACHE_NODE_WORDS - 1)];

  for (uint32_t access = 0; access < ACCESSES; ++access) {
    const uint32_t page =
        fast_mod_dev(fnv32(seed[0] ^ access ^ polynomial[access], mix[access]), page_mod);
    for (unsigned node = 0; node < 4; ++node) {
      uint32_t item_words[CACHE_NODE_WORDS];
      if constexpr (FULL_DAG) {
        const size_t base = (static_cast<size_t>(page) * 4 + node) * CACHE_NODE_WORDS;
        for (unsigned word = 0; word < CACHE_NODE_WORDS; ++word)
          item_words[word] = dag[base + word];
      } else {
        dataset_item(cache, cache_mod, page * 4 + node, item_words);
      }
      for (unsigned word = 0; word < CACHE_NODE_WORDS; ++word)
        mix[node * CACHE_NODE_WORDS + word] =
            fnv32(mix[node * CACHE_NODE_WORDS + word], item_words[word]);
    }
  }

  uint32_t compressed[8];
  for (unsigned lane8 = 0; lane8 < 8; ++lane8) {
    const unsigned left_offset = lane8 * 4;
    const unsigned right_offset = (8 + lane8) * 4;
    uint32_t left = mix[left_offset], right = mix[right_offset];
    for (unsigned j = 1; j < 4; ++j) {
      left = fnv32(left, mix[left_offset + j]);
      right = fnv32(right, mix[right_offset + j]);
    }
    compressed[lane8] = fnv32(left, right);
  }
  uint8_t output[HASH_LEN];
  keccak256_seed_mix(seed, compressed, output);
  if (is_test ? nonce == start : meets_target(output, target))
    save_result(result, nonce, output);
}

inline uint32_t fnv_reduce4(const uint32_t value[4]) {
  return fnv32(fnv32(fnv32(value[0], value[1]), value[2]), value[3]);
}

// Mirrors open-cfxmine's CUDA layout: each native warp evaluates 32 nonces, then
// its two 16-lane teams chase four hashes at once.  Every DAG page is consequently
// read as sixteen contiguous uint4 values instead of 64 scalar private loads.
inline void octopus_hash_cooperative(sycl::nd_item<1> item, const uint8_t* const header,
                                     const uint64_t start, const uint32_t* const dag,
                                     const FastModData page_mod, const uint32_t count,
                                     const uint8_t* const target, Result* const result,
                                     const bool is_test,
                                     sycl::local_accessor<uint32_t, 1> coefficients,
                                     const uint32_t* const points) {
  constexpr uint32_t WARP = 32;
  const uint32_t local = static_cast<uint32_t>(item.get_local_id(0));
  const uint32_t warp = local / WARP;
  const uint32_t lane = local % WARP;
  const uint64_t aligned_start = start & ~(static_cast<uint64_t>(WARP) - 1);
  const uint64_t group_start =
      aligned_start + static_cast<uint64_t>(item.get_group(0)) * item.get_local_range(0);
  const uint64_t block_start = group_start + static_cast<uint64_t>(warp) * WARP;
  const uint64_t nonce = block_start + lane;
  const uint64_t last_nonce = start + count - 1;
  const bool active = nonce >= start && nonce <= last_nonce;
  const uint32_t offset = warp * POLYNOMIAL_SIZE;

  make_polynomial_coefficients(header, block_start, lane, offset, coefficients);
  sycl::group_barrier(item.get_group());
#if defined(MOM_SYCL_ADAPTIVECPP_CUDA)
  // Fewer live Horner chains improve this compiler's CUDA path on both older and newer GPUs.
  // Each nonce still evaluates all 32 points and folds them into FNV in the same order.
  constexpr uint32_t POLYNOMIAL_STEP = 8;
#else
  constexpr uint32_t POLYNOMIAL_STEP = POLYNOMIAL_BATCH;
#endif
  uint32_t polynomial[POLYNOMIAL_LANES];
  uint64_t polynomial_hash = 0;
  for (uint32_t first = 0; first < POLYNOMIAL_LANES; first += POLYNOMIAL_STEP) {
    uint32_t value[POLYNOMIAL_STEP]{};
    uint32_t x[POLYNOMIAL_STEP];
    for (unsigned packed = 0; packed < POLYNOMIAL_STEP; ++packed)
      x[packed] = points[lane + (first + packed) * WARP];
    for (uint32_t coefficient = POLYNOMIAL_SIZE; coefficient-- != 0;) {
      const uint32_t term = coefficients[offset + coefficient];
      for (unsigned packed = 0; packed < POLYNOMIAL_STEP; ++packed)
        value[packed] = mad_field(value[packed], x[packed], term);
    }
    for (unsigned packed = 0; packed < POLYNOMIAL_STEP; ++packed) {
      polynomial[first + packed] = value[packed];
      polynomial_hash = fnv64(polynomial_hash, value[packed]);
    }
  }
  uint32_t seed[CACHE_NODE_WORDS];
  keccak512_header_seed(header, polynomial_hash, seed);
  sycl::group_barrier(item.get_group());
  for (uint32_t access = 0; access < ACCESSES; ++access)
    coefficients[offset + access * WARP + lane] = polynomial[access];
  sycl::group_barrier(item.get_group());

  const sycl::sub_group sg = item.get_sub_group();
  const uint32_t team_base = lane & ~15U;
  const uint32_t team_lane = lane & 15U;
  const uint32_t mix_part = team_lane & 3U;
  uint32_t final_mix[8]{};
  for (uint32_t first = 0; first < 16; first += 4) {
    uint32_t mix[4][4], initial[4];
    for (uint32_t parallel = 0; parallel < 4; ++parallel) {
      const uint32_t source = team_base + first + parallel;
      initial[parallel] = mom_select_from_group(sg, seed[0], source);
      for (uint32_t part = 0; part < 4; ++part)
        for (uint32_t word = 0; word < 4; ++word) {
          const uint32_t selected = mom_select_from_group(sg, seed[part * 4 + word], source);
          if (mix_part == part)
            mix[parallel][word] = selected;
        }
    }
    for (uint32_t access = 0; access < ACCESSES; access += 4) {
      const uint32_t address_lane = team_base + access / 4;
      for (uint32_t step = 0; step < 4; ++step) {
        for (uint32_t parallel = 0; parallel < 4; ++parallel) {
          const uint32_t hash_lane = team_base + first + parallel;
          const uint32_t polynomial_value =
              coefficients[offset + (access + step) * WARP + hash_lane];
          uint32_t page = fast_mod_dev(
              fnv32(initial[parallel] ^ (access + step) ^ polynomial_value, mix[parallel][step]),
              page_mod);
          page = mom_select_from_group(sg, page, address_lane);
          const size_t dag_offset = static_cast<size_t>(page) * PAGE_WORDS + team_lane * 4;
          for (uint32_t word = 0; word < 4; ++word)
            mix[parallel][word] = fnv32(mix[parallel][word], dag[dag_offset + word]);
        }
      }
    }
    for (uint32_t parallel = 0; parallel < 4; ++parallel) {
      const uint32_t reduced = fnv_reduce4(mix[parallel]);
      const uint32_t hash_lane = team_base + first + parallel;
      for (uint32_t word = 0; word < 8; ++word) {
        const uint32_t compressed = fnv32(mom_select_from_group(sg, reduced, team_base + word),
                                          mom_select_from_group(sg, reduced, team_base + 8 + word));
        if (lane == hash_lane)
          final_mix[word] = compressed;
      }
    }
  }
  uint8_t output[HASH_LEN];
  keccak256_seed_mix(seed, final_mix, output);
  if (active && (is_test ? nonce == start : meets_target(output, target)))
    save_result(result, nonce, output);
}
