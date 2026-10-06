#if defined(OCTOPUS_ESIMD)

template <int Width>
inline void native_keccak_esimd(octopus_esimd::simd<uint64_t, Width> (&state)[25]) {
  using Word = octopus_esimd::simd<uint64_t, Width>;
  static constexpr uint64_t round_constants[24] = {
      0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808AULL,
      0x8000000080008000ULL, 0x000000000000808BULL, 0x0000000080000001ULL,
      0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008AULL,
      0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000AULL,
      0x000000008000808BULL, 0x800000000000008BULL, 0x8000000000008089ULL,
      0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
      0x000000000000800AULL, 0x800000008000000AULL, 0x8000000080008081ULL,
      0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL,
  };
  static constexpr unsigned rotations[24] = {
      1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14,
      27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44,
  };
  static constexpr unsigned permutation[24] = {
      10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4,
      15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1,
  };

  // Keep the round loop visible to the compiler; the fixed-width Keccak steps stay in registers.
#pragma unroll 1
  for (unsigned round = 0; round < 24; ++round) {
    Word column[5];
#pragma unroll
    for (unsigned x = 0; x < 5; ++x)
      column[x] = state[x] ^ state[x + 5] ^ state[x + 10] ^ state[x + 15] ^ state[x + 20];
#pragma unroll
    for (unsigned x = 0; x < 5; ++x) {
      const Word rotated = (column[(x + 1) % 5] << 1) | (column[(x + 1) % 5] >> 63);
      const Word d = column[(x + 4) % 5] ^ rotated;
#pragma unroll
      for (unsigned y = 0; y < 25; y += 5)
        state[x + y] ^= d;
    }

    Word t = state[1];
#pragma unroll
    for (unsigned i = 0; i < 24; ++i) {
      const unsigned index = permutation[i];
      const Word previous = state[index];
      state[index] = (t << rotations[i]) | (t >> (64 - rotations[i]));
      t = previous;
    }
#pragma unroll
    for (unsigned y = 0; y < 25; y += 5) {
#pragma unroll
      for (unsigned x = 0; x < 5; ++x)
        column[x] = state[y + x];
#pragma unroll
      for (unsigned x = 0; x < 5; ++x)
        state[y + x] = column[x] ^ ((~column[(x + 1) % 5]) & column[(x + 2) % 5]);
    }
    state[0] ^= Word(round_constants[round]);
  }
}

template <unsigned HashesPerThread> class OctopusSyclNativeFusedIntel;

// Keccak and the DAG mix share an ESIMD work-item, avoiding intermediate USM buffers.
// Two independent hashes reduce per-work-item state on width-8 devices; width-16 retains four.
template <unsigned HashesPerThread>
static sycl::event native_finish_esimd(
    sycl::queue& queue, const uint8_t* const header, const uint32_t* const dag,
    const FastModData page_mod, const NativeProduct* const products, const uint64_t start,
    const uint32_t count, const uint8_t* const target, Result* const result,
    const bool is_test) {
  static_assert(HashesPerThread == 2 || HashesPerThread == 4);
  static_assert(NativeSubgroup % HashesPerThread == 0);
  const size_t hashes = static_cast<size_t>(NativeGroups) * NativeSubgroup;
  return queue.submit([&](sycl::handler& handler) {
    handler.parallel_for<OctopusSyclNativeFusedIntel<HashesPerThread>>(
        sycl::range<1>(hashes / HashesPerThread),
        [header, dag, page_mod, products, start, count, target, result, is_test](sycl::id<1> id)
            SYCL_ESIMD_KERNEL MOM_SYCL_KERNEL_ARGS_RESTRICT {
          using Word = octopus_esimd::simd<uint64_t, HashesPerThread>;
          constexpr auto properties = octopus_esimd::properties{
              octopus_esimd::cache_hint_L1<octopus_esimd::cache_hint::uncached>,
              octopus_esimd::cache_hint_L2<octopus_esimd::cache_hint::cached>,
              octopus_esimd::alignment<4>};
          const unsigned first_nonce = static_cast<unsigned>(id[0]) * HashesPerThread;
          const octopus_esimd::simd<uint32_t, ACCESSES> offsets(0, NativeSubgroup * 4);
          octopus_esimd::simd<uint32_t, ACCESSES> polynomial[HashesPerThread];

#pragma unroll
          for (unsigned parallel = 0; parallel < HashesPerThread; ++parallel) {
            const unsigned nonce_index = first_nonce + parallel;
            const unsigned group = nonce_index / NativeSubgroup;
            const unsigned group_lane = nonce_index & (NativeSubgroup - 1);
            const uint32_t* const row = reinterpret_cast<const uint32_t*>(products) +
                                         static_cast<size_t>(group) * NativeN + group_lane;
            polynomial[parallel] = octopus_esimd::gather<uint32_t, ACCESSES>(row, offsets);
          }

          octopus_esimd::simd<uint64_t, HashesPerThread> polynomial_hash = 0;
#pragma unroll
          for (unsigned point = 0; point < ACCESSES; ++point) {
            octopus_esimd::simd<uint64_t, HashesPerThread> value;
#pragma unroll
            for (unsigned parallel = 0; parallel < HashesPerThread; ++parallel)
              value[parallel] = static_cast<uint64_t>(polynomial[parallel][point]);
            polynomial_hash = polynomial_hash * static_cast<uint64_t>(FNV_PRIME) ^ value;
          }

          Word state[25];
#pragma unroll
          for (unsigned word = 0; word < 25; ++word)
            state[word] = Word(0);
#pragma unroll
          for (unsigned word = 0; word < 4; ++word)
            state[word] = Word(load64(header + word * 8));
          state[4] = polynomial_hash;
          state[5] = Word(1);
          state[8] = Word(0x8000000000000000ULL);
          native_keccak_esimd<HashesPerThread>(state);

          Word seed_state[8];
#pragma unroll
          for (unsigned word = 0; word < 8; ++word)
            seed_state[word] = state[word];

          octopus_esimd::simd<uint32_t, PAGE_WORDS> mix[HashesPerThread];
          uint32_t initial[HashesPerThread];
#pragma unroll
          for (unsigned parallel = 0; parallel < HashesPerThread; ++parallel) {
            initial[parallel] = static_cast<uint32_t>(seed_state[0][parallel]);
#pragma unroll
            for (unsigned word = 0; word < CACHE_NODE_WORDS; ++word) {
              const uint64_t seed_word = seed_state[word / 2][parallel];
              const uint32_t value =
                  (word & 1U) ? static_cast<uint32_t>(seed_word >> 32)
                              : static_cast<uint32_t>(seed_word);
#pragma unroll
              for (unsigned repeat = 0; repeat < PAGE_WORDS / CACHE_NODE_WORDS; ++repeat)
                mix[parallel][repeat * CACHE_NODE_WORDS + word] = value;
            }
          }

#pragma unroll
          for (unsigned access = 0; access < ACCESSES; ++access) {
            octopus_esimd::simd<uint32_t, PAGE_WORDS> values[HashesPerThread];
#pragma unroll
            for (unsigned parallel = 0; parallel < HashesPerThread; ++parallel) {
              const uint32_t address = fnv32(
                  initial[parallel] ^ access ^ polynomial[parallel][access],
                  mix[parallel][access]);
              const uint32_t page = fast_mod_dev(address, page_mod);
              values[parallel] = octopus_esimd::block_load<uint32_t, PAGE_WORDS>(
                  dag + static_cast<size_t>(page) * PAGE_WORDS, properties);
            }
#pragma unroll
            for (unsigned parallel = 0; parallel < HashesPerThread; ++parallel)
              mix[parallel] = (mix[parallel] * FNV_PRIME) ^ values[parallel];
          }

          Word final_state[25];
#pragma unroll
          for (unsigned word = 0; word < 25; ++word)
            final_state[word] = Word(0);
#pragma unroll
          for (unsigned word = 0; word < 8; ++word)
            final_state[word] = seed_state[word];
#pragma unroll
          for (unsigned parallel = 0; parallel < HashesPerThread; ++parallel) {
            octopus_esimd::simd<uint32_t, 16> reduced = mix[parallel].template select<16, 4>(0);
            reduced = (reduced * FNV_PRIME) ^ mix[parallel].template select<16, 4>(1);
            reduced = (reduced * FNV_PRIME) ^ mix[parallel].template select<16, 4>(2);
            reduced = (reduced * FNV_PRIME) ^ mix[parallel].template select<16, 4>(3);
            octopus_esimd::simd<uint32_t, 8> compressed = reduced.select<8, 1>(0);
            compressed = (compressed * FNV_PRIME) ^ reduced.select<8, 1>(8);
#pragma unroll
            for (unsigned word = 0; word < 4; ++word) {
              const uint64_t low = compressed[word * 2];
              const uint64_t high = compressed[word * 2 + 1];
              final_state[8 + word][parallel] = low | (high << 32);
            }
          }
          final_state[12] = Word(1);
          final_state[16] = Word(0x8000000000000000ULL);
          native_keccak_esimd<HashesPerThread>(final_state);

#pragma unroll
          for (unsigned parallel = 0; parallel < HashesPerThread; ++parallel) {
            const uint64_t nonce = (start & ~31ULL) + first_nonce + parallel;
            uint8_t output[HASH_LEN];
#pragma unroll
            for (unsigned word = 0; word < 4; ++word) {
              const uint64_t value = final_state[word][parallel];
              store32(output + word * 8, static_cast<uint32_t>(value));
              store32(output + word * 8 + 4, static_cast<uint32_t>(value >> 32));
            }
            const bool active = nonce >= start && nonce - start < count;
            if (active && (is_test ? nonce == start : meets_target(output, target))) {
              const octopus_esimd::simd<uint32_t, 1> old_index =
                  octopus_esimd::atomic_update<octopus_esimd::atomic_op::add>(
                      &result->count, octopus_esimd::simd<uint32_t, 1>(0),
                      octopus_esimd::simd<uint32_t, 1>(1), octopus_esimd::simd_mask<1>(1));
              const uint32_t index = old_index[0];
              if (index < MAX_RESULTS) {
                result->nonce[index] = nonce;
#pragma unroll
                for (unsigned byte = 0; byte < HASH_LEN; ++byte)
                  result->hash[index][byte] = output[byte];
              }
            }
          }
        });
  });
}

#endif  // defined(OCTOPUS_ESIMD)
