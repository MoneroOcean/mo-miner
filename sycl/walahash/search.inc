// Portable and accelerated search submissions; in-order queues connect the staged routes.
template <bool Native, bool FixedHashes>
static sycl::event search(State& s, uint64_t first, unsigned count, bool is_test,
                          const unsigned work_group) {
  const size_t global = (static_cast<size_t>(count) + work_group - 1) / work_group * work_group;
  const uint8_t* matrix = s.matrix;
  const uint8_t* input = s.input;
#if defined(MOM_SYCL_HAS_CUDA) || defined(MOM_SYCL_ADAPTIVECPP_CUDA)
  const Blake3Cv prehash_cv = s.prehash_cv;
#endif
  const uint8_t* target = s.target;
  Result* result = s.result;
  return s.queue.submit([&](sycl::handler& h) {
    h.parallel_for<SearchKernel<Native, FixedHashes>>(
        sycl::nd_range<1>(global, work_group),
#ifdef MOM_SYCL_HAS_CUDA
        [=](sycl::nd_item<1> it) [[sycl::work_group_size_hint(128)]] MOM_SYCL_KERNEL_ARGS_RESTRICT {
#else
        [=](sycl::nd_item<1> it) MOM_SYCL_KERNEL_ARGS_RESTRICT {
#endif
          const unsigned gid = it.get_global_id(0);
          if (gid >= count)
            return;
          const uint64_t nonce = first + gid;
          uint32_t pow_words[8];
          uint8_t* pow = reinterpret_cast<uint8_t*>(pow_words);
#if defined(MOM_SYCL_HAS_CUDA) || defined(MOM_SYCL_ADAPTIVECPP_CUDA)
          if constexpr (FixedHashes) {
            b3_80_from_cv(pow_words, prehash_cv, input + 64, nonce);
            sha3_words64(pow_words);
          } else
#endif
          {
            uint8_t pre[80];
            for (unsigned i = 0; i < 72; ++i)
              pre[i] = input[i];
            for (unsigned i = 0; i < 8; ++i)
              pre[72 + i] = nonce >> (8 * i);
            blake3_dev(pow, 32, pre, 80);
            sha3_256(pow, pow);
          }
          uint32_t vec[16];
#pragma unroll
          for (unsigned k = 0; k < 16; ++k) {
            const unsigned j = 4 * k;
            vec[k] = static_cast<uint32_t>(pow[j >> 1] >> 4) |
                     static_cast<uint32_t>(pow[j >> 1] & 15) << 8 |
                     static_cast<uint32_t>(pow[(j + 2) >> 1] >> 4) << 16 |
                     static_cast<uint32_t>(pow[(j + 2) >> 1] & 15) << 24;
          }
          auto m4 = sycl::address_space_cast<sycl::access::address_space::global_space,
                                             sycl::access::decorated::no>(
              reinterpret_cast<const uint32_t*>(matrix));
          uint32_t mixed_words[8];
          uint8_t* mixed = reinterpret_cast<uint8_t*>(mixed_words);
          for (unsigned i = 0; i < 64; i += 4) {
            int32_t acc[4]{};
#pragma unroll
            for (unsigned k = 0; k < 16; ++k) {
              const uint32_t b = vec[k];
#pragma unroll
              for (unsigned r = 0; r < 4; ++r)
                acc[r] = dot4<Native>(m4[(i + r) * 16 + k], b, acc[r]);
            }
            uint8_t nibble[4];
            for (unsigned r = 0; r < 4; ++r)
              nibble[r] = (acc[r] & 15) ^ ((acc[r] >> 4) & 15) ^ ((acc[r] >> 8) & 15);
            mixed[i / 2] = pow[i / 2] ^ static_cast<uint8_t>(nibble[0] << 4 | nibble[1]);
            mixed[i / 2 + 1] = pow[i / 2 + 1] ^ static_cast<uint8_t>(nibble[2] << 4 | nibble[3]);
          }
#if defined(MOM_SYCL_HAS_CUDA) || defined(MOM_SYCL_ADAPTIVECPP_CUDA)
          if constexpr (FixedHashes)
            b3_32(pow_words, mixed_words);
          else
#endif
            blake3_dev(pow, 32, mixed, 32);
          bool hit = true;
          for (int i = 31; i >= 0; --i)
            if (pow[i] != target[i]) {
              hit = pow[i] < target[i];
              break;
            }
          if ((is_test && gid == 0) || hit) {
            using A =
                sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                 sycl::access::address_space::global_space>;
            if (A(result->count).fetch_add(1) == 0) {
              result->nonce = nonce;
              for (unsigned i = 0; i < 32; ++i)
                result->hash[i] = pow[i];
            }
          }
        });
  });
}

#ifdef WALAHASH_ESIMD
// Select only Intel matrix-capable non-OpenCL GPUs unless native execution is explicitly disabled.
static bool use_esimd(const sycl::device& d) {
  const char* configured = std::getenv("MOM_WALAHASH_SYCL_NATIVE");
  return (!configured || std::strcmp(configured, "0")) && d.is_gpu() &&
         d.has(sycl::aspect::ext_intel_matrix) && d.get_backend() != sycl::backend::opencl;
}
// Hash each nonce and pack its nibble vector into 16-nonce DPAS tiles.
static sycl::event prehash(State& s, uint64_t first, unsigned count) {
  const uint8_t* input = s.input;
  const Blake3Cv prehash_cv = s.prehash_cv;
  uint8_t* hashes = s.hashes;
  uint8_t* vectors = s.vectors;
  return s.queue.submit([&](sycl::handler& h) {
    h.parallel_for<PrehashKernel>(
        sycl::range<1>(count), [=](sycl::id<1> id) MOM_SYCL_KERNEL_ARGS_RESTRICT {
      const unsigned gid = id[0], lane = gid & 15, tile = gid >> 4;
      uint32_t pow[8];
      b3_80_from_cv(pow, prehash_cv, input + 64, first + gid);
      sha3_words(pow);
      auto hash32 = reinterpret_cast<uint32_t*>(hashes);
      for (unsigned i = 0; i < 8; ++i)
        hash32[(size_t)gid * 8 + i] = pow[i];
      for (unsigned k = 0; k < 64; ++k) {
        const uint8_t byte = pow[k >> 3] >> (8 * ((k >> 1) & 3)),
                      v = (k & 1) ? byte & 15 : byte >> 4;
        vectors[(size_t)tile * 1024 + (k >> 2) * 64 + lane * 4 + (k & 3)] = v;
      }
    });
  });
}
// Multiply packed vectors by the job matrix and fold nibble products into the original hashes.
template <unsigned Width> static sycl::event multiply_esimd(State& s, unsigned count) {
  const int8_t* matrix = reinterpret_cast<int8_t*>(s.matrix);
  const int8_t* vectors = reinterpret_cast<int8_t*>(s.vectors);
  const uint8_t* hashes = s.hashes;
  uint8_t* mixed = s.mixed;
  const unsigned stride = s.capacity;
  return s.queue.submit([&](sycl::handler& h) {
    h.parallel_for(
        sycl::range<1>((count + 15) / 16),
        [=](sycl::id<1> id) SYCL_ESIMD_KERNEL MOM_SYCL_KERNEL_ARGS_RESTRICT {
      const size_t tile = id[0];
      for (unsigned band = 0; band < 8; ++band) {
        esimd::simd<int32_t, 128> acc = 0;
        for (unsigned chunk = 0; chunk < 2; ++chunk) {
          auto a = esimd::block_load<int8_t, 256>(matrix + (band * 2 + chunk) * 256);
          auto b = esimd::block_load<int8_t, 512>(vectors + tile * 1024 + chunk * 512);
          acc = mom::intel_matrix::dpas_8x16x32<Width>(acc, b, a);
        }
        esimd::simd<uint32_t, 128> nibble = (acc & 15) ^ ((acc >> 4) & 15) ^ ((acc >> 8) & 15);
        const esimd::simd<uint32_t, 16> hash_offsets(0, 32);
        for (unsigned pair = 0; pair < 4; ++pair) {
          const unsigned byte = band * 4 + pair;
          const auto hi = nibble.template select<16, 1>(pair * 32),
                     lo = nibble.template select<16, 1>(pair * 32 + 16);
          const esimd::simd<uint8_t, 16> packed = esimd::convert<uint8_t>((hi << 4) | lo);
          const auto original =
              esimd::gather<uint8_t, 16>(hashes + tile * 512 + byte, hash_offsets);
          esimd::block_store<uint8_t, 16>(mixed + (size_t)byte * stride + tile * 16,
                                          original ^ packed);
        }
      }
    });
  });
}
// Hash the ESIMD mixed output and atomically publish one qualifying nonce.
static sycl::event finalize(State& s, uint64_t first, unsigned count, bool is_test) {
  const uint8_t* mixed = s.mixed;
  const unsigned stride = s.capacity;
  const uint8_t* target = s.target;
  Result* result = s.result;
  return s.queue.submit([&](sycl::handler& h) {
    h.parallel_for<FinalKernel>(
        sycl::range<1>(count), [=](sycl::id<1> id) MOM_SYCL_KERNEL_ARGS_RESTRICT {
      const unsigned gid = id[0];
      uint32_t block[8], pow[8];
      for (unsigned i = 0; i < 32; ++i)
        reinterpret_cast<uint8_t*>(block)[i] = mixed[(size_t)i * stride + gid];
      b3_32(pow, block);
      bool hit = true;
      for (int i = 31; i >= 0; --i) {
        const uint8_t byte = pow[i >> 2] >> (8 * (i & 3));
        if (byte != target[i]) {
          hit = byte < target[i];
          break;
        }
      }
      if ((is_test && gid == 0) || hit) {
        using A =
            sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                             sycl::access::address_space::global_space>;
        if (A(result->count).fetch_add(1) == 0) {
          result->nonce = first + gid;
          for (unsigned i = 0; i < 32; ++i)
            result->hash[i] = pow[i >> 2] >> (8 * (i & 3));
        }
      }
    });
  });
}
#endif

#ifdef MOM_SYCL_HAS_HIP
// Hash each nonce into the row-major nibble-vector layout consumed by HIP WMMA.
static sycl::event prehash_amd(State& s, uint64_t first, unsigned count) {
  const uint8_t* input = s.input;
  const Blake3Cv prehash_cv = s.prehash_cv;
  uint8_t* hashes = s.hashes;
  return s.queue.submit([&](sycl::handler& h) {
    h.parallel_for<AmdPrehashKernel>(
        sycl::range<1>(count), [=](sycl::id<1> id) MOM_SYCL_KERNEL_ARGS_RESTRICT {
      const unsigned gid = id[0];
      uint32_t pow[8];
      b3_80_from_cv(pow, prehash_cv, input + 64, first + gid);
      sha3_words(pow);
      const uint8_t* bytes = reinterpret_cast<const uint8_t*>(pow);
      for (unsigned i = 0; i < 32; ++i)
        hashes[(size_t)gid * 32 + i] = bytes[i];
    });
  });
}

// Fold HIP WMMA products into the original hashes before the final target check.
static sycl::event finalize_amd(State& s, uint64_t first, unsigned count, bool is_test) {
  const uint8_t* mixed = s.mixed;
  const uint8_t* target = s.target;
  Result* result = s.result;
  return s.queue.submit([&](sycl::handler& h) {
    h.parallel_for<AmdFinalKernel>(
        sycl::range<1>(count), [=](sycl::id<1> id) MOM_SYCL_KERNEL_ARGS_RESTRICT {
      const unsigned gid = id[0];
      uint32_t pow_words[8];
      uint8_t* pow = reinterpret_cast<uint8_t*>(pow_words);
      uint32_t block_words[8];
      uint8_t* block = reinterpret_cast<uint8_t*>(block_words);
      for (unsigned i = 0; i < 32; ++i)
        block[i] = mixed[(size_t)gid * 32 + i];
      b3_32(pow_words, block_words);
      bool hit = true;
      for (int i = 31; i >= 0; --i)
        if (pow[i] != target[i]) {
          hit = pow[i] < target[i];
          break;
        }
      if ((is_test && gid == 0) || hit) {
        using A =
            sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                             sycl::access::address_space::global_space>;
        if (A(result->count).fetch_add(1) == 0) {
          result->nonce = first + gid;
          for (unsigned i = 0; i < 32; ++i)
            result->hash[i] = pow[i];
        }
      }
    });
  });
}
#endif
