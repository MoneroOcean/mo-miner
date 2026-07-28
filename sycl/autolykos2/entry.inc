// sycl_cleanup() clears the device-owning values before the compiler runtime unloads. Keep only
// this now-empty registry shell alive so C++ static destruction cannot race runtime DLL teardown.
static AutolykosStateMap& autolykos_states() {
  static AutolykosStateMap* const states = new AutolykosStateMap;
  return *states;
}

static std::mutex& autolykos_states_mutex() {
  static std::mutex* const mutex = new std::mutex;
  return *mutex;
}

static AutolykosState& autolykos_state(const std::string& dev_str) {
  std::lock_guard<std::mutex> lock(autolykos_states_mutex());
  auto& state = autolykos_states()[dev_str];
  if (!state) state = std::make_unique<AutolykosState>(dev_str);
  return *state;
}

void autolykos2_cleanup_states() noexcept {
  try {
    std::lock_guard<std::mutex> lock(autolykos_states_mutex());
    autolykos_states().clear();
  } catch (...) {
    std::fprintf(stderr, "autolykos2: ordered SYCL cleanup failed\n");
  }
}

// Return the last winning share found by a mining kernel (0 = none). The kernels may stash up
// to MAX_AUTOLYKOS_OUTPUTS hits but the API reports a single nonce/hash, so pick the last slot.
static int take_autolykos_result(const AutolykosResult* result, uint8_t* output, uint64_t* pnonce) {
  const uint32_t count = result->count;
  if (count == 0) return 0;
  const uint32_t index = std::min(count, MAX_AUTOLYKOS_OUTPUTS) - 1;
  *pnonce = result->nonce[index];
  std::memcpy(output, result->output[index], HASH_LEN);
  return 1;
}

} // namespace mom_autolykos2

using namespace mom_autolykos2;

int autolykos2(
  const unsigned, const uint32_t height, const uint8_t* const input, const unsigned input_size,
  uint8_t* const output, uint64_t* const pnonce, const uint8_t* const target,
  const unsigned intensity, const bool is_test, const bool is_benchmark, const std::string& dev_str
) {
  if (input_size < 40) throw std::string("Bad autolykos2 input length");

  AutolykosState& state = autolykos_state(dev_str);
  std::lock_guard<std::mutex> state_lock(state.mutex);
  state.ensure_result();
  const uint32_t n_len = calc_n(height);
  const uint64_t n_len_M = mo_modM(n_len);   // Lemire magic for the hot `idx % n_len` (see mo_mod_u32)
  const uint64_t n_len_divM = mo_divM64(n_len);
  state.ensure_table(height, n_len, is_test, !is_benchmark);

  AutolykosJobData job{};
  std::memcpy(job.message, input, HASH_LEN);
  std::memcpy(job.target, target, HASH_LEN);

  uint64_t start_nonce = 0;
  std::memcpy(&start_nonce, input + HASH_LEN, sizeof(start_nonce));
  const uint32_t effective_intensity = is_test ? 1 : intensity;
  const uint32_t global_size = round_up(std::max(effective_intensity, state.workgroup), state.workgroup);
  if constexpr (mom_sycl_portable_opencl) {
    state.ensure_portable_results(effective_intensity);
    AutolykosPortableResult empty_portable_result{};
    sycl_wait_and_throw(state.portable_results.fill(state.queue, empty_portable_result), state.device);
  } else {
    std::memset(state.result, 0, sizeof(AutolykosResult));
  }

  sycl::queue& q = state.queue;
  auto& kb = *state.bundle;
  const uint32_t* const __restrict__ d_table = state.table;
  // Split-table kernels are unreachable in this profile, but retain a well-typed placeholder so
  // the common source remains compilable and device-code splitting can discard those images.
  AutolykosResult* const d_result = mom_sycl_portable_opencl ? nullptr : state.result;
  const uint32_t local_size = state.workgroup;
  const bool profile = std::getenv("MOM_AUTOLYKOS2_PROFILE") != nullptr;

  if (d_table && !is_test && AutolykosState::autolykos_split_table()) {
    state.ensure_bhashes(effective_intensity);
    uint32_t* const d_bhashes = state.bhashes;

    const uint64_t profile_t0 = profile ? now_us() : 0;
    sycl::event index_event = q.submit([&](sycl::handler& h) {
      mom_use_bundle(h, kb);
      h.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(local_size)),
        [=](sycl::nd_item<1> item) {
          const uint32_t gid = item.get_global_id(0);
          if (gid >= effective_intensity) return;

          uint8_t index_hash[32];
          autolykos_index_hash_dev(
            job.message, start_nonce + gid, d_table, height, n_len, n_len_divM, index_hash);

          uint32_t* const out = d_bhashes + static_cast<uint64_t>(gid) * 8U;
#pragma unroll
          for (unsigned i = 0; i < 8; ++i) out[i] = load32_be_dev(index_hash + i * 4);
        }
      );
    });

    if (profile) sycl_wait_and_throw(index_event, state.device);
    const uint64_t profile_t1 = profile ? now_us() : 0;
    const bool subgroup_coop = (mom_is_hip(state.device) || mom_is_cuda(state.device)) &&
                               AutolykosState::autolykos_subgroup_coop();
    const uint32_t verify_count = subgroup_coop && AutolykosState::autolykos_subgroup_coop_verify() &&
                                  !state.subgroup_coop_verified
      ? std::min<uint32_t>(effective_intensity, 256U) : 0U;
    uint32_t* const verify_sums = verify_count
      ? sycl::malloc_device<uint32_t>(static_cast<size_t>(verify_count) * 8U, q) : nullptr;
    if (verify_count && !verify_sums)
      throw std::string("Can't allocate Autolykos2 subgroup verification buffer");
    sycl::event search_event;
    if (subgroup_coop) {
      const uint32_t coop_global = round_up(effective_intensity * 8U, local_size);
      search_event = q.submit([&](sycl::handler& h) {
        mom_use_bundle(h, kb);
        h.parallel_for(
          sycl::nd_range<1>(sycl::range<1>(coop_global), sycl::range<1>(local_size)),
          [=](sycl::nd_item<1> item) {
            const uint32_t thread = item.get_global_id(0);
            const uint32_t gid = thread >> 3;
            const bool active = gid < effective_intensity;
            const auto sg = item.get_sub_group();
            const uint32_t lane = static_cast<uint32_t>(sg.get_local_linear_id());
            const uint32_t team_lane = lane & 7U;
            const uint32_t team_base = lane - team_lane;

            // One lane owns one big-endian limb. The eight hash words are loaded coalesced once, then
            // native subgroup permutes broadcast the two words needed for each sliding 32-bit index.
            const uint32_t word = active
              ? d_bhashes[static_cast<uint64_t>(gid) * 8U + team_lane] : 0U;
            uint64_t partial[4] = {};
#pragma unroll
            for (unsigned k = 0; k < K_LEN; ++k) {
              const unsigned wi = k >> 2;
              const uint32_t word0 = mom_select_from_group(sg, word, team_base + wi);
              const uint32_t word1 = mom_select_from_group(sg, word, team_base + ((wi + 1U) & 7U));
              const uint32_t idx = autolykos_index_from_words_dev(
                word0, word1, k & 3U, n_len, n_len_M);
              if (active) {
                partial[k & 3U] += d_table[static_cast<uint64_t>(idx) * 8U + team_lane];
              }
            }
            const uint64_t column = (partial[0] + partial[1]) + (partial[2] + partial[3]);

            // Propagate the carry from least-significant limb 7 toward limb 0. Broadcasting one
            // 64-bit column at each of eight steps keeps every group operation converged.
            uint32_t carry = 0;
            uint32_t output_limb = 0;
#pragma unroll
            for (unsigned step = 0; step < 8; ++step) {
              const uint32_t owner = team_base + 7U - step;
              if (team_lane == 7U - step) {
                const uint64_t with_carry = column + carry;
                output_limb = static_cast<uint32_t>(with_carry);
                carry = static_cast<uint32_t>(with_carry >> 32);
              }
              // Every lane needs the carry for the next, more-significant owner. The column itself
              // is already resident on its owner, so broadcasting its low/high halves is redundant.
              carry = mom_select_from_group(sg, carry, owner);
            }

            uint32_t sum[8];
#pragma unroll
            for (unsigned i = 0; i < 8; ++i) {
              const uint32_t value = mom_select_from_group(sg, output_limb, team_base + i);
              if (team_lane == 0) sum[i] = value;
            }
            if (!active || team_lane != 0) return;
            if (gid < verify_count) {
#pragma unroll
              for (unsigned i = 0; i < 8; ++i)
                verify_sums[static_cast<uint64_t>(gid) * 8U + i] = sum[i];
            }

            uint8_t final_input[32];
#pragma unroll
            for (unsigned i = 0; i < 8; ++i) store32_be_dev(final_input + i * 4, sum[i]);
            uint8_t final_hash[32];
            blake2b256_oneblock_dev(final_input, sizeof(final_input), final_hash);
            if (meets_target_dev(final_hash, job.target))
              store_autolykos_result(d_result, start_nonce + gid, final_hash);
          }
        );
      });
    } else {
      search_event = q.submit([&](sycl::handler& h) {
        mom_use_bundle(h, kb);
        h.parallel_for(
          sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(local_size)),
          [=](sycl::nd_item<1> item) {
            const uint32_t gid = item.get_global_id(0);
            if (gid >= effective_intensity) return;

            const uint32_t* const hash_words = d_bhashes + static_cast<uint64_t>(gid) * 8U;
            uint32_t words[9];
#pragma unroll
            for (unsigned i = 0; i < 8; ++i) words[i] = hash_words[i];
            words[8] = words[0];

            uint32_t sum[8] = {};
            // Unroll the K_LEN table-row reads so the independent random loads issue
            // in parallel (memory-level parallelism) and hide DRAM latency instead
            // of stalling one-at-a-time (the read loop is the autolykos2 bottleneck).
#pragma unroll
            for (unsigned k = 0; k < K_LEN; ++k) {
              const unsigned word_index = k >> 2;
              uint32_t idx;
              switch (k & 3U) {
                case 0: idx = words[word_index]; break;
                case 1: idx = (words[word_index] << 8) | (words[word_index + 1] >> 24); break;
                case 2: idx = (words[word_index] << 16) | (words[word_index + 1] >> 16); break;
                default: idx = (words[word_index] << 24) | (words[word_index + 1] >> 8); break;
              }
              idx = mo_mod_u32(idx, n_len, n_len_M);
              add_limbs_dev(sum, d_table + static_cast<uint64_t>(idx) * (TABLE_ENTRY_BYTES / sizeof(uint32_t)));
            }

            uint8_t final_input[32];
#pragma unroll
            for (unsigned i = 0; i < 8; ++i) store32_be_dev(final_input + i * 4, sum[i]);

            uint8_t final_hash[32];
            blake2b256_oneblock_dev(final_input, sizeof(final_input), final_hash);
            if (meets_target_dev(final_hash, job.target))
              store_autolykos_result(d_result, start_nonce + gid, final_hash);
          }
        );
      });
    }
    sycl_wait_and_throw(search_event, state.device);
    if (verify_count) {
      uint32_t* const mismatches = sycl::malloc_shared<uint32_t>(1, q);
      if (!mismatches) {
        sycl::free(verify_sums, q);
        throw std::string("Can't allocate Autolykos2 subgroup verification counter");
      }
      *mismatches = 0;
      const uint32_t verify_global = round_up(verify_count, local_size);
      sycl_wait_and_throw(q.submit([&](sycl::handler& h) {
        mom_use_bundle(h, kb);
        h.parallel_for(
          sycl::nd_range<1>(sycl::range<1>(verify_global), sycl::range<1>(local_size)),
          [=](sycl::nd_item<1> item) {
            const uint32_t gid = item.get_global_id(0);
            if (gid >= verify_count) return;
            const uint32_t* const hash_words = d_bhashes + static_cast<uint64_t>(gid) * 8U;
            uint32_t words[9];
#pragma unroll
            for (unsigned i = 0; i < 8; ++i) words[i] = hash_words[i];
            words[8] = words[0];
            uint32_t expected[8] = {};
#pragma unroll
            for (unsigned k = 0; k < K_LEN; ++k) {
              const unsigned wi = k >> 2;
              uint32_t idx;
              switch (k & 3U) {
                case 0: idx = words[wi]; break;
                case 1: idx = (words[wi] << 8) | (words[wi + 1] >> 24); break;
                case 2: idx = (words[wi] << 16) | (words[wi + 1] >> 16); break;
                default: idx = (words[wi] << 24) | (words[wi + 1] >> 8); break;
              }
              idx = mo_mod_u32(idx, n_len, n_len_M);
              add_limbs_dev(expected, d_table + static_cast<uint64_t>(idx) * 8U);
            }
            bool differs = false;
#pragma unroll
            for (unsigned i = 0; i < 8; ++i)
              differs |= expected[i] != verify_sums[static_cast<uint64_t>(gid) * 8U + i];
            if (differs) {
              sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                               sycl::access::address_space::global_space>(*mismatches).fetch_add(1U);
            }
          }
        );
      }), state.device);
      const uint32_t mismatch_count = *mismatches;
      sycl::free(mismatches, q);
      sycl::free(verify_sums, q);
      if (mismatch_count)
        throw std::string("Autolykos2 subgroup verification failed: ") +
              std::to_string(mismatch_count) + " of " + std::to_string(verify_count) +
              " accumulator sums differ";
      state.subgroup_coop_verified = true;
      std::fprintf(stderr, "Autolykos2 subgroup sums verified for %u nonces\n", verify_count);
    }
    if (profile) {
      const uint64_t profile_t2 = now_us();
      std::fprintf(stderr,
        "AUTOLYKOS2_PROFILE split%s intensity=%u local=%u index=%.3fms search=%.3fms total=%.3fms\n",
        subgroup_coop ? "-subgroup-coop" : "", effective_intensity, local_size,
        static_cast<double>(profile_t1 - profile_t0) / 1000.0,
        static_cast<double>(profile_t2 - profile_t1) / 1000.0,
        static_cast<double>(profile_t2 - profile_t0) / 1000.0);
    }

    if constexpr (mom_sycl_portable_opencl) {
      throw std::string("Autolykos2 portable OpenCL table path is unavailable");
    } else {
      return take_autolykos_result(state.result, output, pnonce);
    }
  }

  const uint64_t profile_t0 = profile ? now_us() : 0;
  sycl::event search_event;
#if defined(MOM_SYCL_PORTABLE_OPENCL)
    search_event = q.submit([&](sycl::handler& h) {
      auto results = state.portable_results.device_view<sycl::access_mode::write>(h);
      h.parallel_for(sycl::range<1>{effective_intensity}, [=](sycl::id<1> item) {
        const uint32_t gid = item[0];
        const uint64_t nonce = start_nonce + gid;
        uint8_t index_hash[32];
        autolykos_index_hash_portable_dev(
          job.message, nonce, height, n_len, n_len_divM, index_hash);

        uint32_t sum[8] = {};
#pragma unroll
        for (unsigned k = 0; k < K_LEN; ++k) {
          const uint32_t idx = mo_mod_u32(
            (static_cast<uint32_t>(index_hash[(k + 0) & 31U]) << 24) |
            (static_cast<uint32_t>(index_hash[(k + 1) & 31U]) << 16) |
            (static_cast<uint32_t>(index_hash[(k + 2) & 31U]) << 8) |
             static_cast<uint32_t>(index_hash[(k + 3) & 31U]), n_len, n_len_M);
          uint8_t digest[32];
          autolykos_prehash_digest_dev(idx, height, digest);
          uint32_t decoded[8];
          digest_to_limbs_dev(digest, decoded);
          add_limbs_dev(sum, decoded);
        }

        uint8_t final_input[32];
#pragma unroll
        for (unsigned i = 0; i < 8; ++i) store32_be_dev(final_input + i * 4, sum[i]);
        uint8_t final_hash[32];
        blake2b256_oneblock_dev(final_input, sizeof(final_input), final_hash);

        AutolykosPortableResult candidate{};
        if (is_test || meets_target_dev(final_hash, job.target)) {
          candidate.found = 1;
          candidate.nonce = nonce;
#pragma unroll
          for (unsigned i = 0; i < HASH_LEN; ++i) candidate.output[i] = final_hash[i];
        }
        results[gid] = candidate;
      });
    });
#else
    search_event = q.submit([&](sycl::handler& h) {
      mom_use_bundle(h, kb);
      sycl::local_accessor<uint32_t, 1> shared_index(sycl::range<1>(64), h);
      sycl::local_accessor<uint32_t, 1> shared_data(sycl::range<1>(512), h);
      h.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(local_size)),
        [=](sycl::nd_item<1> item) {
        const uint32_t gid = item.get_global_id(0);
        const uint32_t local_id = item.get_local_id(0);
        const bool active = gid < effective_intensity;

        const uint64_t nonce = start_nonce + gid;
        uint32_t indices[K_LEN];
#pragma unroll
        for (unsigned i = 0; i < K_LEN; ++i) indices[i] = 0;

        if (active) {
          uint8_t index_hash[32];
          autolykos_index_hash_dev(job.message, nonce, d_table, height, n_len, n_len_divM, index_hash);

#pragma unroll
          for (unsigned k = 0; k < K_LEN; ++k) {
            indices[k] = mo_mod_u32(
              ((static_cast<uint32_t>(index_hash[(k + 0) & 31U]) << 24) |
               (static_cast<uint32_t>(index_hash[(k + 1) & 31U]) << 16) |
               (static_cast<uint32_t>(index_hash[(k + 2) & 31U]) << 8) |
                static_cast<uint32_t>(index_hash[(k + 3) & 31U])), n_len, n_len_M);
          }
        }

        uint32_t sum[8] = {};

        if (d_table && local_size == 64) {
          const uint32_t thread_id = local_id & 7U;
          const uint32_t hash_id = local_id >> 3;
          for (unsigned k = 0; k < K_LEN; ++k) {
            shared_index[local_id] = indices[k];
            item.barrier(sycl::access::fence_space::local_space);

#pragma unroll
            for (unsigned group = 0; group < 8; ++group) {
              const uint32_t source_lane = hash_id + group * 8U;
              const uint32_t* const table_item =
                d_table + static_cast<uint64_t>(shared_index[source_lane]) * (TABLE_ENTRY_BYTES / sizeof(uint32_t));
              shared_data[(group * 64U) + (hash_id * 8U) + thread_id] = table_item[thread_id];
            }
            item.barrier(sycl::access::fence_space::local_space);

            uint64_t carry = 0;
            for (int i = 7; i >= 0; --i) {
              const uint64_t v = static_cast<uint64_t>(sum[i]) + shared_data[local_id * 8U + i] + carry;
              sum[i] = static_cast<uint32_t>(v);
              carry = v >> 32;
            }
            item.barrier(sycl::access::fence_space::local_space);
          }
        } else {
          if (!active) return;
          for (unsigned k = 0; k < K_LEN; ++k) {
            uint32_t limbs[8];
            table_item_limbs_dev(d_table, indices[k], height, limbs);
            add_limbs_dev(sum, limbs);
          }
        }

        if (!active) return;

        uint8_t final_input[32];
#pragma unroll
        for (unsigned i = 0; i < 8; ++i) store32_be_dev(final_input + i * 4, sum[i]);

        uint8_t final_hash[32];
        blake2b256_oneblock_dev(final_input, sizeof(final_input), final_hash);
        if (is_test || meets_target_dev(final_hash, job.target)) {
          store_autolykos_result(d_result, nonce, final_hash);
        }
        }
      );
    });
#endif
  sycl_wait_and_throw(search_event, state.device);
  if (profile) {
    const uint64_t profile_t1 = now_us();
    std::fprintf(stderr,
      "AUTOLYKOS2_PROFILE single intensity=%u local=%u total=%.3fms table=%u\n",
      effective_intensity, local_size, static_cast<double>(profile_t1 - profile_t0) / 1000.0,
      d_table ? 1U : 0U);
  }

#if defined(MOM_SYCL_PORTABLE_OPENCL)
    std::vector<AutolykosPortableResult> portable_results(effective_intensity);
    state.portable_results.read(portable_results.data(), portable_results.size());
    for (const AutolykosPortableResult& candidate : portable_results) {
      if (!candidate.found) continue;
      *pnonce = candidate.nonce;
      std::memcpy(output, candidate.output, HASH_LEN);
      return 1;
    }
    return 0;
#else
  return take_autolykos_result(state.result, output, pnonce);
#endif
}
