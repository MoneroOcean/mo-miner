int etchash(
  const unsigned, const uint32_t block_height, const uint8_t* const input, const unsigned input_size, uint8_t* const output,
  uint8_t* const mix_hash, uint64_t* const pnonce, const uint8_t* const target, const uint8_t* const seed_hash,
  const unsigned intensity, const bool is_test, const bool is_benchmark, const std::string& dev_str
) {
  if (input_size < 40) throw std::string("Bad etchash input length");

  EtchashState& state = etchash_state(dev_str);
  std::lock_guard<std::mutex> state_lock(state.mutex);
  state.ensure_input(input_size);
  uint32_t epoch = 0;
  uint32_t seed_epoch = 0;
  resolve_epochs(block_height, seed_hash, epoch, seed_epoch);
  state.ensure_epoch(epoch, seed_epoch, !is_benchmark);

  uint64_t start_nonce = 0;
  std::memcpy(&start_nonce, input + 32, sizeof(start_nonce));
  const uint32_t dag_pages = static_cast<uint32_t>(dag_sizes[epoch] / ETHASH_MIX_BYTES);
  const FastModData dag_mod = make_fast_mod_data(dag_pages);

  if (state.shared_io) {
    std::memcpy(state.input, input, input_size);
    std::memcpy(state.target, target, HASH_LEN);
  } else {
    state.queue.memcpy(state.input, input, input_size);
    state.queue.memcpy(state.target, target, HASH_LEN);
  }
  std::memset(state.result, 0, sizeof(EtchashResult));

  sycl::queue& q = state.queue;
  auto& kb = *state.bundle;
  const uint8_t* const d_input = state.input;
  const uint8_t* const d_target = state.target;
  const uint32_t* const __restrict__ d_dag = state.dag;
  EtchashResult* const d_result = state.result;
  constexpr unsigned ETCHASH_LANES = ETHASH_MIX_WORDS;
  constexpr unsigned SEED_OFFSET = 0;
  constexpr unsigned PAGE_OFFSET = SEED_OFFSET + 16;
  constexpr unsigned MIX_OFFSET = PAGE_OFFSET + 2;
  constexpr unsigned CMIX_OFFSET = MIX_OFFSET + ETHASH_MIX_WORDS;
  constexpr unsigned SCRATCH_WORDS = CMIX_OFFSET + 8;

  if (state.device.is_gpu()) {
    // Pick the inlined (fast) or noinline (always-correct) pair-Keccak search kernel per device. The
    // gate is a cached one-time self-test (see EtchashState::use_inline_pair); DAG-gen above is always
    // noinline so the dataset is correct regardless.
    sycl_wait_and_throw(
      state.use_inline_pair()
        ? submit_etchash_search_gpu<true >(q, kb, d_input, start_nonce, d_dag, dag_mod, intensity, d_target, d_result, is_test)
        : submit_etchash_search_gpu<false>(q, kb, d_input, start_nonce, d_dag, dag_mod, intensity, d_target, d_result, is_test),
      state.device);
  } else {
    sycl_wait_and_throw(q.submit([&](sycl::handler& h) {
      mom_use_bundle(h, kb);
      sycl::local_accessor<uint32_t, 1> scratch(sycl::range<1>(SCRATCH_WORDS), h);
      h.parallel_for(
        sycl::nd_range<1>(
          sycl::range<1>(static_cast<size_t>(intensity) * ETCHASH_LANES),
          sycl::range<1>(ETCHASH_LANES)
        ),
        [=](sycl::nd_item<1> item) {
          const uint32_t gid = static_cast<uint32_t>(item.get_group(0));
          const uint32_t lane = static_cast<uint32_t>(item.get_local_id(0));

          const uint64_t nonce = start_nonce + gid;
          if (lane == 0) {
            uint32_t seed[16];
            keccak_512_header_nonce_dev(d_input, nonce, seed);
#pragma unroll
            for (unsigned w = 0; w < 16; ++w) scratch[SEED_OFFSET + w] = seed[w];
          }
          item.barrier(sycl::access::fence_space::local_space);

          const uint32_t seed0 = scratch[SEED_OFFSET];
          uint32_t mix_word = scratch[SEED_OFFSET + (lane & (ETHASH_NODE_WORDS - 1))];

          for (unsigned i = 0; i < ETHASH_ACCESSES; ++i) {
            if (lane == (i & (ETHASH_MIX_WORDS - 1))) {
              scratch[PAGE_OFFSET + (i & 1)] = fast_mod_dev(fnv_dev(seed0 ^ i, mix_word), dag_mod);
            }
            item.barrier(sycl::access::fence_space::local_space);
            const uint32_t page = scratch[PAGE_OFFSET + (i & 1)];
            const uint32_t base = page * ETHASH_MIX_WORDS;
            mix_word = fnv_dev(mix_word, d_dag[base + lane]);
          }

          scratch[MIX_OFFSET + lane] = mix_word;
          item.barrier(sycl::access::fence_space::local_space);

          if (lane < 8) {
            const uint32_t w = lane * 4;
            uint32_t reduction = scratch[MIX_OFFSET + w + 0];
            reduction = fnv_dev(reduction, scratch[MIX_OFFSET + w + 1]);
            reduction = fnv_dev(reduction, scratch[MIX_OFFSET + w + 2]);
            reduction = fnv_dev(reduction, scratch[MIX_OFFSET + w + 3]);
            scratch[CMIX_OFFSET + lane] = reduction;
          }
          item.barrier(sycl::access::fence_space::local_space);

          if (lane != 0) return;

          uint32_t seed[16];
          uint32_t compressed_mix[8];
#pragma unroll
          for (unsigned w = 0; w < 16; ++w) seed[w] = scratch[SEED_OFFSET + w];
#pragma unroll
          for (unsigned w = 0; w < 8; ++w) compressed_mix[w] = scratch[CMIX_OFFSET + w];

          uint8_t final_hash[32];
          keccak_256_seed_mix_dev(seed, compressed_mix, final_hash);
          if ((is_test && gid == 0) || meets_target_dev(final_hash, d_target)) {
            store_etchash_result(d_result, nonce, final_hash, compressed_mix);
          }
        }
      );
    }), state.device);
  }

  const uint32_t count = state.result->count;
  if (count == 0) return 0;
  const uint32_t index = std::min(count, MAX_ETCHASH_OUTPUTS) - 1;
  *pnonce = state.result->nonce[index];
  std::memcpy(output, state.result->output[index], HASH_LEN);
  std::memcpy(mix_hash, state.result->mix_hash[index], HASH_LEN);
  return 1;
}
