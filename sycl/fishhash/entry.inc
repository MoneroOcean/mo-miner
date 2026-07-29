int fishhash(
  const unsigned, const uint32_t, const uint8_t* const input, const unsigned input_size, uint8_t* const output,
  uint8_t* const mix_hash, uint64_t* const pnonce, const uint8_t* const target, const uint8_t* const /*seed_hash*/,
  const unsigned intensity, const bool is_test, const bool is_benchmark, const std::string& dev_str
) {
  if (input_size < 40) throw std::string("Bad fishhash input length");
  FishState& state = fishhash_state(dev_str);
  std::lock_guard<std::mutex> lock(state.mutex);
  state.ensure_io(input_size);
  // MOM_FISHHASH_FULL_TEST makes the normal one-nonce vector build/use the full DAG, so the
  // production cooperative kernel can be checked bit-exactly without slowing the default light-cache
  // vector. The cooperative search is the GPU default; MOM_FISHHASH_COOP=0 retains the scalar path
  // for diagnosis.
  const bool full_test = is_test && std::getenv("MOM_FISHHASH_FULL_TEST") != nullptr;
#if !defined(MOM_SYCL_PORTABLE_OPENCL)
  const char* const coop_env = std::getenv("MOM_FISHHASH_COOP");
  const bool cooperative = state.device.is_gpu() && !(coop_env && coop_env[0] == '0');
#endif
  if (is_test && !full_test) state.ensure_cache(!is_benchmark);  // lazy path: no 4.6-GiB allocation
  else state.ensure_dag(!is_benchmark);                          // mining/bench or explicit full test

  // Iron Fish (180B header) uses nonce@0 big-endian + BE target; the offline vector uses nonce@32 LE.
  const bool ironfish = input_size >= 140;
  const unsigned nonce_off = ironfish ? 0u : 32u;
  const bool nonce_be = ironfish, target_be = ironfish;
  uint64_t start_nonce = 0;
  if (nonce_be) { for (unsigned i = 0; i < 8; ++i) start_nonce = (start_nonce << 8) | input[nonce_off + i]; }
  else { std::memcpy(&start_nonce, input + nonce_off, sizeof(start_nonce)); }
  if (state.shared_io) { std::memcpy(state.input, input, input_size); std::memcpy(state.target, target, HASH_LEN); }
  else { state.queue.memcpy(state.input, input, input_size); state.queue.memcpy(state.target, target, HASH_LEN); }
  std::memset(state.result, 0, sizeof(FishResult));

  sycl::event search;
  if (is_test && !full_test) {
    search = submit_fishhash_search<false>(state.queue, *state.bundle, state.input, input_size, start_nonce,
      state.cache, nullptr, intensity, state.target, state.result, is_test,
      nonce_off, nonce_be, target_be);
#if !defined(MOM_SYCL_PORTABLE_OPENCL)
  } else if (cooperative) {
    search = submit_fishhash_search_coop(state.queue, *state.bundle, state.input, input_size, start_nonce,
      state.dag, intensity, state.target, state.result, is_test,
      nonce_off, nonce_be, target_be, state.coop_workgroup);
#endif
  } else {
    search = submit_fishhash_search<true>(state.queue, *state.bundle, state.input, input_size, start_nonce,
      nullptr, state.dag, intensity, state.target, state.result, is_test,
      nonce_off, nonce_be, target_be);
  }
  sycl_wait_and_throw(search, state.device);

  static const bool perf_log = std::getenv("MOM_FISHHASH_PERF") != nullptr;
  if (perf_log && !is_test) {
    static uint64_t acc = 0; static auto t0 = std::chrono::steady_clock::now();
    acc += intensity;
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (sec >= 2.0) { std::fprintf(stderr, "[fishhash] %.2f MH/s (%s)\n", static_cast<double>(acc) / sec / 1e6, dev_str.c_str()); acc = 0; t0 = std::chrono::steady_clock::now(); }
  }

  const uint32_t count = state.result->count;
  if (count == 0) return 0;
  const uint32_t idx = std::min(count, MAX_OUTPUTS) - 1;
  *pnonce = state.result->nonce[idx];
  std::memcpy(output, state.result->output[idx], HASH_LEN);
  if (mix_hash) std::memset(mix_hash, 0, HASH_LEN);
  return 1;
}

int karlsenhashv2(
  const unsigned, const uint32_t, const uint8_t* const input, const unsigned input_size, uint8_t* const output,
  uint8_t* const mix_hash, uint64_t* const pnonce, const uint8_t* const target, const uint8_t* const /*seed_hash*/,
  const unsigned intensity, const bool is_test, const bool is_benchmark, const std::string& dev_str
) {
  if (input_size < 80) throw std::string("Bad karlsenhashv2 input length");
  FishState& state = fishhash_state(dev_str);  // shares the FishHash 4.6 GB DAG (identical seed/params)
  std::lock_guard<std::mutex> lock(state.mutex);
  state.ensure_io(input_size);
  const bool full_test = is_test && std::getenv("MOM_FISHHASH_FULL_TEST") != nullptr;
#if !defined(MOM_SYCL_PORTABLE_OPENCL)
  const char* const coop_env = std::getenv("MOM_FISHHASH_COOP");
#endif
  if (is_test && !full_test) state.ensure_cache(!is_benchmark);
  else state.ensure_dag(!is_benchmark);

  uint64_t start_nonce = 0; std::memcpy(&start_nonce, input + 72, sizeof(start_nonce));  // nonce@72 LE
  if (state.shared_io) { std::memcpy(state.input, input, input_size); std::memcpy(state.target, target, HASH_LEN); }
  else { state.queue.memcpy(state.input, input, input_size); state.queue.memcpy(state.target, target, HASH_LEN); }
#if !defined(MOM_SYCL_PORTABLE_OPENCL)
  const bool cooperative = state.device.is_gpu() &&
    (coop_env ? coop_env[0] != '0' : (!is_test && select_karlsen_cooperative(state, input_size, start_nonce)));
#endif
  std::memset(state.result, 0, sizeof(FishResult));

  sycl::event search;
  if (is_test && !full_test) {
    search = submit_karlsenhashv2_search<false>(state.queue, *state.bundle, state.input, input_size,
      start_nonce, state.cache, nullptr, intensity, state.target, state.result, is_test);
#if !defined(MOM_SYCL_PORTABLE_OPENCL)
  } else if (cooperative) {
    search = submit_karlsenhashv2_search_coop(state.queue, *state.bundle, state.input, input_size,
      start_nonce, state.dag, intensity, state.target, state.result, is_test, state.coop_workgroup);
#endif
  } else {
    search = submit_karlsenhashv2_search<true>(state.queue, *state.bundle, state.input, input_size,
      start_nonce, nullptr, state.dag, intensity, state.target, state.result, is_test);
  }
  sycl_wait_and_throw(search, state.device);

  static const bool perf_log = std::getenv("MOM_KARLSEN_PERF") != nullptr;
  if (perf_log && !is_test) {
    static uint64_t acc = 0; static auto t0 = std::chrono::steady_clock::now();
    acc += intensity;
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (sec >= 2.0) { std::fprintf(stderr, "[karlsenhashv2] %.2f MH/s (%s)\n", static_cast<double>(acc) / sec / 1e6, dev_str.c_str()); acc = 0; t0 = std::chrono::steady_clock::now(); }
  }

  const uint32_t count = state.result->count;
  if (count == 0) return 0;
  const uint32_t idx = std::min(count, MAX_OUTPUTS) - 1;
  *pnonce = state.result->nonce[idx];
  std::memcpy(output, state.result->output[idx], HASH_LEN);
  if (mix_hash) std::memset(mix_hash, 0, HASH_LEN);
  return 1;
}
