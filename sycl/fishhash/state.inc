class FishState {
public:
  sycl::device device; sycl::queue queue; std::unique_ptr<MomKernelBundle> bundle; bool shared_io;
  uint8_t* input = nullptr; uint8_t* target = nullptr; uint64_t* cache = nullptr; uint64_t* dag = nullptr; FishResult* result = nullptr;
  unsigned input_cap = 0; unsigned coop_workgroup; bool cache_built = false; bool dag_built = false;
  int karlsen_cooperative = -1;
  std::mutex mutex;
  explicit FishState(const std::string& dev_str)
    : device(get_dev(dev_str)), queue(device, sycl::property_list{sycl::property::queue::in_order{}}),
      shared_io(device.is_cpu() || !mom_has_usm_device(device)),
      coop_workgroup(cooperative_workgroup(device)) {
    if (!mom_has_usm_shared(device) || (!device.is_cpu() && !mom_has_usm_device(device)))
      throw std::string("fishhash SYCL device does not support required allocations");
    bundle = std::make_unique<MomKernelBundle>(mom_get_exec_bundle(queue.get_context()));
  }
  static unsigned cooperative_workgroup(const sycl::device& dev) {
    // Both CUDA compiler paths peak at 512 on the RTX 5060 Ti (3.2% over 128 for AdaptiveCpp and
    // 5.1% for DPC++). Intel and AMD are flat-to-slower above 128. Clamp against the device limit.
    const unsigned preferred = mom_is_cuda(dev) ? 512 : 128;
    const unsigned fallback = sycl_default_workgroup(dev, {64, 128, 256, 512}, preferred);
    unsigned long parsed = 0;
    if (!mom_parse_env_ulong("MOM_FISHHASH_WORKGROUP", parsed)) return fallback;
    for (const unsigned candidate : {64u, 128u, 256u, 512u})
      if (parsed == candidate) return candidate;
    return fallback;
  }
  ~FishState() { sycl_cleanup_noexcept("fishhash", [&] { queue.wait_and_throw(); free_all(); }); }
  template <typename T> T* alloc(size_t n) { return shared_io ? sycl::malloc_shared<T>(n, queue) : sycl::malloc_device<T>(n, queue); }
  void free_ptr(auto*& p) { if (p) sycl::free(p, queue); p = nullptr; }
  void free_all() { free_ptr(input); free_ptr(target); free_ptr(cache); free_ptr(dag); free_ptr(result); }
  void ensure_io(unsigned in_sz) {
    if (input && in_sz <= input_cap && target && result) return;
    queue.wait_and_throw(); free_ptr(input); free_ptr(target); free_ptr(result);
    input_cap = in_sz < 180 ? 180 : in_sz;
    input = alloc<uint8_t>(input_cap); target = alloc<uint8_t>(HASH_LEN); result = sycl::malloc_shared<FishResult>(1, queue);
    if (!input || !target || !result) throw std::string("Can't allocate fishhash buffers");
  }
  void ensure_cache(bool log) {
    if (cache_built) return;
    const uint64_t t0 = now_ms();
    std::vector<uint64_t> hcache; build_light_cache_host(hcache);
    free_ptr(cache);  // release a partial buffer from a previously-failed attempt before re-allocating
    cache = alloc<uint64_t>(hcache.size());
    if (!cache) throw std::string("Can't allocate fishhash light cache");
    if (shared_io) std::memcpy(cache, hcache.data(), hcache.size() * sizeof(uint64_t));
    else sycl_wait_and_throw(queue.memcpy(cache, hcache.data(), hcache.size() * sizeof(uint64_t)), device);
    cache_built = true;
    if (log) std::fprintf(stderr, "FishHash light cache (72 MiB) built (%llu ms)\n", (unsigned long long)(now_ms() - t0));
  }
  void ensure_dag(bool log) {
    if (dag_built) return;
    ensure_cache(log);
    const uint64_t t0 = now_ms();
    free_ptr(dag);  // release a partial buffer from a previously-failed attempt before re-allocating
    dag = alloc<uint64_t>((size_t)FULL_DATASET_NUM_ITEMS * 16);  // 4.6 GiB
    if (!dag) throw std::string("Can't allocate fishhash DAG (4.6 GiB)");
    const uint32_t CHUNK = 1u << 21;  // 2M items/launch to avoid the GPU watchdog
    for (uint32_t s = 0; s < (uint32_t)FULL_DATASET_NUM_ITEMS; s += CHUNK) {
      const uint32_t c = std::min(CHUNK, (uint32_t)FULL_DATASET_NUM_ITEMS - s);
      sycl_wait_and_throw(submit_fishhash_dag_gen(queue, *bundle, cache, dag, s, c), device);
    }
    dag_built = true;
    if (log) std::fprintf(stderr, "FishHash DAG (4.6 GiB) built (%llu ms)\n", (unsigned long long)(now_ms() - t0));
  }
  static uint64_t now_ms() { return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
};

using FishStateMap = std::map<std::string, std::unique_ptr<FishState>>;

static FishStateMap& fishhash_states() {
  static FishStateMap* const states = new FishStateMap;
  return *states;
}

static std::mutex& fishhash_states_mutex() {
  static std::mutex* const mutex = new std::mutex;
  return *mutex;
}

static FishState& fishhash_state(const std::string& dev_str) {
  std::lock_guard<std::mutex> lock(fishhash_states_mutex());
  auto& state = fishhash_states()[dev_str];
  if (!state) state = std::make_unique<FishState>(dev_str);
  return *state;
}

static bool select_karlsen_cooperative(
  FishState& state, const unsigned input_size, const uint64_t start_nonce
) {
#if defined(MOM_SYCL_PORTABLE_OPENCL)
  (void)state;
  (void)input_size;
  (void)start_nonce;
  return false;
#else
  if (state.karlsen_cooperative >= 0) return state.karlsen_cooperative != 0;

  // CUDA and HIP have a clear cooperative winner. Generic OpenCL retains the portable scalar path;
  // Level Zero is tuned because Arc generations differ substantially in register pressure versus
  // subgroup-exchange cost. This measures the common kernels instead of branching on a GPU model.
  if (mom_is_cuda(state.device) || mom_is_hip(state.device)) {
    state.karlsen_cooperative = 1;
    return true;
  }
  if (!sycl_is_level_zero_gpu(state.device)) {
    state.karlsen_cooperative = 0;
    return false;
  }
  const auto subgroup_sizes =
    state.device.get_info<sycl::info::device::sub_group_sizes>();
  if (std::find(subgroup_sizes.begin(), subgroup_sizes.end(), FISH_TEAM) == subgroup_sizes.end()) {
    state.karlsen_cooperative = 0;
    return false;
  }

  constexpr uint32_t warmup_intensity = 65536;
  constexpr uint32_t sample_intensity = 1048576;
  const auto run = [&](const bool cooperative, const uint32_t count) {
    const auto begin = std::chrono::steady_clock::now();
    const sycl::event event = cooperative
      ? submit_karlsenhashv2_search_coop(
          state.queue, *state.bundle, state.input, input_size, start_nonce, state.dag, count,
          state.target, state.result, false, state.coop_workgroup)
      : submit_karlsenhashv2_search<true>(
          state.queue, *state.bundle, state.input, input_size, start_nonce, nullptr, state.dag, count,
          state.target, state.result, false);
    sycl_wait_and_throw(event, state.device);
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
  };

  // Exclude each kernel's first-launch JIT cost, then alternate order to avoid a clock/thermal bias.
  run(false, warmup_intensity);
  run(true, warmup_intensity);
  double scalar_seconds = run(false, sample_intensity);
  double cooperative_seconds = run(true, sample_intensity);
  cooperative_seconds += run(true, sample_intensity);
  scalar_seconds += run(false, sample_intensity);
  state.karlsen_cooperative = cooperative_seconds < scalar_seconds ? 1 : 0;
  const double hashes = 2.0 * sample_intensity / 1e6;
  std::fprintf(stderr, "karlsenhashv2: selected %s search (scalar %.2f, cooperative %.2f MH/s)\n",
    state.karlsen_cooperative ? "cooperative" : "scalar",
    hashes / scalar_seconds, hashes / cooperative_seconds);
  return state.karlsen_cooperative != 0;
#endif
}

void fishhash_cleanup_states() noexcept {
  try {
    std::lock_guard<std::mutex> lock(fishhash_states_mutex());
    fishhash_states().clear();
  } catch (...) {
    std::fprintf(stderr, "fishhash: ordered SYCL cleanup failed\n");
  }
}

} // namespace mom_fishhash

using namespace mom_fishhash;
