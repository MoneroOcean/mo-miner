class State {
public:
  sycl::device device;
  sycl::queue queue;
  bool shared_io;
  UsmPtr<uint8_t> input;
  UsmPtr<uint8_t> target;
  UsmPtr<uint64_t> cache;
  UsmPtr<uint64_t> dag;
  UsmPtr<FishResult> result;
  unsigned input_cap = 0;
  unsigned coop_workgroup;
  bool cache_built = false;
  bool dag_built = false;
  int fishhash_cooperative = -1; // search policy per algorithm: -1 unset, 0 scalar, 1 cooperative
  int karlsen_cooperative = -1;
  std::mutex mutex;
  // Bind the fixed dataset and job buffers to a device with required USM support.
  explicit State(const std::string& dev_str)
      : device(get_dev(dev_str)),
        queue(device, sycl::property_list{sycl::property::queue::in_order{}}),
        shared_io(device.is_cpu() || !mom_has_usm_device(device)),
        coop_workgroup(cooperative_workgroup(device)) {
    if (!mom_has_usm_shared(device) || (!device.is_cpu() && !mom_has_usm_device(device)))
      throw std::string("fishhash SYCL device does not support required allocations");
  }
  // Cooperative search workgroup; a MOM_FISHHASH_WORKGROUP override must be supported and fit.
  static unsigned cooperative_workgroup(const sycl::device& dev) {
    // Both CUDA compiler paths peak at 512 on the RTX 5060 Ti (3.2% over 128 for AdaptiveCpp and
    // 5.1% for DPC++); wide pre-sm80 CUDA peaks at 256 (V100S: +2.7%). Intel and AMD are
    // flat-to-slower above 128. Clamp against the device limit.
    unsigned preferred = mom_is_cuda(dev) ? 512 : 128;
#if defined(MOM_SYCL_ADAPTIVECPP_CUDA)
    if (mom_is_cuda(dev) && dev.get_info<sycl::info::device::max_compute_units>() >= 64)
      preferred = 256;
#endif
    const unsigned fallback = sycl_default_workgroup(dev, {64, 128, 256, 512}, preferred);
    unsigned long parsed = 0;
    if (!mom_parse_env_ulong("MOM_FISHHASH_WORKGROUP", parsed))
      return fallback;
    for (const unsigned candidate : {64u, 128u, 256u, 512u})
      if (parsed == candidate && sycl_workgroup_fits(dev, candidate))
        return candidate;
    return fallback;
  }
  // Drain kernels before releasing cache, DAG, and job buffers.
  ~State() {
    sycl_cleanup_noexcept("fishhash wait", [&] { queue.wait_and_throw(); });
  }
  // Grow header buffers after queue completion, retaining production-header capacity.
  void ensure_io(unsigned in_sz) {
    if (input && in_sz <= input_cap && target && result)
      return;
    queue.wait_and_throw();
    input.reset();
    target.reset();
    result.reset();
    input_cap = 0;
    constexpr const char* failure = "Can't allocate fishhash buffers";
    const unsigned requested_cap = in_sz < 180 ? 180 : in_sz;
    input = mom_usm_alloc<uint8_t>(queue, requested_cap, shared_io, failure);
    target = mom_usm_alloc<uint8_t>(queue, HASH_LEN, shared_io, failure);
    result = mom_usm_alloc<FishResult>(queue, 1, true, failure);
    std::memset(result, 0, sizeof(FishResult));
    input_cap = requested_cap;
  }
  // Build/upload the cache once; failures never publish cache_built.
  void ensure_cache(bool log) {
    if (cache_built)
      return;
    const uint64_t t0 = mom_now_ms();
    cache_built = false;
    try {
      cache.reset();
      std::vector<uint64_t> hcache;
      build_light_cache_host(hcache);
      cache = mom_usm_alloc<uint64_t>(queue, hcache.size(), shared_io,
                                      "Can't allocate fishhash light cache");
      MomSyclHostTransferGuard host_cache_transfers(queue, "fishhash cache upload");
      if (shared_io)
        std::memcpy(cache, hcache.data(), hcache.size() * sizeof(uint64_t));
      else
        sycl_wait_and_throw(queue.memcpy(cache, hcache.data(), hcache.size() * sizeof(uint64_t)),
                            device);
      cache_built = true;
      if (log)
        std::fprintf(stderr, "FishHash light cache (72 MiB) built (%llu ms)\n",
                     (unsigned long long)(mom_now_ms() - t0));
    } catch (...) {
      cache.reset();
      cache_built = false;
      throw;
    }
  }
  // Build watchdog-bounded chunks and publish dag_built only after every slice completes.
  void ensure_dag(bool log) {
    if (dag_built)
      return;
    const uint64_t t0 = mom_now_ms();
    dag_built = false;
    try {
      ensure_cache(log);
      dag.reset(); // release a partial buffer from a previously-failed attempt before re-allocating
      dag = mom_usm_alloc<uint64_t>(queue, (size_t)FULL_DATASET_NUM_ITEMS * 16, shared_io,
                                    "Can't allocate fishhash DAG (4.6 GiB)");
      const uint32_t CHUNK = 1u << 21; // 2M items/launch to avoid the GPU watchdog
      for (uint32_t s = 0; s < (uint32_t)FULL_DATASET_NUM_ITEMS; s += CHUNK) {
        const uint32_t c = std::min(CHUNK, (uint32_t)FULL_DATASET_NUM_ITEMS - s);
        sycl_wait_and_throw(submit_fishhash_dag_gen(queue, cache, dag, s, c), device);
      }
      dag_built = true;
      if (log)
        std::fprintf(stderr, "FishHash DAG (4.6 GiB) built (%llu ms)\n",
                     (unsigned long long)(mom_now_ms() - t0));
    } catch (...) {
      dag.reset();
      dag_built = false;
      throw;
    }
  }
};

// Keep the registry shell alive across static teardown; ordered cleanup releases its values.
static DeviceStateRegistry<State>& registry() {
  static auto* const value = new DeviceStateRegistry<State>;
  return *value;
}

static State& state_for(const std::string& dev_str) {
  return registry().get(dev_str, [&] { return std::make_unique<State>(dev_str); });
}

// Cache one algorithm's search policy in `choice`. CUDA and HIP have a clear cooperative winner and
// other non-Level Zero GPUs keep the algorithm's default. On Level Zero the cooperative kernel must
// first reproduce the scalar kernel's nonce-0 hash (the FishHash one silently computes nothing on an
// Arc A770), then outrun it: Arc generations differ in register pressure versus subgroup-exchange
// cost, so the real kernels are measured instead of branching on a GPU model.
// `search(cooperative, count, is_test)` submits one search over the current input and target.
template <class Search>
static bool select_cooperative(State& state, int& choice, const char* const algo,
                               const bool other_gpu_default, const Search& search) {
#if defined(MOM_SYCL_PORTABLE_OPENCL)
  (void)state;
  (void)choice;
  (void)algo;
  (void)other_gpu_default;
  (void)search;
  return false;
#else
  if (choice >= 0)
    return choice != 0;
  if (mom_is_cuda(state.device) || mom_is_hip(state.device)) {
    choice = 1;
    return true;
  }
  if (!sycl_is_level_zero_gpu(state.device)) {
    choice = other_gpu_default ? 1 : 0;
    return other_gpu_default;
  }
  const auto subgroup_sizes = state.device.get_info<sycl::info::device::sub_group_sizes>();
  if (std::find(subgroup_sizes.begin(), subgroup_sizes.end(), FISH_TEAM) == subgroup_sizes.end()) {
    choice = 0;
    return false;
  }

  const auto run = [&](const bool cooperative, const uint32_t count, const bool is_test) {
    std::memset(state.result, 0, sizeof(*state.result));
    const auto begin = std::chrono::steady_clock::now();
    sycl_wait_and_throw(search(cooperative, count, is_test), state.device);
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
  };
  // Test mode stores nonce 0's hash whatever the target; a kernel that stores none fails.
  const auto first_hash = [&](const bool cooperative, uint8_t (&hash)[HASH_LEN]) {
    run(cooperative, 1, true);
    if (state.result->count == 0)
      return false;
    std::memcpy(hash, state.result->hash[0], HASH_LEN);
    return true;
  };
  uint8_t cooperative_hash[HASH_LEN], scalar_hash[HASH_LEN];
  if (!first_hash(true, cooperative_hash) || !first_hash(false, scalar_hash) ||
      std::memcmp(cooperative_hash, scalar_hash, HASH_LEN) != 0) {
    choice = 0;
    std::fprintf(stderr, "%s: cooperative search disagrees with the scalar kernel; using scalar\n",
                 algo);
    return false;
  }

  constexpr uint32_t warmup_intensity = 65536;
  constexpr uint32_t sample_intensity = 1048576;
  // Exclude each kernel's first-launch JIT cost, then alternate order to avoid a clock/thermal bias.
  run(false, warmup_intensity, false);
  run(true, warmup_intensity, false);
  double scalar_seconds = run(false, sample_intensity, false);
  double cooperative_seconds = run(true, sample_intensity, false);
  cooperative_seconds += run(true, sample_intensity, false);
  scalar_seconds += run(false, sample_intensity, false);
  choice = cooperative_seconds < scalar_seconds ? 1 : 0;
  const double hashes = 2.0 * sample_intensity / 1e6;
  std::fprintf(stderr, "%s: selected %s search (scalar %.2f, cooperative %.2f MH/s)\n", algo,
               choice ? "cooperative" : "scalar", hashes / scalar_seconds,
               hashes / cooperative_seconds);
  return choice != 0;
#endif
}

// Release device state before the compiler runtime unloads without escaping exceptions.
void fishhash_cleanup_states() noexcept {
  try {
    registry().clear();
  } catch (...) {
    std::fprintf(stderr, "fishhash: ordered SYCL cleanup failed\n");
  }
}
