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
  const unsigned fish_team;
  const unsigned karlsen_workgroup;
  const unsigned fish_workgroup;
  bool cache_built = false;
  bool dag_built = false;
  // A cached choice is unset (-1), scalar (0), or cooperative (1).
  int fishhashCooperative = -1;
  int karlsenhashv2Cooperative = -1;
  MomSyclRateWindow fishhashRate, karlsenhashv2Rate;
  std::mutex mutex;
  // Bind the fixed dataset and job buffers to a device with required USM support.
  explicit State(const std::string& dev_str)
      : device(get_dev(dev_str)),
        queue(device, sycl::property_list{sycl::property::queue::in_order{}}),
        shared_io(device.is_cpu() || !mom_has_usm_device(device)),
        fish_team(mom_is_linux_alchemist(device) ? 8 : FISH_TEAM),
        karlsen_workgroup(cooperative_workgroup(device)),
        fish_workgroup(cooperative_workgroup(device, fish_team, true)) {
    if (!mom_has_usm_shared(device) || (!device.is_cpu() && !mom_has_usm_device(device)))
      throw std::string("fishhash SYCL device does not support required allocations");
  }
  // Zero keeps scalar search available when cooperative geometry cannot fit the device.
  static unsigned cooperative_workgroup(const sycl::device& dev, unsigned team = FISH_TEAM,
                                        [[maybe_unused]] bool fishhash = false) {
    if constexpr (mom_sycl_portable_opencl)
      return 0;
    if (!dev.is_gpu() || !sycl_workgroup_fits(dev, 64, 0, 0, FISH_STAGE_BYTES, team))
      return 0;
    if (team == FISH_TEAM ? !sycl_reqd_subgroup_16_supported(dev) : !sycl_supports_subgroup_size(dev, team))
      return 0;
    // CUDA defaults to 512 except on wide AdaptiveCpp devices and on SM 12.0 FIP-9 jobs.
    unsigned preferred = mom_is_cuda(dev) ? 512 : 128;
#if defined(MOM_SYCL_ADAPTIVECPP_CUDA)
    if (mom_is_cuda(dev) && dev.get_info<sycl::info::device::max_compute_units>() >= 64)
      preferred = 256;
#endif
#if defined(__linux__) && defined(MOM_SYCL_ADAPTIVECPP_CUDA)
    if (fishhash && mom::nvidia::compute_capability(dev) == 120)
      preferred = 256;
#endif
    // On BMG-G21 a 64-thread group improves FishHash FIP9. KarlsenHashV2 gains from it only with
    // the 256-entry register file (13.95 against 12.68 MH/s at 128 threads; 11.69 at 64 without).
    if ((fishhash || fishhash_large_grf) && mom_is_bmg_g21(dev))
      preferred = 64;
    // intensity.inc admits the same override set when it sizes the batch.
    return sycl_env_workgroup(dev, "MOM_FISHHASH_WORKGROUP", {64, 128, 256, 512}, preferred,
                              0, 0, FISH_STAGE_BYTES, team);
  }
  // Drain kernels before releasing cache, DAG, and job buffers.
  ~State() {
    sycl_cleanup_noexcept("fishhash wait", [&] { queue.wait_and_throw(); });
  }
  // Only offline vectors may use the light cache. Full-test mode exercises the mining DAG.
  bool prepareJob(unsigned inputSize, bool isTest, bool isBenchmark) {
    ensure_io(inputSize);
    const bool lightTest = isTest && !mom_env_flag("MOM_FISHHASH_FULL_TEST").value_or(false);
    if (lightTest)
      ensure_cache(!isBenchmark);
    else
      ensure_dag(!isBenchmark);
    return lightTest;
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

// Compare one hash before timing on Level Zero. The cooperative FishHash kernel can return
// without publishing a result on Arc A770, which would otherwise appear to win the timing probe.
template <class Search>
static bool selectCooperative(State& state, int& choice, const char* algorithm,
                              bool fishhash, const Search& search) {
  if (!(fishhash ? state.fish_workgroup : state.karlsen_workgroup))
    return false;
  // gpu-tuning.js search_mode writes MOM_FISHHASH_COOP; an explicit value skips the probe.
  if (const std::optional<bool> forced = mom_env_flag("MOM_FISHHASH_COOP"))
    return *forced;
  if (choice >= 0)
    return choice != 0;
  if (mom_is_cuda(state.device) || mom_is_hip(state.device))
    return true;
  if (!sycl_is_level_zero_gpu(state.device))
    return fishhash;
  const auto run = [&](bool cooperative, uint32_t count, bool test = false) {
    std::memset(state.result, 0, sizeof(*state.result));
    const auto begin = std::chrono::steady_clock::now();
    sycl_wait_and_throw(search(cooperative, count, test), state.device);
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
  };

  // Test mode writes the first hash independently of the target.
  run(true, 1, true);
  uint8_t cooperativeHash[HASH_LEN];
  if (state.result->count)
    std::memcpy(cooperativeHash, state.result->hash[0], HASH_LEN);
  else {
    choice = 0;
    std::fprintf(stderr, "%s: cooperative search returned no hash; using scalar\n", algorithm);
    return false;
  }
  run(false, 1, true);
  if (!state.result->count || std::memcmp(cooperativeHash, state.result->hash[0], HASH_LEN) != 0) {
    choice = 0;
    std::fprintf(stderr, "%s: cooperative search disagrees with scalar; using scalar\n", algorithm);
    return false;
  }

  // Warm each route, then alternate timing order to reduce thermal and clock bias.
  constexpr uint32_t warmupCount = 65536, sampleCount = 1048576;
  run(false, warmupCount);
  run(true, warmupCount);
  double scalarSeconds = run(false, sampleCount);
  double cooperativeSeconds = run(true, sampleCount);
  cooperativeSeconds += run(true, sampleCount);
  scalarSeconds += run(false, sampleCount);
  choice = cooperativeSeconds < scalarSeconds;
  const double megaHashes = 2.0 * sampleCount / 1e6;
  std::fprintf(stderr, "%s: selected %s search (scalar %.2f, cooperative %.2f MH/s)\n",
               algorithm, choice ? "cooperative" : "scalar", megaHashes / scalarSeconds,
               megaHashes / cooperativeSeconds);
  return choice != 0;
}

// Release device state before the compiler runtime unloads without escaping exceptions.
void fishhash_cleanup_states() noexcept {
  clearPersistentDeviceStates<State>("fishhash");
}
