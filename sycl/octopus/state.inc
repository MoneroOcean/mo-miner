// Per-device epoch/cache/DAG ownership; callers serialize work through State::mutex.
inline bool is_prime(const uint64_t value) {
  if (value < 2 || (value % 2) == 0)
    return value == 2;
  for (uint64_t divisor = 3; divisor <= value / divisor; divisor += 2)
    if (value % divisor == 0)
      return false;
  return true;
}

// Choose the protocol epoch cache size with a prime number of 64-byte nodes.
inline uint64_t cache_size(const uint32_t epoch) {
  uint64_t size = (1ULL << 24) + (1ULL << 16) * epoch - 64;
  while (!is_prime(size / 64))
    size -= 128;
  return size;
}

// Choose the protocol epoch DAG size with a prime number of 256-byte pages.
inline uint64_t dataset_size(const uint32_t epoch) {
  uint64_t size = (1ULL << 32) + (1ULL << 24) * epoch - 256;
  while (!is_prime(size / 256))
    size -= 512;
  return size;
}

class State {
public:
  sycl::device device;
  sycl::queue queue;
  const bool shared_io;
  UsmPtr<uint8_t> header;
  UsmPtr<uint8_t> target;
  UsmPtr<uint32_t> cache;
  UsmPtr<uint32_t> dag;
  UsmPtr<uint32_t> points;
  UsmPtr<Result> result;
#ifdef MOM_OCTOPUS_HAS_SYCL_NATIVE
  OctopusSyclNativeSearch sycl_native;
#endif
  uint32_t epoch = std::numeric_limits<uint32_t>::max();
  uint32_t cache_nodes = 0;
  uint32_t dag_nodes = 0;
  std::array<uint8_t, HASH_LEN> points_header{};
  std::array<uint8_t, HASH_LEN> target_copy{};
  uint64_t points_version = 0;
  bool points_ready = false;
  bool target_ready = false;
  bool full_dag = false;
  bool dag_attempted = false;
  std::mutex mutex;

  // Create the owning in-order queue and bounded work buffers. Shared USM backs CPU I/O and device
  // USM backs discrete-device buffers; a failed allocation frees the earlier ones as it unwinds.
  explicit State(const std::string& dev)
      : device(get_dev(dev)), queue(device, sycl::property_list{sycl::property::queue::in_order{}}),
        shared_io(device.is_cpu() || !mom_has_usm_device(device)) {
    if (!mom_has_usm_shared(device) || (!device.is_cpu() && !mom_has_usm_device(device)))
      throw std::string("octopus requires SYCL device and shared USM");
    const char* const failure = "Can't allocate octopus I/O buffers";
    header = mom_usm_alloc<uint8_t>(queue, HASH_LEN, shared_io, failure);
    target = mom_usm_alloc<uint8_t>(queue, HASH_LEN, shared_io, failure);
    points = mom_usm_alloc<uint32_t>(queue, POLYNOMIAL_SIZE, shared_io, failure);
    result = mom_usm_alloc<Result>(queue, 1, true, failure);
  }

  // Retire native and queued work before the UsmPtr members free the storage it uses.
  ~State() {
    sycl_cleanup_noexcept("octopus", [&] {
#ifdef MOM_OCTOPUS_HAS_SYCL_NATIVE
      sycl_native.release();
#endif
      queue.wait_and_throw();
    });
  }

  // Refresh header-dependent evaluation points only when the problem hash changes.
  bool ensure_points(const uint8_t next_header[HASH_LEN]) {
    if (points_ready && std::memcmp(points_header.data(), next_header, HASH_LEN) == 0)
      return false;
    const uint64_t v0 = load64(next_header), v1 = load64(next_header + 8);
    const uint64_t v2 = load64(next_header + 16), v3 = load64(next_header + 24);
    const uint32_t a = remap(v0), b = remap(v1);
    uint64_t c_input = v2;
    uint32_t c;
    do {
      c = remap(c_input++);
    } while ((static_cast<uint64_t>(b) * b) % MOD == (static_cast<uint64_t>(4) * a * c) % MOD);
    const uint32_t w = remap(v3), w2 = mod_field(static_cast<uint64_t>(w) * w);
    uint32_t wpow = 1, w2pow = 1;
    std::array<uint32_t, POLYNOMIAL_SIZE> host_points;
    for (uint32_t exponent = 0; exponent < POLYNOMIAL_SIZE; ++exponent) {
      host_points[exponent] =
          mod_field(static_cast<uint64_t>(a) * w2pow + static_cast<uint64_t>(b) * wpow + c);
      wpow = mod_field(static_cast<uint64_t>(wpow) * w);
      w2pow = mod_field(static_cast<uint64_t>(w2pow) * w2);
    }
    MomSyclHostTransferGuard host_points_transfers(queue, "octopus points upload");
    // A failed copy must not leave the previous header marked ready.
    points_ready = false;
    if (shared_io)
      std::memcpy(points, host_points.data(), sizeof(host_points));
    else
      sycl_wait_and_throw(queue.memcpy(points, host_points.data(), sizeof(host_points)), device);
    std::memcpy(points_header.data(), next_header, HASH_LEN);
    points_ready = true;
    points_version = next_points_version.fetch_add(1, std::memory_order_relaxed);
    return true;
  }

  // Resolve explicit DAG policy (MOM_OCTOPUS_LIGHT forces the light-cache route) before automatic
  // device-capacity selection.
  bool wants_full_dag() const {
    if (device.is_cpu() || shared_io || mom_env_flag("MOM_OCTOPUS_LIGHT").value_or(false))
      return false;
    const uint64_t bytes = dataset_size(epoch);
    const uint64_t reserve = 512ULL << 20;
    const uint64_t memory = device.get_info<sycl::info::device::global_mem_size>();
    const uint64_t max_alloc = device.get_info<sycl::info::device::max_mem_alloc_size>();
    return max_alloc >= bytes && memory >= bytes + cache_size(epoch) + reserve;
  }

  // The epoch transition has already released the old cache and invalidated its key.
  void build_cache(const uint32_t next_epoch) {
    const uint64_t bytes = cache_size(next_epoch);
    const uint32_t nodes = static_cast<uint32_t>(bytes / (CACHE_NODE_WORDS * sizeof(uint32_t)));
    std::vector<uint32_t> host_cache;
    compute_light_cache(host_cache, bytes, next_epoch, "Can't calculate octopus light cache");
    queue.wait_and_throw();
    cache = mom_usm_alloc<uint32_t>(queue, host_cache.size(), shared_io,
                                    "Can't allocate octopus light cache");
    upload_ethash_light_cache(queue, device, cache, host_cache, shared_io, "octopus cache upload");
    cache_nodes = nodes;
  }

  // Build the optional full DAG lazily; failed allocation leaves the light-cache route available.
  void ensure_dag(const bool log) {
    if (full_dag || dag_attempted)
      return;
    const bool available = wants_full_dag();
    dag_attempted = true;
    if (!available)
      return;
    const uint64_t bytes = dataset_size(epoch);
    const uint32_t nodes = static_cast<uint32_t>(bytes / (CACHE_NODE_WORDS * sizeof(uint32_t)));
    UsmPtr<uint32_t> candidate;
    try {
      candidate = mom_usm_alloc<uint32_t>(queue, static_cast<size_t>(bytes / sizeof(uint32_t)),
                                          shared_io, "Can't allocate octopus DAG");
    } catch (...) {
      return; // The light path remains correct on small or fragmented devices.
    }

    const uint64_t start = mom_now_ms();
    const uint32_t chunk =
        1U << 18; // Keep individual dispatches below Windows WDDM watchdog limits.
    try {
      for (uint32_t offset = 0; offset < nodes; offset += chunk) {
        const uint32_t count = std::min(chunk, nodes - offset);
        sycl_wait_and_throw(build_dag(queue, cache, cache_nodes, candidate, offset, count), device);
      }
    } catch (...) {
      // Drain first: unwinding frees the candidate, which no accepted build may still write.
      sycl_cleanup_noexcept("octopus DAG wait", [&] { queue.wait_and_throw(); });
      // A submitted DAG fault must not become cached light-path availability.
      dag_attempted = false;
      throw;
    }
    dag = std::move(candidate); // Frees the previous DAG first.
    dag_nodes = nodes;
    full_dag = true;
    if (log)
      std::fprintf(stderr, "Octopus DAG epoch %u built (%.2f s)\n", epoch,
                   static_cast<double>(mom_now_ms() - start) / 1000.0);
  }

  // Refresh epoch resources before search; test mode skips the DAG unless
  // MOM_OCTOPUS_TEST_FULL_DAG asks for it.
  void ensure_epoch(const uint32_t next_epoch, const bool log, const bool test) {
    if (epoch != next_epoch) {
      // Invalidate the old epoch before an in-place rebuild can fail; it must not remain reusable.
      epoch = std::numeric_limits<uint32_t>::max();
      dag.reset();
      cache.reset();
      full_dag = false;
      dag_attempted = false;
      dag_nodes = 0;
      cache_nodes = 0;
      build_cache(next_epoch);
      epoch = next_epoch;
    }
    if (!test || mom_env_flag("MOM_OCTOPUS_TEST_FULL_DAG").value_or(false))
      ensure_dag(log);
  }
};
