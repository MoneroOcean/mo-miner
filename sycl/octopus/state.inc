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
  uint8_t* header = nullptr;
  uint8_t* target = nullptr;
  uint32_t* cache = nullptr;
  uint32_t* dag = nullptr;
  uint32_t* points = nullptr;
  Result* result = nullptr;
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

  // Create the owning in-order queue and bounded work buffers, unwinding partial constructor allocation.
  explicit State(const std::string& dev)
      : device(get_dev(dev)), queue(device, sycl::property_list{sycl::property::queue::in_order{}}),
        shared_io(device.is_cpu() || !mom_has_usm_device(device)) {
    if (!mom_has_usm_shared(device) || (!device.is_cpu() && !mom_has_usm_device(device)))
      throw std::string("octopus requires SYCL device and shared USM");
    try {
      header = allocate<uint8_t>(HASH_LEN);
      target = allocate<uint8_t>(HASH_LEN);
      points = allocate<uint32_t>(POLYNOMIAL_SIZE);
      result = sycl::malloc_shared<Result>(1, queue);
      if (!header || !target || !points || !result)
        throw std::string("Can't allocate octopus I/O buffers");
    } catch (...) {
      sycl_cleanup_noexcept("octopus init", [&] {
        release();
      });
      throw;
    }
  }

  ~State() {
    sycl_cleanup_noexcept("octopus", [&] {
      release();
    });
  }

  // Choose shared USM for CPU I/O and device USM for discrete-device buffers.
  template <typename T> T* allocate(const size_t count) {
    return shared_io ? sycl::malloc_shared<T>(count, queue) : sycl::malloc_device<T>(count, queue);
  }

  // Best-effort release on the owning queue; clear the pointer even if runtime cleanup fails.
  template <typename T> void free_ptr(T*& pointer) noexcept {
    if (pointer)
      try {
        sycl::free(pointer, queue);
      } catch (...) {
      }
    pointer = nullptr;
  }

  // Drain outstanding work before releasing the native search and all epoch/work buffers.
  void release() noexcept {
#ifdef MOM_OCTOPUS_HAS_SYCL_NATIVE
    sycl_native.release();
#endif
    try {
      queue.wait_and_throw();
    } catch (...) {
    }
    free_ptr(dag);
    free_ptr(cache);
    free_ptr(points);
    free_ptr(target);
    free_ptr(header);
    free_ptr(result);
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

  static uint64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
  }

  // Resolve explicit DAG policy before automatic device-capacity selection.
  bool wants_full_dag() const {
    if (device.is_cpu() || shared_io || std::getenv("MOM_OCTOPUS_LIGHT"))
      return false;
    const uint64_t bytes = dataset_size(epoch);
    const uint64_t reserve = 512ULL << 20;
    const uint64_t memory = device.get_info<sycl::info::device::global_mem_size>();
    const uint64_t max_alloc = device.get_info<sycl::info::device::max_mem_alloc_size>();
    return max_alloc >= bytes && memory >= bytes + cache_size(epoch) + reserve;
  }

  // Rebuild the host-derived cache on an epoch change and upload it to the owning queue.
  void ensure_cache(const uint32_t next_epoch) {
    const uint64_t bytes = cache_size(next_epoch);
    const uint32_t nodes = static_cast<uint32_t>(bytes / (CACHE_NODE_WORDS * sizeof(uint32_t)));
    if (cache && cache_nodes == nodes)
      return;
    std::vector<uint32_t> host_cache(static_cast<size_t>(bytes / sizeof(uint32_t)));
    const ethash_h256_t seed = ethash_get_seedhash(next_epoch);
    if (!ethash_compute_cache_nodes(host_cache.data(), bytes, &seed))
      throw std::string("Can't calculate octopus light cache");
    queue.wait_and_throw();
    free_ptr(cache);
    cache = allocate<uint32_t>(host_cache.size());
    if (!cache)
      throw std::string("Can't allocate octopus light cache");
    if (shared_io)
      std::memcpy(cache, host_cache.data(), bytes);
    else
      sycl_wait_and_throw(queue.memcpy(cache, host_cache.data(), bytes), device);
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
    uint32_t* candidate = nullptr;
    try {
      candidate = allocate<uint32_t>(static_cast<size_t>(bytes / sizeof(uint32_t)));
    } catch (...) {
      candidate = nullptr;
    }
    if (!candidate)
      return; // The light path remains correct on small or fragmented devices.

    const uint64_t start = now_ms();
    const uint32_t chunk =
        1U << 18; // Keep individual dispatches below Windows WDDM watchdog limits.
    try {
      for (uint32_t offset = 0; offset < nodes; offset += chunk) {
        const uint32_t count = std::min(chunk, nodes - offset);
        sycl_wait_and_throw(build_dag(queue, cache, cache_nodes, candidate, offset, count), device);
      }
    } catch (...) {
      sycl_cleanup_noexcept("octopus DAG wait", [&] { queue.wait_and_throw(); });
      free_ptr(candidate);
      // A submitted DAG fault must not become cached light-path availability.
      dag_attempted = false;
      throw;
    }
    free_ptr(dag);
    dag = candidate;
    dag_nodes = nodes;
    full_dag = true;
    if (log)
      std::fprintf(stderr, "Octopus DAG epoch %u built (%.2f s)\n", epoch,
                   static_cast<double>(now_ms() - start) / 1000.0);
  }

  // Refresh epoch resources before search; test mode retains its bounded DAG policy.
  void ensure_epoch(const uint32_t next_epoch, const bool log, const bool test) {
    if (epoch != next_epoch) {
      // Invalidate the old epoch before an in-place rebuild can fail; it must not remain reusable.
      epoch = std::numeric_limits<uint32_t>::max();
      free_ptr(dag);
      free_ptr(cache);
      full_dag = false;
      dag_attempted = false;
      dag_nodes = 0;
      cache_nodes = 0;
      ensure_cache(next_epoch);
      epoch = next_epoch;
    }
    if (!test || std::getenv("MOM_OCTOPUS_TEST_FULL_DAG"))
      ensure_dag(log);
  }
};

// Keep device owners heap-registered so explicit shutdown controls SYCL destruction order.
static DeviceStateRegistry<State>& registry() {
  static auto* const value = new DeviceStateRegistry<State>;
  return *value;
}

// Reuse one serialized epoch/cache owner for the selected device.
static State& state_for(const std::string& dev) {
  return registry().get(dev, [&] { return std::make_unique<State>(dev); });
}
