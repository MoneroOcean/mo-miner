// Per-device scratch/LUT ownership; callers serialize resizing and search through State::mutex.
class State {
public:
  sycl::device device;
  sycl::queue queue;
  uint8_t *input = nullptr, *target = nullptr;
  uint64_t* scratch = nullptr;
  double* lut = nullptr;
  Result* result = nullptr;
  unsigned capacity = 0;
  std::mutex mutex;
  const bool profile;
  unsigned profile_reports = 0;
  // Allocate fixed work/LUT state on an in-order queue and unwind failed constructor setup.
  explicit State(const std::string& dev)
      : device(get_dev(dev)),
        queue(device,
              sycl::async_handler([](sycl::exception_list errors) {
                for (const std::exception_ptr& error : errors)
                  std::rethrow_exception(error);
              }),
              sycl::property_list{sycl::property::queue::in_order{}}),
        profile([] {
          const char* configured = std::getenv("MOM_XELISHASHV3_PROFILE");
          return configured && *configured && std::strcmp(configured, "0") != 0;
        }()) {
    try {
      if (!xelishashv3_supported(device))
        throw std::string("xelishashv3 requires SYCL device USM");
      input = sycl::malloc_device<uint8_t>(112, queue);
      target = sycl::malloc_device<uint8_t>(32, queue);
      result = sycl::malloc_device<Result>(1, queue);
      if (!input || !target || !result)
        throw std::string("Can't allocate xelishashv3 buffers");
      if (device.has(sycl::aspect::fp64)) {
        lut = sycl::malloc_device<double>(512, queue);
        if (!lut)
          throw std::string("Can't allocate xelishashv3 reciprocal table");
        double host[512];
        for (unsigned i = 0; i < 512; ++i)
          host[i] = 1.0 / ((512.5 + i) * (1u << 22));
        sycl_wait_and_throw(queue.memcpy(lut, host, sizeof(host)), device);
      }
    } catch (...) {
      sycl_cleanup_noexcept("xelishashv3 constructor", [&] {
        cleanup();
      });
      throw;
    }
  }
  ~State() {
    sycl_cleanup_noexcept("xelishashv3", [&] {
      cleanup();
    });
  }
  // Release owned USM after work is drained; runtime free failures propagate to the cleanup caller.
  void free_all() {
    if (lut) {
      sycl::free(lut, queue);
      lut = nullptr;
    }
    if (scratch) {
      sycl::free(scratch, queue);
      scratch = nullptr;
    }
    if (result) {
      sycl::free(result, queue);
      result = nullptr;
    }
    if (target) {
      sycl::free(target, queue);
      target = nullptr;
    }
    if (input) {
      sycl::free(input, queue);
      input = nullptr;
    }
    capacity = 0;
  }
  // Drain kernels before freeing their work, LUT, result, and scratch allocations.
  void cleanup() {
    try {
      queue.wait_and_throw();
    } catch (...) {
      free_all();
      throw;
    }
    free_all();
  }
  // Grow per-nonce scratch within both allocation and total-memory limits.
  void ensure(unsigned count) {
    if (static_cast<uint64_t>(count) > xelishashv3_scratch_capacity(device))
      throw std::string("xelishashv3 intensity exceeds device memory limit");
    const size_t max_count = std::numeric_limits<size_t>::max() / WORDS;
    if (static_cast<uint64_t>(count) > max_count)
      throw std::string("xelishashv3 intensity is too large");
    const size_t count_size = static_cast<size_t>(count);
    if (count <= capacity)
      return;
    queue.wait_and_throw();
    uint64_t* replacement = sycl::malloc_device<uint64_t>(count_size * WORDS, queue);
    if (!replacement)
      throw std::string("Can't allocate xelishashv3 scratchpads");
    uint64_t* old_scratch = scratch;
    scratch = replacement;
    capacity = count;
    if (old_scratch)
      sycl::free(old_scratch, queue);
  }
};
// Keep device owners heap-registered so explicit shutdown controls SYCL destruction order.
static DeviceStateRegistry<State>& registry() {
  static auto* const value = new DeviceStateRegistry<State>;
  return *value;
}
// Reuse one mutex-serialized scratch/LUT owner for the selected device.
static State& state_for(const std::string& dev) {
  return registry().get(dev, [&] { return std::make_unique<State>(dev); });
}
