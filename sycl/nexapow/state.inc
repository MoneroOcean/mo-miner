// Per-device monolithic/staged search resources; callers hold State::mutex.
class SearchKernel;
void nexapow_test_sha256d_49(sycl::queue&, const uint8_t[49], uint8_t[32]);
class State {
public:
  sycl::device device;
  sycl::queue queue;
  uint8_t *header = nullptr, *extranonce = nullptr, *target = nullptr;
  Result* result = nullptr;
  NexaPowSyclSearch portable;
  bool staged_logged = false;
  bool staged_failure_logged = false;
  std::mutex mutex;

  // Create the in-order device owner and retain backend-specific compilation options.
  explicit State(const std::string& dev)
      : device(get_dev(dev)), queue(device, sycl::property::queue::in_order{}) {
    if (!mom_has_usm_device(device))
      throw std::string("nexapow requires SYCL device USM");
#ifdef MOM_NEXAPOW_SYCL_NATIVE_FIELD
    if (mom_is_cuda(device)) {
      const char* options = std::getenv("MOM_NEXAPOW_COMPILE_OPTIONS");
      if (!options)
        options = std::getenv("SYCL_PROGRAM_COMPILE_OPTIONS");
      set_sycl_env("SYCL_PROGRAM_COMPILE_OPTIONS", options ? options : "--maxrregcount=128");
    }
#endif
  }

  // Allocate the small monolithic-search work buffers lazily, unwinding partial allocation.
  void init_portable() {
    if (header)
      return;
    try {
      header = sycl::malloc_device<uint8_t>(32, queue);
      extranonce = sycl::malloc_device<uint8_t>(8, queue);
      target = sycl::malloc_device<uint8_t>(32, queue);
      result = sycl::malloc_device<Result>(1, queue);
      if (!header || !extranonce || !target || !result)
        throw std::string("Can't allocate nexapow buffers");
    } catch (...) {
      free_all();
      throw;
    }
  }

  ~State() {
    sycl_cleanup_noexcept("nexapow wait", [&] { queue.wait_and_throw(); });
    portable.release(queue);
    free_all();
  }

  // Best-effort release of partial/full monolithic work allocations on their owning queue.
  void free_all() noexcept {
    auto release = [&](auto*& pointer) {
      if (!pointer)
        return;
      try {
        sycl::free(pointer, queue);
      } catch (...) {
      }
      pointer = nullptr;
    };
    release(result);
    release(target);
    release(extranonce);
    release(header);
  }
};

// Keep device owners heap-registered so explicit shutdown controls SYCL destruction order.
static DeviceStateRegistry<State>& registry() {
  static auto* const value = new DeviceStateRegistry<State>;
  return *value;
}
// Reuse one mutex-serialized staged/monolithic search owner for the selected device.
static State& state_for(const std::string& dev) {
  return registry().get(dev, [&] { return std::make_unique<State>(dev); });
}
