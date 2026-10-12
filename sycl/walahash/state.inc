// Per-device USM and backend scratch ownership; callers hold State::mutex.
class State {
public:
  sycl::device device;
  sycl::queue queue;
  uint8_t* matrix = nullptr;
  uint8_t* input = nullptr;
  uint8_t* target = nullptr;
  uint8_t matrix_seed[32]{};
  Result* result = nullptr;
  bool matrix_ready = false;
#ifdef MOM_SYCL_HAS_CUDA
  const int cuda_sm = mom::nvidia::compute_capability(device);
#endif
  std::mutex mutex;
#ifdef MOM_SYCL_HAS_HIP
  WalaHashAmdWmma amd_wmma;
#endif
#if defined(WALAHASH_ESIMD) || defined(MOM_SYCL_HAS_CUDA) || defined(MOM_SYCL_ADAPTIVECPP_CUDA) || defined(MOM_SYCL_HAS_HIP)
  Blake3Cv prehash_cv{};
#endif
#if defined(WALAHASH_ESIMD) || defined(MOM_SYCL_HAS_HIP)
  uint8_t *hashes = nullptr, *mixed = nullptr;
  unsigned capacity = 0;
#if defined(WALAHASH_ESIMD)
  uint8_t* vectors = nullptr;
#endif
#endif
  // Create per-device in-order ownership and fixed work buffers, unwinding partial allocation on failure.
  explicit State(const std::string& dev)
      : device(get_dev(dev)), queue(device, sycl::property::queue::in_order{}) {
    if (!mom_has_usm_device(device))
      throw std::string("walahash requires SYCL device USM");
    try {
      matrix = sycl::malloc_device<uint8_t>(4096, queue);
      input = sycl::malloc_device<uint8_t>(72, queue);
      target = sycl::malloc_device<uint8_t>(32, queue);
      result = sycl::malloc_device<Result>(1, queue);
      if (!matrix || !input || !target || !result)
        throw std::string("Can't allocate walahash buffers");
    } catch (...) {
      release();
      throw;
    }
  }
  ~State() {
    release();
  }
  // Best-effort release on the owning queue and clear the stored pointer.
  template <typename T> void free_ptr(T*& pointer) noexcept {
    if (pointer)
      try {
        sycl::free(pointer, queue);
      } catch (...) {
      }
    pointer = nullptr;
  }
#if defined(WALAHASH_ESIMD) || defined(MOM_SYCL_HAS_HIP)
  // Drop all staged hash/vector/mix buffers before resizing their shared capacity.
  void free_accelerated() noexcept {
#if defined(WALAHASH_ESIMD)
    free_ptr(vectors);
#endif
    free_ptr(mixed);
    free_ptr(hashes);
    capacity = 0;
  }
#endif
  // Drain outstanding kernels before releasing the fixed and accelerated USM buffers.
  void release() noexcept {
    sycl_cleanup_noexcept("walahash wait", [&] {
      queue.wait_and_throw();
    });
#if defined(WALAHASH_ESIMD) || defined(MOM_SYCL_HAS_HIP)
    free_accelerated();
#endif
    free_ptr(result);
    free_ptr(target);
    free_ptr(input);
    free_ptr(matrix);
  }
#if defined(WALAHASH_ESIMD) || defined(MOM_SYCL_HAS_HIP)
  // Grow nonce scratch to the padded backend capacity, unwinding partial allocations on failure.
  void ensure(unsigned count) {
    if (count <= capacity)
      return;
    constexpr unsigned alignment =
#if defined(MOM_SYCL_HAS_HIP)
        64;
#else
        16;
#endif
    const unsigned next_capacity = (count + alignment - 1) & ~(alignment - 1);
    queue.wait_and_throw();
    free_accelerated();
    try {
#if defined(WALAHASH_ESIMD)
      vectors = sycl::malloc_device<uint8_t>(static_cast<size_t>(next_capacity) * 64, queue);
#endif
      hashes = sycl::malloc_device<uint8_t>(static_cast<size_t>(next_capacity) * 32, queue);
      mixed = sycl::malloc_device<uint8_t>(static_cast<size_t>(next_capacity) * 32, queue);
#if defined(WALAHASH_ESIMD)
      if (!vectors || !hashes || !mixed)
#else
      if (!hashes || !mixed)
#endif
        throw std::string("Can't allocate accelerated walahash buffers");
      capacity = next_capacity;
    } catch (...) {
      free_accelerated();
      throw;
    }
  }
#endif
};
// Keep device owners heap-registered so explicit shutdown controls SYCL destruction order.
static DeviceStateRegistry<State>& registry() {
  static auto* const value = new DeviceStateRegistry<State>;
  return *value;
}
// Reuse one mutex-serialized work/matrix owner for the selected device.
static State& state_for(const std::string& dev) {
  return registry().get(dev, [&] { return std::make_unique<State>(dev); });
}
