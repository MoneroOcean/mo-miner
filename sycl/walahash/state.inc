// Per-device USM and backend scratch ownership; callers hold State::mutex.
class State {
public:
  sycl::device device;
  sycl::queue queue;
  UsmPtr<uint8_t> matrix, input, target;
  uint8_t matrix_seed[32]{};
  UsmPtr<PowHits<>> result;
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
  UsmPtr<uint8_t> hashes, mixed;
  unsigned capacity = 0;
#if defined(WALAHASH_ESIMD)
  UsmPtr<uint8_t> vectors;
#endif
#endif
  // Create per-device in-order ownership and fixed work buffers.
  explicit State(const std::string& dev)
      : device(get_dev(dev)), queue(device, sycl::property::queue::in_order{}) {
    if (!mom_has_usm_device(device))
      throw std::string("walahash requires SYCL device USM");
    const char* const failure = "Can't allocate walahash buffers";
    matrix = mom_usm_alloc<uint8_t>(queue, 4096, false, failure);
    input = mom_usm_alloc<uint8_t>(queue, 72, false, failure);
    target = mom_usm_alloc<uint8_t>(queue, 32, false, failure);
    result = mom_usm_alloc<PowHits<>>(queue, 1, false, failure);
  }
  // Drain outstanding kernels before the members release their USM.
  ~State() {
    sycl_cleanup_noexcept("walahash wait", [&] { queue.wait_and_throw(); });
  }
#if defined(WALAHASH_ESIMD) || defined(MOM_SYCL_HAS_HIP)
  // Grow nonce scratch to the padded backend capacity; a failed allocation leaves capacity zero.
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
    capacity = 0;
    const char* const failure = "Can't allocate accelerated walahash buffers";
#if defined(WALAHASH_ESIMD)
    vectors.reset();
#endif
    mixed.reset();
    hashes.reset();
#if defined(WALAHASH_ESIMD)
    vectors = mom_usm_alloc<uint8_t>(queue, static_cast<size_t>(next_capacity) * 64, false, failure);
#endif
    hashes = mom_usm_alloc<uint8_t>(queue, static_cast<size_t>(next_capacity) * 32, false, failure);
    mixed = mom_usm_alloc<uint8_t>(queue, static_cast<size_t>(next_capacity) * 32, false, failure);
    capacity = next_capacity;
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
