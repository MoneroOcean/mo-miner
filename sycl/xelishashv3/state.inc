// Per-device scratch/LUT ownership; callers serialize resizing and search through State::mutex.
class State {
public:
  sycl::device device;
  sycl::queue queue;
  UsmPtr<uint8_t> input, target;
  UsmPtr<uint64_t> scratch;
  UsmPtr<double> lut;
  UsmPtr<PowHits<>> result;
  unsigned capacity = 0;
  std::mutex mutex;
  const bool profile;
  unsigned profile_reports = 0;
  // Allocate fixed work/LUT state on an in-order queue.
  explicit State(const std::string& dev)
      : device(get_dev(dev)),
        queue(device,
              sycl::async_handler([](sycl::exception_list errors) {
                for (const std::exception_ptr& error : errors)
                  std::rethrow_exception(error);
              }),
              sycl::property_list{sycl::property::queue::in_order{}}),
        // Diagnostic: report per-stage search timings.
        profile(mom_env_flag("MOM_XELISHASHV3_PROFILE").value_or(false)) {
    if (!xelishashv3_supported(device))
      throw std::string("xelishashv3 requires SYCL device USM");
    const char* const failure = "Can't allocate xelishashv3 buffers";
    input = mom_usm_alloc<uint8_t>(queue, 112, false, failure);
    target = mom_usm_alloc<uint8_t>(queue, 32, false, failure);
    result = mom_usm_alloc<PowHits<>>(queue, 1, false, failure);
    if (device.has(sycl::aspect::fp64)) {
      lut = mom_usm_alloc<double>(queue, 512, false, "Can't allocate xelishashv3 reciprocal table");
      double host[512];
      for (unsigned i = 0; i < 512; ++i)
        host[i] = 1.0 / ((512.5 + i) * (1u << 22));
      sycl_wait_and_throw(queue.memcpy(lut, host, sizeof(host)), device);
    }
  }
  // Drain kernels before the members free their work, LUT, result, and scratch allocations.
  ~State() {
    sycl_cleanup_noexcept("xelishashv3 wait", [&] { queue.wait_and_throw(); });
  }
  // Grow per-nonce scratch within both allocation and total-memory limits. A failed allocation
  // keeps the previous scratch; the old buffer is released only after its replacement exists.
  void ensure(unsigned count) {
    if (static_cast<uint64_t>(count) > xelishashv3_scratch_capacity(device))
      throw std::string("xelishashv3 intensity exceeds device memory limit");
    const size_t max_count = std::numeric_limits<size_t>::max() / WORDS;
    if (static_cast<uint64_t>(count) > max_count)
      throw std::string("xelishashv3 intensity is too large");
    if (count <= capacity)
      return;
    queue.wait_and_throw();
    scratch = mom_usm_alloc<uint64_t>(queue, static_cast<size_t>(count) * WORDS, false,
                                      "Can't allocate xelishashv3 scratchpads");
    capacity = count;
  }
};
