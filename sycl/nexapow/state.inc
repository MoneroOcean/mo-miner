// Per-device monolithic/staged search resources; callers hold State::mutex.
class SearchKernel;
void nexapow_test_sha256d_49(sycl::queue&, const uint8_t[49], uint8_t[32]);
class State {
public:
  sycl::device device;
  sycl::queue queue;
  UsmPtr<uint8_t> header, extranonce, target;
  UsmPtr<Result> result;
  NexaPowSyclSearch portable;
  bool staged_logged = false;
  bool staged_failure_logged = false;
  std::mutex mutex;

  // Create the in-order device owner; sycl_set_compile_options() supplies CUDA build options.
  explicit State(const std::string& dev)
      : device(get_dev(dev)), queue(device, sycl::property::queue::in_order{}) {
    if (!mom_has_usm_device(device))
      throw std::string("nexapow requires SYCL device USM");
  }

  // Allocate the small monolithic-search work buffers lazily. `result` is allocated last, so a
  // partial allocation is retried on the next call.
  void init_portable() {
    if (result)
      return;
    const char* const failure = "Can't allocate nexapow buffers";
    header = mom_usm_alloc<uint8_t>(queue, 32, false, failure);
    extranonce = mom_usm_alloc<uint8_t>(queue, 8, false, failure);
    target = mom_usm_alloc<uint8_t>(queue, 32, false, failure);
    result = mom_usm_alloc<Result>(queue, 1, false, failure);
  }

  // Drain accepted work before the staged search and the members release their allocations.
  ~State() {
    sycl_cleanup_noexcept("nexapow wait", [&] { queue.wait_and_throw(); });
    portable.release(queue);
  }
};

// Program build options for sycl_set_compile_options(): the native-field CUDA kernels need
// --maxrregcount=128 unless MOM_NEXAPOW_COMPILE_OPTIONS or an existing
// SYCL_PROGRAM_COMPILE_OPTIONS says otherwise; other devices take none.
const char* nexapow_compile_options(const sycl::device& device) {
#ifdef MOM_NEXAPOW_SYCL_NATIVE_FIELD
  if (mom_is_cuda(device)) {
    if (const char* const options = std::getenv("MOM_NEXAPOW_COMPILE_OPTIONS"))
      return options;
    return std::getenv("SYCL_PROGRAM_COMPILE_OPTIONS") ? nullptr : "--maxrregcount=128";
  }
#else
  (void)device;
#endif
  return nullptr;
}
