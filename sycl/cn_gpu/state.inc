struct CnGpuState { // Per-device queue and persistent allocations; avoids buffer churn in the hot path.
  sycl::device device;
  sycl::queue queue;
  std::unique_ptr<MomKernelBundle> bundle;
  bool shared_scratch; // CPU or backends without device allocation need shared scratch/IO.
  bool shared_io;
  bool native_notice = false;
  uint8_t* inputs = nullptr, *outputs = nullptr;
  uint64_t* spads = nullptr, *lpads = nullptr;
  MomBufferAllocation<uint8_t> buffered_inputs, buffered_outputs;
  MomBufferAllocation<uint64_t> buffered_spads, buffered_lpads;
  CnGpuCudaExpand cuda_expand;
  unsigned input_cap = 0, batch_cap = 0;
  double wait_ema_us = 0.0; // EMA of recent batch wall times (us); paces the pre-read sleep below
  std::mutex mutex;
  CnGpuState(const std::string& dev_str)
    : device(get_dev(dev_str)),
      queue(device, cn_gpu_exception_handler(), sycl::property_list{sycl::property::queue::in_order{}}),
      shared_scratch(device.is_cpu() || !mom_has_usm_device(device)),
      shared_io(device.is_cpu() || sycl_is_level_zero_gpu(device) ||
                !mom_has_usm_device(device))
  {
    if constexpr (!mom_sycl_portable_opencl) {
      if (!mom_has_usm_shared(device) || (!device.is_cpu() && !mom_has_usm_device(device))) {
        throw std::string("cn/gpu SYCL device does not support required allocations");
      }
    }

    if (!mom_is_cuda(device)) {
      // The OpenCL "-cl-*" build options are an OpenCL/Level-Zero JIT mechanism for the Intel build.
      // On the CUDA backend, setting SYCL_PROGRAM_COMPILE_OPTIONS to them makes the AOT nvptx module
      // fail at runtime with UR_RESULT_ERROR_UNKNOWN, so skip it there; CUDA FP determinism comes
      // instead from -ffp-contract=off in the build flags (binding.gyp).
      set_sycl_env("SYCL_PROGRAM_COMPILE_OPTIONS", cn_gpu_fp_compile_options(device));
    }
    bundle = std::make_unique<MomKernelBundle>(
      mom_get_exec_bundle(queue.get_context())
    );
  }

  ~CnGpuState() { sycl_cleanup_noexcept("cn/gpu", [&] { release(); }); }

  void release() {
    if constexpr (mom_sycl_portable_opencl) {
      buffered_inputs.release(); buffered_spads.release();
      buffered_lpads.release(); buffered_outputs.release();
    } else {
      auto free_ptr = [&](auto*& ptr) { if (ptr) sycl::free(ptr, queue); ptr = nullptr; };
      free_ptr(inputs); free_ptr(spads); free_ptr(lpads); free_ptr(outputs);
    }
    input_cap = batch_cap = 0;
  }

  template <typename T> T* allocate(const size_t count, const bool shared) {
    return shared ? sycl::malloc_shared<T>(count, queue) : sycl::malloc_device<T>(count, queue);
  }

  void ensure_capacity(const unsigned batch, const unsigned input_size) {
    const unsigned input_bytes = input_size * batch;
    if (batch <= batch_cap && input_bytes <= input_cap) return;

    // Resize lazily; steady-state jobs reuse allocations.
    queue.wait_and_throw();
    release();

    input_cap = input_bytes; batch_cap = batch;
    if constexpr (mom_sycl_portable_opencl) {
      buffered_inputs.allocate(input_cap);
      buffered_spads.allocate(25 * batch_cap);
      buffered_lpads.allocate(CN_MEMORY8 * batch_cap);
      buffered_outputs.allocate(HASH_LEN * batch_cap);
    } else {
      inputs = allocate<uint8_t>(input_cap, shared_io);
      spads = allocate<uint64_t>(25 * batch_cap, shared_scratch);
      lpads = allocate<uint64_t>(CN_MEMORY8 * batch_cap, shared_scratch);
      outputs = allocate<uint8_t>(HASH_LEN * batch_cap, shared_io);
      if (!inputs || !spads || !lpads || !outputs) {
        release();
        throw std::string("Can't allocate cn/gpu SYCL buffers");
      }
    }
  }
};

using CnGpuStateMap = std::map<std::string, std::unique_ptr<CnGpuState>>;

static CnGpuStateMap& cn_gpu_states() {
  static CnGpuStateMap* const states = new CnGpuStateMap;
  return *states;
}

static std::mutex& cn_gpu_states_mutex() {
  static std::mutex* const mutex = new std::mutex;
  return *mutex;
}

static CnGpuState& cn_gpu_state(const std::string& dev_str) {
  std::lock_guard<std::mutex> lock(cn_gpu_states_mutex());
  auto& state = cn_gpu_states()[dev_str];
  if (!state) state = std::make_unique<CnGpuState>(dev_str);
  return *state;
}

void cn_gpu_cleanup_states() noexcept {
  try {
    std::lock_guard<std::mutex> lock(cn_gpu_states_mutex());
    cn_gpu_states().clear();
  } catch (...) {
    std::fprintf(stderr, "cn/gpu: ordered SYCL cleanup failed\n");
  }
}

struct CnGpuDevicePointers {
  uint8_t* inputs;
  uint64_t* spads;
  uint64_t* lpads;
  uint8_t* outputs;
  uint32_t* spads4;
  int32_t* lpads4;
  sycl::uint4* lpads16;
};

template <bool Buffered>
class CnGpuViews;

template <>
class CnGpuViews<false> {
  CnGpuDevicePointers pointers_;

public:
  CnGpuViews(CnGpuState& state, sycl::handler& handler)
    : pointers_{state.inputs, state.spads, state.lpads, state.outputs,
                reinterpret_cast<uint32_t*>(state.spads),
                reinterpret_cast<int32_t*>(state.lpads),
                reinterpret_cast<sycl::uint4*>(state.lpads)} {
    (void)handler;
  }

  CnGpuDevicePointers pointers() const { return pointers_; }
};

template <>
class CnGpuViews<true> {
  using ByteView = sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                                  sycl::target::device>;
  using WordView = sycl::accessor<uint64_t, 1, sycl::access_mode::read_write,
                                  sycl::target::device>;
  ByteView inputs_, outputs_;
  WordView spads_, lpads_;

public:
  CnGpuViews(CnGpuState& state, sycl::handler& handler)
    : inputs_(state.buffered_inputs.device_view<sycl::access_mode::read_write>(handler)),
      outputs_(state.buffered_outputs.device_view<sycl::access_mode::read_write>(handler)),
      spads_(state.buffered_spads.device_view<sycl::access_mode::read_write>(handler)),
      lpads_(state.buffered_lpads.device_view<sycl::access_mode::read_write>(handler)) {}

  CnGpuDevicePointers pointers() const {
    auto* const inputs = inputs_.template get_multi_ptr<sycl::access::decorated::no>().get();
    auto* const spads = spads_.template get_multi_ptr<sycl::access::decorated::no>().get();
    auto* const lpads = lpads_.template get_multi_ptr<sycl::access::decorated::no>().get();
    auto* const outputs = outputs_.template get_multi_ptr<sycl::access::decorated::no>().get();
    return {inputs, spads, lpads, outputs, reinterpret_cast<uint32_t*>(spads),
            reinterpret_cast<int32_t*>(lpads), reinterpret_cast<sycl::uint4*>(lpads)};
  }
};

using CnGpuActiveViews = CnGpuViews<mom_sycl_portable_opencl>;
