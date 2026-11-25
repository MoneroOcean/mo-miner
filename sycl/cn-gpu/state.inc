// Per-device queue and persistent allocations; avoids buffer churn in the hot path.
struct CnGpuState {
  sycl::device device;
  sycl::queue queue;
  bool shared_scratch; // CPU or backends without device allocation need shared scratch/IO.
  bool shared_io;
  bool native_notice = false;
  UsmPtr<uint8_t> inputs, outputs;
  UsmPtr<uint64_t> spads, lpads;
  MomBufferAllocation<uint8_t> buffered_inputs, buffered_outputs;
  MomBufferAllocation<uint64_t> buffered_spads, buffered_lpads;
  CnGpuCudaExpand cuda_expand;
  size_t input_cap = 0;
  unsigned batch_cap = 0;
  std::mutex mutex;
  // Bind batch buffers and deterministic FP policy to one in-order queue.
  CnGpuState(const std::string& dev_str)
      : device(get_dev(dev_str)), queue(device, cn_gpu_exception_handler(),
                                        sycl::property_list{sycl::property::queue::in_order{}}),
        shared_scratch(device.is_cpu() || !mom_has_usm_device(device)),
        shared_io(device.is_cpu() || sycl_is_level_zero_gpu(device) ||
                  !mom_has_usm_device(device)) {
    if constexpr (!mom_sycl_portable_opencl) {
      if (!mom_has_usm_shared(device) || (!device.is_cpu() && !mom_has_usm_device(device))) {
        throw std::string("cn/gpu SYCL device does not support required allocations");
      }
    }
  }

  // Drain SYCL kernels before the members release batch allocations.
  ~CnGpuState() {
    sycl_cleanup_noexcept("cn/gpu wait", [&] { queue.wait_and_throw(); });
  }

  // Release buffer or USM storage and invalidate capacity keys.
  void release() {
    if constexpr (mom_sycl_portable_opencl) {
      buffered_inputs.release();
      buffered_spads.release();
      buffered_lpads.release();
      buffered_outputs.release();
    } else {
      inputs.reset();
      spads.reset();
      lpads.reset();
      outputs.reset();
    }
    input_cap = batch_cap = 0;
  }

  // Resize only after queued users finish; failure leaves no reusable capacity.
  void ensure_capacity(const unsigned batch, const unsigned input_size) {
    const size_t requested_input_cap = static_cast<size_t>(input_size) * batch;
    if (batch <= batch_cap && requested_input_cap <= input_cap)
      return;

    // Resize lazily; steady-state jobs reuse allocations.
    queue.wait_and_throw();
    release();

    try {
      if constexpr (mom_sycl_portable_opencl) {
        buffered_inputs.allocate(requested_input_cap);
        buffered_spads.allocate(static_cast<size_t>(25) * batch);
        buffered_lpads.allocate(static_cast<size_t>(CN_MEMORY8) * batch);
        buffered_outputs.allocate(static_cast<size_t>(HASH_LEN) * batch);
      } else {
        const char* const failure = "Can't allocate cn/gpu SYCL buffers";
        inputs = mom_usm_alloc<uint8_t>(queue, requested_input_cap, shared_io, failure);
        spads = mom_usm_alloc<uint64_t>(queue, static_cast<size_t>(25) * batch, shared_scratch,
                                        failure);
        lpads = mom_usm_alloc<uint64_t>(queue, static_cast<size_t>(CN_MEMORY8) * batch,
                                        shared_scratch, failure);
        outputs = mom_usm_alloc<uint8_t>(queue, static_cast<size_t>(HASH_LEN) * batch, shared_io,
                                         failure);
      }
      input_cap = requested_input_cap;
      batch_cap = batch;
    } catch (...) {
      input_cap = batch_cap = 0;
      release();
      throw;
    }
  }
};

// Release device state before the compiler runtime unloads without escaping exceptions.
void cn_gpu_cleanup_states() noexcept {
  clearPersistentDeviceStates<CnGpuState>("cn/gpu");
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

template <bool Buffered> class CnGpuViews;

template <> class CnGpuViews<false> {
  // These typed fields intentionally alias the same scratchpad allocations. Kernels capturing this
  // view therefore cannot make a kernel-argument restrict promise across every build profile.
  CnGpuDevicePointers pointers_;

public:
  // Capture USM pointers with intentionally aliasing typed scratchpad views.
  CnGpuViews(CnGpuState& state, sycl::handler& handler)
      : pointers_{state.inputs,
                  state.spads,
                  state.lpads,
                  state.outputs,
                  reinterpret_cast<uint32_t*>(state.spads.get()),
                  reinterpret_cast<int32_t*>(state.lpads.get()),
                  reinterpret_cast<sycl::uint4*>(state.lpads.get())} {
    (void)handler;
  }

  // Device pointers captured by the kernels of this submission.
  CnGpuDevicePointers pointers() const {
    return pointers_;
  }
};

template <> class CnGpuViews<true> {
  using ByteView = sycl::accessor<uint8_t, 1, sycl::access_mode::read_write, sycl::target::device>;
  using WordView = sycl::accessor<uint64_t, 1, sycl::access_mode::read_write, sycl::target::device>;
  ByteView inputs_, outputs_;
  WordView spads_, lpads_;

public:
  // Bind buffer accessors to this submission before exposing their aliasing typed views.
  CnGpuViews(CnGpuState& state, sycl::handler& handler)
      : inputs_(state.buffered_inputs.device_view<sycl::access_mode::read_write>(handler)),
        outputs_(state.buffered_outputs.device_view<sycl::access_mode::read_write>(handler)),
        spads_(state.buffered_spads.device_view<sycl::access_mode::read_write>(handler)),
        lpads_(state.buffered_lpads.device_view<sycl::access_mode::read_write>(handler)) {
  }

  // Device pointers of the bound accessors, valid only inside this submission's kernels.
  CnGpuDevicePointers pointers() const {
    auto* const inputs = inputs_.template get_multi_ptr<sycl::access::decorated::no>().get();
    auto* const spads = spads_.template get_multi_ptr<sycl::access::decorated::no>().get();
    auto* const lpads = lpads_.template get_multi_ptr<sycl::access::decorated::no>().get();
    auto* const outputs = outputs_.template get_multi_ptr<sycl::access::decorated::no>().get();
    return {inputs,
            spads,
            lpads,
            outputs,
            reinterpret_cast<uint32_t*>(spads),
            reinterpret_cast<int32_t*>(lpads),
            reinterpret_cast<sycl::uint4*>(lpads)};
  }
};

using CnGpuActiveViews = CnGpuViews<mom_sycl_portable_opencl>;
