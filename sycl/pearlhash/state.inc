// Persistent queues, backend buffer layouts, and lazy winning-tile state.
template <typename T> static T* alloc_buffer(sycl::queue& q, size_t count) {
  return q.get_device().is_cpu() ? sycl::malloc_shared<T>(count, q)
                                 : sycl::malloc_device<T>(count, q);
}

// Allocate a mandatory buffer and fail before kernels can observe a null pointer.
template <typename T>
static T* alloc_required(sycl::queue& q, const size_t count, const char* const name) {
  T* const result = alloc_buffer<T>(q, count);
  if (!result)
    throw std::string("Can't allocate pearlhash ") + name;
  return result;
}

// Best-effort release of all partial/full buffer allocations; cleanup must not replace the original error.
static void free_allocations(sycl::queue& q, Buffers& b) noexcept {
  for (void* pointer :
       {static_cast<void*>(b.EAL), static_cast<void*>(b.EBR), static_cast<void*>(b.EBRt),
        static_cast<void*>(b.Ap), static_cast<void*>(b.Bp), static_cast<void*>(b.ApWmma),
        static_cast<void*>(b.BpWmma), static_cast<void*>(b.EARp1), static_cast<void*>(b.EARp2),
        static_cast<void*>(b.EBLq1), static_cast<void*>(b.EBLq2), static_cast<void*>(b.cA),
        static_cast<void*>(b.cB), static_cast<void*>(b.key), static_cast<void*>(b.target),
        static_cast<void*>(b.CVA), static_cast<void*>(b.CVB),
        static_cast<void*>(b.transcript), static_cast<void*>(b.result)}) {
    if (pointer)
      try {
        sycl::free(pointer, q);
      } catch (...) {
      }
  }
  b = Buffers{};
}

// Allocate the selected backend layout with checked dimensions and unwind partial failures.
static Buffers alloc_buffers(sycl::queue& q, int m, int n, int k, int rank, bool hip_jit = false,
                             bool cuda_jit = false) {
  static_assert(sizeof(Result) <= pearlhash_result_allocation_bytes);
  Buffers b{};
  PearlHashAllocationCounts counts{};
  if (!pearlhash_allocation_counts({static_cast<unsigned>(m), static_cast<unsigned>(n),
                                   static_cast<unsigned>(k), static_cast<unsigned>(rank)},
                                  counts, hip_jit))
    throw std::string("Invalid pearlhash allocation dimensions");
  try {
    b.EAL = alloc_required<int8_t>(q, counts.m_rank, "EAL");
    b.EBR = alloc_required<int8_t>(q, counts.n_rank, "EBR");
    b.EBRt = alloc_required<int8_t>(q, counts.n_rank, "EBRt");
    b.Ap = alloc_required<int8_t>(q, counts.m_k, "Ap");
    b.Bp = alloc_required<int8_t>(q, counts.n_k, "Bp");
#if defined(MOM_SYCL_HAS_HIP)
    // HIPRTC WMMA consumes an architecture-native fragment layout; keep its matrices separate
    // from the portable SYCL layout used when source-JIT initialization is unavailable.
    if (hip_jit) {
      b.ApWmma = alloc_required<int8_t>(q, counts.m_k, "WMMA Ap");
      b.BpWmma = alloc_required<int8_t>(q, counts.n_k, "WMMA Bp");
    }
#endif
    b.EARp1 = alloc_required<int32_t>(q, static_cast<size_t>(k), "EARp1");
    b.EARp2 = alloc_required<int32_t>(q, static_cast<size_t>(k), "EARp2");
    b.EBLq1 = alloc_required<int32_t>(q, static_cast<size_t>(k), "EBLq1");
    b.EBLq2 = alloc_required<int32_t>(q, static_cast<size_t>(k), "EBLq2");
    b.cA = alloc_required<uint8_t>(q, 32, "cA");
    b.cB = alloc_required<uint8_t>(q, 32, "cB");
    b.key = alloc_required<uint8_t>(q, 32, "key");
    b.target = alloc_required<uint8_t>(q, 32, "target");
    b.CVA = alloc_required<uint8_t>(q, counts.cva_bytes, "CVA");
    b.CVB = alloc_required<uint8_t>(q, counts.cvb_bytes, "CVB");
    // Keep BLAKE3 separate from the matrix loop to preserve occupancy. CUDA processes row slices:
    // the full reference profile would otherwise need a 16-GiB transcript in addition to A'/B'.
    // Other backends still require a complete transcript and keep their existing 256-MiB cap.
    const int transcript_rows = cuda_jit
        ? std::min(m, static_cast<int>(((1ULL << 32) / static_cast<unsigned>(n)) / 128) * 128)
        : m;
    const uint64_t transcript_bytes =
        static_cast<uint64_t>(transcript_rows / 16) * (n / 16) * 16U * sizeof(uint32_t);
    const uint64_t transcript_limit =
        cuda_jit ? 1ULL << 30 : 256ULL * 1024U * 1024U;
    const sycl::device device = q.get_device();
    const PowDeviceProfile capacity = {
        1, device.get_info<sycl::info::device::global_mem_size>(),
        device.get_info<sycl::info::device::max_mem_alloc_size>()};
    const bool transcript_fits = !cuda_jit ||
        pearlhash_optional_allocation_fits(capacity, counts, transcript_bytes);
    if (transcript_bytes > 0 && transcript_bytes <= transcript_limit && transcript_fits) {
      b.transcript =
          alloc_buffer<uint32_t>(q, static_cast<size_t>(transcript_bytes / sizeof(uint32_t)));
      if (b.transcript)
        b.transcript_rows = transcript_rows;
    }
    b.result = sycl::malloc_shared<Result>(1, q);
    if (!b.result)
      throw std::string("Can't allocate pearlhash result");
    b.result->found = 0;
    return b;
  } catch (...) {
    free_allocations(q, b);
    throw;
  }
}

namespace {
// PlainProof carries M/N/K/rank, so the job can select a device-efficient profile without changing
// proof semantics. mom.js obtains defaults from GPU-CONFIG.md and passes explicit config values.
// One persistent in-order queue + device buffer set per GPU; the seed search reuses them.
struct PearlHashState {
  sycl::queue queue;
  sycl::queue prep_queue;
#ifdef MOM_SYCL_HAS_CUDA
  const bool cuda_tensor = mom::nvidia::has_int8_async_matrix(
      mom::nvidia::compute_capability(queue.get_device()));
#else
  static constexpr bool cuda_tensor = false;
#endif
#if defined(MOM_SYCL_ADAPTIVECPP_CUDA)
  const bool cuda_f16 = [&] {
    const int sm = mom::nvidia::compute_capability(queue.get_device());
    return mom::nvidia::has_f16_mma_8x8x4(sm) && !mom::nvidia::has_int8_mma(sm);
  }();
#else
  static constexpr bool cuda_f16 = false;
#endif
  PearlHashHipSearch hip;
  PearlHashCudaSearch cuda;
  int m = 0, n = 0, k = 0, rank = 0;
  bool hip_jit_buffers = false;
  bool cuda_jit_buffers = false;
  Buffers b{};
  uint8_t key[32] = {0}, header[76] = {0};
  bool have_header = false;
  bool native_notice = false;
  double wait_ema_us = 0.0; // EMA of recent attempt wall times (us); paces the pre-wait sleep below
  std::mutex mutex;
  explicit PearlHashState(const std::string& dev_str)
      : queue(get_dev(dev_str), sycl::property::queue::in_order{}),
        prep_queue(queue.get_context(), queue.get_device(), sycl::property::queue::in_order{}) {
  }
  // Drain both preparation/search queues before releasing native modules and their shared buffer graph.
  void release() noexcept {
    sycl_cleanup_noexcept("pearlhash queue", [&] {
      queue.wait_and_throw();
    });
    sycl_cleanup_noexcept("pearlhash prep queue", [&] {
      prep_queue.wait_and_throw();
    });
    hip.release();
    cuda.release();
    free_allocations(queue, b);
    m = n = k = rank = 0;
    hip_jit_buffers = false;
    cuda_jit_buffers = false;
    have_header = false;
  }
  // Reallocate only when dimensions or native buffer layouts change; invalidate the cached job key afterward.
  void ensure(int m_, int n_, int k_, int rank_, bool hip_jit_, bool cuda_jit_) {
    if (b.Ap && m == m_ && n == n_ && k == k_ && rank == rank_ && hip_jit_buffers == hip_jit_ &&
        cuda_jit_buffers == cuda_jit_)
      return;
    release();
    Buffers next = alloc_buffers(queue, m_, n_, k_, rank_, hip_jit_, cuda_jit_);
    b = next;
    m = m_;
    n = n_;
    k = k_;
    rank = rank_;
    hip_jit_buffers = hip_jit_;
    cuda_jit_buffers = cuda_jit_;
    have_header = false;
  }
  ~PearlHashState() {
    release();
  }
};
// Keep the registry heap-owned to avoid unsafe runtime destruction at process exit.
DeviceStateRegistry<PearlHashState>& pearlhash_registry() {
  static auto* const registry = new DeviceStateRegistry<PearlHashState>;
  return *registry;
}
// Reuse one queue/buffer owner per device; its mutex serializes profile changes and searches.
PearlHashState& pearlhash_state(const std::string& dev_str) {
  return pearlhash_registry().get(dev_str, [&] {
    return std::make_unique<PearlHashState>(dev_str);
  });
}
thread_local std::string
    g_pearlhash_proof; // last built proof, fetched right after pearlhash() returns 1
// Winning tile captured by pearlhash() on a find; consumed by pearlhash_claim() to build the proof lazily.
thread_local struct PearlHashFound {
  uint32_t seed;
  int row, col, m, n, k, rank;
  uint8_t key[32];
  uint8_t jackpot[32];
  uint32_t adjustment_factor;
  bool claim_ready;
  bool valid;
} g_pf{};
} // namespace
