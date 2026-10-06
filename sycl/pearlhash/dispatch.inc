enum class PearlHashSearchBackend {
  sycl,
  hip_jit,
  cuda_jit
};

static PearlHashSearchBackend
attempt(sycl::queue& q, const Buffers& b, uint32_t seed, int m, int n, int k, int rank,
        const std::string& backend = "auto", PearlHashHipSearch* hip = nullptr,
        PearlHashCudaSearch* cuda = nullptr, sycl::queue* hip_prep_queue = nullptr,
        bool validate_native = false, bool cuda_tensor = false) {
  static thread_local unsigned stats_attempts = 0;
  const bool stats = std::getenv("MOM_PEARLHASH_STATS") && stats_attempts++ < 3;
  auto stats_last = std::chrono::steady_clock::now();
  auto mark = [&](const char* stage) {
    if (!stats)
      return;
    q.wait_and_throw();
    const auto now = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(now - stats_last).count();
    std::fprintf(stderr, "PEARLHASH_SYCL_STATS %s=%.3fms\n", stage, ms);
    stats_last = now;
  };
#if defined(_WIN32)
  const bool windows_cpu = q.get_device().is_cpu();
  if (windows_cpu) {
    q.wait_and_throw();
    prepare_roots_cpu(b, seed, m, n, k);
  } else
#endif
    k_roots(q, b, seed, m, n, k); // commitment roots cA/cB (A/Bt regenerated from RNG)
  mark("roots");
#if defined(_WIN32)
  if (windows_cpu)
    prepare_noise_cpu(b, m, n, k, rank);
  else
#endif
    k_noise(q, b, m, n, k, rank); // sparse low-rank noise E_AL/E_AR, E_BL/E_BR
  mark("noise");

  // Pick the search kernel by the running device's backend BEFORE laying out A'/B': compute_ab needs to
  // know whether to write the portable row-major A' / column-major B' (search()) or the ESIMD tile-major
  // layout. CUDA -> search_cuda (mma.sync, nvptx). Any OpenCL device (AMD GPU, the CPU device, and
  // Intel OpenCL) -> the portable dp4a search() -- it is the single matrix-hardware-free path, so
  // the same image runs on AMD and the CPU and Intel OpenCL behaves like another OpenCL GPU. (Set
  // MOM_PEARLHASH_ESIMD to opt an Intel-OpenCL *GPU* back into the faster ESIMD path.) Level-Zero / default
  // Intel -> ESIMD. search_cuda/search_esimd/search() are each compiled only in the builds that can run
  // them, so the references below are guarded to match.
#if defined(MOM_SYCL_HAS_CUDA) || defined(MOM_SYCL_ADAPTIVECPP_CUDA)
#if defined(MOM_SYCL_ADAPTIVECPP)
  const bool is_cuda = q.get_device().get_backend() == sycl::backend::cuda;
#else
  const bool is_cuda = q.get_device().get_backend() == sycl::backend::ext_oneapi_cuda;
#endif
#else
  const bool is_cuda = false;
#endif
  const bool dbg = validate_native || std::getenv("MOM_PEARLHASH_CHK") != nullptr;
  const bool tuned_sycl = backend == "auto" || backend == "sycl-native" || backend == "native";
#if defined(MOM_SYCL_HAS_CUDA)
  const bool cuda_sycl =
      is_cuda && tuned_sycl && cuda_tensor && cuda_sycl_search_supports_shape(m, n);
#else
  (void)cuda_tensor;
  [[maybe_unused]] const bool cuda_sycl = false;
#endif
  // Decide before compute_ab chooses its layout. Optional ESIMD queries fail closed; unsupported
  // devices and explicit overrides retain the ordinary matrix layout and search.
  bool use_portable = !cuda_sycl;
#if defined(PEARLHASH_ESIMD) || defined(MOM_PEARLHASH_HAS_ESIMD)
  const unsigned esimd_width = [&] {
    if (!tuned_sycl || is_cuda)
      return 0u;
    try {
      const sycl::device device = q.get_device();
      const bool supported_backend = sycl_is_level_zero_gpu(device) ||
          (mom_is_opencl(device) && std::getenv("MOM_PEARLHASH_ESIMD") != nullptr);
      // Integrated Intel GPUs can lack the LSC prefetch used by this search even with matrix support.
      const bool discrete_gpu = device.is_gpu() && !is_integrated_gpu(device);
      const bool intel_matrix = discrete_gpu && device.has(sycl::aspect::ext_intel_matrix);
      // Query the DPAS layout before preparation; unknown widths retain the portable reference.
      const unsigned width =
          supported_backend && intel_matrix ? esimd_search_width(device) : 0u;
      return pearlhash_esimd_route(tuned_sycl, supported_backend, discrete_gpu, intel_matrix, width)
          ? width : 0u;
    } catch (...) {
      return 0u;
    }
  }();
  use_portable = !cuda_sycl && esimd_width == 0u;
#endif
  // The SYCL tensor kernel has the same SM80 instruction floor as CUTE; failed capability
  // detection must retain the ordinary matrix layout/search even for an explicit native request.
  if (is_cuda && !cuda_sycl)
    use_portable = true;

  // Only JIT unavailability permits fallback. Preparation, launch and completion faults must
  // escape: retrying another kernel on the same potentially invalid queue is not recovery.
#if defined(MOM_SYCL_HAS_HIP)
  if ((backend == "auto" || backend == "native") && hip && hip->ensure(q, m, n, k, rank)) {
    q.wait_and_throw(); // both independent prep streams consume the completed noise tables
    sycl::queue& prep = hip_prep_queue ? *hip_prep_queue : q;
    PearlHashPrepEvents ready = compute_ab_amd_wmma(q, prep, b, seed, m, n, k, rank);
    while (ready.a.get_info<sycl::info::event::command_execution_status>() !=
               sycl::info::event_command_status::complete ||
           ready.b.get_info<sycl::info::event::command_execution_status>() !=
               sycl::info::event_command_status::complete)
#if defined(_WIN32)
      mom_sycl_poll_pause();
#else
      std::this_thread::sleep_for(std::chrono::microseconds(100));
#endif
    ready.a.wait_and_throw();
    ready.b.wait_and_throw();
    mark("ab");
    hip->launch(b, seed, m, n, k, rank, dbg);
    return PearlHashSearchBackend::hip_jit;
  }
#endif
#if defined(MOM_SYCL_HAS_CUDA)
  if ((backend == "auto" || backend == "native") && is_cuda && cuda &&
      cuda->ensure(q, b, m, n, k, rank)) {
    // CUTE consumes the common row-major A / column-major B representation. The preparation
    // remains the shared SYCL kernel; only the matrix mainloop and transcript hash are native.
    compute_ab(q, b, seed, m, n, k, rank, /*portable=*/true);
    q.wait_and_throw();
    mark("ab");
    cuda->launch(b, seed, m, n, k, dbg);
    return PearlHashSearchBackend::cuda_jit;
  }
#endif
#if defined(PEARLHASH_ESIMD)
  // Paired DG2 proofs/rates are qualified on oneAPI Level Zero; combined AOT/OpenCL stay ordinary.
  const bool use_dpasw = pearlhash_dpasw_route(
      !use_portable && q.get_backend() == sycl::backend::ext_oneapi_level_zero, esimd_width,
      esimd_width == 8u && dpasw_esimd_device(q.get_device()), rank);
#endif
  if (q.get_device().is_cpu()) {
    q.wait_and_throw();
    compute_ab_cpu(b, seed, m, n, k, rank);
  } else {
    // CUDA uses the same column-major B' layout in both generators. Tile an oversized B' to
    // avoid DPC++'s signed 1D range limit without changing the selected CUDA search.
    const bool tiled_cuda_b =
        cuda_sycl && checked_elements(n, k, "N*K") >
                         static_cast<size_t>(std::numeric_limits<int>::max());
#if defined(PEARLHASH_ESIMD)
    if (use_dpasw)
      compute_ab<true>(q, b, seed, m, n, k, rank, /*portable=*/false);
    else
#endif
      compute_ab(q, b, seed, m, n, k, rank, /*portable=*/use_portable || tiled_cuda_b);
  }
  mark("ab");
#if defined(MOM_SYCL_HAS_CUDA)
  if (cuda_sycl) {
    search_cuda(q, b, seed, m, n, k, rank, dbg);
    mark("search_cuda");
    return PearlHashSearchBackend::sycl;
  }
#endif
#ifndef MOM_PEARLHASH_ESIMD_TU // search() is not compiled into the ESIMD-only TU (which only exports search_esimd)
  if (use_portable) {
    if (q.get_device().is_cpu())
      search_cpu(q, b, seed, m, n, k, rank, dbg);
#if defined(MOM_SYCL_ADAPTIVECPP_CUDA)
    else if (is_cuda && tuned_sycl)
      search_t<pearlhash_portable_er, pearlhash_portable_ec, true>(q, b, seed, m, n, k, rank, dbg);
#endif
    else
      search(q, b, seed, m, n, k, rank, dbg);
    mark("search_portable");
    return PearlHashSearchBackend::sycl;
  } // portable dp4a int8 GEMM (AMD / CPU / Intel-OpenCL)
#endif
#if defined(PEARLHASH_ESIMD) || defined(MOM_PEARLHASH_HAS_ESIMD)
  // B580: about 45.4 TH/s with the canonical M=N=131072, K=2048, rank=128 profile.
#if defined(PEARLHASH_ESIMD)
  if (use_dpasw) {
    search_esimd_dpasw(q, b, seed, m, n, k, rank, dbg);
    mark("search_esimd_dpasw");
  } else
#endif
  {
    search_esimd(q, b, seed, m, n, k, rank, dbg, esimd_width);
    mark("search_esimd");
  }
#else
  search(q, b, seed, m, n, k, rank, dbg); // Defensive fallback if future backend selection skips use_portable.
  mark("search_portable");
#endif
  return PearlHashSearchBackend::sycl;
}

} // namespace mom_pearlhash

// The standalone spir64-only ESIMD TU only needs the device kernels inside the namespace above; the
// host-side entry point and PlainProof builder below belong to the main TU.
