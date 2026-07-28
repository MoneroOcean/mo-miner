enum class PearlHashSearchBackend { sycl, amd_sycl_wmma, hip_jit, cuda_jit };

static PearlHashSearchBackend attempt(
    sycl::queue& q, const Buffers& b, uint32_t seed, int m, int n, int k, int rank,
    const std::string& backend = "auto", PearlHashHipSearch* hip = nullptr,
    PearlHashCudaSearch* cuda = nullptr, PearlHashAmdWmmaSearch* amd_wmma = nullptr,
    sycl::queue* hip_prep_queue = nullptr, bool validate_native = false) {
  static thread_local unsigned stats_attempts = 0;
  const bool stats = std::getenv("MOM_PEARLHASH_STATS") && stats_attempts++ < 3;
  auto stats_last = std::chrono::steady_clock::now();
  auto mark = [&](const char* stage) {
    if (!stats) return;
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
    k_roots(q, b, seed, m, n, k);        // commitment roots cA/cB (A/Bt regenerated from RNG)
  mark("roots");
#if defined(_WIN32)
  if (windows_cpu) prepare_noise_cpu(b, m, n, k, rank);
  else
#endif
    k_noise(q, b, m, n, k, rank);        // sparse low-rank noise E_AL/E_AR, E_BL/E_BR
  mark("noise");

  // Pick the search kernel by the running device's backend BEFORE laying out A'/B': compute_ab needs to
  // know whether to write the portable row-major A' / column-major B' (search()) or the ESIMD tile-major
  // layout. CUDA -> search_cuda (mma.sync, nvptx). Any OpenCL device (AMD GPU, the CPU device, and
  // Intel OpenCL) -> the portable dp4a search() -- it is the single matrix-hardware-free path, so
  // the same image runs on AMD and the CPU and Intel OpenCL behaves like another OpenCL GPU. (Set
  // MOM_PEARLHASH_ESIMD to opt an Intel-OpenCL *GPU* back into the faster ESIMD path.) Level-Zero / default
  // Intel -> ESIMD. search_cuda/search_esimd/search() are each compiled only in the builds that can run
  // them, so the references below are guarded to match.
#if defined(MOM_SYCL_HAS_CUDA)
  #if defined(MOM_SYCL_ADAPTIVECPP)
  const bool is_cuda = q.get_device().get_backend() == sycl::backend::cuda;
  #else
  const bool is_cuda = q.get_device().get_backend() == sycl::backend::ext_oneapi_cuda;
  #endif
#else
  const bool is_cuda = false;
#endif
  const bool dbg = validate_native || std::getenv("MOM_PEARLHASH_CHK") != nullptr;
  const bool tuned_sycl = backend == "auto" || backend == "sycl-native" ||
                          backend == "native";
  // The portable search() is compiled into every build that has attempt() (i.e. not the ESIMD-only TU),
  // so it is always callable here -- including the Windows/dpcpp Intel build, which has search_esimd in
  // the SAME TU. Route any OpenCL device (AMD GPU, the CPU device, Intel OpenCL) to it; an
  // Intel-OpenCL GPU can opt back into ESIMD with MOM_PEARLHASH_ESIMD. CUDA -> search_cuda; Level-Zero -> ESIMD.
  bool use_portable = !tuned_sycl;
#if defined(PEARLHASH_ESIMD) || defined(MOM_PEARLHASH_HAS_ESIMD)
#if defined(MOM_SYCL_ADAPTIVECPP)
  if (tuned_sycl && !is_cuda && q.get_device().get_backend() == sycl::backend::ocl) {
#else
  if (tuned_sycl && !is_cuda && q.get_device().get_backend() == sycl::backend::opencl) {
#endif
    const bool is_gpu = q.get_device().is_gpu();
    use_portable = !(is_gpu && std::getenv("MOM_PEARLHASH_ESIMD"));
  }
#else
  use_portable = !tuned_sycl || !is_cuda;
#endif

#if defined(MOM_SYCL_HAS_HIP)
  if ((backend == "auto" || backend == "native") && hip && hip->ensure(q, m, n, k, rank)) {
    try {
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
    } catch (const std::string& error) {
      hip->fail(error);
      b.result->found = 0;
      b.result->chk = 0;
    } catch (const std::exception& error) {
      hip->fail(error.what());
      b.result->found = 0;
      b.result->chk = 0;
    }
  }
  // DPC++ HIP can lower its gfx12 builtin directly. Shipped AdaptiveCpp workers omit this
  // experimental branch and fall through to portable SYCL.
  if (tuned_sycl && amd_wmma && amd_wmma->ensure(q, m, n, k, rank)) {
    compute_ab(q, b, seed, m, n, k, rank, /*portable=*/true);
    mark("ab");
    amd_wmma->launch(q, b, seed, m, n, k, rank, dbg);
    return PearlHashSearchBackend::amd_sycl_wmma;
  }
#endif
#if defined(MOM_SYCL_HAS_CUDA)
  if ((backend == "auto" || backend == "native") && is_cuda && cuda &&
      cuda->ensure(q, b, m, n, k, rank)) {
    try {
      // CUTE consumes the common row-major A / column-major B representation. The preparation
      // remains the shared SYCL kernel; only the matrix mainloop and transcript hash are native.
      compute_ab(q, b, seed, m, n, k, rank, /*portable=*/true);
      q.wait_and_throw();
      mark("ab");
      cuda->launch(b, seed, m, n, dbg);
      return PearlHashSearchBackend::cuda_jit;
    } catch (const std::string& error) {
      cuda->fail(error);
      b.result->found = 0;
      b.result->chk = 0;
    } catch (const std::exception& error) {
      cuda->fail(error.what());
      b.result->found = 0;
      b.result->chk = 0;
    }
  }
#endif
  if (q.get_device().is_cpu()) {
    q.wait_and_throw();
    compute_ab_cpu(b, seed, m, n, k, rank);
  } else {
    compute_ab(q, b, seed, m, n, k, rank, /*portable=*/use_portable);
  }
  mark("ab");
#if defined(MOM_SYCL_HAS_CUDA)
  if (is_cuda && tuned_sycl) {
    search_cuda(q, b, seed, m, n, k, rank);
    mark("search_cuda");
    return PearlHashSearchBackend::sycl;
  }
#endif
#ifndef MOM_PEARLHASH_ESIMD_TU   // search() is not compiled into the ESIMD-only TU (which only exports search_esimd)
  if (use_portable) {
    if (q.get_device().is_cpu())
      search_cpu(q, b, seed, m, n, k, rank, dbg);
    else
      search(q, b, seed, m, n, k, rank, dbg);
    mark("search_portable");
    return PearlHashSearchBackend::sycl;
  }   // portable dp4a int8 GEMM (AMD / CPU / Intel-OpenCL)
#endif
#if defined(PEARLHASH_ESIMD) || defined(MOM_PEARLHASH_HAS_ESIMD)
  search_esimd(q, b, seed, m, n, k, rank, dbg); // ESIMD register-resident DPAS path (Intel GPUs, Level-Zero, ~53 TH/s)
  mark("search_esimd");
#else
  search(q, b, seed, m, n, k, rank, dbg);       // unreached (use_portable already handled non-CUDA above)
  mark("search_portable");
#endif
  return PearlHashSearchBackend::sycl;
}

}  // namespace mom_pearlhash

// The standalone spir64-only ESIMD TU only needs the device kernels inside the namespace above; the
// host-side entry point and PlainProof builder below belong to the main TU.
