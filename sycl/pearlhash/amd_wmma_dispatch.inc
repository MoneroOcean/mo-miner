

struct PearlHashAmdWmmaSearch {
#if defined(MOM_SYCL_HAS_HIP) && !defined(MOM_SYCL_ADAPTIVECPP)
  static bool disabled() {
    const char* value = std::getenv("MOM_PEARLHASH_AMD_WMMA");
    return value && (!std::strcmp(value, "0") || !std::strcmp(value, "false"));
  }

  static bool is_gfx12() {
    int device = 0;
    hipDeviceProp_t properties{};
    const hipError_t current = hipGetDevice(&device);
    const hipError_t info = current == hipSuccess
      ? hipGetDeviceProperties(&properties, device) : current;
    if (std::getenv("MOM_PEARLHASH_AMD_WMMA_LOG"))
      std::fprintf(stderr, "pearlhash: HIP device=%d current=%d properties=%d arch=%s\n",
                   device, static_cast<int>(current), static_cast<int>(info), properties.gcnArchName);
    return current == hipSuccess && info == hipSuccess &&
           !std::strncmp(properties.gcnArchName, "gfx12", 5);
  }

  bool ensure(sycl::queue& q, int m, int n, int k, int rank) {
    const bool is_hip = q.get_device().get_backend() == sycl::backend::ext_oneapi_hip;
    const bool off = disabled();
    const bool gfx12 = is_gfx12();
    if (std::getenv("MOM_PEARLHASH_AMD_WMMA_LOG"))
      std::fprintf(stderr, "pearlhash: DPC++ WMMA hip=%d disabled=%d gfx12=%d shape=%dx%d/%d/%d\n",
                   is_hip, off, gfx12, m, n, k, rank);
    if (!is_hip || off || !gfx12) return false;
    if ((k != 4096 || rank != 256) &&
        (k != 2048 || rank != 128) &&
        (k != 1024 || rank != 64)) return false;
    if (m % 64 || n % 32) return false;
    const uint64_t waves = static_cast<uint64_t>(m / 64) * static_cast<uint64_t>(n / 32);
    if (waves % 8) return false;  // eight wave32 sub-groups per 256-thread work-group
    return true;
  }

  void launch(sycl::queue& q, const Buffers& b, uint32_t seed, int m, int n,
              int k, int rank, bool debug) {
    if (k == 4096 && rank == 256)
      search_amd_wmma_t<4096, 256>(q, b, seed, m, n, debug);
    else if (k == 2048 && rank == 128)
      search_amd_wmma_t<2048, 128, 4, 4>(q, b, seed, m, n, debug);
    else
      search_amd_wmma_t<1024, 64>(q, b, seed, m, n, debug);
  }

  void release(sycl::queue&) noexcept {}
#else
  bool ensure(sycl::queue&, int, int, int, int) { return false; }
  void launch(sycl::queue&, const Buffers&, uint32_t, int, int, int, int, bool) {}
  void release(sycl::queue&) noexcept {}
#endif
};
