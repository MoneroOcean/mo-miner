#if defined(MOM_SYCL_HAS_CUDA) && !defined(__SYCL_DEVICE_ONLY__) && !defined(MOM_PEARLHASH_ESIMD_TU)
using PearlHashCudaDriverApi = mom::CudaDriverApi;
using PearlHashNvrtcApi = mom::NvrtcApi;
using mom::cuda_check;
using mom::rtc_error;

static std::filesystem::path
pearlhash_cuda_header_root(const char* configured,
                           const std::vector<std::filesystem::path>& candidates,
                           const std::filesystem::path& required) {
  if (configured && *configured) {
    const std::filesystem::path path(configured);
    if (std::filesystem::exists(path / required))
      return path;
  }
  for (const auto& path : candidates)
    if (std::filesystem::exists(path / required))
      return path;
  return {};
}
#include "cuda_search.inc"
#else
struct PearlHashCudaSearch {
  static bool is_cuda(const sycl::device&) {
    return false;
  }
  static bool supports(sycl::queue&, std::string* const reason = nullptr) {
    if (reason)
      *reason = "CUDA source-JIT support is not part of this worker";
    return false;
  }
  static bool supports_shape(int, int, int, int) {
    return false;
  }
  bool ensure(sycl::queue&, const Buffers&, int, int, int, int) {
    return false;
  }
  void launch(const Buffers&, uint32_t, int, int, int, bool) {
  }
  void wait(double) {
  }
  void release() noexcept {
  }
  void fail(const std::string&) noexcept {
  }
};
#endif
