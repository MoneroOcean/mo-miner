namespace {

struct WalaHashAmdWmma {
  hipModule_t module = nullptr;
  hipFunction_t kernel = nullptr;
  hipDevice_t device = -1;
  bool checked = false;

  bool ensure(sycl::queue& queue) {
    if (checked)
      return kernel != nullptr;
    checked = true;
    if (!mom::hip_queue_device(queue, device) || hipSetDevice(device) != hipSuccess)
      return false;
    hipDeviceProp_t properties{};
    mom::HiprtcApi& rtc = mom::HiprtcApi::instance();
    if (hipGetDeviceProperties(&properties, device) != hipSuccess ||
        !mom::amd::has_gfx12_int8_wmma(properties.gcnArchName) || !rtc.available())
      return false;
    std::string arch = properties.gcnArchName;
    if (const size_t colon = arch.find(':'); colon != std::string::npos)
      arch.resize(colon);
    const char source[] = R"(
typedef int v2i __attribute__((ext_vector_type(2)));
typedef int v8i __attribute__((ext_vector_type(8)));
__device__ __forceinline__ unsigned expand_nibbles(unsigned x) {
  return ((x >> 4) & 15) | ((x & 15) << 8) | ((x & 0xf000) << 4) |
         ((x & 0x0f00) << 16);
}
extern "C" __global__ void walahash_wmma(
    const signed char* A, const unsigned char* hashes, unsigned char* mixed, unsigned N) {
  constexpr unsigned K = 64, ER = 4, EC = 4;
  const unsigned lane = threadIdx.x & 31;
  const unsigned wave = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  const unsigned tiles = N / 64;
  if (wave >= tiles) {
    return;
  }
  const unsigned col = wave * 64;
  const unsigned rc = lane & 15;
  const unsigned kh = (lane & 16) ? 8 : 0;
  const int* aw = (const int*)A;
  const unsigned* hw = (const unsigned*)hashes;
  unsigned* mw = (unsigned*)mixed;
  v8i acc[ER][EC] = {};
  for (unsigned k = 0; k < K; k += 16) {
    v2i a[ER], b[EC];
    #pragma unroll
    for (unsigned i = 0; i < ER; ++i) {
      const unsigned long long w =
          ((unsigned long long)(i * 16 + rc) * K + k + kh) / 4;
      a[i] = {aw[w], aw[w + 1]};
    }
    #pragma unroll
    for (unsigned j = 0; j < EC; ++j) {
      const unsigned x = hw[(unsigned long long)(col + j * 16 + rc) * 8 + (k + kh) / 8];
      b[j] = {(int)expand_nibbles(x), (int)expand_nibbles(x >> 16)};
    }
    #pragma unroll
    for (unsigned i = 0; i < ER; ++i) {
      #pragma unroll
      for (unsigned j = 0; j < EC; ++j) {
        acc[i][j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
            true, a[i], true, b[j], acc[i][j], false);
      }
    }
  }
  #pragma unroll
  for (unsigned i = 0; i < ER; ++i) {
    #pragma unroll
    for (unsigned j = 0; j < EC; ++j) {
      const unsigned sample = col + j * 16 + rc;
      const unsigned row = i * 16 + 8 * (lane >> 4);
      // Each half-wave row starts on a word boundary; combine its four output bytes.
      unsigned packed = 0;
      #pragma unroll
      for (unsigned e = 0; e < 8; e += 2) {
        const int a = acc[i][j][e];
        const int b = acc[i][j][e + 1];
        const unsigned char na = (a & 15) ^ ((a >> 4) & 15) ^ ((a >> 8) & 15);
        const unsigned char nb = (b & 15) ^ ((b >> 4) & 15) ^ ((b >> 8) & 15);
        packed |= (unsigned)(na << 4 | nb) << (4 * e);
      }
      const unsigned long long word = (unsigned long long)sample * 8 + row / 8;
      mw[word] = hw[word] ^ packed;
    }
  }
})";
    const std::string arch_option = "--offload-arch=" + arch;
    const char* options[] = {"-O3", "--std=c++17", arch_option.c_str()};
    hiprtcProgram program = nullptr;
    if (rtc.create_program(&program, source, "walahash-wmma.hip", 0, nullptr, nullptr) !=
        HIPRTC_SUCCESS)
      return false;
    if (rtc.compile_program(program, 3, options) != HIPRTC_SUCCESS) {
      size_t size = 0;
      rtc.get_log_size(program, &size);
      std::string log(size, '\0');
      if (size)
        rtc.get_log(program, log.data());
      std::fprintf(stderr, "WalaHash gfx12 WMMA compile failed: %s\n", log.c_str());
      rtc.destroy_program(&program);
      return false;
    }
    size_t size = 0;
    if (rtc.get_code_size(program, &size) != HIPRTC_SUCCESS) {
      rtc.destroy_program(&program);
      return false;
    }
    std::vector<char> code(size);
    const bool ok = rtc.get_code(program, code.data()) == HIPRTC_SUCCESS;
    rtc.destroy_program(&program);
    return ok && hipModuleLoadData(&module, code.data()) == hipSuccess &&
           hipModuleGetFunction(&kernel, module, "walahash_wmma") == hipSuccess;
  }

  bool launch(const int8_t* a, const uint8_t* hashes, uint8_t* mixed, const unsigned count,
              hipStream_t stream) {
    if (hipSetDevice(device) != hipSuccess)
      return false;
    void* aa = const_cast<int8_t*>(a);
    void* hh = const_cast<uint8_t*>(hashes);
    void* mm = mixed;
    unsigned n = count;
    void* args[] = {&aa, &hh, &mm, &n};
    constexpr unsigned threads = 256;
    const unsigned waves = count / 64, blocks = (waves * 32 + threads - 1) / threads;
    return hipModuleLaunchKernel(kernel, blocks, 1, 1, threads, 1, 1, 0, stream, args, nullptr) ==
           hipSuccess;
  }

  ~WalaHashAmdWmma() {
    if (device >= 0)
      (void)hipSetDevice(device);
    if (module)
      (void)hipModuleUnload(module);
  }
};

static sycl::event walahash_amd_mix(sycl::queue& queue, WalaHashAmdWmma& wmma,
                                    const int8_t* matrix, const uint8_t* hashes, uint8_t* mixed,
                                    const unsigned count) {
  return queue.AdaptiveCpp_enqueue_custom_operation(
      [&wmma, matrix, hashes, mixed, count](sycl::interop_handle& handle) {
        if (!wmma.launch(matrix, hashes, mixed, count,
                         handle.get_native_queue<sycl::backend::hip>()))
          throw std::string("WalaHash gfx12 WMMA launch failed");
      });
}

} // namespace
