namespace {

struct OctopusAmdWmma {
  hipModule_t module = nullptr;
  hipFunction_t kernel = nullptr;
  hipDevice_t device = -1;
  unsigned rows = 0;
  bool checked = false;

  bool ensure(sycl::queue& queue, const unsigned requested_rows) {
    if (!requested_rows || requested_rows % 64 || !mom_is_hip(queue.get_device()))
      return false;
    if (checked)
      return kernel != nullptr && rows == requested_rows;
    mom::hip_queue_device(queue, device);
    checked = true;
    rows = requested_rows;
    if (hipSetDevice(device) != hipSuccess)
      return false;
    hipDeviceProp_t properties{};
    mom::HiprtcApi& rtc = mom::HiprtcApi::instance();
    if (hipGetDeviceProperties(&properties, device) != hipSuccess ||
        !mom::amd::has_gfx12_int8_wmma(properties.gcnArchName) || !rtc.available())
      return false;
    std::string arch = properties.gcnArchName;
    if (const size_t colon = arch.find(':'); colon != std::string::npos)
      arch.resize(colon);
    const std::string source = "#define OCTOPUS_M " + std::to_string(rows) + R"(
typedef int v2i __attribute__((ext_vector_type(2)));
typedef int v8i __attribute__((ext_vector_type(8)));
extern "C" __global__ void octopus_wmma(const signed char* A, const signed char* B, int* C) {
  constexpr int M = OCTOPUS_M, N = 1024, K = 1024, P = 6, ER = 2, EC = 2;
  constexpr long long base = 128, base2 = base * base, base3 = base2 * base;
  constexpr long long weights[P] = {
      1 - base - base2, -base + base2 - base3,
      -base2 - base3 + base2 * base2, base, base2, base3};
  const unsigned lane = threadIdx.x & 31;
  const unsigned wave = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  const unsigned tiles = (M / (16 * ER)) * (N / (16 * EC));
  if (wave >= tiles) {
    return;
  }
  const unsigned br = wave / (N / (16 * EC));
  const unsigned bc = wave % (N / (16 * EC));
  const unsigned row = br * 16 * ER;
  const unsigned col = bc * 16 * EC;
  const unsigned rc = lane & 15;
  const unsigned kh = (lane & 16) ? 8 : 0;
  // Normalize the six digit products here so finish reads one matrix instead of six.
  long long totals[ER][EC][8] = {};
  #pragma unroll 1
  for (int product = 0; product < P; ++product) {
    const int* aw = (const int*)(A + (unsigned long long)product * M * K);
    const int* bw = (const int*)(B + (unsigned long long)product * N * K);
    v8i acc[ER][EC] = {};
    for (int k = 0; k < K; k += 16) {
      v2i a[ER], b[EC];
      #pragma unroll
      for (int i = 0; i < ER; ++i) {
        const unsigned long long w =
            ((unsigned long long)(row + i * 16 + rc) * K + k + kh) / 4;
        a[i] = {aw[w], aw[w + 1]};
      }
      #pragma unroll
      for (int j = 0; j < EC; ++j) {
        const unsigned long long w =
            ((unsigned long long)(col + j * 16 + rc) * K + k + kh) / 4;
        b[j] = {bw[w], bw[w + 1]};
      }
      #pragma unroll
      for (int i = 0; i < ER; ++i) {
        #pragma unroll
        for (int j = 0; j < EC; ++j) {
          acc[i][j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
              true, a[i], true, b[j], acc[i][j], false);
        }
      }
    }
    #pragma unroll
    for (int i = 0; i < ER; ++i) {
      #pragma unroll
      for (int j = 0; j < EC; ++j) {
        #pragma unroll
        for (int e = 0; e < 8; ++e) {
          totals[i][j][e] += (long long)acc[i][j][e] * weights[product];
        }
      }
    }
  }
  #pragma unroll
  for (int i = 0; i < ER; ++i) {
    #pragma unroll
    for (int j = 0; j < EC; ++j) {
      #pragma unroll
      for (int e = 0; e < 8; ++e) {
        const unsigned long long index =
            (row + i * 16 + 8 * (lane >> 4) + e) * N + col + j * 16 + rc;
        long long value = totals[i][j][e] % 1032193LL;
        if (value < 0) {
          value += 1032193LL;
        }
        C[index] = (int)value;
      }
    }
  }
})";
    const std::string arch_option = "--offload-arch=" + arch;
    const char* options[] = {"-O3", "--std=c++17", arch_option.c_str()};
    mom::HiprtcProgram program(rtc);
    if (rtc.create_program(program.address(), source.c_str(), "octopus-wmma.hip", 0, nullptr, nullptr) !=
        HIPRTC_SUCCESS)
      return false;
    if (rtc.compile_program(program.get(), 3, options) != HIPRTC_SUCCESS) {
      size_t size = 0;
      rtc.get_log_size(program.get(), &size);
      std::string log(size, '\0');
      if (size)
        rtc.get_log(program.get(), log.data());
      std::fprintf(stderr, "Octopus gfx12 WMMA compile failed: %s\n", log.c_str());
      return false;
    }
    size_t size = 0;
    if (rtc.get_code_size(program.get(), &size) != HIPRTC_SUCCESS) {
      return false;
    }
    std::vector<char> code(size);
    const bool ok = rtc.get_code(program.get(), code.data()) == HIPRTC_SUCCESS;
    program.reset();
    if (!ok || hipModuleLoadData(&module, code.data()) != hipSuccess ||
        hipModuleGetFunction(&kernel, module, "octopus_wmma") != hipSuccess)
      return false;
    return true;
  }

  bool launch(const int8_t* a, const int8_t* b, int32_t* c, hipStream_t stream) {
    if (hipSetDevice(device) != hipSuccess)
      return false;
    void* aa = const_cast<int8_t*>(a);
    void* bb = const_cast<int8_t*>(b);
    void* cc = c;
    void* args[] = {&aa, &bb, &cc};
    constexpr unsigned threads = 256;
    const unsigned waves = (rows / 32) * (1024 / 32);
    const unsigned blocks = (waves * 32 + threads - 1) / threads;
    return hipModuleLaunchKernel(kernel, blocks, 1, 1, threads, 1, 1, 0, stream, args, nullptr) ==
           hipSuccess;
  }

  ~OctopusAmdWmma() {
    if (device >= 0)
      (void)hipSetDevice(device);
    if (module)
      (void)hipModuleUnload(module);
  }
};

static sycl::event octopus_amd_gemm(sycl::queue& queue, OctopusAmdWmma& wmma,
                                    const int8_t* a, const int8_t* b, int32_t* c,
                                    bool& logged) {
  if (!logged && std::getenv("MOM_OCTOPUS_AMD_WMMA_LOG")) {
    std::fprintf(stderr, "Octopus gfx12 WMMA active\n");
    logged = true;
  }
  return queue.AdaptiveCpp_enqueue_custom_operation([&wmma, a, b, c](sycl::interop_handle& handle) {
    if (!wmma.launch(a, b, c, handle.get_native_queue<sycl::backend::hip>()))
      throw std::string("Octopus gfx12 WMMA launch failed");
  });
}

} // namespace
