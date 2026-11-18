
struct PearlHashCudaSearch {
  CUmodule module = nullptr;
  CUfunction gemm_kernel = nullptr;
  CUfunction hash_kernel = nullptr;
  CUfunction selector_kernel = nullptr;
  bool compact = false;
  CUevent started = nullptr;
  CUevent done = nullptr;
  CUcontext context = nullptr;
  CUdevice device = 0;
  bool retained_primary = false;
  bool checked = false;
  bool enabled = false;
  bool packed = false;
  unsigned stats_printed = 0;

  static bool is_cuda(const sycl::device& device) {
#if defined(MOM_SYCL_ADAPTIVECPP)
    return device.get_backend() == sycl::backend::cuda;
#else
    return device.get_backend() == sycl::backend::ext_oneapi_cuda;
#endif
  }
  static CUdevice native_device(const sycl::device& device) {
#if defined(MOM_SYCL_ADAPTIVECPP)
    return sycl::get_native<sycl::backend::cuda>(device);
#else
    return sycl::get_native<sycl::backend::ext_oneapi_cuda>(device);
#endif
  }
  // Require the CUDA backend and async int8 tensor hardware before probing the optional NVRTC path.
  static bool supports(sycl::queue& q, std::string* const reason = nullptr) {
    const auto unavailable = [&](const std::string& message) {
      if (reason)
        *reason = message;
      return false;
    };
    if (!is_cuda(q.get_device()))
      return unavailable("the selected SYCL device is not CUDA");
    PearlHashCudaDriverApi& api = PearlHashCudaDriverApi::instance();
    if (!api.extended_available())
      return unavailable(api.error);
    if (api.init(0) != CUDA_SUCCESS)
      return unavailable("cuInit failed");
    int major = 0;
    if (api.device_attribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                             native_device(q.get_device())) != CUDA_SUCCESS)
      return unavailable("the CUDA compute capability could not be read");
    if (major < 8)
      return unavailable("the CUTE int8 mainloop requires compute capability 8.0+");
    return true;
  }

  // Restrict native GEMM to the supported tile-aligned dimensions and noise profile.
  static bool supports_shape(const int m, const int n, const int k, const int rank) {
    if (compact_private::selected(m,n,k,rank)) return true;
    return m > 0 && n > 0 && m % 128 == 0 && n % 256 == 0 &&
           ((k == 4096 && rank == 256) || (k == 8192 && rank == 128));
  }

#if !defined(MOM_EMBEDDED_CUDA_IMAGES)
  // Discovery and dispatch share dependency and resource admission.
#if defined(MOM_EMBEDDED_NVRTC_HEADERS)
  static const mom::JitHeaders& compiler_headers() {
    // Decode once, keep all NVRTC header pointers alive, and never consult an installed SDK.
    static const mom::JitHeaders headers(embedded_resource("jit-headers.gz"), kJitHeadersSize);
    for (const auto* name : {"cuda_runtime.h", "cute/tensor.hpp", "cuda/std/cstdint"})
      if (headers.find(name).empty())
        throw std::string("Missing embedded CUDA JIT header: ") + name;
    return headers;
  }
#else
  static std::array<std::filesystem::path, 3> compiler_headers() {
    std::vector<std::filesystem::path> cuda_candidates;
    if (const char* path = std::getenv("CUDA_PATH"); path && *path)
      cuda_candidates.emplace_back(path);
    if (const char* path = std::getenv("CUDA_HOME"); path && *path)
      cuda_candidates.emplace_back(path);
#if defined(_WIN32)
    cuda_candidates.emplace_back("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0");
    cuda_candidates.emplace_back("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6");
#else
    cuda_candidates.emplace_back("/usr/local/cuda");
    cuda_candidates.emplace_back("/opt/nvidia-cuda-ubuntu/usr");
#endif
    const std::filesystem::path cuda_include = pearlhash_cuda_header_root(
        nullptr,
        [&] {
          std::vector<std::filesystem::path> paths;
          for (const auto& root : cuda_candidates)
            paths.push_back(root / "include");
          return paths;
        }(),
        "cuda_runtime.h");

    std::vector<std::filesystem::path> cutlass_candidates;
#if defined(_WIN32)
    if (const char* program_data = std::getenv("ProgramData"); program_data && *program_data)
      cutlass_candidates.emplace_back(std::filesystem::path(program_data) / "mom" / "cutlass" /
                                      "include");
#else
    cutlass_candidates.emplace_back("/opt/mom/cutlass/include");
    cutlass_candidates.emplace_back("/usr/local/include/cutlass");
#endif
    const std::filesystem::path cutlass_include = pearlhash_cuda_header_root(
        std::getenv("MOM_CUTLASS_INCLUDE_DIR"), cutlass_candidates, "cute/tensor.hpp");

    std::vector<std::filesystem::path> cccl_candidates;
    if (!cuda_include.empty()) {
      cccl_candidates.push_back(cuda_include);
      cccl_candidates.push_back(cuda_include / "cccl");
    }
    const std::filesystem::path cccl_include = pearlhash_cuda_header_root(
        std::getenv("MOM_CCCL_INCLUDE_DIR"), cccl_candidates, "cuda/std/cstdint");
    if (cuda_include.empty())
      throw std::string("CUDA runtime headers are not installed");
    if (cutlass_include.empty())
      throw std::string("CUTLASS headers are not installed");
    if (cccl_include.empty())
      throw std::string("CCCL headers are not installed");

    return {cuda_include, cutlass_include, cccl_include};
  }
#endif
#endif

  static void require_compact_resources(PearlHashCudaDriverApi& driver, CUdevice device) {
    const auto limit = [&](const CUdevice_attribute attribute, const int required) {
      int value = 0;
      cuda_check(driver, driver.device_attribute(&value, attribute, device),
                 "cuDeviceGetAttribute(PearlHash compact limit)");
      return value >= required;
    };
    int warp_size = 0;
    cuda_check(driver, driver.device_attribute(&warp_size, CU_DEVICE_ATTRIBUTE_WARP_SIZE, device),
               "cuDeviceGetAttribute(PearlHash warp size)");
    if (warp_size != 32 || !limit(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, 256) ||
        !limit(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, 100352) ||
        !limit(CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, 255 * 256))
      throw std::string("Insufficient PearlHash compact block resources");
  }

#if defined(MOM_EMBEDDED_CUDA_IMAGES)
  static void require_compact(PearlHashCudaDriverApi& driver, CUdevice device,
                              int capability_major, int capability_minor) {
    // Build-time producer qualification guarantees the embedded sm120a image came from NVRTC 13.3+.
    if (capability_major != 12 || capability_minor != 0)
      throw std::string("Compact PearlHash requires sm120a");
    require_compact_resources(driver, device);
  }
#else
  static void require_compact(PearlHashCudaDriverApi& driver, PearlHashNvrtcApi& rtc,
                              CUdevice device, int capability_major, int capability_minor,
                              int& compiler_major, int& compiler_minor) {
    // The .b8 LDSM source is qualified for the sm120a feature target, not a
    // particular card name/SM count. Keep other architectures on their old
    // profile. A JIT decline happens before the caller derives any compact
    // key/pattern, so the shared path can retain its ordinary configuration.
    if (capability_major != 12 || capability_minor != 0 || !rtc.version ||
        rtc.version(&compiler_major, &compiler_minor) != NVRTC_SUCCESS ||
        compiler_major < 13 || compiler_minor < 0 ||
        (compiler_major == 13 && compiler_minor < 3))
      throw std::string("Compact PearlHash requires sm120a and NVRTC 13.3+");
    require_compact_resources(driver, device);
  }

  static int compiler_arch(PearlHashNvrtcApi& rtc, const int device_arch) {
    int arch_count = 0;
    nvrtcResult arch_status = rtc.get_num_supported_archs(&arch_count);
    if (arch_status != NVRTC_SUCCESS || arch_count <= 0 || arch_count > 256)
      throw rtc_error(rtc, arch_status, nullptr, "nvrtcGetNumSupportedArchs(PearlHash)");
    std::vector<int> supported_archs(static_cast<size_t>(arch_count));
    arch_status = rtc.get_supported_archs(supported_archs.data());
    if (arch_status != NVRTC_SUCCESS)
      throw rtc_error(rtc, arch_status, nullptr, "nvrtcGetSupportedArchs(PearlHash)");
    int target_arch = 0;
    for (const int arch : supported_archs)
      if (arch <= device_arch && arch > target_arch)
        target_arch = arch;
    return target_arch;
  }
#endif

  // Discovery checks the same dependencies as dispatch, without a context,
  // matrix allocations, module loading, or kernel execution. Dispatch rechecks
  // them because an optional runtime may disappear after device enumeration.
  static bool compact_available(const sycl::device& dev) {
    if (!is_cuda(dev))
      return false;
    auto& driver = PearlHashCudaDriverApi::instance();
    if (!driver.extended_available() || driver.init(0) != CUDA_SUCCESS)
      return false;
    const CUdevice device = native_device(dev);
    int major = 0, minor = 0;
    cuda_check(driver, driver.device_attribute(
        &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device), "PearlHash discovery major");
    cuda_check(driver, driver.device_attribute(
        &minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device), "PearlHash discovery minor");
    if (major != 12 || minor != 0)
      return false;
#if defined(MOM_EMBEDDED_CUDA_IMAGES)
    const auto image = mom::cuda_images::select_embedded("pearl-compact", 120);
    if (!image.primary.data)
      return false;
    require_compact(driver, device, major, minor);
#else
    auto& rtc = PearlHashNvrtcApi::instance();
    if (!rtc.cubin_available())
      return false;
    int compiler_major = 0, compiler_minor = 0;
    require_compact(driver, rtc, device, major, minor, compiler_major, compiler_minor);
    if (compiler_arch(rtc, 120) != 120)
      return false;
    (void)compiler_headers();
#endif
    return true;
  }

  // Release the CUDA event/module and clear cached compilation state on the owning context.
  void release() noexcept {
    // Empty fallback state must not instantiate a CUDA driver loader on another vendor.
    if (context || done || started || module || retained_primary) {
      PearlHashCudaDriverApi& api = PearlHashCudaDriverApi::instance();
      if (api.extended_available() && context)
        (void)api.set_current(context);
      if (done && api.event_destroy)
        (void)api.event_destroy(done);
      if (started && api.event_destroy)
        (void)api.event_destroy(started);
      if (module && api.module_unload)
        (void)api.module_unload(module);
      if (retained_primary && api.release_primary)
        (void)api.release_primary(device);
    }
    selector_kernel = nullptr;
    compact = false;
    done = nullptr;
    started = nullptr;
    module = nullptr;
    gemm_kernel = nullptr;
    hash_kernel = nullptr;
    context = nullptr;
    retained_primary = false;
    checked = false;
    enabled = false;
    packed = false;
  }

  // Admit and load the optional native CUDA kernels; unsupported setups retain the SYCL fallback.
  bool ensure(sycl::queue& q, const Buffers& b, int m, int n, int k, int rank) {
    if (!supports(q) || !supports_shape(m, n, k, rank)) {
      packed = false;
      return false;
    }
    const bool want_compact=compact_private::selected(m,n,k,rank);
    // Compact fusion writes only Result; the old P16 transcript is deliberately unused.
    if (!want_compact && (!b.transcript || b.transcript_rows < 128))
      return false;
    // The rank-128 producer and direct CUDA mainloop share 16-byte packet bases and
    // 64-bit offsets. Fail closed to the ordinary SYCL path if the allocation or host
    // address model cannot prove those invariants; rank-256 keeps the staged path.
    const bool packed_candidate = rank == 128 && k == 8192;
    if ((packed_candidate || want_compact) &&
        (sizeof(size_t) < sizeof(uint64_t) || sizeof(uintptr_t) < sizeof(uint64_t) ||
         !b.Ap || !b.Bp || (reinterpret_cast<uintptr_t>(b.Ap) & 15U) != 0 ||
         (reinterpret_cast<uintptr_t>(b.Bp) & 15U) != 0)) {
      packed = false;
      return false;
    }
    if (want_compact && (!b.EAL || !b.EBRt))
      return false;
    packed = packed_candidate;
    if (checked)
      return enabled;
    checked = true;
    try {
      PearlHashCudaDriverApi& driver = PearlHashCudaDriverApi::instance();
#if !defined(MOM_EMBEDDED_CUDA_IMAGES)
      PearlHashNvrtcApi& rtc = PearlHashNvrtcApi::instance();
      if (!rtc.cubin_available())
        throw rtc.error;
#endif

      device = native_device(q.get_device());
      cuda_check(driver, driver.retain_primary(&context, device),
                 "cuDevicePrimaryCtxRetain(PearlHash)");
      retained_primary = true;
      cuda_check(driver, driver.set_current(context), "cuCtxSetCurrent(PearlHash)");

#if defined(MOM_EMBEDDED_CUDA_IMAGES)
      int capability_major = 0, capability_minor = 0;
      cuda_check(driver,
                 driver.device_attribute(&capability_major,
                                         CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device),
                 "cuDeviceGetAttribute(PearlHash compute capability major)");
      cuda_check(driver,
                 driver.device_attribute(&capability_minor,
                                         CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device),
                 "cuDeviceGetAttribute(PearlHash compute capability minor)");
      if (capability_major < 8 || capability_major > 99 || capability_minor < 0 ||
          capability_minor > 9)
        throw std::string("invalid or unsupported PearlHash CUDA capability");
      const int device_arch = capability_major * 10 + capability_minor;
      if (want_compact)
        require_compact(driver, device, capability_major, capability_minor);
      const char* const profile = want_compact ? "pearl-compact"
                                      : rank == 128 ? "pearl-rank128" : "pearl-rank256";
      const auto candidates = mom::cuda_images::select_embedded(profile, device_arch);
      if (!candidates.primary.data)
        throw std::string("no compatible embedded PearlHash image for ") + profile;
      cuda_check(driver,
                 mom::cuda_load_image_candidates(
                     candidates, &module, [&](CUmodule* loaded, const void* image) {
                       return driver.module_load(loaded, image);
                     }),
                 "cuModuleLoadData(PearlHash)");
#else

#if defined(MOM_EMBEDDED_NVRTC_HEADERS)
      const auto& includes = compiler_headers();
#else
      const auto includes = compiler_headers();
      const auto& cuda_include = includes[0];
      const auto& cutlass_include = includes[1];
      const auto& cccl_include = includes[2];
#endif

      int capability_major = 0, capability_minor = 0;
      cuda_check(driver,
                 driver.device_attribute(&capability_major,
                                         CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device),
                 "cuDeviceGetAttribute(PearlHash compute capability major)");
      cuda_check(driver,
                 driver.device_attribute(&capability_minor,
                                         CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device),
                 "cuDeviceGetAttribute(PearlHash compute capability minor)");
      const int device_arch = capability_major * 10 + capability_minor;
      int compiler_major = 0, compiler_minor = 0;
      if (want_compact)
        require_compact(driver, rtc, device, capability_major, capability_minor, compiler_major, compiler_minor);

      const int target_arch = compiler_arch(rtc, device_arch);
      if (target_arch < 80)
        throw std::string("NVRTC does not support PearlHash's required compute capability 8.0+");
      if (want_compact && target_arch != device_arch)
        throw std::string("NVRTC does not support the compact device architecture");
      // A cubin is valid only for the exact architecture it names. When NVRTC knows the selected
      // GPU, emit final machine code locally and avoid a second driver-JIT pass; otherwise retain
      // forward-compatible PTX for newer GPUs (for example CUDA 12.6 on a Blackwell card).
      const bool emit_cubin = target_arch == device_arch;

      const std::string source(want_compact ?
#include "cuda-compact-kernel.inc"
          :
#include "cuda-kernel.inc"
      );
      std::vector<std::string> option_storage{
          "--std=c++17",
          "-DPEARLHASH_RANK=" + std::to_string(rank),
          "--gpu-architecture=" + std::string(emit_cubin ? "sm_" : "compute_") +
              std::to_string(target_arch) + (want_compact ? "a" : ""),
#if !defined(MOM_EMBEDDED_NVRTC_HEADERS)
          "--include-path=" + cutlass_include.generic_string(),
          "--include-path=" + cuda_include.generic_string(),
          "--include-path=" + cccl_include.generic_string(),
#endif
          "--device-as-default-execution-space",
          "--use_fast_math"};
      if (want_compact)
        option_storage.emplace_back("-DDIRECT_RANK_STRIDE=1");
      std::vector<const char*> options;
      options.reserve(option_storage.size());
      for (const auto& option : option_storage)
        options.push_back(option.c_str());

      std::ostringstream key_stream;
      key_stream << source;
      for (const auto& option : option_storage)
        key_stream << '\0' << option;
      key_stream << '\0' << "pearlhash-cuda-cache-v2-unified-vector128";
#if defined(MOM_EMBEDDED_NVRTC_HEADERS)
      // Every embedded include participates, not only the three SDK version headers below.
      key_stream << '\0' << "embedded-headers" << '\0' << kJitHeadersSha256;
#endif
      if (want_compact) {
        key_stream << '\0' << "compact-v1-one-module" << '\0' << compiler_major
                   << '.' << compiler_minor;
#if !defined(MOM_EMBEDDED_NVRTC_HEADERS)
        // Include versioned toolchain identities as well as source/options.
        // These are small, bounded local inputs, not unbounded SDK tree scans.
        for (const auto& header : {cutlass_include / "cutlass/version.h",
                                  cuda_include / "cuda_runtime_api.h",
                                  cccl_include / "cuda/std/__cccl/version.h"}) {
          const auto bytes = mom::jit_cache::read(header, 1024 * 1024);
          if (bytes.empty())
            throw std::string("Missing compact JIT toolchain version header");
          key_stream << '\0' << mom::jit_cache::hash(
              std::string_view(bytes.data(), bytes.size()));
        }
#endif
      }
      std::ostringstream filename;
      filename << "pearlhash-cuda-" << (emit_cubin ? "sm" : "compute") << target_arch << "-"
               << std::hex << std::setw(16) << std::setfill('0')
               << mom::jit_cache::hash(key_stream.str()) << (emit_cubin ? ".cubin" : ".ptx");
      const std::filesystem::path cache_dir = mom::jit_cache::directory();
      const std::filesystem::path cache_path =
          cache_dir.empty() ? std::filesystem::path{} : cache_dir / filename.str();
      std::vector<char> code =
          cache_path.empty() ? std::vector<char>{} : mom::jit_cache::read(cache_path);
      if (!code.empty() && driver.module_load(&module, code.data()) != CUDA_SUCCESS) {
        std::error_code error;
        std::filesystem::remove(cache_path, error);
        code.clear();
      }
      if (code.empty()) {
        mom::NvrtcProgram program(rtc);
        nvrtcResult status = rtc.create_program(
          program.address(), source.c_str(), "pearlhash-cute.cu",
#if defined(MOM_EMBEDDED_NVRTC_HEADERS)
          includes.count(), includes.sources(), includes.names());
#else
          0, nullptr, nullptr);
#endif
        if (status != NVRTC_SUCCESS)
          throw rtc_error(rtc, status, program.get(), "nvrtcCreateProgram(PearlHash)");
        status = rtc.compile_program(program.get(), static_cast<int>(options.size()), options.data());
        if (status != NVRTC_SUCCESS)
          throw rtc_error(rtc, status, program.get(), "nvrtcCompileProgram(PearlHash)");
        size_t code_size = 0;
        status = emit_cubin ? rtc.get_cubin_size(program.get(), &code_size)
                            : rtc.get_ptx_size(program.get(), &code_size);
        if (status != NVRTC_SUCCESS)
          throw rtc_error(rtc, status, program.get(),
                          emit_cubin ? "nvrtcGetCUBINSize(PearlHash)" : "nvrtcGetPTXSize(PearlHash)");
        if (!code_size || code_size > 64 * 1024 * 1024)
          throw std::string("PearlHash compiler output exceeds the cache size bound");
        code.resize(code_size);
        status =
            emit_cubin ? rtc.get_cubin(program.get(), code.data()) : rtc.get_ptx(program.get(), code.data());
        if (status != NVRTC_SUCCESS)
          throw rtc_error(rtc, status, program.get(),
                          emit_cubin ? "nvrtcGetCUBIN(PearlHash)" : "nvrtcGetPTX(PearlHash)");
        program.reset();
        cuda_check(driver, driver.module_load(&module, code.data()),
                   "cuModuleLoadData(PearlHash)");
        mom::jit_cache::write(cache_path, code);
      }
#endif
      if (want_compact) {
        cuda_check(driver, driver.module_function(&gemm_kernel, module, "compact_gemm"),
                   "cuModuleGetFunction(PearlHash compact solver)");
        cuda_check(driver, driver.module_function(&selector_kernel, module, "selector_prepare"),
                   "cuModuleGetFunction(PearlHash compact selector)");
        cuda_check(driver, driver.event_create(&started, CU_EVENT_DEFAULT), "compact start event");
        cuda_check(driver, driver.event_create(&done, CU_EVENT_DEFAULT), "compact done event");
        compact = true;
        enabled = true;
#if defined(MOM_EMBEDDED_CUDA_IMAGES)
        std::fprintf(stderr,
                     "PEARLHASH_COMPACT admitted=1 m=%d n=%d k=%d rank=%d proof=24x8 native_image=1\n",
                     m, n, k, rank);
#else
        std::fprintf(stderr,"PEARLHASH_COMPACT admitted=1 m=%d n=%d k=%d rank=%d proof=24x8 source_jit=1 nvrtc=%d.%d\n",
                     m,n,k,rank,compiler_major,compiler_minor);
#endif
        return true;
      }
      cuda_check(driver, driver.module_function(&gemm_kernel, module, "pearlhash_cuda_gemm"),
                 "cuModuleGetFunction(PearlHash GEMM)");
      cuda_check(driver, driver.module_function(&hash_kernel, module, "pearlhash_cuda_hash"),
                 "cuModuleGetFunction(PearlHash hash)");
      const int shared_bytes = packed ? 0 : 98304;
      cuda_check(driver,
                 driver.function_attribute(
                     gemm_kernel, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shared_bytes),
                 "cuFuncSetAttribute(PearlHash shared memory)");
      cuda_check(driver,
                 driver.function_attribute(gemm_kernel,
                                           CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
                                           packed ? 0 : 100),
                 "cuFuncSetAttribute(PearlHash carveout)");
      cuda_check(driver, driver.event_create(&started, CU_EVENT_DEFAULT),
                 "cuEventCreate(PearlHash start)");
      cuda_check(driver, driver.event_create(&done, CU_EVENT_DEFAULT),
                 "cuEventCreate(PearlHash done)");
      enabled = true;
    } catch (const std::string& error) {
      release();
      checked = true;
#if defined(MOM_EMBEDDED_CUDA_IMAGES)
      std::fprintf(stderr,
                   "PearlHash CUDA embedded image unavailable, using the best SYCL kernel: %s\n",
                   error.c_str());
#else
      std::fprintf(stderr,
                   "PearlHash CUDA source-JIT unavailable, using the best SYCL kernel: %s\n",
                   error.c_str());
#endif
    } catch (const std::exception& error) {
      release();
      checked = true;
#if defined(MOM_EMBEDDED_CUDA_IMAGES)
      std::fprintf(stderr,
                   "PearlHash CUDA embedded image unavailable, using the best SYCL kernel: %s\n",
                   error.what());
#else
      std::fprintf(stderr,
                   "PearlHash CUDA source-JIT unavailable, using the best SYCL kernel: %s\n",
                   error.what());
#endif
    }
    return enabled;
  }

  // Launch GEMM and transcript hashing on the native stream, then record the completion event.
  void launch(const Buffers& b, uint32_t seed, int m, int n, int k, bool debug) {
    PearlHashCudaDriverApi& api = PearlHashCudaDriverApi::instance();
    cuda_check(api, api.set_current(context), "cuCtxSetCurrent(PearlHash launch)");
    void* bp = b.Bp;
    void* transcript = b.transcript;
    void* key = b.cA;
    void* target = b.target;
    void* result = b.result;
    int debug_int = debug ? 1 : 0;
    if (compact) {
      if (!compact_private::selected(m,n,k,128))
        throw std::string("Compact shape changed after admission");
      void* source=b.Ap;void* packed=b.Bp;
      unsigned count=static_cast<unsigned>(k);
      void* prepare_args[]={&source,&packed,&count};
      void* eal=b.EAL;void* ebr=b.EBRt;
      void* unused_a=b.EARp2;void* unused_b=b.EBLq1;void* unused_c=b.EBLq2;
      void* no_transcript=nullptr;int row_offset=0;
      uint32_t b_seed=b.fixed_b_seed;
      void* args[]={&eal,&ebr,&packed,&unused_a,&unused_b,&unused_c,&seed,&b_seed,
                    &m,&n,&k,&no_transcript,&key,&target,&result,&row_offset,&debug_int};
      mom::cuda_check_submitted(api,api.event_record(started,nullptr),"compact start");
      mom::cuda_check_submitted(api,api.launch(selector_kernel,count/128,1,1,128,1,1,0,nullptr,prepare_args,nullptr),"compact selector prepare");
      mom::cuda_check_submitted(api,api.launch(gemm_kernel,static_cast<unsigned>(m/192),static_cast<unsigned>(n/256),1,256,1,1,0,nullptr,args,nullptr),"compact fused solver");
      mom::cuda_check_submitted(api,api.event_record(done,nullptr),"compact done");
      return;
    }
    // Checksum validation also exercises nonzero row offsets on the reduced test matrix.
    const int chunk_rows = debug
        ? std::min(b.transcript_rows, std::max(128, (m / 2 / 128) * 128))
        : b.transcript_rows;
    constexpr unsigned threads = 256;
    const unsigned shared_bytes = packed ? 0U : 98304U;
    mom::cuda_check_submitted(api, api.event_record(started, nullptr),
                              "cuEventRecord(PearlHash start)");
    for (int row_offset = 0; row_offset < m; row_offset += chunk_rows) {
      int rows = std::min(chunk_rows, m - row_offset);
      void* a = b.Ap + static_cast<size_t>(row_offset) * static_cast<size_t>(k);
      void* gemm_args[] = {&a, &bp, &rows, &n, &k, &transcript};
      void* hash_args[] = {
          &transcript, &key, &target, &result, &seed, &rows, &n, &row_offset, &debug_int};
      mom::cuda_check_submitted(
          api, api.launch(gemm_kernel, static_cast<unsigned>(rows / 128),
                          static_cast<unsigned>(n / 256), 1, threads, 1, 1, shared_bytes, nullptr,
                          gemm_args, nullptr),
          "cuLaunchKernel(PearlHash GEMM)");
      const uint64_t tiles = static_cast<uint64_t>(rows / 16) * static_cast<uint64_t>(n / 16);
      mom::cuda_check_submitted(
          api, api.launch(hash_kernel, static_cast<unsigned>((tiles + threads - 1) / threads),
                          1, 1, threads, 1, 1, 0, nullptr, hash_args, nullptr),
          "cuLaunchKernel(PearlHash hash)");
    }
    mom::cuda_check_submitted(api, api.event_record(done, nullptr), "cuEventRecord(PearlHash done)");
  }

  // Poll native completion with the existing learned-duration cadence instead of a CPU-spinning synchronize.
  void wait(double learned_us) {
    PearlHashCudaDriverApi& api = PearlHashCudaDriverApi::instance();
#if !defined(_WIN32)
    const auto poll =
        learned_us > 20000.0 ? std::chrono::microseconds(500) : std::chrono::microseconds(100);
#else
    (void)learned_us;
#endif
    for (;;) {
      const CUresult status = api.event_query(done);
      if (status == CUDA_SUCCESS) {
        if (std::getenv("MOM_PEARLHASH_STATS") && stats_printed++ < 4) {
          float search_ms = 0.0f;
          cuda_check(api, api.event_elapsed(&search_ms, started, done),
                     "cuEventElapsedTime(PearlHash CUDA search)");
          std::fprintf(stderr, "PEARLHASH_CUDA_STATS search=%.3fms sequence=%u\n", search_ms, stats_printed);
        }
        return;
      }
      if (status != CUDA_ERROR_NOT_READY)
        mom::cuda_check_submitted(api, status, "cuEventQuery(PearlHash CUDA)");
#if defined(_WIN32)
      mom_sycl_poll_pause();
#else
      std::this_thread::sleep_for(poll);
#endif
    }
  }
  ~PearlHashCudaSearch() {
    release();
  }
};
