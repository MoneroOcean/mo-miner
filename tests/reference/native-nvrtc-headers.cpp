// Compile maintained Pearl sources with either embedded includes or the existing SDK include
// paths. This is a CPU-only compiler comparison; it never initializes a device or launches code.
#define MOM_SYCL_HAS_CUDA
#include "../../sycl/cuda-api.h"
#ifndef MOM_NVRTC_DISK_HEADERS
#include "../../native/jit-headers.h"
#include "nvrtc-headers.inc"
#endif
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <vector>

static const char standard_source[] =
#include "../../sycl/pearlhash/cuda-kernel.inc"
;
static const char compact_source[] =
#include "../../sycl/pearlhash/cuda-compact-kernel.inc"
;
static void check(mom::NvrtcApi &api, nvrtcResult status, nvrtcProgram program, const char *operation) {
  if (status != NVRTC_SUCCESS) {throw std::runtime_error(mom::rtc_error(api, status, program, operation));}
}

int main(int argc, char **argv) {
#ifdef MOM_NVRTC_DISK_HEADERS
  if (argc < 5 || argc > 10) {return 2;}
#else
  if (argc != 2) {return 2;}
#endif
  try {
    auto &api = mom::NvrtcApi::instance();
    if (!api.cubin_available()) {throw std::runtime_error("NVRTC unavailable: " + api.error);}
#ifndef MOM_NVRTC_DISK_HEADERS
    const mom::JitHeaders headers(embedded_resource("jit-headers.gz"), kJitHeadersSize);
    if (headers.find("cute/tensor.hpp").empty() || headers.find("cuda_runtime.h").empty() ||
        headers.find("cuda/std/cstdint").empty()) {throw std::runtime_error("Incomplete embedded compiler headers");}
#endif
    struct Case {const char *name; const char *target; int rank; bool compact;};
    for (const auto &test : {Case{"sm80-r128", "sm_80", 128, false}, {"sm80-r256", "sm_80", 256, false},
                           {"sm120-r128", "sm_120", 128, false}, {"sm120-r256", "sm_120", 256, false},
                           {"sm120a-compact", "sm_120a", 128, true}}) {
      mom::NvrtcProgram program(api);
      const char *source = test.compact ? compact_source : standard_source;
#ifdef MOM_NVRTC_DISK_HEADERS
      check(api, api.create_program(program.address(), source, "pearlhash-cute.cu", 0, nullptr, nullptr), program.get(), "create");
#else
      check(api, api.create_program(program.address(), source, "pearlhash-cute.cu", headers.count(), headers.sources(), headers.names()), program.get(), "create");
#endif
      std::vector<std::string> storage{"--std=c++17", "-DPEARLHASH_RANK=" + std::to_string(test.rank),
          "--gpu-architecture=" + std::string(test.target), "--device-as-default-execution-space", "--use_fast_math"};
#ifdef MOM_NVRTC_DISK_HEADERS
      // SDK normalization may add the canonical nested CCCL root (at most eight roots).
      for (int i = 2; i < argc; ++i) {storage.emplace_back("--include-path=" + std::string(argv[i]));}
#endif
      if (test.compact) {storage.emplace_back("-DDIRECT_RANK_STRIDE=1");}
      std::vector<const char *> options;
      for (const auto &value : storage) {options.push_back(value.c_str());}
      check(api, api.compile_program(program.get(), static_cast<int>(options.size()), options.data()), program.get(), "compile");
      size_t size = 0; check(api, api.get_cubin_size(program.get(), &size), program.get(), "size");
      if (!size || size > 64U << 20) {throw std::runtime_error("Unbounded cubin output");}
      std::vector<char> cubin(size); check(api, api.get_cubin(program.get(), cubin.data()), program.get(), "cubin");
      std::ofstream file(std::string(argv[1]) + "/" + test.name + ".cubin", std::ios::binary);
      file.write(cubin.data(), static_cast<std::streamsize>(cubin.size())); file.close();
      if (!file) {throw std::runtime_error("Cannot retain compiler output");}
      std::printf("Pearl %s %zu bytes\n", test.name, cubin.size());
    }
    std::puts("Maintained Pearl compiler proof passed; GPU dispatches=0");
    return 0;
  } catch (const std::exception &error) {std::fprintf(stderr, "%s\n", error.what()); return 1;}
}
