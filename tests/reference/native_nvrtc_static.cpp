// Compile the maintained optional CN/GPU source through the actual shared/static compiler API.
// CPU-only code generation: no explicit driver calls or kernel launches. NVRTC may optionally
// probe the system driver internally; the proof also runs without a driver or GPU exposed.
#include "../../sycl/nvrtc-api.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <vector>

static const char source[] =
#include "../../sycl/cn_gpu/cuda_expand_kernel.inc"
;

static void check(mom::NvrtcApi &api, nvrtcResult status, nvrtcProgram program, const char *operation) {
  if (status != NVRTC_SUCCESS) {throw std::runtime_error(mom::rtc_error(api, status, program, operation));}
}

// Retain program ownership through all errors, including compiler diagnostics and host allocation.
static std::vector<char> compile(mom::NvrtcApi &api, const char *code, const std::string &arch, bool cubin) {
  mom::NvrtcProgram program(api);
  check(api, api.create_program(program.address(), code, "cn-gpu-expand.cu", 0, nullptr, nullptr), program.get(), "create");
  const std::string target = "--gpu-architecture=" + arch;
  const char *options[] = {"--std=c++17", target.c_str(), "--device-as-default-execution-space", "--modify-stack-limit=false"};
  check(api, api.compile_program(program.get(), 4, options), program.get(), "compile");
  size_t size = 0;
  check(api, (cubin ? api.get_cubin_size : api.get_ptx_size)(program.get(), &size), program.get(), "size");
  if (!size || size > 64 * 1024 * 1024) {throw std::runtime_error("Unbounded compiler output");}
  std::vector<char> output(size);
  check(api, (cubin ? api.get_cubin : api.get_ptx)(program.get(), output.data()), program.get(), "code");
  return output;
}

int main(int argc, char **argv) {
  if (argc != 2) {std::fputs("Expected an output directory\n", stderr); return 2;}
  try {
    auto &api = mom::NvrtcApi::instance();
    if (!api.cubin_available() || !api.version || !api.error.empty()) {throw std::runtime_error("NVRTC unavailable: " + api.error);}
    int major = 0, minor = 0;
    check(api, api.version(&major, &minor), nullptr, "version");
    int count = 0;
    check(api, api.get_num_supported_archs(&count), nullptr, "architecture count");
    if (count < 1 || count > 128) {throw std::runtime_error("Invalid architecture count");}
    std::vector<int> architectures(static_cast<size_t>(count));
    check(api, api.get_supported_archs(architectures.data()), nullptr, "architectures");
    for (int sm : {80, 120}) {
      if (std::find(architectures.begin(), architectures.end(), sm) == architectures.end()) {
        throw std::runtime_error("Proof compiler lacks a required target");
      }
      for (bool cubin : {false, true}) {
        const auto arch = std::string(cubin ? "sm_" : "compute_") + std::to_string(sm);
        const auto output = compile(api, source, arch, cubin);
        std::ofstream file(std::string(argv[1]) + "/" + arch + (cubin ? ".cubin" : ".ptx"), std::ios::binary);
        file.write(output.data(), static_cast<std::streamsize>(output.size()));
        file.close();
        if (!file) {throw std::runtime_error("Cannot preserve compiler output");}
        std::printf("NVRTC %d.%d %s %zu bytes\n", major, minor, arch.c_str(), output.size());
      }
    }
    unsigned rejected = 0;
    for (const auto &entry : {std::pair<const char *, const char *>{"not CUDA syntax!", "compute_80"},
                             {source, "compute_9999"}}) {
      try {compile(api, entry.first, entry.second, false);} catch (const std::runtime_error &) {++rejected;}
    }
    if (rejected != 2) {throw std::runtime_error("Malformed compiler input unexpectedly succeeded");}
    std::puts("NVRTC source parity fixture passed; GPU dispatches=0");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "%s\n", error.what()); return 1;
  }
}
