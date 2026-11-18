// CPU-only qualification of the maintained HIPRTC adapter and source-JIT kernels. Explicit
// targets avoid GPU discovery; execute in a clean, GPU-free container to audit real dependencies.
#include "../../sycl/hiprtc-api.h"
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
#ifndef _WIN32
#include <sys/stat.h>
#endif

// The builder extracts the expression from Octopus's maintained specialization, not a copy.
#include "octopus-wmma-source.inc"

static std::string pearl_source() {
  return std::string(
#include "../../sycl/pearlhash/hip-kernel.inc"
  );
}

// Every failure releases its own compiler program before a subsequent job is attempted.
static std::vector<char> compile(const std::string& source, const char* arch,
                                 const std::vector<const char*>& options, bool invalid = false,
                                 const char* header_name = nullptr) {
  auto& api = mom::HiprtcApi::instance();
  if (!api.available()) throw std::runtime_error(api.error);
  mom::HiprtcProgram program(api);
  const char* header = "// unused virtual include; its name must still be confined\n";
  if (api.create_program(program.address(), source.c_str(), "mom-native.hip", header_name ? 1 : 0,
                          header_name ? &header : nullptr, header_name ? &header_name : nullptr) != HIPRTC_SUCCESS)
    throw std::runtime_error("Cannot create compiler program");
  std::string target = std::string("--offload-arch=") + arch;
  auto flags = options; flags.push_back(target.c_str());
  const auto status = api.compile_program(program.get(), static_cast<int>(flags.size()), flags.data());
  if (invalid) {
    if (status == HIPRTC_SUCCESS) throw std::runtime_error("Malformed source or virtual filename accepted");
    return {};
  }
  if (status != HIPRTC_SUCCESS) {
    std::size_t size = 0; api.get_log_size(program.get(), &size);
    std::string log(size, '\0'); if (size) api.get_log(program.get(), log.data());
    throw std::runtime_error(std::string(api.error_string(status)) + ": " + log);
  }
  std::size_t size = 0;
  if (api.get_code_size(program.get(), &size) != HIPRTC_SUCCESS || size < 64 || size > (64U << 20))
    throw std::runtime_error("Invalid compiled image size");
  std::vector<char> code(size);
  if (api.get_code(program.get(), code.data()) != HIPRTC_SUCCESS || std::memcmp(code.data(), "\177ELF", 4))
    throw std::runtime_error("Missing ELF image");
  return code;
}

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  try {
#ifndef _WIN32
    umask(0); // Scratch privacy must be guaranteed by atomic creation, not a friendly test umask.
#endif
    const std::vector<const char*> base{"-O3", "--std=c++17"};
    const std::string primitive = "extern \"C\" __global__ void primitive(unsigned* out) { out[threadIdx.x] += 7; }";
    const auto first = compile(primitive, "gfx900", base);
    compile("This is not a HIP kernel!", "gfx900", base, true);
    for (const char* name : {"../escape.h", "/tmp/mom-unexpected-include.h", "nested/../../escape.h", "bad\\name.h"})
      compile(primitive, "gfx900", base, true, name);
    if (compile(primitive, "gfx900", base) != first) throw std::runtime_error("Failed job contaminated later output");
    const auto write = [&](const char* name, const std::vector<char>& bytes) {
      std::ofstream output(std::string(argv[1]) + "/" + name, std::ios::binary);
      output.write(bytes.data(), bytes.size()); output.close();
      if (!output) throw std::runtime_error("Cannot preserve compiled image");
      std::cout << name << ": " << bytes.size() << " bytes\n";
    };
    write("gfx900.hsaco", first);
    write("octopus.hsaco", compile(octopus_source(2048), "gfx1200", base));
    auto pearl = base;
    pearl.insert(pearl.end(), {"-mwavefrontsize64", "-DPEARLHASH_ER=13", "-DPEARLHASH_EC=4",
      "-DPEARLHASH_K=2048", "-DPEARLHASH_RANK=128", "-DPEARLHASH_K_UNROLL=1"});
    write("pearlhash.hsaco", compile(pearl_source(), "gfx1200", pearl));
    std::cout << "Static HIPRTC maintained sources and rejection recovery passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n'; return 1;
  }
}
