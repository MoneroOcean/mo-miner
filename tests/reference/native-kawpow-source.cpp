// Build/CPU-only source generator: execute the maintained period-program and JIT emitters.
// Device math is never dispatched; the receiving proof compiles the returned source in memory.
#define SYCL_DISABLE_FSYCL_SYCLHPP_WARNING
#include <sycl/sycl.hpp>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <limits>
#include <memory>
#include <stdexcept>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>
#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#endif
#include "../../sycl/lib-internal.h"
#include "../../xmrig/3rdparty/libethash/ethash.h"
#include "../../xmrig/3rdparty/libethash/data_sizes.h"

// A combined proof also links the original algorithm TU. Keep this source-generator fixture's
// included host helpers private rather than adding duplicate production definitions.
namespace {
#include "../../sycl/kawpow/epoch.h"
namespace mom_kawpow {
#include "../../sycl/kawpow/program.inc"
}
}

// Same values and assembly order as KawpowState::build_period_bundle. The source/PRNG/emitters
// themselves are the original implementation, not a separate fixture version of the algorithm.
std::pair<std::string, std::string> nativeKawpowSource(const std::string& algo, uint64_t period) {
  using namespace mom_kawpow;
  const auto& variant = algo == "kawpow" ? KAWPOW_VARIANT : algo == "firopow" ? FIROPOW_VARIANT :
      algo == "evrprogpow" ? EVRPROGPOW_VARIANT : MEOWPOW_VARIANT;
  if (algo != "kawpow" && algo != "firopow" && algo != "evrprogpow" && algo != "meowpow")
    throw std::runtime_error("Unknown maintained source-JIT algorithm");
  const auto mod = make_fast_mod_data(67108859);
  const auto name = kawpow_jit_kernel_name(period, mod, variant.seal, variant.magic,
      variant.regs, variant.cnt_cache, variant.cnt_math);
  return {name, std::string(KAWPOW_JIT_PRELUDE) + kawpow_device_src() +
      kawpow_emit_baked(make_program(period, variant.regs, variant.cnt_cache, variant.cnt_math), mod) +
      kawpow_emit_seal(variant.seal, variant.magic) +
      kawpow_emit_shape(variant.regs, variant.cnt_cache, variant.cnt_math) + kawpow_named_jit_wrapper(name)};
}

// Shared byte-exact output for the standalone bootstrap and full compiler proof.
// Windows text-mode translation would change the source and its persistent-cache identity.
void nativeEmitKawpowSource(const char* algo) {
#if defined(_WIN32)
  if (_setmode(_fileno(stdout), _O_BINARY) == -1)
    throw std::runtime_error("Cannot switch source output to binary mode");
#endif
  const auto generated = nativeKawpowSource(algo, 607046);
  if (std::fwrite(generated.second.data(), 1, generated.second.size(), stdout) != generated.second.size() ||
      std::fflush(stdout) != 0 || std::ferror(stdout))
    throw std::runtime_error("Cannot write generated source");
}

// Resource discovery must precede compiler-SDK construction. This host-only bootstrap links
// no SYCL/JIT runtime, while both tools still execute the original generator above.
#if defined(MOM_KAWPOW_SOURCE_EMITTER)
int main(int argc, char** argv) try {
  if (argc != 3 || std::strcmp(argv[1], "--emit-source") != 0)
    throw std::runtime_error("Expected --emit-source <algorithm>");
  nativeEmitKawpowSource(argv[2]);
  return 0;
} catch (const std::exception& error) {
  std::fprintf(stderr, "%s\n", error.what());
  return 1;
}
#endif
