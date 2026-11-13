// Exercise the pinned original UR parser and registration code. Opaque factory spies replace
// every adapter, so selecting a backend never loads a driver or dispatches work.
#include "ur_loader.hpp"
#include <cstdio>
#include <cstdlib>

namespace {
unsigned mask = 0, calls[4]{};
ur_result_t factory(ur_dditable_t* table, unsigned index) {
  mask |= 1U << index;
  ++calls[index];
  *table = {};
  return UR_RESULT_SUCCESS;
}
}
namespace ur::level_zero {
ur_result_t urAdapterGetDdiTables(ur_dditable_t* table) {return factory(table, 0);}
}
namespace ur::opencl {
ur_result_t urAdapterGetDdiTables(ur_dditable_t* table) {return factory(table, 2);}
}
extern "C" ur_result_t mom_ur_cuda_factory(ur_dditable_t* table) {return factory(table, 1);}
extern "C" ur_result_t mom_ur_level_zero_v2_factory(ur_dditable_t* table) {return factory(table, 3);}

int main(int argc, char** argv) {
  if (argc != 2 || !argv[1][0]) return 85;
  char* end = nullptr;
  const auto parsed = std::strtoul(argv[1], &end, 10);
  if (!end || *end || parsed > 15) return 85;
  const auto result = ur_loader::getContext()->init();
  const unsigned expected = unsigned(parsed);
  if (result != UR_RESULT_SUCCESS || mask != expected || calls[0] > 1 || calls[1] > 1 || calls[2] > 1 || calls[3] > 1) {
    std::fprintf(stderr, "Static adapter selection: result=%u mask=%u expected=%u\n", unsigned(result), mask, expected);
    return 86;
  }
  std::printf("Static adapter selection mask %u passed\n", mask);
}
