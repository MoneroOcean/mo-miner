// Actual cleanup bodies must remain lazy on other vendors and release partial CUDA state once.
#include <cstdlib>
#include <cstdio>
#include <stdexcept>

static unsigned queries, selected, events, modules, primaries;
struct Driver {
  static Driver& instance() {++queries; static Driver driver; return driver;}
  bool basic_available() const {return true;}
  bool extended_available() const {return true;}
  int (*set_current)(void*) = +[](void*) {++selected; return 0;};
  int (*event_destroy)(void*) = +[](void*) {++events; return 0;};
  int (*module_unload)(void*) = +[](void*) {++modules; return 0;};
  int (*release_primary)(int) = +[](int) {++primaries; return 0;};
};
using CnGpuCudaDriverApi = Driver;
using PearlHashCudaDriverApi = Driver;
struct State {
  void *module = nullptr, *kernel = nullptr, *gemm_kernel = nullptr, *hash_kernel = nullptr, *selector_kernel = nullptr;
  void *done = nullptr, *started = nullptr, *context = nullptr;
  int device = 7;
  bool retained_primary = false, enabled = true, checked = true, packed = true, compact = true;
#include "release.inc"
};
int main(int argc, char** argv) try {
  if (argc != 2) return 2;
  const int mode = std::atoi(argv[1]);
  State state;
  state.module = mode == 1 || mode == 6 ? &state : nullptr;
  state.done = mode == 2 || mode == 6 ? &state : nullptr;
  state.context = mode == 3 || mode == 6 ? &state : nullptr;
  state.retained_primary = mode == 4 || mode == 6;
#if MOM_TEST_PEARL
  state.started = mode == 5 || mode == 6 ? &state : nullptr;
#endif
  const bool owned = state.module || state.done || state.context || state.retained_primary || state.started;
  state.release(); state.release();
  if (queries != unsigned(owned) || selected != unsigned(mode == 3 || mode == 6) ||
      events != unsigned(mode == 2 || mode == 6) + unsigned(MOM_TEST_PEARL && (mode == 5 || mode == 6)) ||
      modules != unsigned(mode == 1 || mode == 6) || primaries != unsigned(mode == 4 || mode == 6) ||
      state.module || state.done || state.context || state.retained_primary || state.enabled)
    throw std::runtime_error("Cleanup probed an unused driver or changed partial-resource ownership");
#if MOM_TEST_PEARL
  if (state.started || state.checked || state.packed || state.compact || state.selector_kernel)
    throw std::runtime_error("Empty cleanup failed to clear cached compilation state");
#endif
} catch (const std::exception& error) {
  std::fprintf(stderr, "%s\n", error.what());
  return 1;
}
