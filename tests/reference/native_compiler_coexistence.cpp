// Test-only entry wrapper for the complete AOT controller and both original compiler graphs.
// No device discovery/queue is performed: explicit target compilation brackets HIPRTC so
// collisions in globals, constructors, exception handling or repeated entry cannot hide at link time.
#include <cstdio>
#include <cstdlib>
#include <barrier>
#include <future>

int mom_dpcpp_probe(int, char**);
int mom_hiprtc_probe(int, char**);
#ifdef MOM_NATIVE_DPCPP_HOST_JIT
int mom_dpcpp_host_probe();
#endif
extern "C" int __real_main(int, char**);

extern "C" int __wrap_main(int argc, char** argv) {
  if (!std::getenv("MOM_AOT_COMPILER_COEXISTENCE")) return __real_main(argc, argv);
  char probe[] = "compiler-probe", old_target[] = "sm_61", new_target[] = "sm_120";
  char default_algo[] = "kawpow", output[] = "/out";
  char* algo = std::getenv("MOM_AOT_COMPILER_ALGO");
  if (!algo) algo = default_algo;
  char* dpcpp[] = {probe, old_target, algo};
  char* hiprtc[] = {probe, output};
#ifdef MOM_NATIVE_DPCPP_HOST_JIT
  if (mom_dpcpp_host_probe()) return 1;
#endif
  if (mom_dpcpp_probe(3, dpcpp)) return 1;
  dpcpp[1] = new_target;
  if (std::getenv("MOM_AOT_COMPILER_PARALLEL")) {
    // Separate compiler graphs must retain independent registries even on concurrent entry.
    std::barrier start(2);
    auto hip = std::async(std::launch::async, [&] {start.arrive_and_wait(); return mom_hiprtc_probe(2, hiprtc);});
    start.arrive_and_wait();
    const int status = mom_dpcpp_probe(3, dpcpp);
    if (hip.get() || status) return 1;
    std::puts("Concurrent compiler entry passed");
  } else if (mom_hiprtc_probe(2, hiprtc) || mom_dpcpp_probe(3, dpcpp)) return 1;
#ifdef MOM_NATIVE_DPCPP_HOST_JIT
  if (mom_dpcpp_host_probe()) return 1;
#endif
  std::puts("Isolated DPC++ and maintained HIPRTC coexistence passed; entering original AOT controller");
  std::fflush(nullptr); // The controller can exec its worker; preserve parent/child log boundaries.
  return __real_main(argc, argv);
}
