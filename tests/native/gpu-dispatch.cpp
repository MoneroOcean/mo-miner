// CPU-only fake factories exercise the real selector and generated public forwards.
#include "gpu-table.h"
#include "runtime/gpu-dispatch.h"
#include <array>
#include <atomic>
#include <cassert>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

enum Lane {Hip, Cuda, Dpcpp, Intel, Opencl, Count};
static std::array<std::atomic<unsigned>, Count> factories{}, closes{};
static constexpr std::array<unsigned, Count> markers{100, 150, 200, 300, 400};

// Simulate three images defining the same kernel ID with different layouts. Eager registration
// would collide before main; the real dispatcher must defer them and activate exactly one.
static unsigned images[3]{}, registered = 0, registration_calls = 0;
static void* active_image = nullptr;
static void* partial_image = nullptr;
static unsigned unregister_calls = 0;
#ifdef _WIN32
// Windows' real handler destroys ProgramManager through a lazy atexit callback. It must
// have been registered before the offload wrapper registers its image-retirement callback.
extern "C" void mom_native_sycl_prepare_shutdown() {
  static const bool prepared = [] {
    std::atexit([] { if (registered) std::_Exit(90); });
    return true;
  }();
  (void)prepared;
}
#endif
extern "C" void __sycl_register_lib(void* image) {
#ifdef _WIN32
  mom_native_sycl_prepare_shutdown(); // The real first SYCL use also registers that handler.
#endif
  if (active_image) std::_Exit(86); // The old eager topology collides before algorithm dispatch.
  ++registration_calls;
  if (std::getenv("MOM_TEST_REGISTRATION_FAIL")) {
    partial_image = image; // Models mutation before commit, not a recoverable transaction.
    throw std::string("fixture registration fault");
  }
  active_image = image;
  ++registered;
}
extern "C" void __sycl_unregister_lib(void* image) {
  ++unregister_calls;
  assert(!partial_image); // Removing an uncommitted image is not a rollback operation.
  assert(active_image == image && registered == 1);
  active_image = nullptr;
  --registered;
}
#ifdef MOM_NATIVE_GPU_DPCPP
extern "C" void mom_native_register_dpcpp(void*);
extern "C" void mom_native_unregister_dpcpp(void*);
extern "C" void mom_native_register_oneapi(void*);
extern "C" void mom_native_unregister_oneapi(void*);
extern "C" void mom_native_register_dpcpp_opencl(void*);
extern "C" void mom_native_unregister_dpcpp_opencl(void*);
static struct ImageLifetime {
  ImageLifetime() {
    if (std::getenv("MOM_TEST_OMIT_IMAGES")) return;
    // Convert an expected producer-contract abort into a deterministic, core-free death test.
    if (std::getenv("MOM_TEST_ALIAS_IMAGES"))
      std::signal(SIGABRT, [](int) { std::_Exit(88); });
    if (std::getenv("MOM_TEST_EAGER_IMAGES")) {
      __sycl_register_lib(&images[0]);
      __sycl_register_lib(&images[1]);
    }
    mom_native_register_dpcpp(&images[0]);
    if (std::getenv("MOM_TEST_ALIAS_IMAGES")) {
      mom_native_register_oneapi(&images[0]);
      std::_Exit(89); // An accepted alias must not later pass via another slot's duplicate check.
    }
    mom_native_register_oneapi(&images[1]);
    mom_native_register_dpcpp_opencl(&images[2]);
    assert(!active_image && registered == 0);
  }
  ~ImageLifetime() {
    if (std::getenv("MOM_TEST_OMIT_IMAGES")) return;
    mom_native_unregister_dpcpp(&images[0]);
    mom_native_unregister_oneapi(&images[1]);
    mom_native_unregister_dpcpp_opencl(&images[2]);
    assert(!active_image && registered == 0);
  }
} image_lifetime;
#endif
// Mirrors the static runtime's late shutdown: image retirement must already be complete.
__attribute__((destructor(110))) static void runtime_shutdown() {
  if (active_image || registered) std::_Exit(87);
  if (partial_image && unregister_calls) std::_Exit(87);
  partial_image = nullptr; // Only final process/runtime teardown discards failed bookkeeping.
}

template<Lane lane> static uint64_t fakeWork(unsigned a, uint64_t b) {return a + b + markers[lane];}
template<Lane lane> static void fakeCleanup() noexcept {++closes[lane];}
template<Lane lane> static const mom::runtime::GpuTable* fakeFactory() {
  ++factories[lane];
  static const mom::runtime::GpuTable table{fakeWork<lane>, fakeCleanup<lane>};
  return std::getenv("MOM_TEST_NULL_FACTORY") ? nullptr : &table;
}
extern "C" const mom::runtime::GpuTable* mom_native_gpu_acpp_hip() {return fakeFactory<Hip>();}
extern "C" const mom::runtime::GpuTable* mom_native_gpu_acpp_cuda() {return fakeFactory<Cuda>();}
extern "C" const mom::runtime::GpuTable* mom_native_gpu_dpcpp() {return fakeFactory<Dpcpp>();}
extern "C" const mom::runtime::GpuTable* mom_native_gpu_oneapi() {return fakeFactory<Intel>();}
extern "C" const mom::runtime::GpuTable* mom_native_gpu_dpcpp_opencl() {return fakeFactory<Opencl>();}

// Distinct markers catch image aliases, while counts expose initialization/cleanup ownership.
static void expectCounts(int lane, unsigned opened, unsigned closed) {
  for (int i = 0; i < Count; ++i) {
    assert(factories[i] == (i == lane ? opened : 0));
    assert(closes[i] == (i == lane ? closed : 0));
  }
}

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  const std::string mode = argv[1];
  // CPU-only work or an earlier failed selector must not initialize a GPU graph at cleanup.
  sycl_cleanup();
  expectCounts(-1, 0, 0);
  if (mode == "unused") return 0;
  if (mode == "scratch-error" || mode == "scratch-retry") {
    bool rejected = false;
    try {work(7, 11);} catch (const std::string& message) {
      rejected = message.starts_with("Native HIP JIT needs a writable temporary directory:");
    }
    if (!rejected) return 7;
    sycl_cleanup();
    expectCounts(-1, 0, 0);
    if (mode == "scratch-error") return 0;
    assert(setenv("TMPDIR", std::getenv("MOM_TEST_WRITABLE_TMP"), 1) == 0);
  }
  if (mode == "registration-fault" || mode == "missing-images") {
    for (unsigned i = 0; i < 2; ++i) {
      bool rejected = false;
      try {work(7, 11);} catch (const std::string&) {rejected = true;}
      assert(rejected && !active_image && !registered);
      assert(registration_calls == (mode == "registration-fault" ? 1u : 0u));
      assert(unregister_calls == 0);
      assert(bool(partial_image) == (mode == "registration-fault"));
      unsetenv("MOM_TEST_REGISTRATION_FAIL");
      sycl_cleanup();
    }
    return 0;
  }
  if (mode == "reject" || mode == "retry") {
    bool rejected = false;
    try {work(7, 11);} catch (const std::string& message) {
      rejected = message.starts_with("Native GPU compiler is not linked:");
    }
    assert(rejected);
    sycl_cleanup();
    for (const auto& count : closes) assert(count == 0);
    if (mode != "retry") return 0;
    expectCounts(-1, 0, 0);
    assert(setenv("MOM_SYCL_COMPILER", "acpp-hip", 1) == 0);
  }
  const char* key = std::getenv("MOM_SYCL_COMPILER");
  Lane lane = Hip;
  if (key) {
    if (std::strcmp(key, "dpcpp") == 0) lane = Dpcpp;
    else if (std::strcmp(key, "acpp-cuda") == 0) lane = Cuda;
    else if (std::strcmp(key, "oneapi") == 0) lane = Intel;
    else if (std::strcmp(key, "dpcpp-opencl") == 0) lane = Opencl;
  }
  const uint64_t expected = 0x100000003ULL + 7 + markers[lane];
  std::vector<std::thread> threads;
  for (unsigned i = 0; i < 16; ++i) threads.emplace_back([&] {assert(work(7, 0x100000003ULL) == expected);});
  for (auto& thread : threads) thread.join();
  expectCounts(lane, 1, 0);
  const int image = lane == Dpcpp ? 0 : lane == Intel ? 1 : lane == Opencl ? 2 : -1;
  assert(active_image == (image < 0 ? nullptr : &images[image]));
  assert(registered == (image < 0 ? 0u : 1u));
  assert(setenv("MOM_SYCL_COMPILER", lane == Dpcpp ? "acpp-hip" : "dpcpp", 1) == 0);
  assert(work(7, 0x100000003ULL) == expected); // A worker cannot retarget live state by changing env.
  sycl_cleanup(); sycl_cleanup();
  expectCounts(lane, 1, 2);
  std::puts("Generated GPU forwards: immutable route, first-entry concurrency and owned cleanup passed");
}
