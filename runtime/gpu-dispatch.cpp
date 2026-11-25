// Each same-executable worker owns one compiler/runtime graph. The maintained JS policy chooses
// its key before the process starts; algorithm changes already replace the worker generation.
#include "gpu-dispatch.h"
#include "gpu-table.h"
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <string>
#if defined(MOM_NATIVE_GPU_ACPP_HIP) && !defined(_WIN32)
#include <cerrno>
#include <unistd.h>
#endif

#ifndef MOM_NATIVE_GPU_DEFAULT
#error Native builds must select an explicit discovery/default GPU compiler
#endif
#ifdef MOM_NATIVE_GPU_DPCPP
extern "C" const mom::runtime::GpuTable* mom_native_gpu_dpcpp();
#endif
#ifdef MOM_NATIVE_GPU_ONEAPI
extern "C" const mom::runtime::GpuTable* mom_native_gpu_oneapi();
#endif
#ifdef MOM_NATIVE_GPU_DPCPP_OPENCL
extern "C" const mom::runtime::GpuTable* mom_native_gpu_dpcpp_opencl();
#endif
#ifdef MOM_NATIVE_GPU_ACPP_CUDA
extern "C" const mom::runtime::GpuTable* mom_native_gpu_acpp_cuda();
#endif
#ifdef MOM_NATIVE_GPU_ACPP_HIP
extern "C" const mom::runtime::GpuTable* mom_native_gpu_acpp_hip();
#endif

#if defined(MOM_NATIVE_GPU_DPCPP) || defined(MOM_NATIVE_GPU_ONEAPI) || defined(MOM_NATIVE_GPU_DPCPP_OPENCL)
// One merged bundle per compiler graph. Its kernels may have the same C++ names but different
// argument layouts in another profile. Remember descriptors during static initialization;
// register only the selected worker's graph, before its first SYCL call. POD storage survives
// image destructors and does not initialize a GPU/runtime for CPU-only work.
extern "C" void __sycl_register_lib(void*);
extern "C" void __sycl_unregister_lib(void*);
#ifdef _WIN32
extern "C" void mom_native_sycl_prepare_shutdown();
#endif
namespace {
struct StaticImages { void* descriptor; bool registered; bool failed; };
enum ImageProfileSlot : unsigned { dpcppSlot, oneapiSlot, dpcppOpenclSlot, noImageSlot };
StaticImages staticImages[noImageSlot]{};

void rememberImages(ImageProfileSlot profile, void* descriptor) {
  auto& image = staticImages[profile];
  // The producer emits exactly one merged registration unit per profile. Do not silently
  // overwrite another bundle if that build contract changes.
  if (!descriptor || image.descriptor) std::abort();
  for (const auto& other : staticImages)
    if (other.descriptor == descriptor) std::abort();
#ifdef _WIN32
  // The wrapper registers its atexit callback after this gate returns. Prepare the original
  // runtime handler first so image retirement precedes ProgramManager destruction on Windows.
  mom_native_sycl_prepare_shutdown();
#endif
  image.descriptor = descriptor;
}
void forgetImages(ImageProfileSlot profile, void* descriptor) {
  auto& image = staticImages[profile];
  if (image.descriptor != descriptor) std::abort();
  if (image.registered) __sycl_unregister_lib(descriptor);
  image = {};
}
// select() maps the key to its image slot; ACPP graphs have no static images (noImageSlot).
void activateImages(ImageProfileSlot profile, const char* key) {
  if (profile == noImageSlot) return;
  auto& image = staticImages[profile];
  if (!image.descriptor || image.failed)
    throw std::string("Native GPU images are unavailable: ") + key;
  if (!image.registered) {
    // addImages is non-transactional: unregister skips uncommitted images and cannot roll
    // back their partial maps. Never retry or unregister a throwing registration; discard
    // the worker. Ordinary failed factory selection still retries without touching images.
    image.failed = true;
    __sycl_register_lib(image.descriptor);
    image.registered = true;
    image.failed = false;
  }
}
}
// Only static offload-wrapper imports are remapped. Runtime-compiled SYCL JIT images keep
// the original registration API and the already-selected process's ordinary SYCL runtime.
#define MOM_IMAGE_GATE(name, profile) \
  extern "C" void mom_native_register_##name(void* image) { rememberImages(profile, image); } \
  extern "C" void mom_native_unregister_##name(void* image) { forgetImages(profile, image); }
#ifdef MOM_NATIVE_GPU_DPCPP
MOM_IMAGE_GATE(dpcpp, dpcppSlot)
#endif
#ifdef MOM_NATIVE_GPU_ONEAPI
MOM_IMAGE_GATE(oneapi, oneapiSlot)
#endif
#ifdef MOM_NATIVE_GPU_DPCPP_OPENCL
MOM_IMAGE_GATE(dpcpp_opencl, dpcppOpenclSlot)
#endif
#undef MOM_IMAGE_GATE
#endif

namespace mom::runtime {
namespace {
// Cleanup alone must not initialize an unused compiler or turn a failed selection into a
// noexcept termination. Publish only a complete successful selection, immutable until exit.
std::atomic<const GpuTable*> active{nullptr};

#if defined(MOM_NATIVE_GPU_ACPP_HIP) && !defined(_WIN32)
// Embedded COMGR still needs private scratch data for HIP's first stream/blit JIT.
// Its failed-stream cleanup can dereference null if temporary-directory creation fails.
// Probe once before touching that runtime, using LLVM's exact environment precedence;
// mkdtemp is atomic/owner-only and rmdir never recursively removes somebody else's data.
void requireHipScratch() {
  const char* root = "/tmp";
  for (const char* name : {"TMPDIR", "TMP", "TEMP", "TEMPDIR"}) {
    if (const char* value = std::getenv(name)) {root = value; break;}
  }
  const char* prefix = "Native HIP JIT needs a writable temporary directory: ";
  if (*root != '/' || std::strlen(root) > 4096)
    throw std::string(prefix) + "set TMPDIR to an absolute writable directory";
  std::string probe = std::string(root) + "/mom-jit-XXXXXX";
  if (!::mkdtemp(probe.data())) throw std::string(prefix) + std::strerror(errno);
  if (::rmdir(probe.c_str()) != 0) throw std::string(prefix) + std::strerror(errno);
}
#endif

const GpuTable* select() {
  const char* key = std::getenv("MOM_SYCL_COMPILER");
  if (!key || !*key) key = MOM_NATIVE_GPU_DEFAULT;
  const GpuTable* table = nullptr;
#if defined(MOM_NATIVE_GPU_DPCPP) || defined(MOM_NATIVE_GPU_ONEAPI) || defined(MOM_NATIVE_GPU_DPCPP_OPENCL)
  ImageProfileSlot images = noImageSlot;
#endif
#ifdef MOM_NATIVE_GPU_DPCPP
  if (std::strcmp(key, "dpcpp") == 0) {table = mom_native_gpu_dpcpp(); images = dpcppSlot;}
#endif
#ifdef MOM_NATIVE_GPU_ONEAPI
  if (std::strcmp(key, "oneapi") == 0) {table = mom_native_gpu_oneapi(); images = oneapiSlot;}
#endif
#ifdef MOM_NATIVE_GPU_DPCPP_OPENCL
  if (std::strcmp(key, "dpcpp-opencl") == 0) {table = mom_native_gpu_dpcpp_opencl(); images = dpcppOpenclSlot;}
#endif
#ifdef MOM_NATIVE_GPU_ACPP_CUDA
  if (std::strcmp(key, "acpp-cuda") == 0) table = mom_native_gpu_acpp_cuda();
#endif
#ifdef MOM_NATIVE_GPU_ACPP_HIP
  if (std::strcmp(key, "acpp-hip") == 0) {
#ifndef _WIN32
    requireHipScratch();
#endif
    table = mom_native_gpu_acpp_hip();
  }
#endif
  // Unsupported keys fail locally, never through a fallback with the wrong kernel/image ABI.
  // The table is trusted linked code generated from the same lib.h, not external protocol data.
  if (!table || !table->sycl_cleanup)
    throw std::string("Native GPU compiler is not linked: ") + key;
#if defined(MOM_NATIVE_GPU_DPCPP) || defined(MOM_NATIVE_GPU_ONEAPI) || defined(MOM_NATIVE_GPU_DPCPP_OPENCL)
  // Selection's function-static initialization serializes first entry. Failed selection is
  // still retryable: no profile's images are registered before its factory validates.
  activateImages(images, key);
#endif
  active.store(table, std::memory_order_release);
  return table;
}
}

const GpuTable& selectedGpu() {
  static const GpuTable* const table = select();
  return *table;
}

void cleanupGpu() noexcept {
  if (const auto* table = active.load(std::memory_order_acquire)) table->sycl_cleanup();
}
} // namespace mom::runtime
