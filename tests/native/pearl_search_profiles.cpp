// Compile the maintained signature/host dispatch, replacing only GPU execution with observation.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace sycl {
enum class backend {cuda, ext_oneapi_cuda, hip, other};
struct device {backend value; backend get_backend() const {return value;}};
struct queue {device value; device get_device() const {return value;}};
}
static bool mom_is_hip(sycl::device device) {return device.value == sycl::backend::hip;}
struct Buffers {int proof_rows, proof_cols;};
constexpr int pearlhash_portable_er = 4, pearlhash_portable_ec = 4;
static int actual_rows = 0, actual_cols = 0;
static bool actual_pair = false;
#include "signature.inc"
static void search_t(sycl::queue&, const Buffers&, uint32_t, int, int, int, int, bool) {
  actual_rows = Rows; actual_cols = Cols; actual_pair = Pair;
}
#include "dispatch.inc"

int main() try {
#if defined(MOM_SYCL_ADAPTIVECPP_CUDA)
  constexpr auto cuda_backend = sycl::backend::cuda;
#else
  constexpr auto cuda_backend = sycl::backend::ext_oneapi_cuda;
#endif
  sycl::queue hip{{sycl::backend::hip}}, cuda{{cuda_backend}}, other{{sycl::backend::other}};
  for (const auto geometry : {Buffers{4, 64}, Buffers{8, 32}, Buffers{16, 16}}) {
    for (auto* queue : {&hip, &cuda, &other}) {
      search(*queue, geometry, 0, 256, 256, 2048, 128, true);
#if defined(MOM_SYCL_HAS_HIP)
      const bool selected_hip = mom_is_hip(queue->get_device());
#else
      const bool selected_hip = false;
#endif
      if (actual_rows != (selected_hip ? geometry.proof_rows : 16) ||
          actual_cols != (selected_hip ? geometry.proof_cols : 16) || actual_pair)
        throw std::runtime_error("Backend dispatch lost the proof geometry");
    }
  }
  bool rejected = false;
  try {search(hip, {16, 16}, 0, 256, 256, 2048, 128, true, true);}
  catch (const std::string&) {rejected = true;}
  if (!rejected) throw std::runtime_error("Pair dispatch admitted a non-CUDA device");
#if defined(MOM_SYCL_HAS_CUDA) || defined(MOM_SYCL_ADAPTIVECPP_CUDA)
  search(cuda, {16, 16}, 0, 256, 256, 2048, 128, true, true);
  if (!actual_pair || actual_rows != 16 || actual_cols != 16)
    throw std::runtime_error("Pair dispatch changed the canonical proof tile");
#endif
} catch (const std::exception& error) {
  std::fprintf(stderr, "%s\n", error.what());
  return 1;
}
