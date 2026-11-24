// Compile the maintained signature/host dispatch, replacing only GPU execution with observation.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#include "proof-axis.h"

using namespace mom_pearlhash;

namespace sycl {
enum class backend {cuda, ext_oneapi_cuda, hip, other};
struct device {backend value; backend get_backend() const {return value;}};
struct queue {device value; device get_device() const {return value;}};
}
static bool mom_is_hip(sycl::device device) {return device.value == sycl::backend::hip;}
struct Buffers {ProofAxis proof_rows, proof_cols;};
constexpr int pearlhash_portable_er = 4, pearlhash_portable_ec = 4;
static int actual_rows = 0, actual_cols = 0;
#include "signature.inc"
static void search_t(sycl::queue&, const Buffers&, uint32_t, int, int, int, int, bool) {
  actual_rows = Rows; actual_cols = Cols;
}
#include "dispatch.inc"

int main() try {
#if defined(MOM_SYCL_ADAPTIVECPP_CUDA)
  constexpr auto cuda_backend = sycl::backend::cuda;
#else
  constexpr auto cuda_backend = sycl::backend::ext_oneapi_cuda;
#endif
#if defined(MOM_SYCL_HAS_CUDA) || defined(MOM_SYCL_ADAPTIVECPP_CUDA)
  constexpr bool has_cuda = true;
#else
  constexpr bool has_cuda = false;
#endif
  sycl::queue hip{{sycl::backend::hip}}, cuda{{cuda_backend}}, other{{sycl::backend::other}};
  const Buffers square{kSquareAxis, kSquareAxis}, compact{kCompactRows, kCompactCols},
      lanes{kAmdLaneRows, kAmdLaneCols};
  for (const Buffers* geometry : {&square, &compact, &lanes}) {
    for (auto* queue : {&hip, &cuda, &other}) {
      search(*queue, *geometry, 0, 256, 256, 2048, 128, true);
#if defined(MOM_SYCL_HAS_HIP)
      const bool selected_hip = mom_is_hip(queue->get_device());
#else
      const bool selected_hip = false;
#endif
      // The compact CUDA tile and the AMD lane tile are selected by their exact opening patterns;
      // every other pattern keeps the 16x16 work-group tiling.
      const bool compact_tile = has_cuda && geometry == &compact;
      const bool lane_tile = !compact_tile && selected_hip && geometry == &lanes;
      const int rows = compact_tile ? 24 : lane_tile ? 52 : 16;
      const int cols = compact_tile ? 8 : lane_tile ? 4 : 16;
      if (actual_rows != rows || actual_cols != cols)
        throw std::runtime_error("Backend dispatch lost the proof geometry");
    }
  }
} catch (const std::exception& error) {
  std::fprintf(stderr, "%s\n", error.what());
  return 1;
}
