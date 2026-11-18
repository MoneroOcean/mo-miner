// Compile-only stress fixture for deterministic HIP HCF IDs and host-global ordering.
#include <SYCL/sycl.hpp>

#include <cstddef>
#include <utility>

#ifndef HCF_PROOF_VALUE
#error "Compile this fixture with -DHCF_PROOF_VALUE=1 or =2"
#endif

template<std::size_t Index>
class HcfReproKernel;

template<std::size_t Index>
void submitHcfReproKernel(sycl::queue &queue, int *output) {
  queue.submit([&](sycl::handler &handler) {
    handler.single_task<HcfReproKernel<Index>>([=] {
      output[Index] = static_cast<int>(Index) + HCF_PROOF_VALUE;
    });
  });
}

template<std::size_t... Indices>
void submitHcfReproKernels(sycl::queue &queue, int *output, std::index_sequence<Indices...>) {
  (submitHcfReproKernel<Indices>(queue, output), ...);
}

// All 24 distinct named kernels are instantiated in one translation unit; this function is
// never linked or run by the build-only proof.
void native_sycl_hcf_repro_fixture(sycl::queue &queue, int *output) {
  submitHcfReproKernels(queue, output, std::make_index_sequence<24>{});
}
