// Reuse maintained SYCL primitives across CPU-only and explicitly selected GPU proof builds.
// This fixture is not a miner or a performance qualification of all supported algorithms.
#include <SYCL/sycl.hpp>
#include <hipSYCL/runtime/backend.hpp>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>

namespace device_hash {
#include "../../sycl/blake3-device.inc"
}
#include "../../sycl/pearlhash/blake3.inc"

int main(int argc, char **argv) {
  try {
#ifdef MOM_NATIVE_GPU_PROBE
    // One linked multi-vendor image must honor explicit runtime visibility without probing or
    // selecting the other owner's GPU. Existing single-backend/CPU fixtures stay unchanged.
    const char* name = std::getenv("ACPP_VISIBILITY_MASK");
    if (!name || (std::strcmp(name, "hip") && std::strcmp(name, "cuda")))
      throw std::runtime_error("Select exactly one GPU backend for this proof");
    const auto id = std::strcmp(name, "hip") == 0 ? hipsycl::rt::backend_id::hip : hipsycl::rt::backend_id::cuda;
    constexpr auto selector = sycl::gpu_selector_v;
    constexpr bool gpu = true;
#elif defined(MOM_NATIVE_HIP_PROBE)
    constexpr auto name = "hip";
    constexpr auto id = hipsycl::rt::backend_id::hip;
    constexpr auto selector = sycl::gpu_selector_v;
    constexpr bool gpu = true;
#elif defined(MOM_NATIVE_CUDA_PROBE)
    constexpr auto name = "cuda";
    constexpr auto id = hipsycl::rt::backend_id::cuda;
    constexpr auto selector = sycl::gpu_selector_v;
    constexpr bool gpu = true;
#else
    constexpr auto name = "omp";
    constexpr auto id = hipsycl::rt::backend_id::omp;
    constexpr auto selector = sycl::cpu_selector_v;
    constexpr bool gpu = false;
#endif
    hipsycl::rt::backend_loader loader;
    loader.query_backends(); loader.query_backends();
    // GPU backends still need CPU host-memory operations; visibility cannot remove that backend.
    if (loader.get_num_backends() != (gpu ? 2 : 1) || !loader.has_backend(name) ||
        !loader.has_backend("omp") || loader.create("absent"))
      throw std::runtime_error("Static registry failed deduplication/isolation");
    std::unique_ptr<hipsycl::rt::backend> backend(loader.create(name));
    if (!backend || backend->get_unique_backend_id() != id)
      throw std::runtime_error("Static backend factory mismatch");

    sycl::queue queue(selector, sycl::property::queue::in_order{});
    if (gpu ? !queue.get_device().is_gpu() : !queue.get_device().is_cpu())
      throw std::runtime_error("Static probe selected another device type");
    constexpr std::array<unsigned, 10> sizes{0, 1, 63, 64, 65, 127, 128, 255, 1023, 1024};
    std::array<std::uint8_t, 1024> input{};
    std::array<std::uint8_t, sizes.size() * 32> output{};
    for (unsigned i = 0; i < input.size(); ++i) input[i] = static_cast<std::uint8_t>(i * 7 + 3);
    const auto release = [&queue](std::uint8_t *pointer) {sycl::free(pointer, queue);};
    std::unique_ptr<std::uint8_t, decltype(release)> data(sycl::malloc_device<std::uint8_t>(input.size(), queue), release);
    std::unique_ptr<std::uint8_t, decltype(release)> hashes(sycl::malloc_device<std::uint8_t>(output.size(), queue), release);
    if (!data || !hashes) throw std::bad_alloc();
    // A failed submission may leave earlier commands live. Finish them before USM/host arrays
    // unwind; normal execution still uses wait_and_throw below to report asynchronous faults.
    struct Drain {sycl::queue &queue; ~Drain() {try {queue.wait();} catch (...) {}}} drain{queue};
    const auto upload = queue.memcpy(data.get(), input.data(), input.size());
    auto *source = data.get(); auto *target = hashes.get();
    const auto kernel = queue.submit([&](sycl::handler &handler) {
      handler.depends_on(upload);
      handler.parallel_for(sycl::range<1>(sizes.size()), [=](sycl::id<1> index) {
        device_hash::blake3_dev(target + index[0] * 32, 32, source, sizes[index[0]]);
      });
    });
    queue.submit([&](sycl::handler &handler) {
      handler.depends_on(kernel); handler.memcpy(output.data(), hashes.get(), output.size());
    }).wait_and_throw();
    queue.wait_and_throw();
    for (unsigned i = 0; i < sizes.size(); ++i) {
      std::uint8_t expected[32]; pearlhash_b3::b3(input.data(), sizes[i], nullptr, expected);
      if (argc == 2 && std::strcmp(argv[1], "mismatch") == 0 && i == 0) expected[0] ^= 1;
      if (std::memcmp(output.data() + i * 32, expected, 32)) throw std::runtime_error("Static SYCL hash mismatch");
    }
    std::printf("Static SYCL registry/USM/queue/event/BLAKE3 parity passed (%s; %s)\n", name, gpu ? "GPU JIT" : "CPU only; no JIT");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
