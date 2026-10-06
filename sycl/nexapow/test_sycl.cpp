// Compile-only device probe. test.sh builds this translation unit but never links or runs it.

#include <sycl/sycl.hpp>

#include <cstdint>

namespace mom_nexapow {
#include "device.inc"
} // namespace mom_nexapow

class NexaPowCompileProbe;

int main() {
  sycl::queue queue{sycl::default_selector_v};
  sycl::buffer<uint8_t, 1> result(32);
  queue.submit([&](sycl::handler& handler) {
    auto output = result.get_access<sycl::access::mode::write>(handler);
    handler.single_task<NexaPowCompileProbe>([=]() {
      const uint8_t header[32]{};
      const uint8_t extranonce[8]{};
      uint8_t hash[32]{};
      mom_nexapow::np_hash_one(header, extranonce, 0, hash);
      for (unsigned i = 0; i < 32; ++i)
        output[i] = hash[i];
    });
  });
  return 0;
}
