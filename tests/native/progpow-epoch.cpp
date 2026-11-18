#include "../../sycl/kawpow/epoch.h"

#include <cstdint>
#include <cstdlib>

namespace {

struct EpochCase {
  std::uint32_t height;
  std::uint32_t epoch;
};

void check(const EpochCase test) {
  if (mom_kawpow::firopow_epoch(test.height) != test.epoch)
    std::abort();
}

} // namespace

// The DAG word-addressing guard is shared with Etchash and covered by ethash-dag-words.cpp.
int main() {
  check({0, 0});
  check({1299, 0});
  check({1300, 1});
  check({1205099, 926});
  check({1205100, 650});
  check({1370195, 650});
  check({UINT32_MAX, 650});
  return 0;
}
