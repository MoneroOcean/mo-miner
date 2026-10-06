#include "../../sycl/kawpow/epoch.h"

#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <string>

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

static bool rejected_words(const uint64_t new_dag_words) {
  try {
#include "progpow_dag_limit.inc"
  } catch (const std::string& error) {
    return error == "KawPow DAG exceeds 32-bit word addressing";
  }
  return false;
}

int main() {
  // Exercise only the extracted addressing guard; other epoch admission remains separate.
  const uint64_t limit = uint64_t{UINT32_MAX} + 1;
  for (const auto words : {uint64_t{0}, limit - 1, limit, limit + 1, UINT64_MAX})
    if (rejected_words(words) != (words > limit))
      std::abort();
  check({0, 0});
  check({1299, 0});
  check({1300, 1});
  check({1205099, 926});
  check({1205100, 650});
  check({1370195, 650});
  check({UINT32_MAX, 650});
  return 0;
}
