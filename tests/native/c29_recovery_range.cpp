#include "../../sycl/c29/cycle.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

void require(const bool condition, const char* const label) {
  if (!condition) {
    std::fprintf(stderr, "C29 recovery range regression: %s\n", label);
    std::abort();
  }
}

// Spell the signed interpretation without implementation-defined uint32_t-to-int32_t conversion.
int64_t signed_value(const uint32_t value) {
  return value <= static_cast<uint32_t>(INT32_MAX) ? value : int64_t{value} - (int64_t{1} << 32);
}

uint32_t signed_min(const uint32_t left, const uint32_t right) {
  return signed_value(left) < signed_value(right) ? left : right;
}

} // namespace

int main() {
  static_assert(mom_c29::kEdgeBits == 29);
  static_assert(mom_c29::kNumEdges == (uint32_t{1} << 29));
  static_assert(mom_c29::kNoParent == UINT32_MAX);
  constexpr uint32_t num_edges = mom_c29::kNumEdges;

  const std::array<uint32_t, 7> valid = {
    0, 1, 63, num_edges / 2 - 1, num_edges / 2, num_edges - 2, num_edges - 1,
  };
  for (const uint32_t nonce : valid) {
    require(mom_c29::is_valid_nonce(nonce), "legal boundary nonce is valid");
    require(std::min(num_edges, nonce) == nonce && signed_min(num_edges, nonce) == nonce,
            "marker accepts every legal boundary nonce");
    for (const uint32_t other : valid)
      require(std::min(nonce, other) == signed_min(nonce, other),
              "signed and unsigned ordering agree for legal recovery operands");
  }
  require(signed_min(UINT32_MAX, valid.back()) == UINT32_MAX,
          "original high-bit marker reproduces signed-min failure");

  for (const unsigned proof_size : {32U, 42U}) {
    std::vector<uint32_t> nonces(proof_size, num_edges);
    const auto rejects = [&]() {
      return std::any_of(nonces.begin(), nonces.end(),
                         [](const uint32_t nonce) { return !mom_c29::is_valid_nonce(nonce); });
    };
    require(rejects(), "entirely unresolved proof is rejected");
    for (unsigned index = 0; index < proof_size; ++index)
      nonces[index] = valid[index % valid.size()];
    require(!rejects(), "all legal nonce boundaries are accepted");
    for (const uint32_t invalid : {
             num_edges, num_edges + 1, static_cast<uint32_t>(INT32_MAX),
             uint32_t{1} << 31, UINT32_MAX}) {
      nonces.back() = invalid;
      require(rejects(), "one missing or out-of-range edge rejects the whole proof");
    }
    nonces.back() = num_edges - 1;
    require(!rejects(), "maximum legal nonce remains recoverable");
  }
  std::puts("C29 recovery range: shared domain, signed-min ordering, proof32/42 bounds passed");
}
