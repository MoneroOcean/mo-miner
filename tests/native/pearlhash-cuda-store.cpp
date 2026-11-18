// Copyright GNU GPLv3 (c) 2026 MoneroOcean
// CPU-only regression for the actual rank-128 CUDA transcript store body.
//
// The test driver supplies store.inc by extracting the production CUDA source
// at test time.  It deliberately accepts either the existing shuffle/pair
// producer or the direct one-word-per-lane producer: both must realize the
// same canonical 16-word record layout.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string>

#define CUTE_UNROLL

namespace {

constexpr std::uint32_t kLaneCount = 32;
constexpr std::uint32_t kWordsPerLane = 8;
constexpr std::uint32_t kWordsPerRecord = 16;
constexpr std::uint32_t kRecordsPerWarp = 16;
constexpr std::uint64_t kBEndpoint = UINT64_C(0xffffffff);

std::array<std::array<std::uint32_t, kWordsPerLane>, kLaneCount> g_inputs{};
std::uint32_t g_lane = 0;

[[maybe_unused]] std::uint32_t mock_shfl(std::uint32_t mask, std::uint32_t value,
                                         int delta, int word) {
  if (mask != 0xffffffffu || delta != 4 || word < 0 || word >= int(kWordsPerLane) ||
      g_lane >= kLaneCount || (g_lane ^ std::uint32_t(delta)) >= kLaneCount)
    throw std::runtime_error("actual CUDA shuffle contract failed");
  (void)value;
  return g_inputs[g_lane ^ std::uint32_t(delta)][static_cast<std::size_t>(word)];
}

#define __shfl_xor_sync(mask, value, delta) mock_shfl(mask, value, delta, word)

struct OutputSink {
  static constexpr std::uint32_t kGuard = 0xcdcdcdcd;
  static constexpr std::size_t kPrefix = 5;
  static constexpr std::size_t kOutput = kWordsPerRecord * kRecordsPerWarp;
  static constexpr std::size_t kSuffix = 7;

  std::uint64_t base;
  std::array<std::uint32_t, kPrefix + kOutput + kSuffix> storage{};
  std::array<bool, kOutput> written{};

  explicit OutputSink(std::uint64_t absolute_base) : base(absolute_base) {
    storage.fill(kGuard);
  }

  std::uint32_t& operator[](std::uint64_t absolute) {
    if (absolute < base || absolute - base >= kOutput)
      throw std::runtime_error("actual CUDA store escaped bounded output");
    const std::size_t slot = static_cast<std::size_t>(absolute - base);
    if (written[slot])
      throw std::runtime_error("actual CUDA store duplicated an output word");
    written[slot] = true;
    return storage[kPrefix + slot];
  }

  bool complete() const {
    return std::all_of(written.begin(), written.end(), [](bool value) { return value; });
  }

  bool canaries_intact() const {
    for (std::size_t i = 0; i < kPrefix; ++i)
      if (storage[i] != kGuard) return false;
    for (std::size_t i = kPrefix + kOutput; i < storage.size(); ++i)
      if (storage[i] != kGuard) return false;
    return true;
  }

  std::uint32_t value(std::size_t index) const {
    if (index >= kOutput || !written[index])
      throw std::runtime_error("actual CUDA store left an output word missing");
    return storage[kPrefix + index];
  }
};

static void run_actual_store(std::uint32_t block_m, std::uint32_t block_n,
                             std::uint32_t warp_m, std::uint32_t warp_n,
                             std::uint32_t n, std::uint32_t lane,
                             OutputSink& transcript_out) {
  std::uint32_t transcript[kWordsPerLane];
  std::copy(g_inputs[lane].begin(), g_inputs[lane].end(), transcript);
  g_lane = lane;
#include "store.inc"
}

static std::uint64_t dense_base(std::uint32_t block_m, std::uint32_t block_n,
                                std::uint32_t warp_m, std::uint32_t warp_n,
                                std::uint32_t n) {
  const std::uint64_t index =
      ((std::uint64_t(block_m) * 2u + std::uint64_t(warp_m)) * std::uint64_t(n / 64u) +
       std::uint64_t(block_n) * 4u + std::uint64_t(warp_n)) * 16u;
  return index * kWordsPerRecord;
}

static std::array<std::uint32_t, OutputSink::kOutput> expected_output() {
  std::array<std::uint32_t, OutputSink::kOutput> expected{};
  for (std::uint32_t lane = 0; lane < kLaneCount; ++lane) {
    if ((lane & 4u) != 0) continue;
    const std::uint32_t slot = (lane >> 3) * 4u + (lane & 3u);
    for (std::uint32_t word = 0; word < kWordsPerLane; ++word) {
      expected[slot * kWordsPerRecord + word * 2u] = g_inputs[lane][word];
      expected[slot * kWordsPerRecord + word * 2u + 1u] = g_inputs[lane ^ 4u][word];
    }
  }
  return expected;
}

static void fill_inputs(std::uint64_t salt) {
  for (std::uint32_t lane = 0; lane < kLaneCount; ++lane)
    for (std::uint32_t word = 0; word < kWordsPerLane; ++word)
      g_inputs[lane][word] = static_cast<std::uint32_t>(
          UINT32_C(0x9e3779b9) ^ static_cast<std::uint32_t>(salt) ^
          lane * UINT32_C(0x01010101) ^ word * UINT32_C(0x1020304));
}

static void check_case(std::uint32_t m, std::uint32_t n, std::uint32_t block_m,
                       std::uint32_t block_n, std::uint32_t warp_m,
                       std::uint32_t warp_n, std::uint64_t salt) {
  if (m == 0 || n == 0 || m % 128u != 0 || n % 256u != 0 || warp_m >= 2u ||
      warp_n >= 4u || block_m >= m / 128u || block_n >= n / 256u)
    throw std::runtime_error("invalid CUDA store fixture geometry");

  fill_inputs(salt ^ (std::uint64_t(m) << 32) ^ n);
  OutputSink output(dense_base(block_m, block_n, warp_m, warp_n, n));
  for (std::uint32_t lane = 0; lane < kLaneCount; ++lane)
    run_actual_store(block_m, block_n, warp_m, warp_n, n, lane, output);

  if (!output.complete() || !output.canaries_intact())
    throw std::runtime_error("actual CUDA store completeness/canary failure");
  const auto expected = expected_output();
  for (std::size_t word = 0; word < expected.size(); ++word)
    if (output.value(word) != expected[word])
      throw std::runtime_error("actual CUDA store differs from canonical dense map");
}

static void check_duplicate_rejected() {
  fill_inputs(UINT64_C(0xdeadbeef));
  OutputSink output(dense_base(0, 0, 0, 0, 256));
  run_actual_store(0, 0, 0, 0, 256, 0, output);
  bool rejected = false;
  try {
    run_actual_store(0, 0, 0, 0, 256, 0, output);
  } catch (const std::exception&) {
    rejected = true;
  }
  if (!rejected)
    throw std::runtime_error("duplicate-write control was accepted");
  std::puts("CUDA_STORE_NEGATIVE duplicate_write=REJECTED");
}

static void run_all() {
  constexpr std::uint32_t m = 8192;
  constexpr std::uint32_t n = 524288;
  for (std::uint32_t warp_m = 0; warp_m < 2; ++warp_m)
    for (std::uint32_t warp_n = 0; warp_n < 4; ++warp_n)
      check_case(128, 256, 0, 0, warp_m, warp_n,
                 UINT64_C(0x101) + warp_m * 17u + warp_n);
  check_case(m, n, 63, 2047, 1, 3, UINT64_C(0x27182818));

  const std::uint64_t b_endpoint =
      (std::uint64_t(2047) * 256u + std::uint64_t(3) * 64u + 63u) * 8192u + 8191u;
  if (b_endpoint != kBEndpoint)
    throw std::runtime_error("4-GiB B endpoint arithmetic changed");
  std::puts("CUDA_STORE_FIXTURE PASS source=actual rank128 words=256 canaries=PASS "
            "warp_coords=PASS endpoint_B=0xffffffff");
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "--duplicate") {
      check_duplicate_rejected();
      return 0;
    }
    if (argc != 1)
      return 64;
    run_all();
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "CUDA_STORE_FIXTURE FAIL %s\n", error.what());
    return 1;
  }
}
