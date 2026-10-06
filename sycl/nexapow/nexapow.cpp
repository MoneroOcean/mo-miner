// Copyright GNU GPLv3 (c) 2026 MoneroOcean <support@moneroocean.stream>
// NexaPow/Echelon portable SYCL search.
//
// Work blob ABI:
//   40 bytes: [0..31] header hash, [32..39] miner nonce
//   44 bytes: [0..31] header hash, [32..35] fixed extranonce, [36..43] miner nonce
//   48 bytes: [0..31] header hash, [32..39] fixed extranonce, [40..47] miner nonce
// All fields are in display/big-endian order.
// `target` and the returned hash are also display/big-endian. `pnonce` carries the 64-bit miner
// nonce; the fixed Echelon prefix, when present, remains separate in the work blob.
//
// Performance/portability notes:
// - The capacity-selected 7-GiB table raised B580 from 19.65 to 21.27 MH/s on Linux and from 18.60
//   to 20.23 MH/s on Windows. Fixed-base secp256k1 scalar multiplication and its carry-heavy 256-bit
//   integer arithmetic still consume most of the pipeline and dominate the remaining gap.
// - The staged fixed-base table selects its small or large layout by device capacity; this changes the
//   fixed-base window count while field representation remains backend-selected. The portable 8x32
//   field representation remains the production baseline for CPU, iGPU, OpenCL, and unsupported
//   devices; profiles that select the 10x26 field representation retain it. Table construction and
//   each search complete through the shared low-CPU GPU event wait before host-copy submission. The
//   tested ESIMD SIMD8/SIMD16 gather variants were exact but slower, so they are not built.
// - On Windows RX 9060 XT with AdaptiveCpp HIP, the small table measured 26.79 MH/s versus 25.60
//   MH/s for the large table; Windows HIP therefore prefers small, while Linux and other capable
//   devices retain capacity-selected layout.
// - This is a compiler/code-generation and algorithm-structure gap, not proof that Intel/NVIDIA parity
//   is impossible. Native paths must remain capability-gated with the portable fallback intact.

#ifndef MOM_NEXAPOW_HOST_TEST
#include <sycl/sycl.hpp>

#include <cstdint>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>

#include "../lib-internal.h"
#else
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#endif

#include "../../native/job-boundary.h"

namespace mom_nexapow {

#if !defined(MOM_NEXAPOW_HOST_TEST) && (defined(__clang__) || defined(__GNUC__))
#define MOM_NEXAPOW_SHA_ATTR inline __attribute__((always_inline))
#define MOM_NEXAPOW_RFC6979_ATTR inline __attribute__((always_inline))
#endif
#include "device.inc"
#undef MOM_NEXAPOW_RFC6979_ATTR
#undef MOM_NEXAPOW_SHA_ATTR

struct Result {
  uint32_t count;
  uint64_t miner_nonce;
  uint8_t hash[32];
};

static constexpr uint8_t kRecordedHeader[32] = {
    0x0a, 0x4a, 0xc4, 0x9b, 0x2d, 0x02, 0xe3, 0xc8, 0xd1, 0x2c, 0x70, 0x93, 0x25, 0x5b, 0xa7, 0xc4,
    0x96, 0x24, 0xf9, 0xc3, 0x74, 0xd9, 0xf1, 0xc2, 0xf8, 0xe3, 0x7c, 0x58, 0x70, 0x5e, 0x74, 0xb0,
};
static constexpr uint8_t kRecordedExtranonce[8] = {0x10, 0, 0, 0, 0, 0, 0, 0};
static constexpr uint8_t kRecordedMinerNonceBytes[8] = {0x11, 0x82, 0xdc, 0x58, 0, 0, 0, 0};
static constexpr uint64_t kRecordedMinerNonce = 0x1182dc5800000000ULL;
static constexpr uint8_t kRecordedHash[32] = {
    0x00, 0x00, 0x00, 0x42, 0xcb, 0xc2, 0x40, 0x37, 0x52, 0x42, 0xe1, 0x46, 0x41, 0x48, 0x8a, 0x0e,
    0x2d, 0xca, 0x7b, 0x54, 0x45, 0x8a, 0x2c, 0xea, 0x23, 0xdc, 0x1d, 0x2c, 0x17, 0x8b, 0xb1, 0x88,
};
static constexpr uint8_t kRecordedMiningHash[32] = {
    0xef, 0xdf, 0xd3, 0x3a, 0x56, 0x0f, 0x2e, 0x9f, 0xca, 0x9b, 0xc7, 0x5b, 0x1b, 0x68, 0xea, 0xf6,
    0x71, 0xc1, 0xbb, 0x91, 0x22, 0x1c, 0x68, 0xff, 0x01, 0x80, 0x08, 0x9e, 0x54, 0xc3, 0x77, 0x70,
};

inline bool np_equal(const uint8_t* a, const uint8_t* b, const unsigned count) {
  uint8_t difference = 0;
  for (unsigned i = 0; i < count; ++i)
    difference |= static_cast<uint8_t>(a[i] ^ b[i]);
  return difference == 0;
}

inline bool np_meets_target(const uint8_t hash[32], const uint8_t target[32]) {
  for (unsigned i = 0; i < 32; ++i) {
    if (hash[i] != target[i])
      return hash[i] < target[i];
  }
  return true;
}

static int scan_host(const uint8_t* input, unsigned extra_bytes, uint64_t first, unsigned count,
                     const uint8_t* target, bool is_test, uint8_t* output, uint64_t* pnonce) {
  Result found{};
  for (unsigned i = 0; i < count; ++i) {
    uint8_t hash[32];
    const uint64_t nonce = first + i;
    if (np_hash_one(input, input + 32, nonce, hash, extra_bytes) &&
        (is_test || np_meets_target(hash, target)) && !found.count) {
      found.count = 1;
      found.miner_nonce = nonce;
      std::memcpy(found.hash, hash, 32);
    }
  }
  // The caller accounts for the entire batch even when its first nonce wins.
  if (!found.count)
    return 0;
  *pnonce = found.miner_nonce;
  std::memcpy(output, found.hash, 32);
  return 1;
}

#ifndef MOM_NEXAPOW_HOST_TEST
#include "sycl_pipeline.inc"

class SearchKernel;
void nexapow_test_sha256d_49(sycl::queue&, const uint8_t[49], uint8_t[32]);
class State {
public:
  sycl::device device;
  sycl::queue queue;
  uint8_t *header = nullptr, *extranonce = nullptr, *target = nullptr;
  Result* result = nullptr;
  NexaPowSyclSearch portable;
  bool staged_logged = false;
  bool staged_failure_logged = false;
  std::mutex mutex;

  explicit State(const std::string& dev)
      : device(get_dev(dev)), queue(device, sycl::property::queue::in_order{}) {
    if (!mom_has_usm_device(device))
      throw std::string("nexapow requires SYCL device USM");
#ifdef MOM_NEXAPOW_SYCL_NATIVE_FIELD
    if (mom_is_cuda(device)) {
      const char* options = std::getenv("MOM_NEXAPOW_COMPILE_OPTIONS");
      if (!options)
        options = std::getenv("SYCL_PROGRAM_COMPILE_OPTIONS");
      set_sycl_env("SYCL_PROGRAM_COMPILE_OPTIONS", options ? options : "--maxrregcount=128");
    }
#endif
  }

  void init_portable() {
    if (header)
      return;
    try {
      header = sycl::malloc_device<uint8_t>(32, queue);
      extranonce = sycl::malloc_device<uint8_t>(8, queue);
      target = sycl::malloc_device<uint8_t>(32, queue);
      result = sycl::malloc_device<Result>(1, queue);
      if (!header || !extranonce || !target || !result)
        throw std::string("Can't allocate nexapow buffers");
    } catch (...) {
      free_all();
      throw;
    }
  }

  ~State() {
    sycl_cleanup_noexcept("nexapow wait", [&] { queue.wait_and_throw(); });
    portable.release(queue);
    free_all();
  }

  void free_all() noexcept {
    auto release = [&](auto*& pointer) {
      if (!pointer)
        return;
      try {
        sycl::free(pointer, queue);
      } catch (...) {
      }
      pointer = nullptr;
    };
    release(result);
    release(target);
    release(extranonce);
    release(header);
  }
};

static DeviceStateRegistry<State>& registry() {
  static auto* const value = new DeviceStateRegistry<State>;
  return *value;
}
static State& state_for(const std::string& dev) {
  return registry().get(dev, [&] { return std::make_unique<State>(dev); });
}

void nexapow_cleanup_states() noexcept {
  try {
    registry().clear();
  } catch (...) {
    std::fprintf(stderr, "nexapow: ordered SYCL cleanup failed\n");
  }
}

static void test_recorded_vector(State& state, const uint8_t* input, uint8_t* output,
                                 uint64_t* nonce) {
  uint8_t serialized[49], mining_hash[32], final_hash[32];
  np_nonce_bytes(input, input + 32, kRecordedMinerNonce, serialized);
  nexapow_test_sha256d_49(state.queue, serialized, mining_hash);
  if (!np_equal(mining_hash, kRecordedMiningHash, sizeof(mining_hash)) ||
      !np_hash_one(input, input + 32, kRecordedMinerNonce, final_hash) ||
      !np_equal(final_hash, kRecordedHash, sizeof(final_hash)))
    throw std::string("NexaPoW recorded Echelon vector mismatch");
  *nonce = kRecordedMinerNonce;
  std::memcpy(output, final_hash, sizeof(final_hash));
}

static bool env_enabled(const char* name) {
  const char* value = std::getenv(name);
  return value && *value && std::strcmp(value, "0");
}

static const char* staged_field_name(const sycl::device& device) {
#if defined(MOM_NEXAPOW_SYCL_NATIVE_FIELD)
  if (mom_is_cuda(device))
    return "nvptx-native";
  return "portable-10x26";
#elif defined(MOM_NEXAPOW_PORTABLE_FIELD32)
  (void)device;
  return "portable-8x32";
#else
  (void)device;
  return "portable-10x26";
#endif
}

int nexapow(unsigned, uint32_t, const uint8_t* input, unsigned input_size, uint8_t* output,
            uint8_t*, uint64_t* pnonce, const uint8_t* target, const uint8_t*, unsigned intensity,
            bool is_test, bool, const std::string& dev) {
  if ((input_size != 40 && input_size != 44 && input_size != 48) || !intensity)
    throw std::string("nexapow requires a 40-, 44-, or 48-byte work blob and nonzero intensity");
  if (!input || !output || !pnonce || (!target && !is_test))
    throw std::string("nexapow received null work pointers");
  if (!mom::job_boundary::padded_u32_range_fits(intensity, sycl_pipeline::kWorkGroup))
    throw std::string("nexapow intensity is too large");
  const unsigned extra_bytes = input_size == 40u ? 0u : input_size == 44u ? 4u : 8u;
  const bool recorded_test = is_test && input_size == 48u;
  if (recorded_test &&
      (!np_equal(input, kRecordedHeader, 32) || !np_equal(input + 32, kRecordedExtranonce, 8) ||
       !np_equal(input + 40, kRecordedMinerNonceBytes, 8)))
    throw std::string("nexapow test requires the recorded Echelon vector");

  State& state = state_for(dev);
  std::lock_guard<std::mutex> lock(state.mutex);
  const uint64_t first = recorded_test ? kRecordedMinerNonce : *pnonce;
  const unsigned count = is_test ? 1u : intensity;
  const bool staged_test = is_test && env_enabled("MOM_NEXAPOW_STAGED_TEST");
  const bool staged_required = staged_test || env_enabled("MOM_NEXAPOW_STAGED_REQUIRE");
  if (recorded_test && !staged_test) {
    test_recorded_vector(state, input, output, pnonce);
    return 1;
  }
  // Explicit CPU work reuses host consensus math instead of building GPU tables/search kernels.
  if (state.device.is_cpu() && !staged_required)
    return scan_host(input, extra_bytes, first, count, target, is_test, output, pnonce);

  std::string reason;
  if (state.portable.ensure(state.queue, state.device, &reason)) {
    if (env_enabled("MOM_NEXAPOW_STAGED_LOG") && !state.staged_logged) {
      std::fprintf(
          stderr,
          "NexaPoW staged SYCL active (field=%s, table=%s, points=%u, bytes=%llu, count=%u)\n",
          staged_field_name(state.device), state.portable.table_name(),
          state.portable.table_points(),
          static_cast<unsigned long long>(state.portable.table_bytes()),
          count);
      state.staged_logged = true;
    }
    uint64_t found = 0;
    uint8_t test_target[32];
    if (is_test)
      std::memset(test_target, 0xff, sizeof(test_target));
    const uint8_t* portable_target = is_test ? test_target : target;
    // Once search can submit work, a failure must abort rather than reuse its queue for fallback.
    const bool found_candidate = state.portable.search(
        state.queue, input, input + 32, first, portable_target, count, extra_bytes, found);
    if (!found_candidate) {
      if (recorded_test)
        throw std::string("NexaPoW staged SYCL recorded vector found no candidate");
      if (is_test)
        throw std::string("NexaPoW staged SYCL test found no candidate");
      return 0;
    } else {
      uint8_t hash[32];
      if (found - first >= count || !np_hash_one(input, input + 32, found, hash, extra_bytes) ||
          !np_meets_target(hash, portable_target))
        throw std::string("NexaPoW staged SYCL candidate verification failed");
      if (recorded_test &&
          (found != kRecordedMinerNonce || !np_equal(hash, kRecordedHash, sizeof(hash))))
        throw std::string("NexaPoW staged SYCL recorded vector mismatch");
      *pnonce = found;
      std::memcpy(output, hash, sizeof(hash));
      return 1;
    }
  }

  if (!state.staged_failure_logged) {
    std::fprintf(stderr, "NexaPoW staged SYCL unavailable: %s; using monolithic fallback\n",
                 reason.empty() ? "unknown initialization failure" : reason.c_str());
    state.staged_failure_logged = true;
  }
  if (staged_required)
    throw std::string("NexaPoW staged SYCL required but unavailable: ") +
        (reason.empty() ? "unknown initialization failure" : reason);

  state.init_portable();
  Result found{};
  MomSyclHostTransferGuard host_transfers(state.queue, "nexapow monolithic host transfers");
  state.queue.memcpy(state.header, input, 32);
  state.queue.memcpy(state.extranonce, input + 32, 8);
  if (is_test)
    state.queue.memset(state.target, 0xff, 32);
  else
    state.queue.memcpy(state.target, target, 32);
  sycl_wait_and_throw(state.queue.memset(state.result, 0, sizeof(Result)), state.device);
  const uint8_t* header = state.header;
  const uint8_t* extranonce = state.extranonce;
  const uint8_t* device_target = state.target;
  Result* result = state.result;
  const sycl::event search_event = state.queue.submit([&](sycl::handler& handler) {
    handler.parallel_for<SearchKernel>(
        sycl::range<1>(count), [=](sycl::id<1> id) MOM_SYCL_KERNEL_ARGS_RESTRICT {
      const uint64_t miner_nonce = first + id[0];
      uint8_t hash[32];
      if (!np_hash_one(header, extranonce, miner_nonce, hash, extra_bytes))
        return;
      if (np_meets_target(hash, device_target)) {
        using Atomic =
            sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                             sycl::access::address_space::global_space>;
        if (Atomic(result->count).fetch_add(1) == 0) {
          result->miner_nonce = miner_nonce;
          for (unsigned i = 0; i < 32; ++i)
            result->hash[i] = hash[i];
        }
      }
    });
  });
  sycl_wait_and_throw(search_event, state.device);
  sycl_wait_and_throw(state.queue.memcpy(&found, state.result, sizeof(found)), state.device);
  if (!found.count)
    return 0;
  *pnonce = found.miner_nonce;
  std::memcpy(output, found.hash, 32);
  return 1;
}
#else
int nexapow(unsigned, uint32_t, const uint8_t* input, unsigned input_size, uint8_t* output,
            uint8_t*, uint64_t* pnonce, const uint8_t* target, const uint8_t*, unsigned intensity,
            bool is_test, bool, const std::string&) {
  if ((input_size != 40 && input_size != 44 && input_size != 48) ||
      !input || !output || !pnonce || !intensity)
    throw std::runtime_error("invalid nexapow host-test arguments");
  const unsigned extra_bytes = input_size == 40u ? 0u : input_size == 44u ? 4u : 8u;
  const bool recorded_test = is_test && input_size == 48u;
  if (recorded_test &&
      (!np_equal(input, kRecordedHeader, 32) || !np_equal(input + 32, kRecordedExtranonce, 8) ||
       !np_equal(input + 40, kRecordedMinerNonceBytes, 8)))
    throw std::runtime_error("nexapow test requires the recorded Echelon vector");
  const uint64_t first = recorded_test ? kRecordedMinerNonce : *pnonce;
  Result result{};
  result.miner_nonce = first;
  const int found = scan_host(input, extra_bytes, first, is_test ? 1u : intensity,
                              target, is_test, result.hash, &result.miner_nonce);
  if (found && recorded_test && !np_equal(result.hash, kRecordedHash, 32))
    throw std::runtime_error("nexapow host vector mismatch");
  if (found) {
    *pnonce = result.miner_nonce;
    std::memcpy(output, result.hash, 32);
  }
  return found;
}
#endif

} // namespace mom_nexapow

#ifndef MOM_NEXAPOW_HOST_TEST
int nexapow(unsigned job_id, uint32_t height, const uint8_t* input, unsigned input_size,
            uint8_t* output, uint8_t* mix_hash, uint64_t* pnonce, const uint8_t* target,
            const uint8_t* seed_hash, unsigned intensity, bool is_test, bool is_benchmark,
            const std::string& dev) {
  return mom_nexapow::nexapow(job_id, height, input, input_size, output, mix_hash, pnonce, target,
                              seed_hash, intensity, is_test, is_benchmark, dev);
}
#endif

#ifdef MOM_NEXAPOW_HOST_TEST
int main() {
  uint8_t input[48]{};
  const uint8_t header[32] = {
      0x0a, 0x4a, 0xc4, 0x9b, 0x2d, 0x02, 0xe3, 0xc8, 0xd1, 0x2c, 0x70,
      0x93, 0x25, 0x5b, 0xa7, 0xc4, 0x96, 0x24, 0xf9, 0xc3, 0x74, 0xd9,
      0xf1, 0xc2, 0xf8, 0xe3, 0x7c, 0x58, 0x70, 0x5e, 0x74, 0xb0,
  };
  for (unsigned i = 0; i < 32; ++i)
    input[i] = header[i];
  input[32] = 0x10;
  const uint64_t nonce = 0x1182dc5800000000ULL;
  for (unsigned i = 0; i < 8; ++i)
    input[40 + i] = static_cast<uint8_t>(nonce >> (8u * (7u - i)));
  const uint8_t header2[32] = {
      0x0a, 0x4a, 0xc4, 0x9b, 0x2d, 0x02, 0xe3, 0xc8, 0xd1, 0x2c, 0x70,
      0x93, 0x25, 0x5b, 0xa7, 0xc4, 0x96, 0x24, 0xf9, 0xc3, 0x74, 0xd9,
      0xf1, 0xc2, 0xf8, 0xe3, 0x7c, 0x58, 0x70, 0x5e, 0x74, 0xb0,
  };
  const uint8_t extranonce2[8] = {0x10, 0, 0, 0, 0, 0, 0, 0};
  const uint64_t nonce2 = 0xb787915d00000000ULL;
  const uint8_t expected2[32] = {
      0x00, 0x00, 0x00, 0x5f, 0x0b, 0x59, 0xe1, 0x10, 0x86, 0x35, 0x66,
      0xe7, 0x7d, 0x85, 0xe1, 0xb4, 0xfc, 0x71, 0x37, 0x54, 0xe5, 0xc7,
      0x5f, 0x8f, 0xc3, 0xdf, 0x44, 0x13, 0x38, 0x66, 0xa6, 0x69,
  };
  uint8_t hash2[32];
  if (!mom_nexapow::np_hash_one(header2, extranonce2, nonce2, hash2) ||
      !mom_nexapow::np_equal(hash2, expected2, sizeof(hash2)))
    return 1;
  uint8_t target[32];
  for (uint8_t& byte : target)
    byte = 0xff;
  uint8_t output[32]{};
  uint64_t found = 0;
  if (mom_nexapow::nexapow(0, 0, input, sizeof(input), output, nullptr, &found, target, nullptr, 1,
                           true, false, "") != 1 ||
      found != nonce)
    return 1;

  const uint8_t short_expected[32] = {
      0xb0, 0x34, 0xdd, 0x02, 0xa3, 0xba, 0x15, 0xd1, 0x6b, 0x1f, 0x03,
      0x25, 0xe4, 0x4d, 0xe3, 0xe3, 0x74, 0x48, 0x91, 0x82, 0xb2, 0x30,
      0x5c, 0xa1, 0x37, 0x40, 0x77, 0x21, 0x84, 0xc3, 0xbc, 0xd9,
  };
  uint8_t short_input[40]{};
  uint8_t short_output[32]{};
  uint64_t short_nonce = 0;
  if (mom_nexapow::nexapow(0, 0, short_input, sizeof(short_input), short_output, nullptr,
                           &short_nonce, target, nullptr, 1, true, false, "") != 1 ||
      short_nonce != 0 ||
      !mom_nexapow::np_equal(short_output, short_expected, sizeof(short_output)))
    return 1;

  uint8_t prefixed_input[44]{};
  const uint8_t prefix[4] = {0x10, 0x20, 0x30, 0x40};
  std::memcpy(prefixed_input, mom_nexapow::kRecordedHeader,
              sizeof(mom_nexapow::kRecordedHeader));
  std::memcpy(prefixed_input + 32, prefix, sizeof(prefix));
  uint64_t prefixed_nonce = 0x0102030405060708ULL;
  uint8_t prefixed_output[32]{};
  const uint8_t prefixed_expected_hash[32] = {
      0xa9, 0xbd, 0x7f, 0xfd, 0xfc, 0xc1, 0x81, 0x9e, 0xa1, 0xe4, 0x06, 0x87,
      0x45, 0xb4, 0x3e, 0xe2, 0x33, 0xa2, 0xcf, 0xe0, 0x4f, 0x42, 0xe3, 0x36,
      0xe3, 0xd9, 0x3c, 0x24, 0xf3, 0xaf, 0x9d, 0xee,
  };
  uint8_t prefixed_serialized[49]{};
  mom_nexapow::np_nonce_bytes(mom_nexapow::kRecordedHeader, prefix, prefixed_nonce,
                              prefixed_serialized, sizeof(prefix));
  bool serialized_ok = prefixed_serialized[32] == 0x0c;
  for (unsigned i = 0; i < 32; ++i)
    serialized_ok = serialized_ok &&
        prefixed_serialized[i] == mom_nexapow::kRecordedHeader[31u - i];
  for (unsigned i = 0; i < sizeof(prefix); ++i)
    serialized_ok = serialized_ok && prefixed_serialized[33u + i] == prefix[i];
  for (unsigned i = 0; i < 8; ++i)
    serialized_ok = serialized_ok && prefixed_serialized[37u + i] ==
        static_cast<uint8_t>(prefixed_nonce >> (8u * (7u - i)));
  if (mom_nexapow::nexapow(0, 0, prefixed_input, sizeof(prefixed_input), prefixed_output,
                           nullptr, &prefixed_nonce, target, nullptr, 1, true, false, "") != 1 ||
      prefixed_nonce != 0x0102030405060708ULL || !serialized_ok ||
      !mom_nexapow::np_equal(prefixed_output, prefixed_expected_hash, sizeof(prefixed_output)))
    return 1;
  return 0;
}
#endif
