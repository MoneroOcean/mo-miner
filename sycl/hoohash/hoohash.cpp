// Copyright GNU GPLv3 (c) 2026 MoneroOcean <support@moneroocean.stream>
// HooHash v1.1.0 GPU search, ported from HoosatNetwork/HTND.
//
// Consensus uses FP64 transcendental functions whose last bits are not portable across GPU stacks.
// The periodic GPU path is therefore only a search filter: every candidate is recomputed on the host
// in the reference operation order before its target test and submission. A rejected filter candidate
// resumes the remaining batch instead of discarding later nonces.
// Portable OpenCL and AdaptiveCpp avoid 64-bit sycl::mul_hi because affected compiler stacks do
// not lower it reliably. Vendor-native DPC++ keeps the intrinsic.

#include <sycl/sycl.hpp>

#include <cfenv>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>

#include "../lib-internal.h"
#include "../../native/consts.h"
#include "host_math.h"

namespace mom_hoohash {

#include "../blake3-device.inc"

#if defined(__clang__)
#pragma clang fp reassociate(off) contract(off)
#endif

struct Result {
  uint32_t offset;
  uint8_t hash[32];
};
struct FloatPair {
  float hi, lo;
};
class SearchKernel;
class TestKernel;

inline uint64_t load64le(const uint8_t* p) {
  uint64_t value = 0;
  for (unsigned i = 0; i < 8; ++i)
    value |= static_cast<uint64_t>(p[i]) << (8 * i);
  return value;
}
inline uint64_t rotl64(uint64_t x, unsigned n) {
  return (x << n) | (x >> (64 - n));
}

static void make_matrix(const uint8_t hash[32], double matrix[4096]) {
  uint64_t s0 = load64le(hash), s1 = load64le(hash + 8), s2 = load64le(hash + 16),
           s3 = load64le(hash + 24);
  for (unsigned i = 0; i < 4096; ++i) {
    const uint64_t value = rotl64(s0 + s3, 23) + s0, t = s1 << 17;
    s2 ^= s0;
    s3 ^= s1;
    s1 ^= s2;
    s0 ^= s3;
    s2 ^= t;
    s3 = rotl64(s3, 45);
    matrix[i] = static_cast<double>(static_cast<uint32_t>(value)) / 4294967295.0 * 1000000.0;
  }
}

class RoundingToNearest {
  int previous_;

public:
  RoundingToNearest() : previous_(std::fegetround()) {
    if (previous_ < 0 || (previous_ != FE_TONEAREST && std::fesetround(FE_TONEAREST)))
      throw std::string("hoohash cannot select round-to-nearest arithmetic");
  }
  ~RoundingToNearest() {
    if (previous_ != FE_TONEAREST)
      std::fesetround(previous_);
  }
};

static constexpr uint64_t FOUR_OVER_PI[] = {
    0x0000000000000001ULL, 0x45f306dc9c882a53ULL, 0xf84eafa3ea69bb81ULL, 0xb6c52b3278872083ULL,
    0xfca2c757bd778ac3ULL, 0x6e48dc74849ba5c0ULL, 0x0c925dd413a32439ULL, 0xfc3bd63962534e7dULL,
    0xd1046bea5d768909ULL, 0xd338e04d68befc82ULL, 0x7323ac7306a673e9ULL, 0x3908bf177bf25076ULL,
    0x3ff12fffbc0b301fULL, 0xde5e2316b414da3eULL, 0xda6cfd9e4f96136eULL, 0x9e8c7ecd3cbfd45aULL,
    0xea4f758fd7cbe2f6ULL, 0x7a0e73ef14a525d4ULL, 0xd7f6bf623f1aba10ULL, 0xac06608df8f6d757ULL};

inline uint64_t shl_join(uint64_t a, uint64_t b, unsigned shift) {
  return shift ? (a << shift) | (b >> (64 - shift)) : a;
}
inline uint64_t mul_hi64(uint64_t a, uint64_t b) {
  return mo_mul_hi_u64(a, b);
}
inline void trig_reduce(double x, uint64_t& octant, double& reduced) {
  uint64_t bits = sycl::bit_cast<uint64_t>(x);
  const int exponent = static_cast<int>((bits >> 52) & 0x7ff) - 1023 - 52;
  bits = (bits & ((1ULL << 52) - 1)) | (1ULL << 52);
  const unsigned digit = static_cast<unsigned>(exponent + 61) / 64;
  const unsigned shift = static_cast<unsigned>(exponent + 61) % 64;
  const uint64_t z0 = shl_join(FOUR_OVER_PI[digit], FOUR_OVER_PI[digit + 1], shift);
  const uint64_t z1 = shl_join(FOUR_OVER_PI[digit + 1], FOUR_OVER_PI[digit + 2], shift);
  const uint64_t z2 = shl_join(FOUR_OVER_PI[digit + 2], FOUR_OVER_PI[digit + 3], shift);
  const uint64_t z2hi = mul_hi64(z2, bits), z1hi = mul_hi64(z1, bits), z1lo = z1 * bits;
  const uint64_t lo = z1lo + z2hi;
  uint64_t hi = z0 * bits + z1hi + (lo < z1lo);
  octant = hi >> 61;
  hi = hi << 3 | lo >> 61;
  const unsigned leading = static_cast<unsigned>(sycl::clz(hi));
  const uint64_t e = 1023 - (leading + 1);
  hi = (hi << (leading + 1)) | (lo >> (64 - (leading + 1)));
  reduced = sycl::bit_cast<double>((hi >> 12) | (e << 52));
  if (octant & 1) {
    octant = (octant + 1) & 7;
    reduced -= 1;
  }
  reduced *= 0.785398163397448309615660845819875721;
}

static constexpr unsigned PERIODIC_LUT_SIZE = 98304;
static constexpr unsigned MAX_FILTER_RETRIES = 8;
inline bool periodic_position(double x, double& position) {
  double reduced;
  if (sycl::fabs(x) >= 14100000000000000.0) {
    const bool negative = x < 0;
    uint64_t octant;
    trig_reduce(sycl::fabs(x), octant, reduced);
    reduced += octant * 0.785398163397448309616;
    if (negative)
      reduced = -reduced;
  } else {
    constexpr double TWO_OVER_PI = 0.636619772367581343076;
    const double turns = sycl::rint(x * TWO_OVER_PI);
    reduced = sycl::fma(-turns, 1.57079632679489661923, x);
    reduced = sycl::fma(-turns, sycl::bit_cast<double>(0x3c91a62633145c07ULL), reduced);
    reduced = sycl::fma(-turns, sycl::bit_cast<double>(0x3ae8a2e03707344aULL), reduced);
    reduced += static_cast<double>(static_cast<int64_t>(turns) & 3) * 1.57079632679489661923;
  }
  if (reduced < 0)
    reduced += 6.28318530717958647693;
  else if (reduced >= 6.28318530717958647693)
    reduced -= 6.28318530717958647693;
  position = reduced * 15645.56752570568075;
  return true;
}
inline double periodic_lookup(const double* table, double x) {
  double position;
  periodic_position(x, position);
  unsigned index = static_cast<unsigned>(position);
  if (index >= PERIODIC_LUT_SIZE)
    index = PERIODIC_LUT_SIZE - 1;
  return sycl::fma(position - index, table[index + 1] - table[index], table[index]);
}
inline double complex_value(double x, const double* exp_lut, const double* sin2_lut) {
  const double a = x * .000001 / 8 - sycl::floor(x * .000001 / 8),
               b = x * .000001 / 4 - sycl::floor(x * .000001 / 4);
  const double y = b < .25   ? x + (1 + b)
                   : b < .5  ? x - (1 + b)
                   : b < .75 ? x * (1 + b)
                             : x / (1 + b);
  if (a < .33)
    return periodic_lookup(exp_lut, y);
  if (a < .66)
    return periodic_lookup(sin2_lut, y);
  return static_cast<double>(sycl::rsqrt(static_cast<float>(sycl::fabs(y) + 1)));
}
inline double finite_complex(double x, const double* exp_lut, const double* sin2_lut) {
  double result = complex_value(x, exp_lut, sin2_lut);
  unsigned rounds = 1;
  while (sycl::isnan(result) || sycl::isinf(result)) {
    x *= .1;
    if (x <= 1e-13)
      return 0;
    ++rounds;
    result = complex_value(x, exp_lut, sin2_lut);
  }
  return result * rounds;
}
inline void pair_add(float& hi, float& lo, float add_hi, float add_lo) {
  const float sum = hi + add_hi, bv = sum - hi;
  float error = (hi - (sum - bv)) + (add_hi - bv) + lo + add_lo;
  hi = sum + error;
  lo = error - (hi - sum);
}
inline bool pair_complex_branch(float hi, float lo) {
  int quotient = static_cast<int>(hi * (1.0f / 1024));
  pair_add(hi, lo, -static_cast<float>(quotient) * 1024, 0);
  if (hi < 0 || (hi == 0 && lo < 0)) {
    --quotient;
    pair_add(hi, lo, 1024, 0);
  } else if (hi > 1024 || (hi == 1024 && lo >= 0)) {
    ++quotient;
    pair_add(hi, lo, -1024, 0);
  }
  constexpr float threshold_hi = 20.48f;
  constexpr float threshold_lo = static_cast<float>(20.48 - static_cast<double>(threshold_hi));
  return hi < threshold_hi || (hi == threshold_hi && lo <= threshold_lo);
}
class State {
public:
  sycl::device device;
  sycl::queue queue;
  unsigned workgroup;
  double *matrix = nullptr, *exp_lut = nullptr, *sin2_lut = nullptr;
  double canonical_matrix[4096]{};
  FloatPair* normal = nullptr;
  uint8_t *input = nullptr, *target = nullptr;
  Result* result = nullptr;
  std::mutex mutex;
  uint8_t matrix_seed[32]{};
  bool matrix_ready = false;
  explicit State(const std::string& dev_str)
      : device(get_dev(dev_str)),
        queue(device, sycl::property_list{sycl::property::queue::in_order{}}),
        workgroup(sycl_default_workgroup(device, {1, 2, 4, 8, 16, 32, 64, 128},
                                         mom_is_cuda(device) ? 128 : 64)) {
    if (!device.has(sycl::aspect::fp64) || !mom_has_usm_device(device))
      throw std::string("hoohash requires FP64 and SYCL device USM");
    unsigned long override;
    if (mom_parse_env_ulong("MOM_HOOHASH_WORKGROUP", override)) {
      const size_t maximum = device.get_info<sycl::info::device::max_work_group_size>();
      if (!override || override > maximum)
        throw std::string("MOM_HOOHASH_WORKGROUP exceeds the device limit");
      workgroup = static_cast<unsigned>(override);
    }
    try {
      matrix = sycl::malloc_device<double>(4096, queue);
      normal = sycl::malloc_device<FloatPair>(4096 * 16, queue);
      exp_lut = sycl::malloc_device<double>(PERIODIC_LUT_SIZE + 1, queue);
      sin2_lut = sycl::malloc_device<double>(PERIODIC_LUT_SIZE + 1, queue);
      input = sycl::malloc_device<uint8_t>(72, queue);
      target = sycl::malloc_device<uint8_t>(32, queue);
      result = sycl::malloc_device<Result>(1, queue);
      if (!matrix || !normal || !exp_lut || !sin2_lut || !input || !target || !result)
        throw std::string("Can't allocate hoohash buffers");
      auto exp_host = std::make_unique<double[]>(PERIODIC_LUT_SIZE + 1);
      auto sin2_host = std::make_unique<double[]>(PERIODIC_LUT_SIZE + 1);
      for (unsigned i = 0; i <= PERIODIC_LUT_SIZE; ++i) {
        const double angle = 6.28318530717958647693 * i / PERIODIC_LUT_SIZE;
        const double sine = std::sin(angle);
        exp_host[i] = std::exp(sine + std::cos(angle));
        sin2_host[i] = sine * sine;
      }
      MomSyclHostTransferGuard host_transfers(queue, "hoohash LUT uploads");
      queue.memcpy(exp_lut, exp_host.get(), sizeof(double) * (PERIODIC_LUT_SIZE + 1));
      sycl_wait_and_throw(
          queue.memcpy(sin2_lut, sin2_host.get(), sizeof(double) * (PERIODIC_LUT_SIZE + 1)),
          device);
    } catch (...) {
      free_all();
      throw;
    }
  }
  ~State() {
    try {
      queue.wait_and_throw();
    } catch (...) {
    }
    free_all();
  }
  template <typename T> void free_ptr(T*& pointer) noexcept {
    if (pointer)
      try {
        sycl::free(pointer, queue);
      } catch (...) {
      }
    pointer = nullptr;
  }
  void free_all() noexcept {
    free_ptr(result);
    free_ptr(target);
    free_ptr(input);
    free_ptr(sin2_lut);
    free_ptr(exp_lut);
    free_ptr(normal);
    free_ptr(matrix);
  }
};
static DeviceStateRegistry<State>& registry() {
  static auto* const value = new DeviceStateRegistry<State>;
  return *value;
}
static State& state_for(const std::string& dev) {
  return registry().get(dev, [&] { return std::make_unique<State>(dev); });
}
void hoohash_cleanup_states() noexcept {
  try {
    registry().clear();
  } catch (...) {
    std::fprintf(stderr, "hoohash: ordered SYCL cleanup failed\n");
  }
}

inline uint32_t prepare_first(const uint8_t* input, uint64_t nonce, uint8_t first[32]) {
  uint8_t pre[80];
  for (unsigned i = 0; i < 72; ++i)
    pre[i] = input[i];
  for (unsigned i = 0; i < 8; ++i)
    pre[72 + i] = static_cast<uint8_t>(nonce >> (8 * i));
  blake3_dev(first, 32, pre, 80);
  uint32_t hash_mod = 0;
  for (unsigned i = 0; i < 8; ++i)
    hash_mod ^= static_cast<uint32_t>(first[4 * i]) << 24 |
                static_cast<uint32_t>(first[4 * i + 1]) << 16 |
                static_cast<uint32_t>(first[4 * i + 2]) << 8 | first[4 * i + 3];
  return hash_mod;
}

inline __attribute__((always_inline)) void
filter_hash(const uint8_t first[32], uint32_t hash_mod, uint64_t nonce, const double* matrix,
            const double* exp_lut, const double* sin2_lut, const FloatPair* normal,
            uint8_t out[32]) {
  uint8_t mixed[32];
  uint64_t even = 0;
  bool complex_branch = true;
  // A zero leading nibble resets the product-derived branch state at each row; later zeros do not.
  const bool zero_first_nibble = !(first[0] >> 4);
  for (unsigned row = 0; row < 64; ++row) {
    float product_hi = 0, product_lo = 0;
    if (zero_first_nibble)
      complex_branch = true;
    for (unsigned j = 0; j < 64; ++j) {
      const uint8_t value = (j & 1) ? first[j / 2] & 15 : first[j / 2] >> 4;
      if (value) {
        const unsigned k = row * 64 + j;
        if (complex_branch) {
          const double term = finite_complex(matrix[k] * hash_mod * value +
                                                 static_cast<double>(nonce & 255),
                                             exp_lut, sin2_lut) *
                              value * 1234;
          const float hi = static_cast<float>(term),
                      lo = static_cast<float>(term - static_cast<double>(hi));
          pair_add(product_hi, product_lo, hi, lo);
        } else {
          const FloatPair term = normal[k * 16 + value];
          pair_add(product_hi, product_lo, term.hi, term.lo);
        }
        complex_branch = pair_complex_branch(product_hi, product_lo);
      }
    }
    const uint64_t value = static_cast<uint64_t>(static_cast<double>(product_hi) + product_lo);
    if (row & 1)
      mixed[row / 2] = first[row / 2] ^ static_cast<uint8_t>(even + value);
    else
      even = value;
  }
  blake3_dev(out, 32, mixed, 32);
}

inline bool meets_target(const uint8_t hash[32], const uint8_t target[32]) {
  for (int i = 31; i >= 0; --i)
    if (hash[i] != target[i])
      return hash[i] < target[i];
  return true;
}

static sycl::event search(State& s, uint64_t first_nonce, unsigned count) {
  const double* matrix = s.matrix;
  const double* exp_lut = s.exp_lut;
  const double* sin2_lut = s.sin2_lut;
  const FloatPair* normal = s.normal;
  const uint8_t* input = s.input;
  const uint8_t* target = s.target;
  Result* result = s.result;
  return s.queue.submit([&](sycl::handler& h) {
    const size_t global =
        (static_cast<size_t>(count) + s.workgroup - 1) / s.workgroup * s.workgroup;
    h.parallel_for<SearchKernel>(
        sycl::nd_range<1>(global, s.workgroup),
        [=](sycl::nd_item<1> item) MOM_SYCL_KERNEL_ARGS_RESTRICT {
          const size_t id = item.get_global_linear_id();
          if (id >= count)
            return;
          const uint64_t nonce = first_nonce + id;
          uint8_t first[32], out[32];
          const uint32_t hash_mod = prepare_first(input, nonce, first);
          filter_hash(first, hash_mod, nonce, matrix, exp_lut, sin2_lut, normal, out);
          if (meets_target(out, target)) {
            using Atomic =
                sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                 sycl::access::address_space::global_space>;
            Atomic(result->offset).fetch_min(static_cast<uint32_t>(id));
          }
        });
  });
}

static sycl::event test_hash(State& s, uint64_t nonce) {
  const double* matrix = s.matrix;
  const double* exp_lut = s.exp_lut;
  const double* sin2_lut = s.sin2_lut;
  const FloatPair* normal = s.normal;
  const uint8_t* input = s.input;
  Result* result = s.result;
  return s.queue.submit([&](sycl::handler& h) {
    h.single_task<TestKernel>([=] {
      uint8_t first[32], out[32];
      const uint32_t hash_mod = prepare_first(input, nonce, first);
      filter_hash(first, hash_mod, nonce, matrix, exp_lut, sin2_lut, normal, out);
      result->offset = 0;
      for (unsigned i = 0; i < 32; ++i)
        result->hash[i] = out[i];
    });
  });
}

} // namespace mom_hoohash

using namespace mom_hoohash;
int hoohash(unsigned, uint32_t, const uint8_t* input, unsigned input_size, uint8_t* output,
            uint8_t*, uint64_t* pnonce, const uint8_t* target, const uint8_t*, unsigned intensity,
            bool is_test, bool, const std::string& dev_str) {
  if (!input || !output || !pnonce || !target || input_size != 80 || !intensity)
    throw std::string(
      "hoohash requires input, output, nonce, target, an 80-byte header, and nonzero intensity");
  if (const char* error = mom_hoohash_math_error())
    throw std::string(error);
  RoundingToNearest rounding;
  State& state = state_for(dev_str);
  std::lock_guard<std::mutex> lock(state.mutex);
  std::unique_ptr<FloatPair[]> normal;
  MomSyclHostTransferGuard host_transfers(state.queue, "hoohash host transfers");
  if (!state.matrix_ready || std::memcmp(state.matrix_seed, input, 32)) {
    state.matrix_ready = false;
    normal = std::make_unique<FloatPair[]>(4096 * 16);
    make_matrix(input, state.canonical_matrix);
    state.queue.memcpy(state.matrix, state.canonical_matrix, sizeof(state.canonical_matrix));
    for (unsigned k = 0; k < 4096; ++k)
      for (unsigned value = 0; value < 16; ++value) {
        const double term = state.canonical_matrix[k] * .0001 * value;
        const float hi = static_cast<float>(term);
        normal[k * 16 + value] = {hi, static_cast<float>(term - static_cast<double>(hi))};
      }
    sycl_wait_and_throw(
        state.queue.memcpy(state.normal, normal.get(), sizeof(FloatPair) * 4096 * 16),
        state.device);
    std::memcpy(state.matrix_seed, input, 32);
    state.matrix_ready = true;
  }
  state.queue.memcpy(state.input, input, 72);
  state.queue.memcpy(state.target, target, 32);
  if (is_test) {
    state.queue.memset(state.result, 0xff, sizeof(Result));
    sycl_wait_and_throw(test_hash(state, *pnonce), state.device);
    Result result;
    sycl_wait_and_throw(state.queue.memcpy(&result, state.result, sizeof(result)), state.device);
    if (result.offset)
      return 0;
    mom_hoohash_canonical_hash(input, *pnonce, output);
    return 1;
  }

  uint64_t first_nonce = *pnonce;
  unsigned remaining = intensity, rejected_candidates = 0;
  while (remaining) {
    state.queue.memset(state.result, 0xff, sizeof(Result));
    const sycl::event work_event = search(state, first_nonce, remaining);
    // Some runtimes busy-spin while submitting a host copy behind live kernels. Complete the GPU
    // work with the shared low-CPU wait before enqueuing the already-ready result copy.
    sycl_wait_and_throw(work_event, state.device);
    Result result;
    sycl_wait_and_throw(state.queue.memcpy(&result, state.result, sizeof(result)), state.device);
    if (result.offset == UINT32_MAX)
      return 0;

    if (result.offset >= remaining)
      throw std::string("hoohash GPU returned a nonce outside its search range");
    const uint64_t candidate_nonce = first_nonce + result.offset;
    mom_hoohash_canonical_hash(input, candidate_nonce, output);
    if (meets_target(output, target)) {
      *pnonce = candidate_nonce;
      return 1;
    }

    // Bound work controlled by an untrusted pool target while retaining later candidates in the
    // ordinary rare-mismatch case.
    if (++rejected_candidates == MAX_FILTER_RETRIES)
      return 0;
    const unsigned consumed = result.offset + 1;
    remaining -= consumed;
    first_nonce = candidate_nonce + 1;
  }
  return 0;
}
