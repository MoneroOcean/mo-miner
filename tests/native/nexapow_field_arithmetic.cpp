// Compile with actual sycl_pipeline.inc arithmetic prefix and feature predicate extracted
// as nexapow_fields.inc and nexapow_field_gate.inc; no SYCL runtime is loaded.
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <type_traits>
#include <vector>

#ifdef TEST_NEXAPOW_ACPP_HOST
#define MOM_SYCL_ADAPTIVECPP_CUDA
#define __acpp_if_target_sscp(...)
#endif
namespace nexa {
#include "../../sycl/nexapow/device.inc"
#include "nexapow_fields.inc"
}
#ifdef TEST_NEXAPOW_ACPP_HOST
#undef __acpp_if_target_sscp
#endif

// Execute the actual source predicate with SDK-shaped reflection stubs. Unknown
// queries must not be reflected; this tests admission, not PTX materialization.
namespace sycl::AdaptiveCpp_jit {
namespace reflection_query {
struct compiler_backend {};
struct target_vendor_id {};
struct target_arch {};
}
namespace compiler_backend { constexpr int ptx = 1; }
namespace vendor_id { constexpr int nvidia = 4318; }
static std::array<bool, 3> known;
static std::array<int, 3> value;
template <typename Query> constexpr unsigned index() {
  return std::is_same_v<Query, reflection_query::compiler_backend> ? 0 :
    std::is_same_v<Query, reflection_query::target_vendor_id> ? 1 : 2;
}
template <typename Query> bool knows() { return known[index<Query>()]; }
template <typename Query> int reflect() {
  assert(known[index<Query>()]);
  return value[index<Query>()];
}
}
#define __acpp_if_target_sscp(...) __VA_ARGS__
namespace gate_fixture {
#include "nexapow_field_gate.inc"
}
#undef __acpp_if_target_sscp

int main() {
  using namespace nexa;
  namespace field = nexa::sycl_pipeline;
  static_assert(sizeof(field::Field) == 40);
  static_assert(std::is_trivially_copyable_v<field::Field>);
#ifdef MOM_NEXAPOW_ACPP_PTX_FIELD
  assert(!field::field_uses_acpp_ptx()); // Host materialization cannot select device PTX.
#endif
  namespace jit = sycl::AdaptiveCpp_jit;
  unsigned gate_cases = 0;
  for (const int backend : {0, 1, 2}) {
    for (const int vendor : {0, 4098, 4318, 32902}) {
      for (const int arch : {-1, 0, 50, 60, 61, 62, 70, 79, 80, 120}) {
        for (unsigned mask = 0; mask < 8; ++mask) {
          jit::known = {(mask & 1u) != 0, (mask & 2u) != 0, (mask & 4u) != 0};
          jit::value = {backend, vendor, arch};
          assert(gate_fixture::field_uses_acpp_ptx() ==
            (mask == 7 && backend == 1 && vendor == 4318 && arch >= 61));
          ++gate_cases;
        }
      }
    }
  }

  const U256 two = np_add_mod(np_one(), np_one(), kFieldPrime);
  std::vector<U256> inputs{np_zero(), np_one(), np_sub_mod(kFieldPrime, np_one(), kFieldPrime),
    np_sub_mod(kFieldPrime, two, kFieldPrime), np_sub_mod(kGroupOrder, np_one(), kGroupOrder),
    np_sub_mod(kGroupOrder, two, kGroupOrder)};
  for (const unsigned bit : {25u, 26u, 31u, 32u, 63u, 64u, 127u, 128u, 254u, 255u}) {
    U256 value{};
    value.w[bit / 32] = uint32_t{1} << (bit % 32);
    inputs.push_back(np_sub_mod(value, np_one(), kFieldPrime));
    inputs.push_back(value);
    inputs.push_back(np_add_mod(value, np_one(), kFieldPrime));
  }
  uint32_t random = 0x87654321;
  for (unsigned i = 0; i < 16; ++i) {
    U256 value{};
    for (auto& word : value.w) {
      random ^= random << 13; random ^= random >> 17; random ^= random << 5;
      word = random;
    }
    inputs.push_back(np_reduce_mod(value, kFieldPrime));
  }
  const auto equal = [](const U256& a, const U256& b) { assert(np_cmp(a, b) == 0); };
  auto prime = field::field_from_u256(kFieldPrime);
  field::field_normalize(prime);
  equal(field::field_to_u256(prime), np_zero());
  U256 prime_plus_one{};
  assert(np_add_raw(kFieldPrime, np_one(), prime_plus_one) == 0);
  auto above_prime = field::field_from_u256(prime_plus_one);
  field::field_normalize_weak(above_prime);
  equal(field::field_to_u256(above_prime), np_one());
  for (unsigned i = 0; i < inputs.size(); ++i) {
    const U256 a = inputs[i], b = inputs[(i + 1) % inputs.size()];
    auto fa = field::field_from_u256(a), fb = field::field_from_u256(b);
    equal(field::field_to_u256(fa), a);
    for (unsigned limb = 0; limb < 10; ++limb)
      assert(fa.n[limb] <= (limb == 9 ? 0x3fffffu : 0x3ffffffu));
    auto normalized = fa;
    field::field_normalize_weak(normalized); field::field_normalize(normalized);
    equal(field::field_to_u256(normalized), a);
    equal(field::field_to_u256(field::field_mul(fa, fb)), np_field_mul(a, b));
    equal(field::field_to_u256(field::field_square(fa)), np_field_square(a));
    auto difference = field::field_sub(fa, fb);
    field::field_normalize_weak(difference);
    equal(field::field_to_u256(difference), np_field_sub(a, b));
    equal(field::field_to_u256(field::field_mul_sub_mul(fa, fb, fb, fb)),
      np_field_sub(np_field_mul(a, b), np_field_square(b)));
    equal(field::field_to_u256(field::field_inverse(fa)), np_field_inverse(a));
    if (!np_zero_p(a)) {
      equal(field::field_to_u256(field::field_mul(fa, field::field_inverse(fa))), np_one());
      assert(field::field_quadratic_residue(fa) == np_quadratic_residue(a));
    }
    const U256 ga = np_reduce_mod(a, kGroupOrder), gb = np_reduce_mod(b, kGroupOrder);
    equal(field::group_op<true>(ga, gb), np_add_mod(ga, gb, kGroupOrder));
    equal(field::group_op<false>(ga, gb), np_sub_mod(ga, gb, kGroupOrder));
    equal(field::group_mul(ga, gb), np_mul_mod(ga, gb, kGroupOrder));
  }
  // Fallback exponentiation defines inverse(0)=0; this does not establish a
  // zero-input contract for the PTX binary-GCD helper.
  equal(field::field_to_u256(field::field_inverse(field::field_zero())), np_zero());
  equal(field::field_to_u256(field::field_one()), np_one());
  std::printf("Nexa field fallback: %zu arithmetic cases; %u reflected admission cases passed\n",
    inputs.size(), gate_cases);
}
