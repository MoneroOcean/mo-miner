// Host-only differential regression for the operation selector in the XelisHashV3 kernel.
#define MOM_XELISHASHV3_HOST_TEST 1
#include "../../sycl/xelishashv3/xelishashv3.cpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>

namespace {

using U128 = unsigned __int128;

struct Input {
  uint64_t a, b, c, result;
  uint32_t r, i, j;
};

U128 combine(uint64_t hi, uint64_t lo) {
  return (static_cast<U128>(hi) << 64) | lo;
}

uint64_t isqrt_reference(uint64_t value) {
  uint64_t root = static_cast<uint64_t>(std::sqrt(static_cast<long double>(value)));
  while (static_cast<U128>(root) * root > value) {
    --root;
  }
  while (static_cast<U128>(root + 1) * (root + 1) <= value) {
    ++root;
  }
  return root;
}

uint64_t rotl_reference(uint64_t value, uint32_t count) {
  const uint32_t shift = count & 63;
  return (value << shift) | (value >> ((64 - shift) & 63));
}

uint64_t murmur_reference(uint64_t value) {
  value ^= value >> 55;
  value *= 0xff51afd7ed558ccdULL;
  value ^= value >> 32;
  value *= 0xc4ceb9fe1a85ec53ULL;
  return value ^ (value >> 15);
}

uint64_t modpow_reference(uint64_t base, uint64_t exp, uint64_t mod) {
  mod += mod == 0;
  if (mod == 1)
    return 0;
  base %= mod;
  if (base < 2)
    return base;
  if (!exp)
    return 1;
  uint64_t out = 1;
  while (exp) {
    if (exp & 1)
      out = static_cast<uint64_t>(static_cast<U128>(out) * base % mod);
    exp >>= 1;
    if (exp)
      base = static_cast<uint64_t>(static_cast<U128>(base) * base % mod);
  }
  return out;
}

uint64_t reference(uint32_t op, const Input& in) {
  const uint64_t t0 = rotl_reference(in.result, in.r);
  const uint64_t i = in.i, j = in.j;
  switch (op) {
    case 0:
      return static_cast<uint64_t>(combine(in.a + i, isqrt_reference(in.b + j)) %
                                   static_cast<U128>(murmur_reference(in.c ^ in.result ^ i ^ j) | 1));
    case 1: {
      const uint64_t divisor = isqrt_reference(in.b | 2);
      return rotl_reference((in.c + i) % divisor, in.i + in.j) * isqrt_reference(in.a + j);
    }
    case 2:
      return isqrt_reference(in.a + i) * isqrt_reference(in.c + j) ^ (in.b + i + j);
    case 3:
      return (in.a + in.b) * in.c;
    case 4:
      return (in.b - in.c) * in.a;
    case 5:
      return in.c - in.a + in.b;
    case 6:
      return in.a - in.b + in.c;
    case 7:
      return in.b * in.c + in.a;
    case 8:
      return in.c * in.a + in.b;
    case 9:
      return in.a * in.b * in.c;
    case 10:
      return static_cast<uint64_t>(combine(in.a, in.b) % static_cast<U128>(in.c | 1));
    case 11: {
      const U128 t1 = combine(in.b, in.c), t2 = combine(t0, in.a | 2);
      return t2 > t1 ? in.c : static_cast<uint64_t>(t1 % t2);
    }
    case 12:
      return static_cast<uint64_t>(combine(in.c, in.a) / static_cast<U128>(in.b | 4));
    case 13: {
      const U128 t1 = combine(t0, in.b), t2 = combine(in.a, in.c | 8);
      return t1 > t2 ? static_cast<uint64_t>(t1 / t2) : in.a ^ in.b;
    }
    case 14:
      return static_cast<uint64_t>((combine(in.b, in.a) * in.c) >> 64);
    case 15:
      return static_cast<uint64_t>((combine(in.a, in.c) * combine(in.result >> (in.r & 63) |
        in.result << ((64 - (in.r & 63)) & 63), in.b)) >> 64);
    default:
      return 0;
  }
}

template <bool Fp64>
bool check(uint32_t op, const Input& in, const double* lut, uint64_t case_number) {
  const uint64_t expected = reference(op, in);
  const uint64_t actual = mom_xelishashv3::operation<false, Fp64>(
      op, in.a, in.b, in.c, in.r, in.result, in.i, in.j, Fp64 ? lut : nullptr);
  if (expected == actual) return true;
  std::cerr << "operation fp64=" << Fp64 << ' ' << op << " mismatch at case " << case_number
            << ": a=0x" << std::hex << in.a << " b=0x" << in.b << " c=0x" << in.c
            << " result=0x" << in.result << " r=" << std::dec << in.r << " i=" << in.i << " j=" << in.j
            << " expected=0x" << std::hex << expected << " actual=0x" << actual << std::dec << '\n';
  return false;
}

bool check_divmod64(uint64_t numerator, uint64_t divisor, uint64_t case_number) {
  if (!divisor) {
    std::cerr << "divmod64 zero divisor at case " << case_number << '\n';
    return false;
  }
  uint64_t actual_q, actual_r;
  mom_xelishashv3::divmod64(numerator, divisor, actual_q, actual_r);
  const uint64_t expected_q = numerator / divisor, expected_r = numerator % divisor;
  if (actual_q == expected_q && actual_r == expected_r)
    return true;
  std::cerr << "divmod64 mismatch at case " << case_number << ": n=0x" << std::hex << numerator
            << " d=0x" << divisor << " expected_q=0x" << expected_q << " actual_q=0x" << actual_q
            << " expected_r=0x" << expected_r << " actual_r=0x" << actual_r << std::dec << '\n';
  return false;
}

template <bool Fast>
bool check_divmod128_fallback(uint64_t hi, uint64_t lo, uint64_t divisor, uint64_t case_number) {
  const uint64_t effective_divisor = divisor ? divisor : 1;
  const U128 numerator = combine(hi, lo);
  const uint64_t expected_q = static_cast<uint64_t>(numerator / effective_divisor);
  const uint64_t expected_r = static_cast<uint64_t>(numerator % effective_divisor);
  uint64_t actual_q, actual_r;
  if constexpr (Fast) {
    mom_xelishashv3::divmod128_fast<false, false>(hi, lo, divisor, actual_q, actual_r);
  } else {
    mom_xelishashv3::divmod128_newton<false, false>(
        hi, lo, divisor, nullptr, actual_q, actual_r);
  }
  if (actual_q == expected_q && actual_r == expected_r)
    return true;
  std::cerr << "divmod128 fp64=0 fast=" << Fast << " mismatch at case " << case_number
            << ": hi=0x" << std::hex << hi << " lo=0x" << lo << " d=0x" << divisor
            << " expected_q=0x" << expected_q << " actual_q=0x" << actual_q
            << " expected_r=0x" << expected_r << " actual_r=0x" << actual_r << std::dec << '\n';
  return false;
}

bool check_div96_fallback(uint64_t r, uint32_t x, uint64_t divisor, uint64_t case_number) {
  const U128 numerator = (static_cast<U128>(r) << 32) | x;
  const U128 expected_q = numerator / divisor;
  const uint64_t expected_r = static_cast<uint64_t>(numerator % divisor);
  uint64_t actual_r;
  const uint32_t actual_q = mom_xelishashv3::div96<false, false>(
      r, x, divisor, 0, actual_r);
  if (expected_q <= UINT32_MAX && actual_q == static_cast<uint32_t>(expected_q) &&
      actual_r == expected_r)
    return true;
  std::cerr << "div96 fp64=0 mismatch at case " << case_number
            << ": r=0x" << std::hex << r << " x=0x" << x << " d=0x" << divisor
            << " expected_q=0x" << static_cast<uint64_t>(expected_q)
            << " actual_q=0x" << actual_q << " expected_r=0x" << expected_r
            << " actual_r=0x" << actual_r << std::dec << '\n';
  return false;
}

bool check_divmod128_by128_fallback(U128 numerator, U128 divisor, uint64_t case_number) {
  const U128 effective_divisor = divisor ? divisor : 1;
  const uint64_t expected_q = static_cast<uint64_t>(numerator / effective_divisor);
  const uint64_t expected_r = static_cast<uint64_t>(numerator % effective_divisor);
  uint64_t actual_q, actual_r;
  mom_xelishashv3::divmod128_by128<false, false>(
      numerator >> 64, numerator, divisor >> 64, divisor, nullptr, actual_q, actual_r);
  if (actual_q == expected_q && actual_r == expected_r)
    return true;
  std::cerr << "divmod128_by128 fp64=0 mismatch at case " << case_number
            << ": hi=0x" << std::hex << static_cast<uint64_t>(numerator >> 64)
            << " lo=0x" << static_cast<uint64_t>(numerator)
            << " divisor_hi=0x" << static_cast<uint64_t>(divisor >> 64)
            << " divisor_lo=0x" << static_cast<uint64_t>(divisor)
            << " expected_q=0x" << expected_q << " actual_q=0x" << actual_q
            << " expected_r=0x" << expected_r << " actual_r=0x" << actual_r << std::dec << '\n';
  return false;
}

bool check_mul_hi64_portable(uint64_t a, uint64_t b, uint64_t case_number) {
  const uint64_t expected = static_cast<uint64_t>((static_cast<U128>(a) * b) >> 64);
  const uint64_t actual = mom_xelishashv3::mul_hi64_portable(a, b);
  if (actual == expected)
    return true;
  std::cerr << "mul_hi64_portable mismatch at case " << case_number
            << ": a=0x" << std::hex << a << " b=0x" << b
            << " expected=0x" << expected << " actual=0x" << actual << std::dec << '\n';
  return false;
}

template <bool Fp64>
bool check_modpow(uint64_t base, uint64_t exp, uint64_t mod, uint64_t case_number) {
  const uint64_t expected = modpow_reference(base, exp, mod);
  const uint64_t actual = mom_xelishashv3::modpow<false, Fp64>(base, exp, mod);
  if (expected == actual)
    return true;
  std::cerr << "modpow fp64=" << Fp64 << " mismatch at case " << case_number
            << ": base=0x" << std::hex << base << " exp=0x" << exp << " mod=0x" << mod
            << " expected=0x" << expected << " actual=0x" << actual << std::dec << '\n';
  return false;
}

struct DirectFacts {
  uint64_t mod64_cases = 0, mod64_mismatches = 0;
  uint64_t divmod128_cases = 0, divmod128_mismatches = 0, domain_failures = 0;
};

void direct_mod64(DirectFacts& f, uint64_t a, uint64_t d, const double* lut) {
  if (d <= UINT32_MAX) {
    ++f.domain_failures;
    return;
  }
  ++f.mod64_cases;
  f.mod64_mismatches += mom_xelishashv3::mod64_newton<>(a, d, lut) != a % d;
}

void direct_divmod128(DirectFacts& f, U128 n, uint64_t d, const double* lut) {
  if (d <= UINT32_MAX) {
    ++f.domain_failures;
    return;
  }
  ++f.divmod128_cases;
  uint64_t actual_q, actual_r;
  mom_xelishashv3::divmod128_newton<>(
      static_cast<uint64_t>(n >> 64), static_cast<uint64_t>(n), d, lut, actual_q, actual_r);
  const U128 expected_q = n / d, expected_r = n % d;
  f.divmod128_mismatches += actual_q != static_cast<uint64_t>(expected_q) ||
                             actual_r != static_cast<uint64_t>(expected_r);
}

void direct_divisor(DirectFacts& f, uint64_t d, const double* lut) {
  constexpr uint64_t max64 = UINT64_MAX;
  const uint64_t largest = static_cast<uint64_t>(static_cast<U128>(max64 / d) * d);
  // Edge values and exact large multiples cover cases random sampling may miss.
  const std::array<U128, 10> values = {{
    0, 1, static_cast<U128>(d - 1), d, static_cast<U128>(d) + 1,
    max64 - 1, max64, static_cast<U128>(largest - 1), largest,
    static_cast<U128>(largest) + 1,
  }};
  const U128 max128 = ~static_cast<U128>(0);
  for (U128 value : values)
    if (value <= max64)
      direct_mod64(f, static_cast<uint64_t>(value), d, lut);

  const std::array<uint64_t, 4> quotients = {{0, 1, max64 - 1, max64}};
  const std::array<uint64_t, 3> remainders = {{0, 1, d - 1}};
  for (uint64_t q : quotients) {
    const U128 product = static_cast<U128>(q) * d;
    for (uint64_t r : remainders)
      if (product <= max128 - r)
        direct_divmod128(f, product + r, d, lut);
  }
  const std::array<uint64_t, 4> high_words = {{0, d - 1, d, max64}};
  const std::array<uint64_t, 3> low_words = {{0, 1, max64}};
  for (uint64_t hi : high_words)
    for (uint64_t lo : low_words)
      direct_divmod128(f, (static_cast<U128>(hi) << 64) | lo, d, lut);
}

DirectFacts run_direct_checks(const double* lut) {
  constexpr uint64_t base = UINT64_C(1) << 32;
  DirectFacts f;
  for (uint64_t d = base; d <= base + 4096; ++d)
    direct_divisor(f, d, lut);
  for (unsigned bit = 33; bit < 64; ++bit) {
    const uint64_t power = UINT64_C(1) << bit;
    direct_divisor(f, power - 1, lut);
    direct_divisor(f, power, lut);
    if (power != UINT64_MAX)
      direct_divisor(f, power + 1, lut);
  }
  direct_divisor(f, UINT64_MAX - 1, lut);
  direct_divisor(f, UINT64_MAX, lut);
  return f;
}

struct ModPowInput {
  uint64_t base, exp, mod;
};

bool run_integer_carry_checks() {
  constexpr std::array<uint64_t, 8> WORD_EDGES = {{
    0, 1, UINT32_MAX, UINT64_C(0x100000000), UINT64_C(0x7fffffffffffffff),
    UINT64_C(0x8000000000000000), UINT64_MAX - 1, UINT64_MAX,
  }};
  uint64_t multiply_cases = 0;
  for (uint64_t a : WORD_EDGES)
    for (uint64_t b : WORD_EDGES)
      if (!check_mul_hi64_portable(a, b, multiply_cases++))
        return false;

  std::mt19937_64 random(0x6d756c6869636172ULL);
  for (uint64_t n = 0; n < 20000; ++n) {
    const uint64_t a = random(), b = random();
    if (!check_mul_hi64_portable(a, b, multiply_cases++))
      return false;
  }

  // These two estimates need a quotient correction, with and without a carry in r + v1.
  uint64_t division_cases = 0;
  if (!check_divmod128_by128_fallback(combine(2, 1), combine(1, 1), division_cases++) ||
      !check_divmod128_by128_fallback(combine(5, UINT64_MAX - 3), combine(1, UINT64_MAX),
                                     division_cases++))
    return false;

  constexpr U128 max128 = ~static_cast<U128>(0);
  for (unsigned bit = 0; bit < 64; ++bit) {
    for (uint64_t low : std::array<uint64_t, 3>{{0, 1, UINT64_MAX}}) {
      const U128 divisor = combine(UINT64_C(1) << bit, low);
      const U128 largest_multiple = (max128 / divisor) * divisor;
      const std::array<U128, 6> numerators = {{
        divisor - 1, divisor, divisor + 1, largest_multiple - 1, largest_multiple, max128,
      }};
      for (U128 numerator : numerators)
        if (!check_divmod128_by128_fallback(numerator, divisor, division_cases++))
          return false;
    }
  }
  std::cout << "XelisHashV3 integer carry regression passed: " << multiply_cases
            << " portable multiply cases, " << division_cases << " divmod128_by128 cases\n";
  return true;
}

bool run_integer_fallback_checks() {
  constexpr std::array<uint64_t, 14> DIVISORS = {{
    0, 1, 2, 3, 0x7fffffffULL, 0x80000000ULL, UINT32_MAX,
    UINT64_C(0x100000000), UINT64_C(0x100000001), 0x7fffffffffffffffULL,
    0x8000000000000000ULL, 0x8000000000000001ULL, UINT64_MAX - 1, UINT64_MAX,
  }};
  constexpr std::array<uint64_t, 4> LOW_WORDS = {{
    0, 1, UINT64_C(0x0123456789abcdef), UINT64_MAX,
  }};
  uint64_t div_cases = 0;
  for (const uint64_t divisor : DIVISORS) {
    const uint64_t effective = divisor ? divisor : 1;
    const std::array<uint64_t, 5> HIGH_WORDS = {{
      0,
      effective > 1 ? effective - 1 : 0,
      effective,
      effective < UINT64_MAX ? effective + 1 : effective,
      UINT64_MAX,
    }};
    for (const uint64_t hi : HIGH_WORDS) {
      for (const uint64_t lo : LOW_WORDS) {
        const uint64_t current_case = div_cases++;
        if (!check_divmod128_fallback<true>(hi, lo, divisor, current_case) ||
            !check_divmod128_fallback<false>(hi, lo, divisor, current_case))
          return false;
      }
    }
  }

  // Normalized d with r/(d >> 32) two high: the fallback must apply both overestimate corrections.
  if (!check_div96_fallback(UINT64_C(0x7fffffff80000000), 0,
                            UINT64_C(0x80000000ffffffff), 0))
    return false;

  constexpr std::array<ModPowInput, 18> MODPOW_EDGES = {{
    {0, 0, 0}, {0, 0, 1}, {1, 0, 0}, {1, 0, 1}, {0, 1, 2}, {2, 0, 3},
    {2, 1, 3}, {3, 2, 5}, {UINT64_MAX, UINT64_MAX, UINT64_MAX},
    {UINT64_MAX, 1, 2}, {UINT64_MAX, 2, 0x8000000000000000ULL},
    {UINT64_C(0x8000000000000000), UINT64_MAX, UINT64_MAX - 1},
    {UINT64_C(0x0123456789abcdef), UINT64_C(0xfedcba9876543210), 7},
    {UINT64_C(0x0123456789abcdef), 0, UINT64_C(0x100000000)},
    {UINT64_C(0x100000001), 3, UINT64_C(0x100000000)},
    {UINT64_C(0x7fffffffffffffff), 17, UINT64_C(0x8000000000000001)},
    {UINT64_C(0x8000000000000001), 31, UINT64_C(0x7fffffffffffffff)},
    {UINT64_MAX - 1, UINT64_MAX, UINT64_MAX - 2},
  }};
  uint64_t modpow_cases = 0;
  for (const ModPowInput& input : MODPOW_EDGES) {
    const uint64_t current_case = modpow_cases++;
    if (!check_modpow<true>(input.base, input.exp, input.mod, current_case) ||
        !check_modpow<false>(input.base, input.exp, input.mod, current_case))
      return false;
  }

  constexpr uint64_t RANDOM_FALLBACK_CASES = 20000;
  std::mt19937_64 random(0x4d4f4d58454c4953ULL);
  for (uint64_t n = 0; n < RANDOM_FALLBACK_CASES; ++n) {
    uint64_t divisor = random();
    if ((n & 7) == 0)
      divisor = 0;
    else if ((n & 7) == 1)
      divisor = 1;
    const uint64_t hi = random(), lo = random();
    const uint64_t current_div_case = div_cases++;
    if (!check_divmod128_fallback<true>(hi, lo, divisor, current_div_case) ||
        !check_divmod128_fallback<false>(hi, lo, divisor, current_div_case))
      return false;

    uint64_t mod = random();
    if ((n & 7) == 0)
      mod = 0;
    else if ((n & 7) == 1)
      mod = 1;
    const uint64_t base = random(), exp = random();
    const uint64_t current_modpow_case = modpow_cases++;
    if (!check_modpow<true>(base, exp, mod, current_modpow_case) ||
        !check_modpow<false>(base, exp, mod, current_modpow_case))
      return false;
  }
  std::cout << "XelisHashV3 integer fallback regression passed: " << div_cases
            << " divmod128 cases, " << modpow_cases << " modpow cases\n";
  return true;
}

}  // namespace

int main() {
  double lut[512];
  for (unsigned i = 0; i < 512; ++i) {
    lut[i] = 1.0 / ((512.5 + i) * (1u << 22));
  }

  const std::array<Input, 12> edges = {{
    {0, 0, 0, 0, 0, 0, 0},
    {1, 1, 1, 1, 1, 1, 1},
    {2, 4, 8, 16, 2, 0, 3},
    {UINT64_MAX, UINT64_MAX, UINT64_MAX, UINT64_MAX,
      63, 1, 0xffffffffU},
    {UINT64_MAX, 0, 1, 0xaaaaaaaaaaaaaaaaULL, 0, 0, 0},
    {0, UINT64_MAX, 0, 0x5555555555555555ULL, 63, 1, 0xffffffffU},
    {0x8000000000000000ULL, 0x7fffffffffffffffULL, 0x8000000000000000ULL,
      0x8000000000000000ULL, 32, 0, 1},
    {0x7fffffffffffffffULL, 0x8000000000000000ULL, 0x7fffffffffffffffULL,
      0x7fffffffffffffffULL, 31, 1, 0},
    {0, 0x8000000000000000ULL, UINT64_MAX, 17, 47, 0, 0xffffffffU},
    {UINT64_MAX, 0, 0x8000000000000000ULL, 23, 48, 1, 0},
    {0x0123456789abcdefULL, 0xfedcba9876543210ULL, 0x1111111111111111ULL,
      0x2222222222222222ULL, 7, 0, 0xffffffffU},
    {0xfedcba9876543210ULL, 0x0123456789abcdefULL, 0xeeeeeeeeeeeeeeeeULL,
      0xddddddddddddddddULL, 56, 1, 0},
  }};

  uint64_t case_number = 0;
  for (const Input& in : edges) {
    for (uint32_t op = 0; op < 16; ++op) {
      const uint64_t current_case = case_number++;
      if (!check<true>(op, in, lut, current_case) ||
          !check<false>(op, in, lut, current_case))
        return 1;
    }
  }

  std::mt19937_64 random(0x58454c4953563301ULL);
  constexpr uint32_t RANDOM_CASES = 2000000;
  for (uint32_t n = 0; n < RANDOM_CASES; ++n) {
    const Input in = {
      random(), random(), random(), random(),
      static_cast<uint32_t>(random() % (531 * 128)), n & 1,
      static_cast<uint32_t>(random() % (531 * 64)),
    };
    for (uint32_t op = 0; op < 16; ++op) {
      const uint64_t current_case = case_number++;
      if (!check<true>(op, in, lut, current_case) ||
          !check<false>(op, in, lut, current_case))
        return 1;
    }
  }

  constexpr std::array<uint64_t, 9> DIVISOR_CENTERS = {{
    1ULL,
    2ULL,
    0x7fffffffULL,
    0x80000000ULL,
    UINT32_MAX,
    static_cast<uint64_t>(UINT32_MAX) + 1,
    0x7fffffffffffffffULL,
    0x8000000000000000ULL,
    UINT64_MAX,
  }};
  const U128 max_u64 = static_cast<U128>(UINT64_MAX);
  uint64_t directed_divmod_cases = 0;
  for (uint64_t center : DIVISOR_CENTERS) {
    for (int delta = -4; delta <= 4; ++delta) {
      U128 divisor128 = static_cast<U128>(center);
      const U128 magnitude = static_cast<U128>(delta < 0 ? -delta : delta);
      if (delta < 0) {
        if (divisor128 < magnitude)
          continue;
        divisor128 -= magnitude;
      } else {
        divisor128 += magnitude;
      }
      if (!divisor128 || divisor128 > max_u64)
        continue;
      const std::array<U128, 10> numerators = {{
        0,
        1,
        divisor128 - 1,
        divisor128,
        divisor128 + 1,
        divisor128 * 2 - 1,
        divisor128 * 2,
        divisor128 * 2 + 1,
        max_u64 - 1,
        max_u64,
      }};
      for (U128 numerator128 : numerators) {
        if (numerator128 > max_u64)
          continue;
        if (!check_divmod64(static_cast<uint64_t>(numerator128), static_cast<uint64_t>(divisor128),
                            directed_divmod_cases++))
          return 1;
      }
    }
  }

  constexpr uint64_t RANDOM_DIVISOR_CASES = 5000000;
  std::mt19937_64 divisor_random(0xd1f640b3a2c19e77ULL);
  for (uint64_t n = 0; n < RANDOM_DIVISOR_CASES; ++n) {
    const uint64_t numerator = divisor_random();
    uint64_t divisor = divisor_random();
    if (!divisor)
      divisor = 1;
    if (!check_divmod64(numerator, divisor, directed_divmod_cases + n))
      return 1;
  }

  if (!run_integer_carry_checks() || !run_integer_fallback_checks())
    return 1;

  const DirectFacts direct = run_direct_checks(lut);
  if (direct.domain_failures || direct.mod64_mismatches || direct.divmod128_mismatches) {
    std::cerr << "XelisHashV3 direct division aggregate mismatch: mod64_cases="
              << direct.mod64_cases << " mod64_mismatches=" << direct.mod64_mismatches
              << " divmod128_cases=" << direct.divmod128_cases
              << " divmod128_mismatches=" << direct.divmod128_mismatches
              << " domain_failures=" << direct.domain_failures << '\n';
    return 1;
  }
  std::cout << "XelisHashV3 direct division regression passed: mod64_cases="
            << direct.mod64_cases << ", divmod128_cases=" << direct.divmod128_cases << '\n';

  std::cout << "XelisHashV3 divmod64 regression passed: " << directed_divmod_cases
            << " directed cases, " << RANDOM_DIVISOR_CASES << " random nonzero divisors\n";
  std::cout << "XelisHashV3 operation regression passed: 16 operations, "
            << edges.size() << " edge cases, " << RANDOM_CASES << " random cases\n";
  return 0;
}
