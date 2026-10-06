#define _GNU_SOURCE

// Copyright GNU GPLv3 (c) 2026 MoneroOcean <support@moneroocean.stream>

#include "host_math.h"

#include <math.h>

typedef double (*MathFunction)(double);

#if defined(__GLIBC__)
#include <dlfcn.h>
#include <pthread.h>

static pthread_once_t math_once = PTHREAD_ONCE_INIT;
static MathFunction sin_function, cos_function, exp_function;
static MathFunction sqrt_function, floor_function, fabs_function;

static void resolve_math(void) {
#define MOM_RESOLVE_MATH(name) \
  name##_function = (MathFunction)dlvsym(RTLD_NEXT, #name, "GLIBC_2.2.5")
  MOM_RESOLVE_MATH(sin);
  MOM_RESOLVE_MATH(cos);
  MOM_RESOLVE_MATH(exp);
  MOM_RESOLVE_MATH(sqrt);
  MOM_RESOLVE_MATH(floor);
  MOM_RESOLVE_MATH(fabs);
#undef MOM_RESOLVE_MATH
}

const char* mom_hoohash_math_error(void) {
  pthread_once(&math_once, resolve_math);
  return sin_function && cos_function && exp_function && sqrt_function && floor_function &&
                 fabs_function
             ? 0
             : "hoohash cannot resolve canonical system math";
}

#define MOM_HOST_MATH_WRAPPER(name)                 \
  static double mom_hoohash_##name(double value) {  \
    pthread_once(&math_once, resolve_math);          \
    return name##_function ? name##_function(value) \
                           : NAN;                    \
  }
#else
const char* mom_hoohash_math_error(void) { return 0; }

#define MOM_HOST_MATH_WRAPPER(name)                 \
  static double mom_hoohash_##name(double value) {  \
    static MathFunction volatile function = name; \
    return function(value);                       \
  }
#endif

// Keep the canonical host check outside SYCL frontend math substitution.
MOM_HOST_MATH_WRAPPER(sin)
MOM_HOST_MATH_WRAPPER(cos)
MOM_HOST_MATH_WRAPPER(exp)
MOM_HOST_MATH_WRAPPER(sqrt)
MOM_HOST_MATH_WRAPPER(floor)
MOM_HOST_MATH_WRAPPER(fabs)

#undef MOM_HOST_MATH_WRAPPER

static const uint32_t B3_IV[8] = {0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
                                  0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u};
static const uint8_t B3_MSG[7][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8},
    {3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1},
    {10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6},
    {12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4},
    {9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7},
    {11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13}};

static uint32_t load32le(const uint8_t* input) {
  return (uint32_t)input[0] | (uint32_t)input[1] << 8 | (uint32_t)input[2] << 16 |
         (uint32_t)input[3] << 24;
}

static void store32le(uint8_t* output, uint32_t value) {
  output[0] = (uint8_t)value;
  output[1] = (uint8_t)(value >> 8);
  output[2] = (uint8_t)(value >> 16);
  output[3] = (uint8_t)(value >> 24);
}

static uint32_t rotr32(uint32_t value, unsigned count) {
  return value >> count | value << (32 - count);
}

static void b3_mix(uint32_t state[16], unsigned a, unsigned b, unsigned c, unsigned d,
                   uint32_t x, uint32_t y) {
  state[a] += state[b] + x;
  state[d] = rotr32(state[d] ^ state[a], 16);
  state[c] += state[d];
  state[b] = rotr32(state[b] ^ state[c], 12);
  state[a] += state[b] + y;
  state[d] = rotr32(state[d] ^ state[a], 8);
  state[c] += state[d];
  state[b] = rotr32(state[b] ^ state[c], 7);
}

static void b3_round(uint32_t state[16], const uint32_t message[16], unsigned round) {
  const uint8_t* schedule = B3_MSG[round];
  b3_mix(state, 0, 4, 8, 12, message[schedule[0]], message[schedule[1]]);
  b3_mix(state, 1, 5, 9, 13, message[schedule[2]], message[schedule[3]]);
  b3_mix(state, 2, 6, 10, 14, message[schedule[4]], message[schedule[5]]);
  b3_mix(state, 3, 7, 11, 15, message[schedule[6]], message[schedule[7]]);
  b3_mix(state, 0, 5, 10, 15, message[schedule[8]], message[schedule[9]]);
  b3_mix(state, 1, 6, 11, 12, message[schedule[10]], message[schedule[11]]);
  b3_mix(state, 2, 7, 8, 13, message[schedule[12]], message[schedule[13]]);
  b3_mix(state, 3, 4, 9, 14, message[schedule[14]], message[schedule[15]]);
}

static void b3_compress(const uint32_t cv[8], const uint8_t block[64], uint8_t block_length,
                        uint8_t flags, uint8_t output[64]) {
  uint32_t message[16], state[16];
  for (unsigned i = 0; i < 16; ++i)
    message[i] = load32le(block + 4 * i);
  for (unsigned i = 0; i < 8; ++i) {
    state[i] = cv[i];
    state[i + 8] = i < 4 ? B3_IV[i] : 0;
  }
  state[14] = block_length;
  state[15] = flags;
  for (unsigned round = 0; round < 7; ++round)
    b3_round(state, message, round);
  for (unsigned i = 0; i < 8; ++i) {
    store32le(output + 4 * i, state[i] ^ state[i + 8]);
    store32le(output + 4 * (i + 8), state[i + 8] ^ cv[i]);
  }
}

static void b3_hash(const uint8_t* input, unsigned length, uint8_t output[32]) {
  uint32_t cv[8];
  for (unsigned i = 0; i < 8; ++i)
    cv[i] = B3_IV[i];

  unsigned offset = 0;
  uint8_t first = 1;
  while (length - offset > 64) {
    uint8_t block[64], compressed[64];
    for (unsigned i = 0; i < 64; ++i)
      block[i] = input[offset + i];
    b3_compress(cv, block, 64, first, compressed);
    for (unsigned i = 0; i < 8; ++i)
      cv[i] = load32le(compressed + 4 * i);
    offset += 64;
    first = 0;
  }

  const unsigned tail = length - offset;
  uint8_t block[64] = {0}, compressed[64];
  for (unsigned i = 0; i < tail; ++i)
    block[i] = input[offset + i];
  b3_compress(cv, block, (uint8_t)tail, (uint8_t)(first | 2u | 8u), compressed);
  for (unsigned i = 0; i < 32; ++i)
    output[i] = compressed[i];
}

static uint64_t load64le(const uint8_t* input) {
  uint64_t value = 0;
  for (unsigned i = 0; i < 8; ++i)
    value |= (uint64_t)input[i] << (8 * i);
  return value;
}

static uint64_t rotl64(uint64_t value, unsigned count) {
  return value << count | value >> (64 - count);
}

static void make_matrix(const uint8_t seed[32], double matrix[4096]) {
  uint64_t s0 = load64le(seed), s1 = load64le(seed + 8), s2 = load64le(seed + 16),
           s3 = load64le(seed + 24);
  for (unsigned i = 0; i < 4096; ++i) {
    const uint64_t value = rotl64(s0 + s3, 23) + s0;
    const uint64_t temporary = s1 << 17;
    s2 ^= s0;
    s3 ^= s1;
    s1 ^= s2;
    s0 ^= s3;
    s2 ^= temporary;
    s3 = rotl64(s3, 45);
    matrix[i] = (double)(uint32_t)value / 4294967295.0 * 1000000.0;
  }
}

static double canonical_complex_value(double x) {
  const double a = x * .000001 / 8 - mom_hoohash_floor(x * .000001 / 8);
  const double b = x * .000001 / 4 - mom_hoohash_floor(x * .000001 / 4);
  const double y = b < .25   ? x + (1 + b)
                   : b < .5  ? x - (1 + b)
                   : b < .75 ? x * (1 + b)
                             : x / (1 + b);
  if (a < .33)
    return mom_hoohash_exp(mom_hoohash_sin(y) + mom_hoohash_cos(y));
  if (a < .66) {
    const double pi = 3.141592653589793238462643383279502884;
    if (y == pi / 2 || y == 3 * pi / 2)
      return 0;
    const double sine = mom_hoohash_sin(y);
    return sine * sine;
  }
  return 1 / mom_hoohash_sqrt(mom_hoohash_fabs(y) + 1);
}

static double canonical_finite_complex(double x) {
  double result = canonical_complex_value(x);
  unsigned rounds = 1;
  while (!isfinite(result)) {
    x *= .1;
    if (x <= 1e-13)
      return 0;
    ++rounds;
    result = canonical_complex_value(x);
  }
  return result * rounds;
}

static void canonical_mix(const uint8_t first[32], uint32_t hash_mod, uint64_t nonce,
                          const double matrix[4096], uint8_t mixed[32]) {
  uint64_t even = 0;
  double sw = 0;
  for (unsigned row = 0; row < 64; ++row) {
    double product = 0;
    for (unsigned j = 0; j < 64; ++j) {
      const uint8_t value = (j & 1) ? first[j / 2] & 15 : first[j / 2] >> 4;
      if (value) {
        const unsigned k = row * 64 + j;
        if (sw <= .02)
          product += canonical_finite_complex(matrix[k] * hash_mod * value +
                                              (double)(nonce & 255)) *
                     value * 1234;
        else
          product += matrix[k] * .0001 * value;
      }
      // A leading zero nibble resets the branch state for the new row.
      sw = product / 1024 - mom_hoohash_floor(product / 1024);
    }
    const uint64_t value = (uint64_t)product;
    if (row & 1)
      mixed[row / 2] = first[row / 2] ^ (uint8_t)(even + value);
    else
      even = value;
  }
}

void mom_hoohash_canonical_hash(const uint8_t input[72], uint64_t nonce, uint8_t output[32]) {
  uint8_t header[80], first[32], mixed[32];
  for (unsigned i = 0; i < 72; ++i)
    header[i] = input[i];
  for (unsigned i = 0; i < 8; ++i)
    header[72 + i] = (uint8_t)(nonce >> (8 * i));
  b3_hash(header, 80, first);

  uint32_t hash_mod = 0;
  for (unsigned i = 0; i < 8; ++i)
    hash_mod ^= (uint32_t)first[4 * i] << 24 | (uint32_t)first[4 * i + 1] << 16 |
                (uint32_t)first[4 * i + 2] << 8 | first[4 * i + 3];

  double matrix[4096];
  make_matrix(input, matrix);
  canonical_mix(first, hash_mod, nonce, matrix, mixed);
  b3_hash(mixed, 32, output);
}
