// Copyright GNU GPLv3 (c) 2026 MoneroOcean <support@moneroocean.stream>

#pragma once

#include <cstdint>

// BLAKE2b (RFC 7693) building blocks shared by the Equihash-family solvers: the scalar compression
// used by host midstates and reference paths, and a 32-bit-pair G primitive for device hot loops on
// targets without a native 64-bit ALU. Device and host safe.
namespace mom_blake2b {

static constexpr std::uint64_t IV[8] = {
    0x6a09e667f3bcc908ull, 0xbb67ae8584caa73bull, 0x3c6ef372fe94f82bull, 0xa54ff53a5f1d36f1ull,
    0x510e527fade682d1ull, 0x9b05688c2b3e6c1full, 0x1f83d9abfb41bd6bull, 0x5be0cd19137e2179ull,
};

static constexpr std::uint8_t SIGMA[12][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
    {11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4},
    {7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8},
    {9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13},
    {2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9},
    {12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11},
    {13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10},
    {6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5},
    {10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0},
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
};

inline std::uint64_t rotr64(const std::uint64_t x, const unsigned n) {
  return (x >> n) | (x << (64 - n));
}

inline std::uint64_t load64_le(const std::uint8_t* const p) {
  return static_cast<std::uint64_t>(p[0]) | (static_cast<std::uint64_t>(p[1]) << 8) |
         (static_cast<std::uint64_t>(p[2]) << 16) | (static_cast<std::uint64_t>(p[3]) << 24) |
         (static_cast<std::uint64_t>(p[4]) << 32) | (static_cast<std::uint64_t>(p[5]) << 40) |
         (static_cast<std::uint64_t>(p[6]) << 48) | (static_cast<std::uint64_t>(p[7]) << 56);
}

// The BLAKE2b G mixing primitive on four state words.
inline void g(std::uint64_t& a, std::uint64_t& b, std::uint64_t& c, std::uint64_t& d,
              const std::uint64_t x, const std::uint64_t y) {
  a += b + x;
  d = rotr64(d ^ a, 32);
  c += d;
  b = rotr64(b ^ c, 24);
  a += b + y;
  d = rotr64(d ^ a, 16);
  c += d;
  b = rotr64(b ^ c, 63);
}

// Compress one block of 16 little-endian message words. `count` is the total number of bytes
// absorbed so far (messages here stay below 2^64 bytes); `last` marks the final block.
inline void compress(std::uint64_t h[8], const std::uint64_t m[16], const std::uint64_t count,
                     const bool last) {
  std::uint64_t v[16];
  for (unsigned i = 0; i < 8; ++i) {
    v[i] = h[i];
    v[i + 8] = IV[i];
  }
  v[12] ^= count;
  if (last)
    v[14] = ~v[14];
  for (unsigned r = 0; r < 12; ++r) {
    const std::uint8_t* const s = SIGMA[r];
    g(v[0], v[4], v[8], v[12], m[s[0]], m[s[1]]);
    g(v[1], v[5], v[9], v[13], m[s[2]], m[s[3]]);
    g(v[2], v[6], v[10], v[14], m[s[4]], m[s[5]]);
    g(v[3], v[7], v[11], v[15], m[s[6]], m[s[7]]);
    g(v[0], v[5], v[10], v[15], m[s[8]], m[s[9]]);
    g(v[1], v[6], v[11], v[12], m[s[10]], m[s[11]]);
    g(v[2], v[7], v[8], v[13], m[s[12]], m[s[13]]);
    g(v[3], v[4], v[9], v[14], m[s[14]], m[s[15]]);
  }
  for (unsigned i = 0; i < 8; ++i)
    h[i] ^= v[i] ^ v[i + 8];
}

// Load and compress one 128-byte block.
inline void compress_bytes(std::uint64_t h[8], const std::uint8_t block[128],
                           const std::uint64_t count, const bool last) {
  std::uint64_t m[16];
  for (unsigned i = 0; i < 16; ++i)
    m[i] = load64_le(block + 8 * i);
  compress(h, m, count, last);
}

// Initialise an unkeyed sequential chain for an `out_len`-byte digest with a 16-byte personalisation.
inline void init(std::uint64_t h[8], const unsigned out_len, const std::uint8_t personal[16]) {
  std::uint8_t parameter[64] = {};
  parameter[0] = static_cast<std::uint8_t>(out_len);
  parameter[2] = 1; // fanout
  parameter[3] = 1; // depth
  for (unsigned i = 0; i < 16; ++i)
    parameter[48 + i] = personal[i];
  for (unsigned i = 0; i < 8; ++i)
    h[i] = IV[i] ^ load64_le(parameter + 8 * i);
}

// Equihash-family personalisation: an 8-byte ASCII tag followed by le32(n) and le32(k).
inline void equihash_personal(const char tag[8], const std::uint32_t n, const std::uint32_t k,
                              std::uint8_t personal[16]) {
  for (unsigned i = 0; i < 8; ++i)
    personal[i] = static_cast<std::uint8_t>(tag[i]);
  for (unsigned i = 0; i < 4; ++i) {
    personal[8 + i] = static_cast<std::uint8_t>(n >> (8 * i));
    personal[12 + i] = static_cast<std::uint8_t>(k >> (8 * i));
  }
}

// One 64-bit BLAKE2b word held as native 32-bit halves.
struct Word {
  std::uint32_t lo;
  std::uint32_t hi;
};

// BLAKE2b's 64-bit G primitive expressed as two native 32-bit halves. PTX uses explicit carry
// chains and funnel shifts; other compilers receive the same operations as portable C++.
inline void mix(Word& a, Word& b, Word& c, Word& d, const Word x, const Word y) {
#if defined(__NVPTX__)
  std::uint32_t al = a.lo, ah = a.hi, bl = b.lo, bh = b.hi;
  std::uint32_t cl = c.lo, ch = c.hi, dl = d.lo, dh = d.hi;
  asm volatile("{ .reg .u32 t0, t1;\n"
               "add.cc.u32 %0, %0, %2; addc.u32 %1, %1, %3;\n"
               "add.cc.u32 %0, %0, %8; addc.u32 %1, %1, %9;\n"
               "xor.b32 t0, %6, %0; xor.b32 t1, %7, %1; mov.b32 %6, t1; mov.b32 %7, t0;\n"
               "add.cc.u32 %4, %4, %6; addc.u32 %5, %5, %7;\n"
               "xor.b32 t0, %2, %4; xor.b32 t1, %3, %5;\n"
               "shf.r.wrap.b32 %2, t0, t1, 24; shf.r.wrap.b32 %3, t1, t0, 24;\n"
               "add.cc.u32 %0, %0, %2; addc.u32 %1, %1, %3;\n"
               "add.cc.u32 %0, %0, %10; addc.u32 %1, %1, %11;\n"
               "xor.b32 t0, %6, %0; xor.b32 t1, %7, %1;\n"
               "shf.r.wrap.b32 %6, t0, t1, 16; shf.r.wrap.b32 %7, t1, t0, 16;\n"
               "add.cc.u32 %4, %4, %6; addc.u32 %5, %5, %7;\n"
               "xor.b32 t0, %2, %4; xor.b32 t1, %3, %5;\n"
               "shf.l.wrap.b32 %2, t1, t0, 1; shf.l.wrap.b32 %3, t0, t1, 1;\n"
               "}\n"
               : "+&r"(al), "+&r"(ah), "+&r"(bl), "+&r"(bh), "+&r"(cl), "+&r"(ch), "+&r"(dl),
                 "+&r"(dh)
               : "r"(x.lo), "r"(x.hi), "r"(y.lo), "r"(y.hi));
  a = {al, ah};
  b = {bl, bh};
  c = {cl, ch};
  d = {dl, dh};
#else
  std::uint32_t lo = a.lo + b.lo;
  std::uint32_t carry = lo < b.lo ? 1u : 0u;
  a.lo = lo + x.lo;
  carry += a.lo < x.lo ? 1u : 0u;
  a.hi = a.hi + b.hi + x.hi + carry;
  {
    const std::uint32_t old_lo = d.lo ^ a.lo;
    d.lo = d.hi ^ a.hi;
    d.hi = old_lo;
  }

  lo = c.lo + d.lo;
  c.hi = c.hi + d.hi + (lo < d.lo ? 1u : 0u);
  c.lo = lo;
  {
    const std::uint32_t old_lo = b.lo ^ c.lo;
    const std::uint32_t old_hi = b.hi ^ c.hi;
    b.lo = (old_lo >> 24) | (old_hi << 8);
    b.hi = (old_hi >> 24) | (old_lo << 8);
  }

  lo = a.lo + b.lo;
  carry = lo < b.lo ? 1u : 0u;
  a.lo = lo + y.lo;
  carry += a.lo < y.lo ? 1u : 0u;
  a.hi = a.hi + b.hi + y.hi + carry;
  {
    const std::uint32_t old_lo = d.lo ^ a.lo;
    const std::uint32_t old_hi = d.hi ^ a.hi;
    d.lo = (old_lo >> 16) | (old_hi << 16);
    d.hi = (old_hi >> 16) | (old_lo << 16);
  }

  lo = c.lo + d.lo;
  c.hi = c.hi + d.hi + (lo < d.lo ? 1u : 0u);
  c.lo = lo;
  {
    const std::uint32_t old_lo = b.lo ^ c.lo;
    const std::uint32_t old_hi = b.hi ^ c.hi;
    b.lo = (old_lo << 1) | (old_hi >> 31);
    b.hi = (old_hi << 1) | (old_lo >> 31);
  }
#endif
}

} // namespace mom_blake2b
