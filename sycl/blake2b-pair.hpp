#pragma once

#include <cstdint>

namespace mom_blake2b_pair {

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

} // namespace mom_blake2b_pair
