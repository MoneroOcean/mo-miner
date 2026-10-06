/*
MIT License

Copyright (c) 2026 Vano Chkheidze

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#pragma once

// Source commit: bd39cfbc0802ecd6eb21b7ee3b582a89617d20f0
// Source path: src/cuda/include/secp256k1_32_hybrid_final.cuh
// Mechanical adaptation notes:
// - Retains only the 32-bit multiply, square, and reduction logic.
// - Removes device qualifiers and typed wrappers; no unused compatibility APIs
//   remain; canonical field values use eight little-endian uint32_t limbs.
// - Applies early-clobber constraints only where a later instruction still
//   consumes an input operand in the same multi-instruction asm block.

#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)

#ifndef MOM_NEXAPOW_NO_STDINT
#include <stdint.h>
#endif

namespace mom_nexa_field32_hybrid_ptx {
namespace detail {

#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
#define MOM_NEXAPOW_FIELD32_INLINE inline
#elif defined(__clang__) || defined(__GNUC__)
#define MOM_NEXAPOW_FIELD32_INLINE inline __attribute__((always_inline))
#else
#define MOM_NEXAPOW_FIELD32_INLINE inline
#endif

MOM_NEXAPOW_FIELD32_INLINE void mul_256_comba32(const uint32_t a32[8], const uint32_t b32[8],
                                                uint32_t t32[16]) {
  uint32_t r0 = 0, r1 = 0, r2 = 0;

#define MUL32_ACC(ai, bj)                                                                          \
  {                                                                                                \
    asm volatile("mad.lo.cc.u32 %0, %3, %4, %0; \n\t"                                              \
                 "madc.hi.cc.u32 %1, %3, %4, %1; \n\t"                                             \
                 "addc.u32 %2, %2, 0; \n\t"                                                        \
                 : "+&r"(r0), "+r"(r1), "+r"(r2)                                                   \
                 : "r"(a32[ai]), "r"(b32[bj]));                                                    \
  }

  // Column 0
  MUL32_ACC(0, 0);
  t32[0] = r0;
  r0 = r1;
  r1 = r2;
  r2 = 0;

  // Column 1
  MUL32_ACC(0, 1);
  MUL32_ACC(1, 0);
  t32[1] = r0;
  r0 = r1;
  r1 = r2;
  r2 = 0;

  // Column 2
  MUL32_ACC(0, 2);
  MUL32_ACC(1, 1);
  MUL32_ACC(2, 0);
  t32[2] = r0;
  r0 = r1;
  r1 = r2;
  r2 = 0;

  // Column 3
  MUL32_ACC(0, 3);
  MUL32_ACC(1, 2);
  MUL32_ACC(2, 1);
  MUL32_ACC(3, 0);
  t32[3] = r0;
  r0 = r1;
  r1 = r2;
  r2 = 0;

  // Column 4
  MUL32_ACC(0, 4);
  MUL32_ACC(1, 3);
  MUL32_ACC(2, 2);
  MUL32_ACC(3, 1);
  MUL32_ACC(4, 0);
  t32[4] = r0;
  r0 = r1;
  r1 = r2;
  r2 = 0;

  // Column 5
  MUL32_ACC(0, 5);
  MUL32_ACC(1, 4);
  MUL32_ACC(2, 3);
  MUL32_ACC(3, 2);
  MUL32_ACC(4, 1);
  MUL32_ACC(5, 0);
  t32[5] = r0;
  r0 = r1;
  r1 = r2;
  r2 = 0;

  // Column 6
  MUL32_ACC(0, 6);
  MUL32_ACC(1, 5);
  MUL32_ACC(2, 4);
  MUL32_ACC(3, 3);
  MUL32_ACC(4, 2);
  MUL32_ACC(5, 1);
  MUL32_ACC(6, 0);
  t32[6] = r0;
  r0 = r1;
  r1 = r2;
  r2 = 0;

  // Column 7
  MUL32_ACC(0, 7);
  MUL32_ACC(1, 6);
  MUL32_ACC(2, 5);
  MUL32_ACC(3, 4);
  MUL32_ACC(4, 3);
  MUL32_ACC(5, 2);
  MUL32_ACC(6, 1);
  MUL32_ACC(7, 0);
  t32[7] = r0;
  r0 = r1;
  r1 = r2;
  r2 = 0;

  // Column 8
  MUL32_ACC(1, 7);
  MUL32_ACC(2, 6);
  MUL32_ACC(3, 5);
  MUL32_ACC(4, 4);
  MUL32_ACC(5, 3);
  MUL32_ACC(6, 2);
  MUL32_ACC(7, 1);
  t32[8] = r0;
  r0 = r1;
  r1 = r2;
  r2 = 0;

  // Column 9
  MUL32_ACC(2, 7);
  MUL32_ACC(3, 6);
  MUL32_ACC(4, 5);
  MUL32_ACC(5, 4);
  MUL32_ACC(6, 3);
  MUL32_ACC(7, 2);
  t32[9] = r0;
  r0 = r1;
  r1 = r2;
  r2 = 0;

  // Column 10
  MUL32_ACC(3, 7);
  MUL32_ACC(4, 6);
  MUL32_ACC(5, 5);
  MUL32_ACC(6, 4);
  MUL32_ACC(7, 3);
  t32[10] = r0;
  r0 = r1;
  r1 = r2;
  r2 = 0;

  // Column 11
  MUL32_ACC(4, 7);
  MUL32_ACC(5, 6);
  MUL32_ACC(6, 5);
  MUL32_ACC(7, 4);
  t32[11] = r0;
  r0 = r1;
  r1 = r2;
  r2 = 0;

  // Column 12
  MUL32_ACC(5, 7);
  MUL32_ACC(6, 6);
  MUL32_ACC(7, 5);
  t32[12] = r0;
  r0 = r1;
  r1 = r2;
  r2 = 0;

  // Column 13
  MUL32_ACC(6, 7);
  MUL32_ACC(7, 6);
  t32[13] = r0;
  r0 = r1;
  r1 = r2;
  r2 = 0;

  // Column 14
  MUL32_ACC(7, 7);
  t32[14] = r0;
  t32[15] = r1;

#undef MUL32_ACC
}

MOM_NEXAPOW_FIELD32_INLINE uint32_t sub_mul_256_comba32(uint32_t t[16], const uint32_t a[8],
                                                        const uint32_t b[8]) {
  uint32_t r0 = 0, r1 = 0, r2 = 0, borrow = 0;

#define MUL32_ACC(ai, bj)                                                                          \
  {                                                                                                \
    asm volatile("mad.lo.cc.u32 %0, %3, %4, %0; \n\t"                                              \
                 "madc.hi.cc.u32 %1, %3, %4, %1; \n\t"                                             \
                 "addc.u32 %2, %2, 0; \n\t"                                                        \
                 : "+&r"(r0), "+r"(r1), "+r"(r2)                                                   \
                 : "r"(a[ai]), "r"(b[bj]));                                                        \
  }
#define SUB32_OUT(i)                                                                               \
  {                                                                                                \
    const uint64_t difference = static_cast<uint64_t>(t[i]) - r0 - borrow;                         \
    t[i] = static_cast<uint32_t>(difference);                                                      \
    borrow = static_cast<uint32_t>(difference >> 63);                                              \
    r0 = r1;                                                                                       \
    r1 = r2;                                                                                       \
    r2 = static_cast<uint32_t>(static_cast<int32_t>(r2) >> 31);                                    \
  }

  MUL32_ACC(0, 0);
  SUB32_OUT(0);
  MUL32_ACC(0, 1);
  MUL32_ACC(1, 0);
  SUB32_OUT(1);
  MUL32_ACC(0, 2);
  MUL32_ACC(1, 1);
  MUL32_ACC(2, 0);
  SUB32_OUT(2);
  MUL32_ACC(0, 3);
  MUL32_ACC(1, 2);
  MUL32_ACC(2, 1);
  MUL32_ACC(3, 0);
  SUB32_OUT(3);
  MUL32_ACC(0, 4);
  MUL32_ACC(1, 3);
  MUL32_ACC(2, 2);
  MUL32_ACC(3, 1);
  MUL32_ACC(4, 0);
  SUB32_OUT(4);
  MUL32_ACC(0, 5);
  MUL32_ACC(1, 4);
  MUL32_ACC(2, 3);
  MUL32_ACC(3, 2);
  MUL32_ACC(4, 1);
  MUL32_ACC(5, 0);
  SUB32_OUT(5);
  MUL32_ACC(0, 6);
  MUL32_ACC(1, 5);
  MUL32_ACC(2, 4);
  MUL32_ACC(3, 3);
  MUL32_ACC(4, 2);
  MUL32_ACC(5, 1);
  MUL32_ACC(6, 0);
  SUB32_OUT(6);
  MUL32_ACC(0, 7);
  MUL32_ACC(1, 6);
  MUL32_ACC(2, 5);
  MUL32_ACC(3, 4);
  MUL32_ACC(4, 3);
  MUL32_ACC(5, 2);
  MUL32_ACC(6, 1);
  MUL32_ACC(7, 0);
  SUB32_OUT(7);
  MUL32_ACC(1, 7);
  MUL32_ACC(2, 6);
  MUL32_ACC(3, 5);
  MUL32_ACC(4, 4);
  MUL32_ACC(5, 3);
  MUL32_ACC(6, 2);
  MUL32_ACC(7, 1);
  SUB32_OUT(8);
  MUL32_ACC(2, 7);
  MUL32_ACC(3, 6);
  MUL32_ACC(4, 5);
  MUL32_ACC(5, 4);
  MUL32_ACC(6, 3);
  MUL32_ACC(7, 2);
  SUB32_OUT(9);
  MUL32_ACC(3, 7);
  MUL32_ACC(4, 6);
  MUL32_ACC(5, 5);
  MUL32_ACC(6, 4);
  MUL32_ACC(7, 3);
  SUB32_OUT(10);
  MUL32_ACC(4, 7);
  MUL32_ACC(5, 6);
  MUL32_ACC(6, 5);
  MUL32_ACC(7, 4);
  SUB32_OUT(11);
  MUL32_ACC(5, 7);
  MUL32_ACC(6, 6);
  MUL32_ACC(7, 5);
  SUB32_OUT(12);
  MUL32_ACC(6, 7);
  MUL32_ACC(7, 6);
  SUB32_OUT(13);
  MUL32_ACC(7, 7);
  SUB32_OUT(14);
  SUB32_OUT(15);

#undef SUB32_OUT
#undef MUL32_ACC
  return 0u - borrow;
}

MOM_NEXAPOW_FIELD32_INLINE void sqr_256_comba32(const uint32_t a[8], uint32_t t[16]) {
  uint32_t r0 = 0, r1 = 0, r2 = 0;
  uint32_t s0 = 0, s1 = 0, s2 = 0;

#define SQR32_OFF(ai, aj)                                                                          \
  asm volatile("mad.lo.cc.u32 %0, %3, %4, %0; \n\t"                                                \
               "madc.hi.cc.u32 %1, %3, %4, %1; \n\t"                                               \
               "addc.u32 %2, %2, 0; \n\t"                                                          \
               : "+&r"(s0), "+r"(s1), "+r"(s2)                                                     \
               : "r"(a[ai]), "r"(a[aj]))
#define SQR32_DIAGONAL(ai)                                                                         \
  asm volatile("mad.lo.cc.u32 %0, %3, %3, %0; \n\t"                                                \
               "madc.hi.cc.u32 %1, %3, %3, %1; \n\t"                                               \
               "addc.u32 %2, %2, 0; \n\t"                                                          \
               : "+&r"(r0), "+r"(r1), "+r"(r2)                                                     \
               : "r"(a[ai]))
#define SQR32_MERGE()                                                                              \
  do {                                                                                             \
    asm volatile("add.cc.u32 %0, %0, %0; \n\t"                                                     \
                 "addc.cc.u32 %1, %1, %1; \n\t"                                                    \
                 "addc.u32 %2, %2, %2; \n\t"                                                       \
                 : "+&r"(s0), "+&r"(s1), "+r"(s2));                                                \
    asm volatile("add.cc.u32 %0, %0, %3; \n\t"                                                     \
                 "addc.cc.u32 %1, %1, %4; \n\t"                                                    \
                 "addc.u32 %2, %2, %5; \n\t"                                                       \
                 : "+&r"(r0), "+&r"(r1), "+r"(r2)                                                  \
                 : "r"(s0), "r"(s1), "r"(s2));                                                     \
    s0 = s1 = s2 = 0;                                                                              \
  } while (0)
#define SQR32_OUT(i)                                                                               \
  do {                                                                                             \
    t[i] = r0;                                                                                     \
    r0 = r1;                                                                                       \
    r1 = r2;                                                                                       \
    r2 = 0;                                                                                        \
  } while (0)

  SQR32_DIAGONAL(0);
  SQR32_OUT(0);
  SQR32_OFF(0, 1);
  SQR32_MERGE();
  SQR32_OUT(1);
  SQR32_OFF(0, 2);
  SQR32_MERGE();
  SQR32_DIAGONAL(1);
  SQR32_OUT(2);
  SQR32_OFF(0, 3);
  SQR32_OFF(1, 2);
  SQR32_MERGE();
  SQR32_OUT(3);
  SQR32_OFF(0, 4);
  SQR32_OFF(1, 3);
  SQR32_MERGE();
  SQR32_DIAGONAL(2);
  SQR32_OUT(4);
  SQR32_OFF(0, 5);
  SQR32_OFF(1, 4);
  SQR32_OFF(2, 3);
  SQR32_MERGE();
  SQR32_OUT(5);
  SQR32_OFF(0, 6);
  SQR32_OFF(1, 5);
  SQR32_OFF(2, 4);
  SQR32_MERGE();
  SQR32_DIAGONAL(3);
  SQR32_OUT(6);
  SQR32_OFF(0, 7);
  SQR32_OFF(1, 6);
  SQR32_OFF(2, 5);
  SQR32_OFF(3, 4);
  SQR32_MERGE();
  SQR32_OUT(7);
  SQR32_OFF(1, 7);
  SQR32_OFF(2, 6);
  SQR32_OFF(3, 5);
  SQR32_MERGE();
  SQR32_DIAGONAL(4);
  SQR32_OUT(8);
  SQR32_OFF(2, 7);
  SQR32_OFF(3, 6);
  SQR32_OFF(4, 5);
  SQR32_MERGE();
  SQR32_OUT(9);
  SQR32_OFF(3, 7);
  SQR32_OFF(4, 6);
  SQR32_MERGE();
  SQR32_DIAGONAL(5);
  SQR32_OUT(10);
  SQR32_OFF(4, 7);
  SQR32_OFF(5, 6);
  SQR32_MERGE();
  SQR32_OUT(11);
  SQR32_OFF(5, 7);
  SQR32_MERGE();
  SQR32_DIAGONAL(6);
  SQR32_OUT(12);
  SQR32_OFF(6, 7);
  SQR32_MERGE();
  SQR32_OUT(13);
  SQR32_DIAGONAL(7);
  t[14] = r0;
  t[15] = r1;

#undef SQR32_OUT
#undef SQR32_MERGE
#undef SQR32_DIAGONAL
#undef SQR32_OFF
}

MOM_NEXAPOW_FIELD32_INLINE void reduce_512_to_256_32(uint32_t t32[16], uint32_t r[8]) {
  constexpr uint32_t k_mod = 977U;

  uint32_t t0 = t32[0], t1 = t32[1], t2 = t32[2], t3 = t32[3];
  uint32_t t4 = t32[4], t5 = t32[5], t6 = t32[6], t7 = t32[7];
  const uint32_t t8 = t32[8], t9 = t32[9], t10 = t32[10], t11 = t32[11];
  const uint32_t t12 = t32[12], t13 = t32[13], t14 = t32[14], t15 = t32[15];

  // Phase 1: T_hi multiplied by 977.
  uint32_t a0, a1, a2, a3, a4, a5, a6, a7, a8;
  asm volatile("mul.lo.u32 %0, %9, 977;\n\t"
               "mul.hi.u32 %1, %9, 977;\n\t"
               "mad.lo.cc.u32 %1, %10, 977, %1;\n\t"
               "madc.hi.u32 %2, %10, 977, 0;\n\t"
               "mad.lo.cc.u32 %2, %11, 977, %2;\n\t"
               "madc.hi.u32 %3, %11, 977, 0;\n\t"
               "mad.lo.cc.u32 %3, %12, 977, %3;\n\t"
               "madc.hi.u32 %4, %12, 977, 0;\n\t"
               "mad.lo.cc.u32 %4, %13, 977, %4;\n\t"
               "madc.hi.u32 %5, %13, 977, 0;\n\t"
               "mad.lo.cc.u32 %5, %14, 977, %5;\n\t"
               "madc.hi.u32 %6, %14, 977, 0;\n\t"
               "mad.lo.cc.u32 %6, %15, 977, %6;\n\t"
               "madc.hi.u32 %7, %15, 977, 0;\n\t"
               "mad.lo.cc.u32 %7, %16, 977, %7;\n\t"
               "madc.hi.u32 %8, %16, 977, 0;\n\t"
               : "=&r"(a0), "=&r"(a1), "=&r"(a2), "=&r"(a3), "=&r"(a4), "=&r"(a5), "=&r"(a6),
                 "=&r"(a7), "=r"(a8)
               : "r"(t8), "r"(t9), "r"(t10), "r"(t11), "r"(t12), "r"(t13), "r"(t14), "r"(t15));

  // Phase 1b: add T_hi shifted by one 32-bit limb.
  uint32_t a9;
  asm volatile("add.cc.u32 %0, %0, %9;\n\t"
               "addc.cc.u32 %1, %1, %10;\n\t"
               "addc.cc.u32 %2, %2, %11;\n\t"
               "addc.cc.u32 %3, %3, %12;\n\t"
               "addc.cc.u32 %4, %4, %13;\n\t"
               "addc.cc.u32 %5, %5, %14;\n\t"
               "addc.cc.u32 %6, %6, %15;\n\t"
               "addc.cc.u32 %7, %7, %16;\n\t"
               "addc.u32 %8, 0, 0;\n\t"
               : "+&r"(a1), "+&r"(a2), "+&r"(a3), "+&r"(a4), "+&r"(a5), "+&r"(a6), "+&r"(a7),
                 "+r"(a8), "=r"(a9)
               : "r"(t8), "r"(t9), "r"(t10), "r"(t11), "r"(t12), "r"(t13), "r"(t14), "r"(t15));

  // Phase 2: add the low eight limbs.
  uint32_t carry;
  asm volatile("add.cc.u32 %0, %0, %9;\n\t"
               "addc.cc.u32 %1, %1, %10;\n\t"
               "addc.cc.u32 %2, %2, %11;\n\t"
               "addc.cc.u32 %3, %3, %12;\n\t"
               "addc.cc.u32 %4, %4, %13;\n\t"
               "addc.cc.u32 %5, %5, %14;\n\t"
               "addc.cc.u32 %6, %6, %15;\n\t"
               "addc.cc.u32 %7, %7, %16;\n\t"
               "addc.u32 %8, 0, 0;\n\t"
               : "+&r"(t0), "+&r"(t1), "+&r"(t2), "+&r"(t3), "+&r"(t4), "+&r"(t5), "+&r"(t6),
                 "+r"(t7), "=r"(carry)
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(a4), "r"(a5), "r"(a6), "r"(a7));

  // Phase 3: fold the overflow using K_MOD = 2^32 + 977.
  uint32_t e_lo, e_carry;
  asm volatile("add.cc.u32 %0, %2, %3;\n\t"
               "addc.u32 %1, 0, 0;\n\t"
               : "=r"(e_lo), "=r"(e_carry)
               : "r"(a8), "r"(carry));
  uint32_t e_hi = a9 + e_carry;

  uint32_t p_lo, p_hi;
  asm volatile("mul.lo.u32 %0, %2, 977;\n\t"
               "mul.hi.u32 %1, %2, 977;\n\t"
               : "=&r"(p_lo), "=r"(p_hi)
               : "r"(e_lo));
  uint32_t q = e_hi * k_mod;
  uint32_t m1 = p_hi + q;

  uint32_t ek0 = p_lo;
  uint32_t ek1, ek1_carry;
  asm volatile("add.cc.u32 %0, %2, %3;\n\t"
               "addc.u32 %1, 0, 0;\n\t"
               : "=r"(ek1), "=r"(ek1_carry)
               : "r"(m1), "r"(e_lo));
  uint32_t ek2 = e_hi + ek1_carry;

  uint32_t final_carry;
  asm volatile("add.cc.u32 %0, %0, %9;\n\t"
               "addc.cc.u32 %1, %1, %10;\n\t"
               "addc.cc.u32 %2, %2, %11;\n\t"
               "addc.cc.u32 %3, %3, 0;\n\t"
               "addc.cc.u32 %4, %4, 0;\n\t"
               "addc.cc.u32 %5, %5, 0;\n\t"
               "addc.cc.u32 %6, %6, 0;\n\t"
               "addc.cc.u32 %7, %7, 0;\n\t"
               "addc.u32 %8, 0, 0;\n\t"
               : "+&r"(t0), "+&r"(t1), "+&r"(t2), "+&r"(t3), "+&r"(t4), "+&r"(t5), "+&r"(t6),
                 "+r"(t7), "=r"(final_carry)
               : "r"(ek0), "r"(ek1), "r"(ek2));

  if (final_carry) {
    asm volatile("add.cc.u32 %0, %0, 977;\n\t"
                 "addc.cc.u32 %1, %1, 1;\n\t"
                 "addc.cc.u32 %2, %2, 0;\n\t"
                 "addc.cc.u32 %3, %3, 0;\n\t"
                 "addc.cc.u32 %4, %4, 0;\n\t"
                 "addc.cc.u32 %5, %5, 0;\n\t"
                 "addc.cc.u32 %6, %6, 0;\n\t"
                 "addc.u32 %7, %7, 0;\n\t"
                 : "+&r"(t0), "+&r"(t1), "+&r"(t2), "+&r"(t3), "+&r"(t4), "+&r"(t5), "+&r"(t6),
                   "+r"(t7));
  }

  // Subtract p only in the top 2^32+977 values by adding its complement.
  if (t7 == 0xffffffffu && t6 == 0xffffffffu && t5 == 0xffffffffu && t4 == 0xffffffffu &&
      t3 == 0xffffffffu && t2 == 0xffffffffu &&
      (t1 == 0xffffffffu || (t1 == 0xfffffffeu && t0 >= 0xfffffc2fu))) {
    asm volatile("add.cc.u32 %0, %0, 977;\n\t"
                 "addc.cc.u32 %1, %1, 1;\n\t"
                 "addc.cc.u32 %2, %2, 0;\n\t"
                 "addc.cc.u32 %3, %3, 0;\n\t"
                 "addc.cc.u32 %4, %4, 0;\n\t"
                 "addc.cc.u32 %5, %5, 0;\n\t"
                 "addc.cc.u32 %6, %6, 0;\n\t"
                 "addc.u32 %7, %7, 0;\n\t"
                 : "+&r"(t0), "+&r"(t1), "+&r"(t2), "+&r"(t3), "+&r"(t4), "+&r"(t5), "+&r"(t6),
                   "+r"(t7));
  }
  r[0] = t0;
  r[1] = t1;
  r[2] = t2;
  r[3] = t3;
  r[4] = t4;
  r[5] = t5;
  r[6] = t6;
  r[7] = t7;
}

MOM_NEXAPOW_FIELD32_INLINE void field_mul_hybrid(const uint32_t a[8], const uint32_t b[8],
                                                 uint32_t r[8]) {
  uint32_t t32[16];
  mul_256_comba32(a, b, t32);
  reduce_512_to_256_32(t32, r);
}

MOM_NEXAPOW_FIELD32_INLINE void field_mul_sub_mul_hybrid(const uint32_t a[8], const uint32_t b[8],
                                                         const uint32_t c[8], const uint32_t d[8],
                                                         uint32_t r[8]) {
  uint32_t t[16];
  mul_256_comba32(a, b, t);
  const uint32_t underflow = sub_mul_256_comba32(t, c, d);
  reduce_512_to_256_32(t, r);
  uint32_t borrow;
  const uint32_t k0 = underflow & 0xe90a1u;
  const uint32_t k1 = underflow & 0x7a2u;
  const uint32_t k2 = underflow & 1u;
  asm volatile("sub.cc.u32 %0, %0, %9;\n\t"
               "subc.cc.u32 %1, %1, %10;\n\t"
               "subc.cc.u32 %2, %2, %11;\n\t"
               "subc.cc.u32 %3, %3, 0;\n\t"
               "subc.cc.u32 %4, %4, 0;\n\t"
               "subc.cc.u32 %5, %5, 0;\n\t"
               "subc.cc.u32 %6, %6, 0;\n\t"
               "subc.cc.u32 %7, %7, 0;\n\t"
               "subc.u32 %8, 0, 0;\n\t"
               : "+&r"(r[0]), "+&r"(r[1]), "+&r"(r[2]), "+&r"(r[3]), "+&r"(r[4]), "+&r"(r[5]),
                 "+&r"(r[6]), "+r"(r[7]), "=r"(borrow)
               : "r"(k0), "r"(k1), "r"(k2));
  const uint32_t p0 = borrow & 977u, p1 = borrow & 1u;
  asm volatile("sub.cc.u32 %0, %0, %8;\n\t"
               "subc.cc.u32 %1, %1, %9;\n\t"
               "subc.cc.u32 %2, %2, 0;\n\t"
               "subc.cc.u32 %3, %3, 0;\n\t"
               "subc.cc.u32 %4, %4, 0;\n\t"
               "subc.cc.u32 %5, %5, 0;\n\t"
               "subc.cc.u32 %6, %6, 0;\n\t"
               "subc.u32 %7, %7, 0;\n\t"
               : "+&r"(r[0]), "+&r"(r[1]), "+&r"(r[2]), "+&r"(r[3]), "+&r"(r[4]), "+&r"(r[5]),
                 "+&r"(r[6]), "+r"(r[7])
               : "r"(p0), "r"(p1));
}

MOM_NEXAPOW_FIELD32_INLINE void field_sqr_hybrid(const uint32_t a[8], uint32_t r[8]) {
  uint32_t t32[16];
  sqr_256_comba32(a, t32);
  reduce_512_to_256_32(t32, r);
}

} // namespace detail

MOM_NEXAPOW_FIELD32_INLINE void mul(uint32_t out[8], const uint32_t lhs[8], const uint32_t rhs[8]) {
  detail::field_mul_hybrid(lhs, rhs, out);
}

MOM_NEXAPOW_FIELD32_INLINE void square(uint32_t out[8], const uint32_t value[8]) {
  detail::field_sqr_hybrid(value, out);
}

MOM_NEXAPOW_FIELD32_INLINE void sub(uint32_t out[8], const uint32_t lhs[8], const uint32_t rhs[8]) {
  uint32_t borrow;
  asm volatile("sub.cc.u32 %0, %9, %17;\n\t"
               "subc.cc.u32 %1, %10, %18;\n\t"
               "subc.cc.u32 %2, %11, %19;\n\t"
               "subc.cc.u32 %3, %12, %20;\n\t"
               "subc.cc.u32 %4, %13, %21;\n\t"
               "subc.cc.u32 %5, %14, %22;\n\t"
               "subc.cc.u32 %6, %15, %23;\n\t"
               "subc.cc.u32 %7, %16, %24;\n\t"
               "subc.u32 %8, 0, 0;\n\t"
               : "=&r"(out[0]), "=&r"(out[1]), "=&r"(out[2]), "=&r"(out[3]), "=&r"(out[4]),
                 "=&r"(out[5]), "=&r"(out[6]), "=&r"(out[7]), "=r"(borrow)
               : "r"(lhs[0]), "r"(lhs[1]), "r"(lhs[2]), "r"(lhs[3]), "r"(lhs[4]), "r"(lhs[5]),
                 "r"(lhs[6]), "r"(lhs[7]), "r"(rhs[0]), "r"(rhs[1]), "r"(rhs[2]), "r"(rhs[3]),
                 "r"(rhs[4]), "r"(rhs[5]), "r"(rhs[6]), "r"(rhs[7]));
  const uint32_t p0 = borrow & 977u, p1 = borrow & 1u;
  asm volatile("sub.cc.u32 %0, %0, %8;\n\t"
               "subc.cc.u32 %1, %1, %9;\n\t"
               "subc.cc.u32 %2, %2, 0;\n\t"
               "subc.cc.u32 %3, %3, 0;\n\t"
               "subc.cc.u32 %4, %4, 0;\n\t"
               "subc.cc.u32 %5, %5, 0;\n\t"
               "subc.cc.u32 %6, %6, 0;\n\t"
               "subc.u32 %7, %7, 0;\n\t"
               : "+&r"(out[0]), "+&r"(out[1]), "+&r"(out[2]), "+&r"(out[3]), "+&r"(out[4]),
                 "+&r"(out[5]), "+&r"(out[6]), "+r"(out[7])
               : "r"(p0), "r"(p1));
}

MOM_NEXAPOW_FIELD32_INLINE void mul_sub_mul(uint32_t out[8], const uint32_t a[8],
                                            const uint32_t b[8], const uint32_t c[8],
                                            const uint32_t d[8]) {
  detail::field_mul_sub_mul_hybrid(a, b, c, d, out);
}

} // namespace mom_nexa_field32_hybrid_ptx

#undef MOM_NEXAPOW_FIELD32_INLINE
#endif
