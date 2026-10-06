/*
* This file is part of the VanitySearch distribution (https://github.com/JeanLucPons/VanitySearch).
* Copyright (c) 2019 Jean Luc PONS.
*
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, version 3.
*
* This program is distributed in the hope that it will be useful, but
* WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
* General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)

#ifndef MOM_NEXAPOW_NO_STDINT
#include <stdint.h>
#endif

#define MOM_NEXA_F64_PTX_UADDO(c, a, b)                                                            \
  asm volatile("add.cc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b) : "memory");
#define MOM_NEXA_F64_PTX_UADDC(c, a, b)                                                            \
  asm volatile("addc.cc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b) : "memory");
#define MOM_NEXA_F64_PTX_UADD(c, a, b)                                                             \
  asm volatile("addc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b));

#define MOM_NEXA_F64_PTX_UADDO1(c, a)                                                              \
  asm volatile("add.cc.u64 %0, %0, %1;" : "+l"(c) : "l"(a) : "memory");
#define MOM_NEXA_F64_PTX_UADDC1(c, a)                                                              \
  asm volatile("addc.cc.u64 %0, %0, %1;" : "+l"(c) : "l"(a) : "memory");
#define MOM_NEXA_F64_PTX_UADD1(c, a) asm volatile("addc.u64 %0, %0, %1;" : "+l"(c) : "l"(a));

#define MOM_NEXA_F64_PTX_USUBO(c, a, b)                                                            \
  asm volatile("sub.cc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b) : "memory");
#define MOM_NEXA_F64_PTX_USUBC(c, a, b)                                                            \
  asm volatile("subc.cc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b) : "memory");
#define MOM_NEXA_F64_PTX_USUB(c, a, b)                                                             \
  asm volatile("subc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b));

#define MOM_NEXA_F64_PTX_USUBO1(c, a)                                                              \
  asm volatile("sub.cc.u64 %0, %0, %1;" : "+l"(c) : "l"(a) : "memory");
#define MOM_NEXA_F64_PTX_USUBC1(c, a)                                                              \
  asm volatile("subc.cc.u64 %0, %0, %1;" : "+l"(c) : "l"(a) : "memory");
#define MOM_NEXA_F64_PTX_USUB1(c, a) asm volatile("subc.u64 %0, %0, %1;" : "+l"(c) : "l"(a));

#define MOM_NEXA_F64_PTX_UMULLO(lo, a, b)                                                          \
  asm volatile("mul.lo.u64 %0, %1, %2;" : "=l"(lo) : "l"(a), "l"(b));
#define MOM_NEXA_F64_PTX_UMULHI(hi, a, b)                                                          \
  asm volatile("mul.hi.u64 %0, %1, %2;" : "=l"(hi) : "l"(a), "l"(b));
#define MOM_NEXA_F64_PTX_MADDO(r, a, b, c)                                                         \
  asm volatile("mad.hi.cc.u64 %0, %1, %2, %3;" : "=l"(r) : "l"(a), "l"(b), "l"(c) : "memory");
#define MOM_NEXA_F64_PTX_MADDC(r, a, b, c)                                                         \
  asm volatile("madc.hi.cc.u64 %0, %1, %2, %3;" : "=l"(r) : "l"(a), "l"(b), "l"(c) : "memory");
#define MOM_NEXA_F64_PTX_MADD(r, a, b, c)                                                          \
  asm volatile("madc.hi.u64 %0, %1, %2, %3;" : "=l"(r) : "l"(a), "l"(b), "l"(c));
#define MOM_NEXA_F64_PTX_MADDS(r, a, b, c)                                                         \
  asm volatile("madc.hi.s64 %0, %1, %2, %3;" : "=l"(r) : "l"(a), "l"(b), "l"(c));

#define MOM_NEXA_F64_PTX_IS_POSITIVE(x) (((int64_t)(x[4])) >= 0LL)

#define MOM_NEXA_F64_PTX_SUBP(r)                                                                   \
  {                                                                                                \
    MOM_NEXA_F64_PTX_USUBO1(r[0], 0xFFFFFFFEFFFFFC2FULL);                                          \
    MOM_NEXA_F64_PTX_USUBC1(r[1], 0xFFFFFFFFFFFFFFFFULL);                                          \
    MOM_NEXA_F64_PTX_USUBC1(r[2], 0xFFFFFFFFFFFFFFFFULL);                                          \
    MOM_NEXA_F64_PTX_USUBC1(r[3], 0xFFFFFFFFFFFFFFFFULL);                                          \
    MOM_NEXA_F64_PTX_USUB1(r[4], 0ULL);                                                            \
  }

#define MOM_NEXA_F64_PTX_UMULT(r, a, b)                                                            \
  {                                                                                                \
    MOM_NEXA_F64_PTX_UMULLO(r[0], a[0], b);                                                        \
    MOM_NEXA_F64_PTX_UMULLO(r[1], a[1], b);                                                        \
    MOM_NEXA_F64_PTX_MADDO(r[1], a[0], b, r[1]);                                                   \
    MOM_NEXA_F64_PTX_UMULLO(r[2], a[2], b);                                                        \
    MOM_NEXA_F64_PTX_MADDC(r[2], a[1], b, r[2]);                                                   \
    MOM_NEXA_F64_PTX_UMULLO(r[3], a[3], b);                                                        \
    MOM_NEXA_F64_PTX_MADDC(r[3], a[2], b, r[3]);                                                   \
    MOM_NEXA_F64_PTX_MADD(r[4], a[3], b, 0ULL);                                                    \
  }

#define MOM_NEXA_F64_PTX_LOAD256(r, a)                                                             \
  {                                                                                                \
    (r)[0] = (a)[0];                                                                               \
    (r)[1] = (a)[1];                                                                               \
    (r)[2] = (a)[2];                                                                               \
    (r)[3] = (a)[3];                                                                               \
  }

#define MOM_NEXA_F64_PTX_MM64 0xD838091DD2253531ULL
#define MOM_NEXA_F64_PTX_NBBLOCK 5
#define MOM_NEXA_F64_PTX_MSK62 0x3FFFFFFFFFFFFFFFULL

#define MOM_NEXA_F64_PTX_IS_NEGATIVE(x) (((int64_t)(x[4])) < 0LL)
#define MOM_NEXA_F64_PTX_IS_ZERO(a) ((a[4] | a[3] | a[2] | a[1] | a[0]) == 0ULL)
#define MOM_NEXA_F64_PTX_IS_ONE(a)                                                                 \
  ((a[4] == 0ULL) && (a[3] == 0ULL) && (a[2] == 0ULL) && (a[1] == 0ULL) && (a[0] == 1ULL))
#define MOM_NEXA_F64_PTX_SLEFT128(a, b, n) (((b) << (n)) | ((a) >> (64 - (n))))

#define MOM_NEXA_F64_PTX_ADDP(r)                                                                   \
  {                                                                                                \
    MOM_NEXA_F64_PTX_UADDO1(r[0], 0xFFFFFFFEFFFFFC2FULL);                                          \
    MOM_NEXA_F64_PTX_UADDC1(r[1], 0xFFFFFFFFFFFFFFFFULL);                                          \
    MOM_NEXA_F64_PTX_UADDC1(r[2], 0xFFFFFFFFFFFFFFFFULL);                                          \
    MOM_NEXA_F64_PTX_UADDC1(r[3], 0xFFFFFFFFFFFFFFFFULL);                                          \
    MOM_NEXA_F64_PTX_UADD1(r[4], 0ULL);                                                            \
  }

#define MOM_NEXA_F64_PTX_NEG(r)                                                                    \
  {                                                                                                \
    MOM_NEXA_F64_PTX_USUBO(r[0], 0ULL, r[0]);                                                      \
    MOM_NEXA_F64_PTX_USUBC(r[1], 0ULL, r[1]);                                                      \
    MOM_NEXA_F64_PTX_USUBC(r[2], 0ULL, r[2]);                                                      \
    MOM_NEXA_F64_PTX_USUBC(r[3], 0ULL, r[3]);                                                      \
    MOM_NEXA_F64_PTX_USUB(r[4], 0ULL, r[4]);                                                       \
  }

#define MOM_NEXA_F64_PTX_LOAD(r, a)                                                                \
  {                                                                                                \
    (r)[0] = (a)[0];                                                                               \
    (r)[1] = (a)[1];                                                                               \
    (r)[2] = (a)[2];                                                                               \
    (r)[3] = (a)[3];                                                                               \
    (r)[4] = (a)[4];                                                                               \
  }

#define MOM_NEXA_F64_PTX_SWAP(tmp, x, y)                                                           \
  tmp = x;                                                                                         \
  x = y;                                                                                           \
  y = tmp;

namespace mom_nexa_field64_ptx {
namespace detail {

inline void mom_nexa_field64_ptx_mod_add256(uint64_t* r, const uint64_t* a, const uint64_t* b) {
  uint64_t rr[5];

  MOM_NEXA_F64_PTX_UADDO(rr[0], a[0], b[0]);
  MOM_NEXA_F64_PTX_UADDC(rr[1], a[1], b[1]);
  MOM_NEXA_F64_PTX_UADDC(rr[2], a[2], b[2]);
  MOM_NEXA_F64_PTX_UADDC(rr[3], a[3], b[3]);
  MOM_NEXA_F64_PTX_UADD(rr[4], 0UL, 0UL);

  MOM_NEXA_F64_PTX_LOAD256(r, rr);

  MOM_NEXA_F64_PTX_SUBP(rr);

  if (MOM_NEXA_F64_PTX_IS_POSITIVE(rr)) {
    MOM_NEXA_F64_PTX_LOAD256(r, rr);
  }
}

inline void mom_nexa_field64_ptx_mod_sub256(uint64_t* r, const uint64_t* a, const uint64_t* b) {
  uint64_t t;
  uint64_t T[4];

  MOM_NEXA_F64_PTX_USUBO(r[0], a[0], b[0]);
  MOM_NEXA_F64_PTX_USUBC(r[1], a[1], b[1]);
  MOM_NEXA_F64_PTX_USUBC(r[2], a[2], b[2]);
  MOM_NEXA_F64_PTX_USUBC(r[3], a[3], b[3]);
  MOM_NEXA_F64_PTX_USUB(t, 0ULL, 0ULL);

  T[0] = 0xFFFFFFFEFFFFFC2FULL & t;
  T[1] = 0xFFFFFFFFFFFFFFFFULL & t;
  T[2] = 0xFFFFFFFFFFFFFFFFULL & t;
  T[3] = 0xFFFFFFFFFFFFFFFFULL & t;

  MOM_NEXA_F64_PTX_UADDO1(r[0], T[0]);
  MOM_NEXA_F64_PTX_UADDC1(r[1], T[1]);
  MOM_NEXA_F64_PTX_UADDC1(r[2], T[2]);
  MOM_NEXA_F64_PTX_UADD1(r[3], T[3]);
}

// Compute a*b*(mod n)
// a and b must be lower than n
inline void mom_nexa_field64_ptx_mod_mult(uint64_t* r, const uint64_t* a, const uint64_t* b) {
  uint64_t r512[8];
  uint64_t t[5];
  uint64_t ah, al;

  r512[5] = 0;
  r512[6] = 0;
  r512[7] = 0;

  // 256*256 multiplier
  MOM_NEXA_F64_PTX_UMULT(r512, a, b[0]);
  MOM_NEXA_F64_PTX_UMULT(t, a, b[1]);
  MOM_NEXA_F64_PTX_UADDO1(r512[1], t[0]);
  MOM_NEXA_F64_PTX_UADDC1(r512[2], t[1]);
  MOM_NEXA_F64_PTX_UADDC1(r512[3], t[2]);
  MOM_NEXA_F64_PTX_UADDC1(r512[4], t[3]);
  MOM_NEXA_F64_PTX_UADD1(r512[5], t[4]);
  MOM_NEXA_F64_PTX_UMULT(t, a, b[2]);
  MOM_NEXA_F64_PTX_UADDO1(r512[2], t[0]);
  MOM_NEXA_F64_PTX_UADDC1(r512[3], t[1]);
  MOM_NEXA_F64_PTX_UADDC1(r512[4], t[2]);
  MOM_NEXA_F64_PTX_UADDC1(r512[5], t[3]);
  MOM_NEXA_F64_PTX_UADD1(r512[6], t[4]);
  MOM_NEXA_F64_PTX_UMULT(t, a, b[3]);
  MOM_NEXA_F64_PTX_UADDO1(r512[3], t[0]);
  MOM_NEXA_F64_PTX_UADDC1(r512[4], t[1]);
  MOM_NEXA_F64_PTX_UADDC1(r512[5], t[2]);
  MOM_NEXA_F64_PTX_UADDC1(r512[6], t[3]);
  MOM_NEXA_F64_PTX_UADD1(r512[7], t[4]);

  // Reduce from 512 to 320
  MOM_NEXA_F64_PTX_UMULT(t, (r512 + 4), 0x1000003D1ULL);
  MOM_NEXA_F64_PTX_UADDO1(r512[0], t[0]);
  MOM_NEXA_F64_PTX_UADDC1(r512[1], t[1]);
  MOM_NEXA_F64_PTX_UADDC1(r512[2], t[2]);
  MOM_NEXA_F64_PTX_UADDC1(r512[3], t[3]);

  // Reduce from 320 to 256
  MOM_NEXA_F64_PTX_UADD1(t[4], 0ULL);
  MOM_NEXA_F64_PTX_UMULLO(al, t[4], 0x1000003D1ULL);
  MOM_NEXA_F64_PTX_UMULHI(ah, t[4], 0x1000003D1ULL);
  MOM_NEXA_F64_PTX_UADDO(r[0], r512[0], al);
  MOM_NEXA_F64_PTX_UADDC(r[1], r512[1], ah);
  MOM_NEXA_F64_PTX_UADDC(r[2], r512[2], 0ULL);
  MOM_NEXA_F64_PTX_UADD(r[3], r512[3], 0ULL);
}

inline void mom_nexa_field64_ptx_mod_sqr(uint64_t* rp, const uint64_t* up) {
  uint64_t r512[8];

  uint64_t u10, u11;

  uint64_t r0;
  uint64_t r1;
  uint64_t r3;
  uint64_t r4;

  uint64_t t1;
  uint64_t t2;

  //k=0
  MOM_NEXA_F64_PTX_UMULLO(r512[0], up[0], up[0]);
  MOM_NEXA_F64_PTX_UMULHI(r1, up[0], up[0]);

  //k=1
  MOM_NEXA_F64_PTX_UMULLO(r3, up[0], up[1]);
  MOM_NEXA_F64_PTX_UMULHI(r4, up[0], up[1]);
  MOM_NEXA_F64_PTX_UADDO1(r3, r3);
  MOM_NEXA_F64_PTX_UADDC1(r4, r4);
  MOM_NEXA_F64_PTX_UADD(t1, 0x0ULL, 0x0ULL);
  MOM_NEXA_F64_PTX_UADDO1(r3, r1);
  MOM_NEXA_F64_PTX_UADDC1(r4, 0x0ULL);
  MOM_NEXA_F64_PTX_UADD1(t1, 0x0ULL);
  r512[1] = r3;

  //k=2
  MOM_NEXA_F64_PTX_UMULLO(r0, up[0], up[2]);
  MOM_NEXA_F64_PTX_UMULHI(r1, up[0], up[2]);
  MOM_NEXA_F64_PTX_UADDO1(r0, r0);
  MOM_NEXA_F64_PTX_UADDC1(r1, r1);
  MOM_NEXA_F64_PTX_UADD(t2, 0x0ULL, 0x0ULL);
  MOM_NEXA_F64_PTX_UMULLO(u10, up[1], up[1]);
  MOM_NEXA_F64_PTX_UMULHI(u11, up[1], up[1]);
  MOM_NEXA_F64_PTX_UADDO1(r0, u10);
  MOM_NEXA_F64_PTX_UADDC1(r1, u11);
  MOM_NEXA_F64_PTX_UADD1(t2, 0x0ULL);
  MOM_NEXA_F64_PTX_UADDO1(r0, r4);
  MOM_NEXA_F64_PTX_UADDC1(r1, t1);
  MOM_NEXA_F64_PTX_UADD1(t2, 0x0ULL);

  r512[2] = r0;

  //k=3
  MOM_NEXA_F64_PTX_UMULLO(r3, up[0], up[3]);
  MOM_NEXA_F64_PTX_UMULHI(r4, up[0], up[3]);
  MOM_NEXA_F64_PTX_UMULLO(u10, up[1], up[2]);
  MOM_NEXA_F64_PTX_UMULHI(u11, up[1], up[2]);
  MOM_NEXA_F64_PTX_UADDO1(r3, u10);
  MOM_NEXA_F64_PTX_UADDC1(r4, u11);
  MOM_NEXA_F64_PTX_UADD(t1, 0x0ULL, 0x0ULL);
  t1 += t1;
  MOM_NEXA_F64_PTX_UADDO1(r3, r3);
  MOM_NEXA_F64_PTX_UADDC1(r4, r4);
  MOM_NEXA_F64_PTX_UADD1(t1, 0x0ULL);
  MOM_NEXA_F64_PTX_UADDO1(r3, r1);
  MOM_NEXA_F64_PTX_UADDC1(r4, t2);
  MOM_NEXA_F64_PTX_UADD1(t1, 0x0ULL);

  r512[3] = r3;

  //k=4
  MOM_NEXA_F64_PTX_UMULLO(r0, up[1], up[3]);
  MOM_NEXA_F64_PTX_UMULHI(r1, up[1], up[3]);
  MOM_NEXA_F64_PTX_UADDO1(r0, r0);
  MOM_NEXA_F64_PTX_UADDC1(r1, r1);
  MOM_NEXA_F64_PTX_UADD(t2, 0x0ULL, 0x0ULL);
  MOM_NEXA_F64_PTX_UMULLO(u10, up[2], up[2]);
  MOM_NEXA_F64_PTX_UMULHI(u11, up[2], up[2]);
  MOM_NEXA_F64_PTX_UADDO1(r0, u10);
  MOM_NEXA_F64_PTX_UADDC1(r1, u11);
  MOM_NEXA_F64_PTX_UADD1(t2, 0x0ULL);
  MOM_NEXA_F64_PTX_UADDO1(r0, r4);
  MOM_NEXA_F64_PTX_UADDC1(r1, t1);
  MOM_NEXA_F64_PTX_UADD1(t2, 0x0ULL);

  r512[4] = r0;

  //k=5
  MOM_NEXA_F64_PTX_UMULLO(r3, up[2], up[3]);
  MOM_NEXA_F64_PTX_UMULHI(r4, up[2], up[3]);
  MOM_NEXA_F64_PTX_UADDO1(r3, r3);
  MOM_NEXA_F64_PTX_UADDC1(r4, r4);
  MOM_NEXA_F64_PTX_UADD(t1, 0x0ULL, 0x0ULL);
  MOM_NEXA_F64_PTX_UADDO1(r3, r1);
  MOM_NEXA_F64_PTX_UADDC1(r4, t2);
  MOM_NEXA_F64_PTX_UADD1(t1, 0x0ULL);

  r512[5] = r3;

  //k=6
  MOM_NEXA_F64_PTX_UMULLO(r0, up[3], up[3]);
  MOM_NEXA_F64_PTX_UMULHI(r1, up[3], up[3]);
  MOM_NEXA_F64_PTX_UADDO1(r0, r4);
  MOM_NEXA_F64_PTX_UADD1(r1, t1);
  r512[6] = r0;

  //k=7
  r512[7] = r1;

  // Reduce from 512 to 320
  MOM_NEXA_F64_PTX_UMULLO(r0, r512[4], 0x1000003D1ULL);
  MOM_NEXA_F64_PTX_UMULLO(r1, r512[5], 0x1000003D1ULL);
  MOM_NEXA_F64_PTX_MADDO(r1, r512[4], 0x1000003D1ULL, r1);
  MOM_NEXA_F64_PTX_UMULLO(t2, r512[6], 0x1000003D1ULL);
  MOM_NEXA_F64_PTX_MADDC(t2, r512[5], 0x1000003D1ULL, t2);
  MOM_NEXA_F64_PTX_UMULLO(r3, r512[7], 0x1000003D1ULL);
  MOM_NEXA_F64_PTX_MADDC(r3, r512[6], 0x1000003D1ULL, r3);
  MOM_NEXA_F64_PTX_MADD(r4, r512[7], 0x1000003D1ULL, 0ULL);

  MOM_NEXA_F64_PTX_UADDO1(r512[0], r0);
  MOM_NEXA_F64_PTX_UADDC1(r512[1], r1);
  MOM_NEXA_F64_PTX_UADDC1(r512[2], t2);
  MOM_NEXA_F64_PTX_UADDC1(r512[3], r3);

  // Reduce from 320 to 256
  MOM_NEXA_F64_PTX_UADD1(r4, 0ULL);
  MOM_NEXA_F64_PTX_UMULLO(u10, r4, 0x1000003D1ULL);
  MOM_NEXA_F64_PTX_UMULHI(u11, r4, 0x1000003D1ULL);
  MOM_NEXA_F64_PTX_UADDO(rp[0], r512[0], u10);
  MOM_NEXA_F64_PTX_UADDC(rp[1], r512[1], u11);
  MOM_NEXA_F64_PTX_UADDC(rp[2], r512[2], 0ULL);
  MOM_NEXA_F64_PTX_UADD(rp[3], r512[3], 0ULL);
}

inline void mom_nexa_field64_ptx_shift_r62(uint64_t* r) {
  r[0] = (r[1] << 2) | (r[0] >> 62);
  r[1] = (r[2] << 2) | (r[1] >> 62);
  r[2] = (r[3] << 2) | (r[2] >> 62);
  r[3] = (r[4] << 2) | (r[3] >> 62);
  r[4] = (int64_t)(r[4]) >> 62;
}

inline void mom_nexa_field64_ptx_shift_r62(uint64_t dest[5], uint64_t r[5], uint64_t carry) {
  dest[0] = (r[1] << 2) | (r[0] >> 62);
  dest[1] = (r[2] << 2) | (r[1] >> 62);
  dest[2] = (r[3] << 2) | (r[2] >> 62);
  dest[3] = (r[4] << 2) | (r[3] >> 62);
  dest[4] = (carry << 2) | (r[4] >> 62);
}

inline void mom_nexa_field64_ptx_imult(uint64_t* r, uint64_t* a, int64_t b) {
  uint64_t t[MOM_NEXA_F64_PTX_NBBLOCK];

  if (b < 0) {
    b = -b;
    MOM_NEXA_F64_PTX_USUBO(t[0], 0ULL, a[0]);
    MOM_NEXA_F64_PTX_USUBC(t[1], 0ULL, a[1]);
    MOM_NEXA_F64_PTX_USUBC(t[2], 0ULL, a[2]);
    MOM_NEXA_F64_PTX_USUBC(t[3], 0ULL, a[3]);
    MOM_NEXA_F64_PTX_USUB(t[4], 0ULL, a[4]);
  } else {
    MOM_NEXA_F64_PTX_LOAD(t, a);
  }

  MOM_NEXA_F64_PTX_UMULLO(r[0], t[0], b);
  MOM_NEXA_F64_PTX_UMULLO(r[1], t[1], b);
  MOM_NEXA_F64_PTX_MADDO(r[1], t[0], b, r[1]);
  MOM_NEXA_F64_PTX_UMULLO(r[2], t[2], b);
  MOM_NEXA_F64_PTX_MADDC(r[2], t[1], b, r[2]);
  MOM_NEXA_F64_PTX_UMULLO(r[3], t[3], b);
  MOM_NEXA_F64_PTX_MADDC(r[3], t[2], b, r[3]);
  MOM_NEXA_F64_PTX_UMULLO(r[4], t[4], b);
  MOM_NEXA_F64_PTX_MADD(r[4], t[3], b, r[4]);
}

inline uint64_t mom_nexa_field64_ptx_imult_c(uint64_t* r, uint64_t* a, int64_t b) {
  uint64_t t[MOM_NEXA_F64_PTX_NBBLOCK];
  uint64_t carry;

  if (b < 0) {
    b = -b;
    MOM_NEXA_F64_PTX_USUBO(t[0], 0ULL, a[0]);
    MOM_NEXA_F64_PTX_USUBC(t[1], 0ULL, a[1]);
    MOM_NEXA_F64_PTX_USUBC(t[2], 0ULL, a[2]);
    MOM_NEXA_F64_PTX_USUBC(t[3], 0ULL, a[3]);
    MOM_NEXA_F64_PTX_USUB(t[4], 0ULL, a[4]);
  } else {
    MOM_NEXA_F64_PTX_LOAD(t, a);
  }

  MOM_NEXA_F64_PTX_UMULLO(r[0], t[0], b);
  MOM_NEXA_F64_PTX_UMULLO(r[1], t[1], b);
  MOM_NEXA_F64_PTX_MADDO(r[1], t[0], b, r[1]);
  MOM_NEXA_F64_PTX_UMULLO(r[2], t[2], b);
  MOM_NEXA_F64_PTX_MADDC(r[2], t[1], b, r[2]);
  MOM_NEXA_F64_PTX_UMULLO(r[3], t[3], b);
  MOM_NEXA_F64_PTX_MADDC(r[3], t[2], b, r[3]);
  MOM_NEXA_F64_PTX_UMULLO(r[4], t[4], b);
  MOM_NEXA_F64_PTX_MADDC(r[4], t[3], b, r[4]);
  MOM_NEXA_F64_PTX_MADDS(carry, t[4], b, 0ULL);

  return carry;
}

inline void mom_nexa_field64_ptx_mul_p(uint64_t* r, uint64_t a) {
  uint64_t ah;
  uint64_t al;

  MOM_NEXA_F64_PTX_UMULLO(al, a, 0x1000003D1ULL);
  MOM_NEXA_F64_PTX_UMULHI(ah, a, 0x1000003D1ULL);

  MOM_NEXA_F64_PTX_USUBO(r[0], 0ULL, al);
  MOM_NEXA_F64_PTX_USUBC(r[1], 0ULL, ah);
  MOM_NEXA_F64_PTX_USUBC(r[2], 0ULL, 0ULL);
  MOM_NEXA_F64_PTX_USUBC(r[3], 0ULL, 0ULL);
  MOM_NEXA_F64_PTX_USUB(r[4], a, 0ULL);
}

inline uint32_t mom_nexa_field64_ptx_ctz(uint64_t x) {
  uint32_t n;
  asm("{\n\t"
      " .reg .u64 tmp;\n\t"
      " brev.b64 tmp, %1;\n\t"
      " clz.b64 %0, tmp;\n\t"
      "}"
      : "=r"(n)
      : "l"(x));
  return n;
}

inline void mom_nexa_field64_ptx_div_step62(uint64_t u[5], uint64_t v[5], int32_t* pos, int64_t* uu,
                                            int64_t* uv, int64_t* vu, int64_t* vv,
                                            int* jac = nullptr) {
  *uu = 1;
  *uv = 0;
  *vu = 0;
  *vv = 1;

  uint32_t bitCount = 62;
  uint32_t zeros;
  uint64_t u0 = u[0];
  uint64_t v0 = v[0];

  uint64_t uh;
  uint64_t vh;
  int64_t w, x, y, z;

  while (*pos > 0 && (u[*pos] | v[*pos]) == 0)
    (*pos)--;
  if (*pos == 0) {
    uh = u[0];
    vh = v[0];
  } else {
    uint32_t s = __builtin_clzll(u[*pos] | v[*pos]);
    if (s == 0) {
      uh = u[*pos];
      vh = v[*pos];
    } else {
      uh = MOM_NEXA_F64_PTX_SLEFT128(u[*pos - 1], u[*pos], s);
      vh = MOM_NEXA_F64_PTX_SLEFT128(v[*pos - 1], v[*pos], s);
    }
  }

  while (true) {
    zeros = mom_nexa_field64_ptx_ctz(v0 | (1ULL << bitCount));

    if (jac)
      *jac ^= zeros & ((u0 >> 1) ^ (u0 >> 2));
    v0 >>= zeros;
    vh >>= zeros;
    *uu <<= zeros;
    *uv <<= zeros;
    bitCount -= zeros;

    if (bitCount == 0)
      break;

    if (vh < uh) {
      if (jac)
        *jac ^= (u0 & v0) >> 1;
      MOM_NEXA_F64_PTX_SWAP(w, uh, vh);
      MOM_NEXA_F64_PTX_SWAP(x, u0, v0);
      MOM_NEXA_F64_PTX_SWAP(y, *uu, *vu);
      MOM_NEXA_F64_PTX_SWAP(z, *uv, *vv);
    }

    vh -= uh;
    v0 -= u0;
    *vv -= *uv;
    *vu -= *uu;
  }
}

inline void mom_nexa_field64_ptx_matrix_vec_mul_half(uint64_t dest[5], uint64_t u[5], uint64_t v[5],
                                                     int64_t _11, int64_t _12, uint64_t* carry) {
  uint64_t t1[MOM_NEXA_F64_PTX_NBBLOCK];
  uint64_t t2[MOM_NEXA_F64_PTX_NBBLOCK];
  uint64_t c1, c2;

  c1 = mom_nexa_field64_ptx_imult_c(t1, u, _11);
  c2 = mom_nexa_field64_ptx_imult_c(t2, v, _12);

  MOM_NEXA_F64_PTX_UADDO(dest[0], t1[0], t2[0]);
  MOM_NEXA_F64_PTX_UADDC(dest[1], t1[1], t2[1]);
  MOM_NEXA_F64_PTX_UADDC(dest[2], t1[2], t2[2]);
  MOM_NEXA_F64_PTX_UADDC(dest[3], t1[3], t2[3]);
  MOM_NEXA_F64_PTX_UADDC(dest[4], t1[4], t2[4]);
  MOM_NEXA_F64_PTX_UADD(*carry, c1, c2);
}

inline void mom_nexa_field64_ptx_matrix_vec_mul(uint64_t u[5], uint64_t v[5], int64_t _11,
                                                int64_t _12, int64_t _21, int64_t _22) {
  uint64_t t1[MOM_NEXA_F64_PTX_NBBLOCK];
  uint64_t t2[MOM_NEXA_F64_PTX_NBBLOCK];
  uint64_t t3[MOM_NEXA_F64_PTX_NBBLOCK];
  uint64_t t4[MOM_NEXA_F64_PTX_NBBLOCK];

  mom_nexa_field64_ptx_imult(t1, u, _11);
  mom_nexa_field64_ptx_imult(t2, v, _12);
  mom_nexa_field64_ptx_imult(t3, u, _21);
  mom_nexa_field64_ptx_imult(t4, v, _22);

  MOM_NEXA_F64_PTX_UADDO(u[0], t1[0], t2[0]);
  MOM_NEXA_F64_PTX_UADDC(u[1], t1[1], t2[1]);
  MOM_NEXA_F64_PTX_UADDC(u[2], t1[2], t2[2]);
  MOM_NEXA_F64_PTX_UADDC(u[3], t1[3], t2[3]);
  MOM_NEXA_F64_PTX_UADD(u[4], t1[4], t2[4]);

  MOM_NEXA_F64_PTX_UADDO(v[0], t3[0], t4[0]);
  MOM_NEXA_F64_PTX_UADDC(v[1], t3[1], t4[1]);
  MOM_NEXA_F64_PTX_UADDC(v[2], t3[2], t4[2]);
  MOM_NEXA_F64_PTX_UADDC(v[3], t3[3], t4[3]);
  MOM_NEXA_F64_PTX_UADD(v[4], t3[4], t4[4]);
}

inline uint64_t mom_nexa_field64_ptx_add_ch(uint64_t r[5], uint64_t a[5], uint64_t carry) {
  uint64_t carryOut;

  MOM_NEXA_F64_PTX_UADDO1(r[0], a[0]);
  MOM_NEXA_F64_PTX_UADDC1(r[1], a[1]);
  MOM_NEXA_F64_PTX_UADDC1(r[2], a[2]);
  MOM_NEXA_F64_PTX_UADDC1(r[3], a[3]);
  MOM_NEXA_F64_PTX_UADDC1(r[4], a[4]);
  MOM_NEXA_F64_PTX_UADD(carryOut, carry, 0ULL);

  return carryOut;
}

inline void mom_nexa_field64_ptx_mod_inv(uint64_t* R) {
  uint64_t u[MOM_NEXA_F64_PTX_NBBLOCK];
  uint64_t v[MOM_NEXA_F64_PTX_NBBLOCK];
  uint64_t r[MOM_NEXA_F64_PTX_NBBLOCK];
  uint64_t s[MOM_NEXA_F64_PTX_NBBLOCK];
  uint64_t tr[MOM_NEXA_F64_PTX_NBBLOCK];
  uint64_t ts[MOM_NEXA_F64_PTX_NBBLOCK];
  uint64_t r0[MOM_NEXA_F64_PTX_NBBLOCK];
  uint64_t s0[MOM_NEXA_F64_PTX_NBBLOCK];

  int64_t uu;
  int64_t uv;
  int64_t vu;
  int64_t vv;

  uint64_t mr0;
  uint64_t ms0;

  uint64_t carryR;
  uint64_t carryS;

  int32_t pos = MOM_NEXA_F64_PTX_NBBLOCK - 1;

  u[0] = 0xFFFFFFFEFFFFFC2FULL;
  u[1] = 0xFFFFFFFFFFFFFFFFULL;
  u[2] = 0xFFFFFFFFFFFFFFFFULL;
  u[3] = 0xFFFFFFFFFFFFFFFFULL;
  u[4] = 0;
  MOM_NEXA_F64_PTX_LOAD(v, R);
  r[0] = 0;
  s[0] = 1;
  r[1] = 0;
  s[1] = 0;
  r[2] = 0;
  s[2] = 0;
  r[3] = 0;
  s[3] = 0;
  r[4] = 0;
  s[4] = 0;

  while (true) {
    mom_nexa_field64_ptx_div_step62(u, v, &pos, &uu, &uv, &vu, &vv);

    mom_nexa_field64_ptx_matrix_vec_mul(u, v, uu, uv, vu, vv);

    if (MOM_NEXA_F64_PTX_IS_NEGATIVE(u)) {
      MOM_NEXA_F64_PTX_NEG(u);
      uu = -uu;
      uv = -uv;
    }
    if (MOM_NEXA_F64_PTX_IS_NEGATIVE(v)) {
      MOM_NEXA_F64_PTX_NEG(v);
      vu = -vu;
      vv = -vv;
    }

    mom_nexa_field64_ptx_shift_r62(u);
    mom_nexa_field64_ptx_shift_r62(v);

    mom_nexa_field64_ptx_matrix_vec_mul_half(tr, r, s, uu, uv, &carryR);
    mr0 = (tr[0] * MOM_NEXA_F64_PTX_MM64) & MOM_NEXA_F64_PTX_MSK62;
    mom_nexa_field64_ptx_mul_p(r0, mr0);
    carryR = mom_nexa_field64_ptx_add_ch(tr, r0, carryR);

    if (MOM_NEXA_F64_PTX_IS_ZERO(v)) {
      mom_nexa_field64_ptx_shift_r62(r, tr, carryR);
      break;
    } else {
      mom_nexa_field64_ptx_matrix_vec_mul_half(ts, r, s, vu, vv, &carryS);
      ms0 = (ts[0] * MOM_NEXA_F64_PTX_MM64) & MOM_NEXA_F64_PTX_MSK62;
      mom_nexa_field64_ptx_mul_p(s0, ms0);
      carryS = mom_nexa_field64_ptx_add_ch(ts, s0, carryS);
    }

    mom_nexa_field64_ptx_shift_r62(r, tr, carryR);
    mom_nexa_field64_ptx_shift_r62(s, ts, carryS);
  }

  if (!MOM_NEXA_F64_PTX_IS_ONE(u)) {
    R[0] = 0ULL;
    R[1] = 0ULL;
    R[2] = 0ULL;
    R[3] = 0ULL;
    R[4] = 0ULL;
    return;
  }

  while (MOM_NEXA_F64_PTX_IS_NEGATIVE(r))
    MOM_NEXA_F64_PTX_ADDP(r);
  while (!MOM_NEXA_F64_PTX_IS_NEGATIVE(r))
    MOM_NEXA_F64_PTX_SUBP(r);
  MOM_NEXA_F64_PTX_ADDP(r);

  MOM_NEXA_F64_PTX_LOAD(R, r);
}

inline bool mom_nexa_field64_ptx_is_quad(const uint64_t r[4]) {
  uint64_t u[5] = {0xFFFFFFFEFFFFFC2FULL, UINT64_MAX, UINT64_MAX, UINT64_MAX, 0};
  uint64_t v[5] = {r[0], r[1], r[2], r[3], 0};
  int32_t pos = 4;
  int jac = 0;
  while (!MOM_NEXA_F64_PTX_IS_ZERO(v)) {
    int64_t uu, uv, vu, vv;
    mom_nexa_field64_ptx_div_step62(u, v, &pos, &uu, &uv, &vu, &vv, &jac);
    mom_nexa_field64_ptx_matrix_vec_mul(u, v, uu, uv, vu, vv);
    const bool u_negative = MOM_NEXA_F64_PTX_IS_NEGATIVE(u);
    const bool v_negative = MOM_NEXA_F64_PTX_IS_NEGATIVE(v);
    if (u_negative)
      MOM_NEXA_F64_PTX_NEG(u);
    if (v_negative)
      MOM_NEXA_F64_PTX_NEG(v);
    mom_nexa_field64_ptx_shift_r62(u);
    mom_nexa_field64_ptx_shift_r62(v);
    if (v_negative && ((u[0] & 3u) == 3u))
      jac ^= 1;
    if (u_negative && v_negative)
      jac ^= 1;
  }
  return MOM_NEXA_F64_PTX_IS_ONE(u) && !(jac & 1);
}

} // namespace detail

inline void add(uint64_t* r, const uint64_t* a, const uint64_t* b) {
  detail::mom_nexa_field64_ptx_mod_add256(r, a, b);
}

inline void sub(uint64_t* r, const uint64_t* a, const uint64_t* b) {
  detail::mom_nexa_field64_ptx_mod_sub256(r, a, b);
}

inline void add_mod(uint64_t* r, const uint64_t* a, const uint64_t* b, const uint64_t* modulus) {
  uint64_t sum[5], reduced[5];
  MOM_NEXA_F64_PTX_UADDO(sum[0], a[0], b[0]);
  MOM_NEXA_F64_PTX_UADDC(sum[1], a[1], b[1]);
  MOM_NEXA_F64_PTX_UADDC(sum[2], a[2], b[2]);
  MOM_NEXA_F64_PTX_UADDC(sum[3], a[3], b[3]);
  MOM_NEXA_F64_PTX_UADD(sum[4], 0ULL, 0ULL);
  MOM_NEXA_F64_PTX_USUBO(reduced[0], sum[0], modulus[0]);
  MOM_NEXA_F64_PTX_USUBC(reduced[1], sum[1], modulus[1]);
  MOM_NEXA_F64_PTX_USUBC(reduced[2], sum[2], modulus[2]);
  MOM_NEXA_F64_PTX_USUBC(reduced[3], sum[3], modulus[3]);
  MOM_NEXA_F64_PTX_USUB(reduced[4], sum[4], 0ULL);
  const uint64_t underflow = reduced[4];
  for (unsigned i = 0; i != 4; ++i)
    r[i] = (reduced[i] & ~underflow) | (sum[i] & underflow);
}

inline void sub_mod(uint64_t* r, const uint64_t* a, const uint64_t* b, const uint64_t* modulus) {
  uint64_t borrow;
  MOM_NEXA_F64_PTX_USUBO(r[0], a[0], b[0]);
  MOM_NEXA_F64_PTX_USUBC(r[1], a[1], b[1]);
  MOM_NEXA_F64_PTX_USUBC(r[2], a[2], b[2]);
  MOM_NEXA_F64_PTX_USUBC(r[3], a[3], b[3]);
  MOM_NEXA_F64_PTX_USUB(borrow, 0ULL, 0ULL);
  MOM_NEXA_F64_PTX_UADDO1(r[0], modulus[0] & borrow);
  MOM_NEXA_F64_PTX_UADDC1(r[1], modulus[1] & borrow);
  MOM_NEXA_F64_PTX_UADDC1(r[2], modulus[2] & borrow);
  MOM_NEXA_F64_PTX_UADD1(r[3], modulus[3] & borrow);
}

inline void mul(uint64_t* r, const uint64_t* a, const uint64_t* b) {
  detail::mom_nexa_field64_ptx_mod_mult(r, a, b);
}

inline void square(uint64_t* r, const uint64_t* a) {
  detail::mom_nexa_field64_ptx_mod_sqr(r, a);
}

inline void inverse(uint64_t r[4]) {
  uint64_t work[5] = {r[0], r[1], r[2], r[3], 0ULL};
  detail::mom_nexa_field64_ptx_mod_inv(work);
  r[0] = work[0];
  r[1] = work[1];
  r[2] = work[2];
  r[3] = work[3];
}

inline bool is_quadratic_residue(const uint64_t r[4]) {
  return detail::mom_nexa_field64_ptx_is_quad(r);
}

} // namespace mom_nexa_field64_ptx

#undef MOM_NEXA_F64_PTX_UADDO
#undef MOM_NEXA_F64_PTX_UADDC
#undef MOM_NEXA_F64_PTX_UADD
#undef MOM_NEXA_F64_PTX_UADDO1
#undef MOM_NEXA_F64_PTX_UADDC1
#undef MOM_NEXA_F64_PTX_UADD1
#undef MOM_NEXA_F64_PTX_USUBO
#undef MOM_NEXA_F64_PTX_USUBC
#undef MOM_NEXA_F64_PTX_USUB
#undef MOM_NEXA_F64_PTX_USUBO1
#undef MOM_NEXA_F64_PTX_USUBC1
#undef MOM_NEXA_F64_PTX_USUB1
#undef MOM_NEXA_F64_PTX_UMULLO
#undef MOM_NEXA_F64_PTX_UMULHI
#undef MOM_NEXA_F64_PTX_MADDO
#undef MOM_NEXA_F64_PTX_MADDC
#undef MOM_NEXA_F64_PTX_MADD
#undef MOM_NEXA_F64_PTX_MADDS
#undef MOM_NEXA_F64_PTX_IS_POSITIVE
#undef MOM_NEXA_F64_PTX_SUBP
#undef MOM_NEXA_F64_PTX_UMULT
#undef MOM_NEXA_F64_PTX_LOAD256
#undef MOM_NEXA_F64_PTX_MM64
#undef MOM_NEXA_F64_PTX_NBBLOCK
#undef MOM_NEXA_F64_PTX_MSK62
#undef MOM_NEXA_F64_PTX_IS_NEGATIVE
#undef MOM_NEXA_F64_PTX_IS_ZERO
#undef MOM_NEXA_F64_PTX_IS_ONE
#undef MOM_NEXA_F64_PTX_SLEFT128
#undef MOM_NEXA_F64_PTX_ADDP
#undef MOM_NEXA_F64_PTX_NEG
#undef MOM_NEXA_F64_PTX_LOAD
#undef MOM_NEXA_F64_PTX_SWAP

#endif
