/*
 * Derived mechanically from nvidia-secp256k1-reference/src/field_10x26_impl.h
 * at commit 497f14cbf19e26d486f6e26280b9645c89b2b5a8; copyright (c) 2013, 2014 Pieter Wuille.
 * Distributed under the MIT software license; see
 * https://www.opensource.org/licenses/mit-license.php.
 */
#include <cstdint>

namespace mom_nexa_field10_ref {

inline void mul(uint32_t r[10], const uint32_t a[10], const uint32_t b[10]) {
  uint64_t c, d;
  uint64_t u0, u1, u2, u3, u4, u5, u6, u7, u8;
  uint32_t t9, t1, t0, t2, t3, t4, t5, t6, t7;
  const uint32_t M = 0x3FFFFFFUL, R0 = 0x3D10UL, R1 = 0x400UL;

  /** [... a b c] is a shorthand for ... + a<<52 + b<<26 + c<<0 mod n.
     *  for 0 <= x <= 9, px is a shorthand for sum(a[i]*b[x-i], i=0..x).
     *  for 9 <= x <= 18, px is a shorthand for sum(a[i]*b[x-i], i=(x-9)..9)
     *  Note that [x 0 0 0 0 0 0 0 0 0 0] = [x*R1 x*R0].
     */

  d = (uint64_t)a[0] * b[9] + (uint64_t)a[1] * b[8] + (uint64_t)a[2] * b[7] +
      (uint64_t)a[3] * b[6] + (uint64_t)a[4] * b[5] + (uint64_t)a[5] * b[4] +
      (uint64_t)a[6] * b[3] + (uint64_t)a[7] * b[2] + (uint64_t)a[8] * b[1] + (uint64_t)a[9] * b[0];
  /* [d 0 0 0 0 0 0 0 0 0] = [p9 0 0 0 0 0 0 0 0 0] */
  t9 = d & M;
  d >>= 26;
  /* [d t9 0 0 0 0 0 0 0 0 0] = [p9 0 0 0 0 0 0 0 0 0] */

  c = (uint64_t)a[0] * b[0];
  /* [d t9 0 0 0 0 0 0 0 0 c] = [p9 0 0 0 0 0 0 0 0 p0] */
  d += (uint64_t)a[1] * b[9] + (uint64_t)a[2] * b[8] + (uint64_t)a[3] * b[7] +
       (uint64_t)a[4] * b[6] + (uint64_t)a[5] * b[5] + (uint64_t)a[6] * b[4] +
       (uint64_t)a[7] * b[3] + (uint64_t)a[8] * b[2] + (uint64_t)a[9] * b[1];
  /* [d t9 0 0 0 0 0 0 0 0 c] = [p10 p9 0 0 0 0 0 0 0 0 p0] */
  u0 = d & M;
  d >>= 26;
  c += u0 * R0;
  /* [d u0 t9 0 0 0 0 0 0 0 0 c-u0*R0] = [p10 p9 0 0 0 0 0 0 0 0 p0] */
  t0 = c & M;
  c >>= 26;
  c += u0 * R1;
  /* [d u0 t9 0 0 0 0 0 0 0 c-u0*R1 t0-u0*R0] = [p10 p9 0 0 0 0 0 0 0 0 p0] */
  /* [d 0 t9 0 0 0 0 0 0 0 c t0] = [p10 p9 0 0 0 0 0 0 0 0 p0] */

  c += (uint64_t)a[0] * b[1] + (uint64_t)a[1] * b[0];
  /* [d 0 t9 0 0 0 0 0 0 0 c t0] = [p10 p9 0 0 0 0 0 0 0 p1 p0] */
  d += (uint64_t)a[2] * b[9] + (uint64_t)a[3] * b[8] + (uint64_t)a[4] * b[7] +
       (uint64_t)a[5] * b[6] + (uint64_t)a[6] * b[5] + (uint64_t)a[7] * b[4] +
       (uint64_t)a[8] * b[3] + (uint64_t)a[9] * b[2];
  /* [d 0 t9 0 0 0 0 0 0 0 c t0] = [p11 p10 p9 0 0 0 0 0 0 0 p1 p0] */
  u1 = d & M;
  d >>= 26;
  c += u1 * R0;
  /* [d u1 0 t9 0 0 0 0 0 0 0 c-u1*R0 t0] = [p11 p10 p9 0 0 0 0 0 0 0 p1 p0] */
  t1 = c & M;
  c >>= 26;
  c += u1 * R1;
  /* [d u1 0 t9 0 0 0 0 0 0 c-u1*R1 t1-u1*R0 t0] = [p11 p10 p9 0 0 0 0 0 0 0 p1 p0] */
  /* [d 0 0 t9 0 0 0 0 0 0 c t1 t0] = [p11 p10 p9 0 0 0 0 0 0 0 p1 p0] */

  c += (uint64_t)a[0] * b[2] + (uint64_t)a[1] * b[1] + (uint64_t)a[2] * b[0];
  /* [d 0 0 t9 0 0 0 0 0 0 c t1 t0] = [p11 p10 p9 0 0 0 0 0 0 p2 p1 p0] */
  d += (uint64_t)a[3] * b[9] + (uint64_t)a[4] * b[8] + (uint64_t)a[5] * b[7] +
       (uint64_t)a[6] * b[6] + (uint64_t)a[7] * b[5] + (uint64_t)a[8] * b[4] +
       (uint64_t)a[9] * b[3];
  /* [d 0 0 t9 0 0 0 0 0 0 c t1 t0] = [p12 p11 p10 p9 0 0 0 0 0 0 p2 p1 p0] */
  u2 = d & M;
  d >>= 26;
  c += u2 * R0;
  /* [d u2 0 0 t9 0 0 0 0 0 0 c-u2*R0 t1 t0] = [p12 p11 p10 p9 0 0 0 0 0 0 p2 p1 p0] */
  t2 = c & M;
  c >>= 26;
  c += u2 * R1;
  /* [d u2 0 0 t9 0 0 0 0 0 c-u2*R1 t2-u2*R0 t1 t0] = [p12 p11 p10 p9 0 0 0 0 0 0 p2 p1 p0] */
  /* [d 0 0 0 t9 0 0 0 0 0 c t2 t1 t0] = [p12 p11 p10 p9 0 0 0 0 0 0 p2 p1 p0] */

  c +=
      (uint64_t)a[0] * b[3] + (uint64_t)a[1] * b[2] + (uint64_t)a[2] * b[1] + (uint64_t)a[3] * b[0];
  /* [d 0 0 0 t9 0 0 0 0 0 c t2 t1 t0] = [p12 p11 p10 p9 0 0 0 0 0 p3 p2 p1 p0] */
  d += (uint64_t)a[4] * b[9] + (uint64_t)a[5] * b[8] + (uint64_t)a[6] * b[7] +
       (uint64_t)a[7] * b[6] + (uint64_t)a[8] * b[5] + (uint64_t)a[9] * b[4];
  /* [d 0 0 0 t9 0 0 0 0 0 c t2 t1 t0] = [p13 p12 p11 p10 p9 0 0 0 0 0 p3 p2 p1 p0] */
  u3 = d & M;
  d >>= 26;
  c += u3 * R0;
  /* [d u3 0 0 0 t9 0 0 0 0 0 c-u3*R0 t2 t1 t0] = [p13 p12 p11 p10 p9 0 0 0 0 0 p3 p2 p1 p0] */
  t3 = c & M;
  c >>= 26;
  c += u3 * R1;
  /* [d u3 0 0 0 t9 0 0 0 0 c-u3*R1 t3-u3*R0 t2 t1 t0] = [p13 p12 p11 p10 p9 0 0 0 0 0 p3 p2 p1 p0] */
  /* [d 0 0 0 0 t9 0 0 0 0 c t3 t2 t1 t0] = [p13 p12 p11 p10 p9 0 0 0 0 0 p3 p2 p1 p0] */

  c += (uint64_t)a[0] * b[4] + (uint64_t)a[1] * b[3] + (uint64_t)a[2] * b[2] +
       (uint64_t)a[3] * b[1] + (uint64_t)a[4] * b[0];
  /* [d 0 0 0 0 t9 0 0 0 0 c t3 t2 t1 t0] = [p13 p12 p11 p10 p9 0 0 0 0 p4 p3 p2 p1 p0] */
  d += (uint64_t)a[5] * b[9] + (uint64_t)a[6] * b[8] + (uint64_t)a[7] * b[7] +
       (uint64_t)a[8] * b[6] + (uint64_t)a[9] * b[5];
  /* [d 0 0 0 0 t9 0 0 0 0 c t3 t2 t1 t0] = [p14 p13 p12 p11 p10 p9 0 0 0 0 p4 p3 p2 p1 p0] */
  u4 = d & M;
  d >>= 26;
  c += u4 * R0;
  /* [d u4 0 0 0 0 t9 0 0 0 0 c-u4*R0 t3 t2 t1 t0] = [p14 p13 p12 p11 p10 p9 0 0 0 0 p4 p3 p2 p1 p0] */
  t4 = c & M;
  c >>= 26;
  c += u4 * R1;
  /* [d u4 0 0 0 0 t9 0 0 0 c-u4*R1 t4-u4*R0 t3 t2 t1 t0] = [p14 p13 p12 p11 p10 p9 0 0 0 0 p4 p3 p2 p1 p0] */
  /* [d 0 0 0 0 0 t9 0 0 0 c t4 t3 t2 t1 t0] = [p14 p13 p12 p11 p10 p9 0 0 0 0 p4 p3 p2 p1 p0] */

  c += (uint64_t)a[0] * b[5] + (uint64_t)a[1] * b[4] + (uint64_t)a[2] * b[3] +
       (uint64_t)a[3] * b[2] + (uint64_t)a[4] * b[1] + (uint64_t)a[5] * b[0];
  /* [d 0 0 0 0 0 t9 0 0 0 c t4 t3 t2 t1 t0] = [p14 p13 p12 p11 p10 p9 0 0 0 p5 p4 p3 p2 p1 p0] */
  d +=
      (uint64_t)a[6] * b[9] + (uint64_t)a[7] * b[8] + (uint64_t)a[8] * b[7] + (uint64_t)a[9] * b[6];
  /* [d 0 0 0 0 0 t9 0 0 0 c t4 t3 t2 t1 t0] = [p15 p14 p13 p12 p11 p10 p9 0 0 0 p5 p4 p3 p2 p1 p0] */
  u5 = d & M;
  d >>= 26;
  c += u5 * R0;
  /* [d u5 0 0 0 0 0 t9 0 0 0 c-u5*R0 t4 t3 t2 t1 t0] = [p15 p14 p13 p12 p11 p10 p9 0 0 0 p5 p4 p3 p2 p1 p0] */
  t5 = c & M;
  c >>= 26;
  c += u5 * R1;
  /* [d u5 0 0 0 0 0 t9 0 0 c-u5*R1 t5-u5*R0 t4 t3 t2 t1 t0] = [p15 p14 p13 p12 p11 p10 p9 0 0 0 p5 p4 p3 p2 p1 p0] */
  /* [d 0 0 0 0 0 0 t9 0 0 c t5 t4 t3 t2 t1 t0] = [p15 p14 p13 p12 p11 p10 p9 0 0 0 p5 p4 p3 p2 p1 p0] */

  c += (uint64_t)a[0] * b[6] + (uint64_t)a[1] * b[5] + (uint64_t)a[2] * b[4] +
       (uint64_t)a[3] * b[3] + (uint64_t)a[4] * b[2] + (uint64_t)a[5] * b[1] +
       (uint64_t)a[6] * b[0];
  /* [d 0 0 0 0 0 0 t9 0 0 c t5 t4 t3 t2 t1 t0] = [p15 p14 p13 p12 p11 p10 p9 0 0 p6 p5 p4 p3 p2 p1 p0] */
  d += (uint64_t)a[7] * b[9] + (uint64_t)a[8] * b[8] + (uint64_t)a[9] * b[7];
  /* [d 0 0 0 0 0 0 t9 0 0 c t5 t4 t3 t2 t1 t0] = [p16 p15 p14 p13 p12 p11 p10 p9 0 0 p6 p5 p4 p3 p2 p1 p0] */
  u6 = d & M;
  d >>= 26;
  c += u6 * R0;
  /* [d u6 0 0 0 0 0 0 t9 0 0 c-u6*R0 t5 t4 t3 t2 t1 t0] = [p16 p15 p14 p13 p12 p11 p10 p9 0 0 p6 p5 p4 p3 p2 p1 p0] */
  t6 = c & M;
  c >>= 26;
  c += u6 * R1;
  /* [d u6 0 0 0 0 0 0 t9 0 c-u6*R1 t6-u6*R0 t5 t4 t3 t2 t1 t0] = [p16 p15 p14 p13 p12 p11 p10 p9 0 0 p6 p5 p4 p3 p2 p1 p0] */
  /* [d 0 0 0 0 0 0 0 t9 0 c t6 t5 t4 t3 t2 t1 t0] = [p16 p15 p14 p13 p12 p11 p10 p9 0 0 p6 p5 p4 p3 p2 p1 p0] */

  c += (uint64_t)a[0] * b[7] + (uint64_t)a[1] * b[6] + (uint64_t)a[2] * b[5] +
       (uint64_t)a[3] * b[4] + (uint64_t)a[4] * b[3] + (uint64_t)a[5] * b[2] +
       (uint64_t)a[6] * b[1] + (uint64_t)a[7] * b[0];
  /* [d 0 0 0 0 0 0 0 t9 0 c t6 t5 t4 t3 t2 t1 t0] = [p16 p15 p14 p13 p12 p11 p10 p9 0 p7 p6 p5 p4 p3 p2 p1 p0] */
  d += (uint64_t)a[8] * b[9] + (uint64_t)a[9] * b[8];
  /* [d 0 0 0 0 0 0 0 t9 0 c t6 t5 t4 t3 t2 t1 t0] = [p17 p16 p15 p14 p13 p12 p11 p10 p9 0 p7 p6 p5 p4 p3 p2 p1 p0] */
  u7 = d & M;
  d >>= 26;
  c += u7 * R0;
  /* [d u7 0 0 0 0 0 0 0 t9 0 c-u7*R0 t6 t5 t4 t3 t2 t1 t0] = [p17 p16 p15 p14 p13 p12 p11 p10 p9 0 p7 p6 p5 p4 p3 p2 p1 p0] */
  t7 = c & M;
  c >>= 26;
  c += u7 * R1;
  /* [d u7 0 0 0 0 0 0 0 t9 c-u7*R1 t7-u7*R0 t6 t5 t4 t3 t2 t1 t0] = [p17 p16 p15 p14 p13 p12 p11 p10 p9 0 p7 p6 p5 p4 p3 p2 p1 p0] */
  /* [d 0 0 0 0 0 0 0 0 t9 c t7 t6 t5 t4 t3 t2 t1 t0] = [p17 p16 p15 p14 p13 p12 p11 p10 p9 0 p7 p6 p5 p4 p3 p2 p1 p0] */

  c += (uint64_t)a[0] * b[8] + (uint64_t)a[1] * b[7] + (uint64_t)a[2] * b[6] +
       (uint64_t)a[3] * b[5] + (uint64_t)a[4] * b[4] + (uint64_t)a[5] * b[3] +
       (uint64_t)a[6] * b[2] + (uint64_t)a[7] * b[1] + (uint64_t)a[8] * b[0];
  /* [d 0 0 0 0 0 0 0 0 t9 c t7 t6 t5 t4 t3 t2 t1 t0] = [p17 p16 p15 p14 p13 p12 p11 p10 p9 p8 p7 p6 p5 p4 p3 p2 p1 p0] */
  d += (uint64_t)a[9] * b[9];
  /* [d 0 0 0 0 0 0 0 0 t9 c t7 t6 t5 t4 t3 t2 t1 t0] = [p18 p17 p16 p15 p14 p13 p12 p11 p10 p9 p8 p7 p6 p5 p4 p3 p2 p1 p0] */
  u8 = d & M;
  d >>= 26;
  c += u8 * R0;
  /* [d u8 0 0 0 0 0 0 0 0 t9 c-u8*R0 t7 t6 t5 t4 t3 t2 t1 t0] = [p18 p17 p16 p15 p14 p13 p12 p11 p10 p9 p8 p7 p6 p5 p4 p3 p2 p1 p0] */

  r[3] = t3;
  /* [d u8 0 0 0 0 0 0 0 0 t9 c-u8*R0 t7 t6 t5 t4 r3 t2 t1 t0] = [p18 p17 p16 p15 p14 p13 p12 p11 p10 p9 p8 p7 p6 p5 p4 p3 p2 p1 p0] */
  r[4] = t4;
  /* [d u8 0 0 0 0 0 0 0 0 t9 c-u8*R0 t7 t6 t5 r4 r3 t2 t1 t0] = [p18 p17 p16 p15 p14 p13 p12 p11 p10 p9 p8 p7 p6 p5 p4 p3 p2 p1 p0] */
  r[5] = t5;
  /* [d u8 0 0 0 0 0 0 0 0 t9 c-u8*R0 t7 t6 r5 r4 r3 t2 t1 t0] = [p18 p17 p16 p15 p14 p13 p12 p11 p10 p9 p8 p7 p6 p5 p4 p3 p2 p1 p0] */
  r[6] = t6;
  /* [d u8 0 0 0 0 0 0 0 0 t9 c-u8*R0 t7 r6 r5 r4 r3 t2 t1 t0] = [p18 p17 p16 p15 p14 p13 p12 p11 p10 p9 p8 p7 p6 p5 p4 p3 p2 p1 p0] */
  r[7] = t7;
  /* [d u8 0 0 0 0 0 0 0 0 t9 c-u8*R0 r7 r6 r5 r4 r3 t2 t1 t0] = [p18 p17 p16 p15 p14 p13 p12 p11 p10 p9 p8 p7 p6 p5 p4 p3 p2 p1 p0] */

  r[8] = c & M;
  c >>= 26;
  c += u8 * R1;
  /* [d u8 0 0 0 0 0 0 0 0 t9+c-u8*R1 r8-u8*R0 r7 r6 r5 r4 r3 t2 t1 t0] = [p18 p17 p16 p15 p14 p13 p12 p11 p10 p9 p8 p7 p6 p5 p4 p3 p2 p1 p0] */
  /* [d 0 0 0 0 0 0 0 0 0 t9+c r8 r7 r6 r5 r4 r3 t2 t1 t0] = [p18 p17 p16 p15 p14 p13 p12 p11 p10 p9 p8 p7 p6 p5 p4 p3 p2 p1 p0] */
  c += d * R0 + t9;
  /* [d 0 0 0 0 0 0 0 0 0 c-d*R0 r8 r7 r6 r5 r4 r3 t2 t1 t0] = [p18 p17 p16 p15 p14 p13 p12 p11 p10 p9 p8 p7 p6 p5 p4 p3 p2 p1 p0] */
  r[9] = c & (M >> 4);
  c >>= 22;
  c += d * (R1 << 4);
  /* [d 0 0 0 0 0 0 0 0 r9+((c-d*R1<<4)<<22)-d*R0 r8 r7 r6 r5 r4 r3 t2 t1 t0] = [p18 p17 p16 p15 p14 p13 p12 p11 p10 p9 p8 p7 p6 p5 p4 p3 p2 p1 p0] */
  /* [d 0 0 0 0 0 0 0 -d*R1 r9+(c<<22)-d*R0 r8 r7 r6 r5 r4 r3 t2 t1 t0] = [p18 p17 p16 p15 p14 p13 p12 p11 p10 p9 p8 p7 p6 p5 p4 p3 p2 p1 p0] */
  /* [r9+(c<<22) r8 r7 r6 r5 r4 r3 t2 t1 t0] = [p18 p17 p16 p15 p14 p13 p12 p11 p10 p9 p8 p7 p6 p5 p4 p3 p2 p1 p0] */

  d = c * (R0 >> 4) + t0;
  /* [r9+(c<<22) r8 r7 r6 r5 r4 r3 t2 t1 d-c*R0>>4] = [p18 p17 p16 p15 p14 p13 p12 p11 p10 p9 p8 p7 p6 p5 p4 p3 p2 p1 p0] */
  r[0] = d & M;
  d >>= 26;
  /* [r9+(c<<22) r8 r7 r6 r5 r4 r3 t2 t1+d r0-c*R0>>4] = [p18 p17 p16 p15 p14 p13 p12 p11 p10 p9 p8 p7 p6 p5 p4 p3 p2 p1 p0] */
  d += c * (R1 >> 4) + t1;
  /* [r9+(c<<22) r8 r7 r6 r5 r4 r3 t2 d-c*R1>>4 r0-c*R0>>4] = [p18 p17 p16 p15 p14 p13 p12 p11 p10 p9 p8 p7 p6 p5 p4 p3 p2 p1 p0] */
  /* [r9 r8 r7 r6 r5 r4 r3 t2 d r0] = [p18 p17 p16 p15 p14 p13 p12 p11 p10 p9 p8 p7 p6 p5 p4 p3 p2 p1 p0] */
  r[1] = d & M;
  d >>= 26;
  /* [r9 r8 r7 r6 r5 r4 r3 t2+d r1 r0] = [p18 p17 p16 p15 p14 p13 p12 p11 p10 p9 p8 p7 p6 p5 p4 p3 p2 p1 p0] */
  d += t2;
  /* [r9 r8 r7 r6 r5 r4 r3 d r1 r0] = [p18 p17 p16 p15 p14 p13 p12 p11 p10 p9 p8 p7 p6 p5 p4 p3 p2 p1 p0] */
  r[2] = d;
  /* [r9 r8 r7 r6 r5 r4 r3 r2 r1 r0] = [p18 p17 p16 p15 p14 p13 p12 p11 p10 p9 p8 p7 p6 p5 p4 p3 p2 p1 p0] */
}

inline void square(uint32_t r[10], const uint32_t a[10]) {
  uint64_t c, d;
  uint64_t u0, u1, u2, u3, u4, u5, u6, u7, u8;
  uint32_t t9, t0, t1, t2, t3, t4, t5, t6, t7;
  const uint32_t M = 0x3FFFFFFUL, R0 = 0x3D10UL, R1 = 0x400UL;

  /** [... a b c] is a shorthand for ... + a<<52 + b<<26 + c<<0 mod n.
     *  px is a shorthand for sum(a[i]*a[x-i], i=0..x).
     *  Note that [x 0 0 0 0 0 0 0 0 0 0] = [x*R1 x*R0].
     */

  d = (uint64_t)(a[0] * 2) * a[9] + (uint64_t)(a[1] * 2) * a[8] + (uint64_t)(a[2] * 2) * a[7] +
      (uint64_t)(a[3] * 2) * a[6] + (uint64_t)(a[4] * 2) * a[5];
  /* [d 0 0 0 0 0 0 0 0 0] = [p9 0 0 0 0 0 0 0 0 0] */
  t9 = d & M;
  d >>= 26;
  /* [d t9 0 0 0 0 0 0 0 0 0] = [p9 0 0 0 0 0 0 0 0 0] */

  c = (uint64_t)a[0] * a[0];
  /* [d t9 0 0 0 0 0 0 0 0 c] = [p9 0 0 0 0 0 0 0 0 p0] */
  d += (uint64_t)(a[1] * 2) * a[9] + (uint64_t)(a[2] * 2) * a[8] + (uint64_t)(a[3] * 2) * a[7] +
       (uint64_t)(a[4] * 2) * a[6] + (uint64_t)a[5] * a[5];
  /* [d t9 0 0 0 0 0 0 0 0 c] = [p10 p9 0 0 0 0 0 0 0 0 p0] */
  u0 = d & M;
  d >>= 26;
  c += u0 * R0;
  /* [d u0 t9 0 0 0 0 0 0 0 0 c-u0*R0] = [p10 p9 0 0 0 0 0 0 0 0 p0] */
  t0 = c & M;
  c >>= 26;
  c += u0 * R1;
  /* [d u0 t9 0 0 0 0 0 0 0 c-u0*R1 t0-u0*R0] = [p10 p9 0 0 0 0 0 0 0 0 p0] */
  /* [d 0 t9 0 0 0 0 0 0 0 c t0] = [p10 p9 0 0 0 0 0 0 0 0 p0] */

  c += (uint64_t)(a[0] * 2) * a[1];
  /* [d 0 t9 0 0 0 0 0 0 0 c t0] = [p10 p9 0 0 0 0 0 0 0 p1 p0] */
  d += (uint64_t)(a[2] * 2) * a[9] + (uint64_t)(a[3] * 2) * a[8] + (uint64_t)(a[4] * 2) * a[7] +
       (uint64_t)(a[5] * 2) * a[6];
  /* [d 0 t9 0 0 0 0 0 0 0 c t0] = [p11 p10 p9 0 0 0 0 0 0 0 p1 p0] */
  u1 = d & M;
  d >>= 26;
  c += u1 * R0;
  /* [d u1 0 t9 0 0 0 0 0 0 0 c-u1*R0 t0] = [p11 p10 p9 0 0 0 0 0 0 0 p1 p0] */
  t1 = c & M;
  c >>= 26;
  c += u1 * R1;
  /* [d u1 0 t9 0 0 0 0 0 0 c-u1*R1 t1-u1*R0 t0] = [p11 p10 p9 0 0 0 0 0 0 0 p1 p0] */
  /* [d 0 0 t9 0 0 0 0 0 0 c t1 t0] = [p11 p10 p9 0 0 0 0 0 0 0 p1 p0] */

  c += (uint64_t)(a[0] * 2) * a[2] + (uint64_t)a[1] * a[1];
  /* [d 0 0 t9 0 0 0 0 0 0 c t1 t0] = [p11 p10 p9 0 0 0 0 0 0 p2 p1 p0] */
  d += (uint64_t)(a[3] * 2) * a[9] + (uint64_t)(a[4] * 2) * a[8] + (uint64_t)(a[5] * 2) * a[7] +
       (uint64_t)a[6] * a[6];
  /* [d 0 0 t9 0 0 0 0 0 0 c t1 t0] = [p12 p11 p10 p9 0 0 0 0 0 0 p2 p1 p0] */
  u2 = d & M;
  d >>= 26;
  c += u2 * R0;
  /* [d u2 0 0 t9 0 0 0 0 0 0 c-u2*R0 t1 t0] = [p12 p11 p10 p9 0 0 0 0 0 0 p2 p1 p0] */
  t2 = c & M;
  c >>= 26;
  c += u2 * R1;
  /* [d u2 0 0 t9 0 0 0 0 0 c-u2*R1 t2-u2*R0 t1 t0] = [p12 p11 p10 p9 0 0 0 0 0 0 p2 p1 p0] */
  /* [d 0 0 0 t9 0 0 0 0 0 c t2 t1 t0] = [p12 p11 p10 p9 0 0 0 0 0 0 p2 p1 p0] */

  c += (uint64_t)(a[0] * 2) * a[3] + (uint64_t)(a[1] * 2) * a[2];
  /* [d 0 0 0 t9 0 0 0 0 0 c t2 t1 t0] = [p12 p11 p10 p9 0 0 0 0 0 p3 p2 p1 p0] */
  d += (uint64_t)(a[4] * 2) * a[9] + (uint64_t)(a[5] * 2) * a[8] + (uint64_t)(a[6] * 2) * a[7];
  /* [d 0 0 0 t9 0 0 0 0 0 c t2 t1 t0] = [p13 p12 p11 p10 p9 0 0 0 0 0 p3 p2 p1 p0] */
  u3 = d & M;
  d >>= 26;
  c += u3 * R0;
  /* [d u3 0 0 0 t9 0 0 0 0 0 c-u3*R0 t2 t1 t0] = [p13 p12 p11 p10 p9 0 0 0 0 0 p3 p2 p1 p0] */
  t3 = c & M;
  c >>= 26;
  c += u3 * R1;
  /* [d u3 0 0 0 t9 0 0 0 0 c-u3*R1 t3-u3*R0 t2 t1 t0] = [p13 p12 p11 p10 p9 0 0 0 0 0 p3 p2 p1 p0] */
  /* [d 0 0 0 0 t9 0 0 0 0 c t3 t2 t1 t0] = [p13 p12 p11 p10 p9 0 0 0 0 0 p3 p2 p1 p0] */

  c += (uint64_t)(a[0] * 2) * a[4] + (uint64_t)(a[1] * 2) * a[3] + (uint64_t)a[2] * a[2];
  /* [d 0 0 0 0 t9 0 0 0 0 c t3 t2 t1 t0] = [p13 p12 p11 p10 p9 0 0 0 0 p4 p3 p2 p1 p0] */
  d += (uint64_t)(a[5] * 2) * a[9] + (uint64_t)(a[6] * 2) * a[8] + (uint64_t)a[7] * a[7];
  /* [d 0 0 0 0 t9 0 0 0 0 c t3 t2 t1 t0] = [p14 p13 p12 p11 p10 p9 0 0 0 0 p4 p3 p2 p1 p0] */
  u4 = d & M;
  d >>= 26;
  c += u4 * R0;
  /* [d u4 0 0 0 0 t9 0 0 0 0 c-u4*R0 t3 t2 t1 t0] = [p14 p13 p12 p11 p10 p9 0 0 0 0 p4 p3 p2 p1 p0] */
  t4 = c & M;
  c >>= 26;
  c += u4 * R1;
  /* [d u4 0 0 0 0 t9 0 0 0 c-u4*R1 t4-u4*R0 t3 t2 t1 t0] = [p14 p13 p12 p11 p10 p9 0 0 0 0 p4 p3 p2 p1 p0] */
  /* [d 0 0 0 0 0 t9 0 0 0 c t4 t3 t2 t1 t0] = [p14 p13 p12 p11 p10 p9 0 0 0 0 p4 p3 p2 p1 p0] */

  c += (uint64_t)(a[0] * 2) * a[5] + (uint64_t)(a[1] * 2) * a[4] + (uint64_t)(a[2] * 2) * a[3];
  /* [d 0 0 0 0 0 t9 0 0 0 c t4 t3 t2 t1 t0] = [p14 p13 p12 p11 p10 p9 0 0 0 p5 p4 p3 p2 p1 p0] */
  d += (uint64_t)(a[6] * 2) * a[9] + (uint64_t)(a[7] * 2) * a[8];
  /* [d 0 0 0 0 0 t9 0 0 0 c t4 t3 t2 t1 t0] = [p15 p14 p13 p12 p11 p10 p9 0 0 0 p5 p4 p3 p2 p1 p0] */
  u5 = d & M;
  d >>= 26;
  c += u5 * R0;
  /* [d u5 0 0 0 0 0 t9 0 0 0 c-u5*R0 t4 t3 t2 t1 t0] = [p15 p14 p13 p12 p11 p10 p9 0 0 0 p5 p4 p3 p2 p1 p0] */
  t5 = c & M;
  c >>= 26;
  c += u5 * R1;
  /* [d u5 0 0 0 0 0 t9 0 0 c-u5*R1 t5-u5*R0 t4 t3 t2 t1 t0] = [p15 p14 p13 p12 p11 p10 p9 0 0 0 p5 p4 p3 p2 p1 p0] */
  /* [d 0 0 0 0 0 0 t9 0 0 c t5 t4 t3 t2 t1 t0] = [p15 p14 p13 p12 p11 p10 p9 0 0 0 p5 p4 p3 p2 p1 p0] */

  c += (uint64_t)(a[0] * 2) * a[6] + (uint64_t)(a[1] * 2) * a[5] + (uint64_t)(a[2] * 2) * a[4] +
       (uint64_t)a[3] * a[3];
  /* [d 0 0 0 0 0 0 t9 0 0 c t5 t4 t3 t2 t1 t0] = [p15 p14 p13 p12 p11 p10 p9 0 0 p6 p5 p4 p3 p2 p1 p0] */
  d += (uint64_t)(a[7] * 2) * a[9] + (uint64_t)a[8] * a[8];
  /* [d 0 0 0 0 0 0 t9 0 0 c t5 t4 t3 t2 t1 t0] = [p16 p15 p14 p13 p12 p11 p10 p9 0 0 p6 p5 p4 p3 p2 p1 p0] */
  u6 = d & M;
  d >>= 26;
  c += u6 * R0;
  /* [d u6 0 0 0 0 0 0 t9 0 0 c-u6*R0 t5 t4 t3 t2 t1 t0] = [p16 p15 p14 p13 p12 p11 p10 p9 0 0 p6 p5 p4 p3 p2 p1 p0] */
  t6 = c & M;
  c >>= 26;
  c += u6 * R1;
  /* [d u6 0 0 0 0 0 0 t9 0 c-u6*R1 t6-u6*R0 t5 t4 t3 t2 t1 t0] = [p16 p15 p14 p13 p12 p11 p10 p9 0 0 p6 p5 p4 p3 p2 p1 p0] */
  /* [d 0 0 0 0 0 0 0 t9 0 c t6 t5 t4 t3 t2 t1 t0] = [p16 p15 p14 p13 p12 p11 p10 p9 0 0 p6 p5 p4 p3 p2 p1 p0] */

  c += (uint64_t)(a[0] * 2) * a[7] + (uint64_t)(a[1] * 2) * a[6] + (uint64_t)(a[2] * 2) * a[5] +
       (uint64_t)(a[3] * 2) * a[4];
  /* [d 0 0 0 0 0 0 0 t9 0 c t6 t5 t4 t3 t2 t1 t0] = [p16 p15 p14 p13 p12 p11 p10 p9 0 p7 p6 p5 p4 p3 p2 p1 p0] */
  d += (uint64_t)(a[8] * 2) * a[9];
  /* [d 0 0 0 0 0 0 0 t9 0 c t6 t5 t4 t3 t2 t1 t0] = [p17 p16 p15 p14 p13 p12 p11 p10 p9 0 p7 p6 p5 p4 p3 p2 p1 p0] */
  u7 = d & M;
  d >>= 26;
  c += u7 * R0;
  /* [d u7 0 0 0 0 0 0 0 t9 0 c-u7*R0 t6 t5 t4 t3 t2 t1 t0] = [p17 p16 p15 p14 p13 p12 p11 p10 p9 0 p7 p6 p5 p4 p3 p2 p1 p0] */
  t7 = c & M;
  c >>= 26;
  c += u7 * R1;
  /* [d u7 0 0 0 0 0 0 0 t9 c-u7*R1 t7-u7*R0 t6 t5 t4 t3 t2 t1 t0] = [p17 p16 p15 p14 p13 p12 p11 p10 p9 0 p7 p6 p5 p4 p3 p2 p1 p0] */
  /* [d 0 0 0 0 0 0 0 0 t9 c t7 t6 t5 t4 t3 t2 t1 t0] = [p17 p16 p15 p14 p13 p12 p11 p10 p9 0 p7 p6 p5 p4 p3 p2 p1 p0] */

  c += (uint64_t)(a[0] * 2) * a[8] + (uint64_t)(a[1] * 2) * a[7] + (uint64_t)(a[2] * 2) * a[6] +
       (uint64_t)(a[3] * 2) * a[5] + (uint64_t)a[4] * a[4];
  /* [d 0 0 0 0 0 0 0 0 t9 c t7 t6 t5 t4 t3 t2 t1 t0] = [p17 p16 p15 p14 p13 p12 p11 p10 p9 p8 p7 p6 p5 p4 p3 p2 p1 p0] */
  d += (uint64_t)a[9] * a[9];
  /* [d 0 0 0 0 0 0 0 0 t9 c t7 t6 t5 t4 t3 t2 t1 t0] = [p18 p17 p16 p15 p14 p13 p12 p11 p10 p9 p8 p7 p6 p5 p4 p3 p2 p1 p0] */
  u8 = d & M;
  d >>= 26;
  c += u8 * R0;
  /* [d u8 0 0 0 0 0 0 0 0 t9 c-u8*R0 t7 t6 t5 t4 t3 t2 t1 t0] = [p18 p17 p16 p15 p14 p13 p12 p11 p10 p9 p8 p7 p6 p5 p4 p3 p2 p1 p0] */

  r[3] = t3;
  /* [d u8 0 0 0 0 0 0 0 0 t9 c-u8*R0 t7 t6 t5 t4 r3 t2 t1 t0] = [p18 p17 p16 p15 p14 p13 p12 p11 p10 p9 p8 p7 p6 p5 p4 p3 p2 p1 p0] */
  r[4] = t4;
  /* [d u8 0 0 0 0 0 0 0 0 t9 c-u8*R0 t7 t6 t5 r4 r3 t2 t1 t0] = [p18 p17 p16 p15 p14 p13 p12 p11 p10 p9 p8 p7 p6 p5 p4 p3 p2 p1 p0] */
  r[5] = t5;
  /* [d u8 0 0 0 0 0 0 0 0 t9 c-u8*R0 t7 t6 r5 r4 r3 t2 t1 t0] = [p18 p17 p16 p15 p14 p13 p12 p11 p10 p9 p8 p7 p6 p5 p4 p3 p2 p1 p0] */
  r[6] = t6;
  /* [d u8 0 0 0 0 0 0 0 0 t9 c-u8*R0 t7 r6 r5 r4 r3 t2 t1 t0] = [p18 p17 p16 p15 p14 p13 p12 p11 p10 p9 p8 p7 p6 p5 p4 p3 p2 p1 p0] */
  r[7] = t7;
  /* [d u8 0 0 0 0 0 0 0 0 t9 c-u8*R0 r7 r6 r5 r4 r3 t2 t1 t0] = [p18 p17 p16 p15 p14 p13 p12 p11 p10 p9 p8 p7 p6 p5 p4 p3 p2 p1 p0] */

  r[8] = c & M;
  c >>= 26;
  c += u8 * R1;
  /* [d u8 0 0 0 0 0 0 0 0 t9+c-u8*R1 r8-u8*R0 r7 r6 r5 r4 r3 t2 t1 t0] = [p18 p17 p16 p15 p14 p13 p12 p11 p10 p9 p8 p7 p6 p5 p4 p3 p2 p1 p0] */
  /* [d 0 0 0 0 0 0 0 0 0 t9+c r8 r7 r6 r5 r4 r3 t2 t1 t0] = [p18 p17 p16 p15 p14 p13 p12 p11 p10 p9 p8 p7 p6 p5 p4 p3 p2 p1 p0] */
  c += d * R0 + t9;
  /* [d 0 0 0 0 0 0 0 0 0 c-d*R0 r8 r7 r6 r5 r4 r3 t2 t1 t0] = [p18 p17 p16 p15 p14 p13 p12 p11 p10 p9 p8 p7 p6 p5 p4 p3 p2 p1 p0] */
  r[9] = c & (M >> 4);
  c >>= 22;
  c += d * (R1 << 4);
  /* [d 0 0 0 0 0 0 0 0 r9+((c-d*R1<<4)<<22)-d*R0 r8 r7 r6 r5 r4 r3 t2 t1 t0] = [p18 p17 p16 p15 p14 p13 p12 p11 p10 p9 p8 p7 p6 p5 p4 p3 p2 p1 p0] */
  /* [d 0 0 0 0 0 0 0 -d*R1 r9+(c<<22)-d*R0 r8 r7 r6 r5 r4 r3 t2 t1 t0] = [p18 p17 p16 p15 p14 p13 p12 p11 p10 p9 p8 p7 p6 p5 p4 p3 p2 p1 p0] */
  /* [r9+(c<<22) r8 r7 r6 r5 r4 r3 t2 t1 t0] = [p18 p17 p16 p15 p14 p13 p12 p11 p10 p9 p8 p7 p6 p5 p4 p3 p2 p1 p0] */

  d = c * (R0 >> 4) + t0;
  /* [r9+(c<<22) r8 r7 r6 r5 r4 r3 t2 t1 d-c*R0>>4] = [p18 p17 p16 p15 p14 p13 p12 p11 p10 p9 p8 p7 p6 p5 p4 p3 p2 p1 p0] */
  r[0] = d & M;
  d >>= 26;
  /* [r9+(c<<22) r8 r7 r6 r5 r4 r3 t2 t1+d r0-c*R0>>4] = [p18 p17 p16 p15 p14 p13 p12 p11 p10 p9 p8 p7 p6 p5 p4 p3 p2 p1 p0] */
  d += c * (R1 >> 4) + t1;
  /* [r9+(c<<22) r8 r7 r6 r5 r4 r3 t2 d-c*R1>>4 r0-c*R0>>4] = [p18 p17 p16 p15 p14 p13 p12 p11 p10 p9 p8 p7 p6 p5 p4 p3 p2 p1 p0] */
  /* [r9 r8 r7 r6 r5 r4 r3 t2 d r0] = [p18 p17 p16 p15 p14 p13 p12 p11 p10 p9 p8 p7 p6 p5 p4 p3 p2 p1 p0] */
  r[1] = d & M;
  d >>= 26;
  /* [r9 r8 r7 r6 r5 r4 r3 t2+d r1 r0] = [p18 p17 p16 p15 p14 p13 p12 p11 p10 p9 p8 p7 p6 p5 p4 p3 p2 p1 p0] */
  d += t2;
  /* [r9 r8 r7 r6 r5 r4 r3 d r1 r0] = [p18 p17 p16 p15 p14 p13 p12 p11 p10 p9 p8 p7 p6 p5 p4 p3 p2 p1 p0] */
  r[2] = d;
  /* [r9 r8 r7 r6 r5 r4 r3 r2 r1 r0] = [p18 p17 p16 p15 p14 p13 p12 p11 p10 p9 p8 p7 p6 p5 p4 p3 p2 p1 p0] */
}

} // namespace mom_nexa_field10_ref
