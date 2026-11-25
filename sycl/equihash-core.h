#pragma once

// Portable, backend-neutral Equihash primitives shared by the ZHash and Equihash(192,7)
// adapters. Keeping hashing, packed row layout, bucket/scatter contracts, and proof validation
// here prevents the two algorithms from drifting apart while leaving backend-specific occupancy
// decisions to their existing solvers.

#include <cstddef>
#include <cstdint>

#include "blake2b.h"

namespace mom_equihash {

template <unsigned N, unsigned K, unsigned IndicesPerHash, unsigned HeaderLength = 140>
struct EquihashConfig {
  static constexpr unsigned n = N, k = K, rounds = K + 1;
  static constexpr unsigned collision_bits = N / rounds;
  static constexpr unsigned collision_bytes = (collision_bits + 7) / 8;
  static constexpr unsigned indices_per_hash = IndicesPerHash;
  static constexpr unsigned segment_bytes = rounds * collision_bytes;
  static constexpr unsigned hash_length = indices_per_hash * segment_bytes;
  static constexpr unsigned index_bits = collision_bits + 1;
  static constexpr unsigned proof_indices = 1u << k;
  static constexpr unsigned solution_length = proof_indices * index_bits / 8;
  static constexpr unsigned row_count = 1u << index_bits;
  static constexpr unsigned hash_count = (row_count + indices_per_hash - 1) / indices_per_hash;
  static constexpr unsigned header_length = HeaderLength;
  static constexpr unsigned parent_ref_bits = 32;
  static constexpr unsigned level0_bits = rounds * collision_bits + index_bits;
  static constexpr unsigned level0_words = (level0_bits + 31) / 32;
  static constexpr unsigned collision_record_bits =
      (rounds - 1) * collision_bits + 2 * parent_ref_bits;
  static constexpr unsigned collision_record_words = (collision_record_bits + 31) / 32;

  static_assert(N % rounds == 0, "Equihash n must divide evenly across rounds");
  static_assert(collision_bits <= 24 && collision_bytes <= 3,
                "the packed record format supports at most 24-bit collision fields");
  static_assert((proof_indices * index_bits) % 8 == 0, "proof must be byte aligned");
};

struct ZHash144_5 : EquihashConfig<144, 5, 3> {
  // ZHash uses 16,384 global buckets and 1,024 exact-key bins in each bucket.  The capacity is
  // mean(2^25 / 2^14) + 7.1 sigma, rounded to a cache-friendly multiple of 16.
  static constexpr unsigned bucket_bits = 14, local_bits = 10;
  static constexpr unsigned bucket_count = 1u << bucket_bits, local_bins = 1u << local_bits;
  static constexpr unsigned bucket_slot_capacity = 2368;
  static constexpr unsigned collision_work_group = 1024;
  static constexpr unsigned parent_ref_bits = 26;
  static constexpr unsigned collision_record_bits =
      (rounds - 1) * collision_bits + 2 * parent_ref_bits;
  static constexpr unsigned collision_record_words = (collision_record_bits + 31) / 32;
  static constexpr unsigned bucket_for(const std::uint32_t field) {
    return field >> local_bits;
  }
  static constexpr unsigned local_bin(const std::uint32_t field) {
    return field & (local_bins - 1u);
  }
  static_assert(bucket_bits + local_bits == collision_bits, "bucket split must cover the collision key");

  static constexpr std::uint8_t personal_byte(const unsigned i) {
    return i == 0 ? 'B' : i == 1 ? 'g' : i == 2 ? 'o' : i == 3 ? 'l' :
           i == 4 ? 'd' : i == 5 ? 'P' : i == 6 ? 'o' : i == 7 ? 'W' :
           i == 8 ? 144 : i == 12 ? 5 : 0;
  }
};

struct Equihash192_7 : EquihashConfig<192, 7, 2> {

  static constexpr std::uint8_t personal_byte(const unsigned i) {
    return i == 0 ? 'Z' : i == 1 ? 'c' : i == 2 ? 'a' : i == 3 ? 's' :
           i == 4 ? 'h' : i == 5 ? 'P' : i == 6 ? 'o' : i == 7 ? 'W' :
           i == 8 ? 192 : i == 12 ? 7 : i == 9 ? 0 : i == 10 ? 0 : i == 11 ? 0 :
           i == 13 ? 0 : i == 14 ? 0 : i == 15 ? 0 : 0;
  }
};

template <typename Spec>
struct Row {
  std::uint32_t fields[Spec::rounds];
  std::uint32_t first_index;
};

template <typename Spec>
using Node = Row<Spec>;

// Preserve compact natural record strides where backend measurements favor them over wider alignment.
constexpr unsigned packed_record_alignment(const unsigned words) {
  return words == 4 ? 16 : words >= 5 ? 8 : 4;
}

template <typename Spec, unsigned FieldCount = Spec::rounds - 1>
struct alignas(packed_record_alignment(
    (FieldCount * Spec::collision_bits + 2 * Spec::parent_ref_bits + 31) / 32))
    PackedCollisionRecord {
  static constexpr unsigned field_count = FieldCount;
  static constexpr unsigned bit_count =
      FieldCount * Spec::collision_bits + 2 * Spec::parent_ref_bits;
  static constexpr unsigned word_count = (bit_count + 31) / 32;
  std::uint32_t words[word_count];
};

// A round number identifies the number of collisions already performed.  Round one records have
// five active fields for ZHash; round five records have one active field, whose zero value marks a
// complete Equihash root.  Parent references always point into the immediately preceding round's
// bucket arena (round zero is the level-zero arena).
template <typename Spec, unsigned Round>
using CollisionRoundRecord = PackedCollisionRecord<Spec, Spec::rounds - Round>;

// Select collision-record alignment without changing its bit layout or parent-reference width.
// The Intel profile is explicit: upstream DPC++ does not necessarily define the branded
// compiler macro. Host and device must select the same layout, never a per-device-pass ABI.
constexpr unsigned packed_bucket_collision_alignment(const unsigned n,
                                                       const unsigned field_count) {
#if defined(MOM_SYCL_HAS_HIP) || defined(__INTEL_LLVM_COMPILER) || defined(MOM_SYCL_INTEL_LAYOUT)
  constexpr unsigned wide_fields = 7;
#else
  constexpr unsigned wide_fields = 6;
#endif
#if defined(MOM_SYCL_HAS_HIP)
  if (n == 144 && (field_count == 2 || field_count == 3)) return 4;
#endif
  if (n == 192) {
#if defined(__INTEL_LLVM_COMPILER) || defined(MOM_SYCL_INTEL_LAYOUT)
    if (field_count == 7) return 4;
#endif
    if (field_count >= wide_fields) return 32;
    if (field_count >= 5) return 8;
#if defined(MOM_SYCL_HAS_HIP) || defined(__INTEL_LLVM_COMPILER) || defined(MOM_SYCL_INTEL_LAYOUT)
    if (field_count == 2) return 4;
#endif
  }
  return field_count >= 5 ? 4 : 16;
}

// Return the bits needed for bounded slot references, including non-power-of-two capacities.
constexpr unsigned bits_for_capacity(std::uint32_t capacity) {
  unsigned bits = 0;
  for (--capacity; capacity != 0; capacity >>= 1) {
    ++bits;
  }
  return bits;
}

// Keep level-zero allocation alignment consistent with the backend's selected record stride.
constexpr unsigned packed_bucket_level0_alignment([[maybe_unused]] const unsigned n,
                                                   [[maybe_unused]] const unsigned k,
                                                   const unsigned word_count) {
#if defined(__INTEL_LLVM_COMPILER) || defined(MOM_SYCL_INTEL_LAYOUT)
  if (n == 192 && k == 7) return 4;
#endif
  return word_count == 7 ? 32 : 4;
}

template <typename Spec, unsigned LocalBits>
struct alignas(packed_bucket_level0_alignment(
    Spec::n, Spec::k,
    (LocalBits + (Spec::rounds - 1) * Spec::collision_bits + Spec::index_bits + 31) / 32))
    PackedBucketLevel0Record {
  static constexpr unsigned local_bits = LocalBits;
  // Tuned ZHash layouts keep the next collision key in the 16-byte head and omit only trailing
  // hash bits. Proof rehashing still validates every result; the omitted bits merely admit a few
  // extra final candidates while avoiding a tail load in every first-round bucket scan.
#ifdef MOM_SYCL_HAS_HIP
  static constexpr bool compact_selection_head =
      Spec::n == 144 && Spec::k == 5 && LocalBits == 12;
#elif defined(MOM_ZHASH_INTEL_LATE_BUCKETS)
  static constexpr bool compact_selection_head =
      Spec::n == 144 && Spec::k == 5 && LocalBits == 11;
#else
  static constexpr bool compact_selection_head = false;
#endif
  static constexpr unsigned bit_count =
      LocalBits + (Spec::rounds - 1) * Spec::collision_bits + Spec::index_bits;
  static constexpr unsigned word_count = (bit_count + 31) / 32;
  std::uint32_t words[word_count];
};

// Later bucket rounds need only the low part of their first field. Parent locations use the
// preceding bucket plus two local slots instead of two full arena offsets.
template <typename Spec, unsigned FieldCount, unsigned LocalBits, unsigned ParentBucketBits,
          unsigned ParentSlots>
struct alignas(packed_bucket_collision_alignment(Spec::n, FieldCount))
    PackedBucketCollisionRecord {
  static constexpr unsigned field_count = FieldCount;
  static constexpr unsigned local_bits = LocalBits;
  static constexpr unsigned parent_bucket_bits = ParentBucketBits;
  static constexpr unsigned parent_slot_bits = bits_for_capacity(ParentSlots);
  static constexpr unsigned full_bit_count =
      LocalBits + (FieldCount - 1) * Spec::collision_bits + ParentBucketBits +
      2 * parent_slot_bits;
#if defined(MOM_SYCL_HAS_HIP)
  // A ZHash round-three record needs only enough of its trailing field to find candidates. The
  // complete proof is rehashed before submission, so fitting the record into 12 bytes can add
  // false candidates but cannot lose or accept an invalid solution.
  static constexpr bool compact_trailing_field =
      Spec::n == 144 && Spec::k == 5 && FieldCount == 3 && full_bit_count >= 96 &&
      full_bit_count < 120;
#else
  static constexpr bool compact_trailing_field = false;
#endif
  static constexpr unsigned trailing_bits_omitted =
      compact_trailing_field ? full_bit_count - 96 : 0;
  static_assert(!compact_trailing_field ||
                (full_bit_count >= 96 && trailing_bits_omitted < Spec::collision_bits));
  static constexpr unsigned bit_count = full_bit_count - trailing_bits_omitted;
  static constexpr unsigned parent_bit =
      LocalBits + (FieldCount - 1) * Spec::collision_bits - trailing_bits_omitted;
  static constexpr unsigned word_count = (bit_count + 31) / 32;
  static_assert(parent_bit + ParentBucketBits + 2 * parent_slot_bits == bit_count);
  std::uint32_t words[word_count];
};

template <typename Spec>
using EquihashRootRecord = CollisionRoundRecord<Spec, Spec::rounds - 1>;

inline std::uint64_t packed_mask(const unsigned bits) {
  return bits == 32 ? 0xffffffffull : ((1ull << bits) - 1ull);
}

// Extract a bounded bit field across adjacent words; absent high words read as zero.
inline std::uint32_t packed_load(const std::uint32_t* const words, const unsigned word_count,
                                 const unsigned bit, const unsigned width) {
  const unsigned word = bit >> 5;
  const unsigned shift = bit & 31u;
  std::uint64_t value = words[word];
  if (shift != 0 && word + 1 < word_count) {
    value |= static_cast<std::uint64_t>(words[word + 1]) << 32;
  }
  return static_cast<std::uint32_t>((value >> shift) & packed_mask(width));
}

// Replace one bit field without disturbing neighboring packed fields, including word crossings.
inline void packed_store(std::uint32_t* const words, const unsigned word_count, const unsigned bit,
                         const unsigned width, const std::uint32_t value) {
  const unsigned word = bit >> 5;
  const unsigned shift = bit & 31u;
  std::uint64_t pair = words[word];
  if (shift != 0 && word + 1 < word_count) {
    pair |= static_cast<std::uint64_t>(words[word + 1]) << 32;
  }
  const std::uint64_t mask = packed_mask(width) << shift;
  pair = (pair & ~mask) | ((static_cast<std::uint64_t>(value) & packed_mask(width)) << shift);
  words[word] = static_cast<std::uint32_t>(pair);
  if (shift != 0 && word + 1 < word_count) {
    words[word + 1] = static_cast<std::uint32_t>(pair >> 32);
  }
}

template <typename Spec, unsigned FieldCount>
inline void clear_packed(PackedCollisionRecord<Spec, FieldCount>& record) {
  for (unsigned i = 0; i < PackedCollisionRecord<Spec, FieldCount>::word_count; ++i)
    record.words[i] = 0;
}

template <typename Spec, unsigned FieldCount>
// Pack remaining XOR fields and both parent references for later provenance recovery.
inline void store_collision(PackedCollisionRecord<Spec, FieldCount>& record,
                            const std::uint32_t fields[FieldCount],
                            const std::uint32_t left_ref, const std::uint32_t right_ref) {
  clear_packed<Spec, FieldCount>(record);
  for (unsigned i = 0; i < FieldCount; ++i)
    packed_store(record.words, PackedCollisionRecord<Spec, FieldCount>::word_count,
                 i * Spec::collision_bits, Spec::collision_bits, fields[i]);
  packed_store(record.words, PackedCollisionRecord<Spec, FieldCount>::word_count,
               FieldCount * Spec::collision_bits, Spec::parent_ref_bits, left_ref);
  packed_store(record.words, PackedCollisionRecord<Spec, FieldCount>::word_count,
               FieldCount * Spec::collision_bits + Spec::parent_ref_bits,
               Spec::parent_ref_bits, right_ref);
}

template <typename Spec, unsigned FieldCount>
inline std::uint32_t load_collision_field(const PackedCollisionRecord<Spec, FieldCount>& record,
                                           const unsigned field) {
  return packed_load(record.words, PackedCollisionRecord<Spec, FieldCount>::word_count,
                     field * Spec::collision_bits, Spec::collision_bits);
}

template <typename Spec, unsigned FieldCount>
inline std::uint32_t load_collision_left(const PackedCollisionRecord<Spec, FieldCount>& record) {
  return packed_load(record.words, PackedCollisionRecord<Spec, FieldCount>::word_count,
                     FieldCount * Spec::collision_bits, Spec::parent_ref_bits);
}

template <typename Spec, unsigned FieldCount>
inline std::uint32_t load_collision_right(const PackedCollisionRecord<Spec, FieldCount>& record) {
  return packed_load(record.words, PackedCollisionRecord<Spec, FieldCount>::word_count,
                     FieldCount * Spec::collision_bits + Spec::parent_ref_bits,
                     Spec::parent_ref_bits);
}

template <typename Spec, unsigned FieldCount, unsigned LocalBits, unsigned ParentBucketBits,
          unsigned ParentSlots>
// Store the bucket-relative collision representation, retaining enough parent bits for exact recovery.
inline void store_bucket_collision(
    PackedBucketCollisionRecord<Spec, FieldCount, LocalBits, ParentBucketBits, ParentSlots>& record,
    const std::uint32_t fields[FieldCount], const std::uint32_t parent_bucket,
    const std::uint32_t left_slot, const std::uint32_t right_slot) {
  using Record = PackedBucketCollisionRecord<Spec, FieldCount, LocalBits, ParentBucketBits,
                                             ParentSlots>;
  if constexpr (Spec::collision_bits == 24 && FieldCount == 5 && LocalBits == 12 &&
                ParentBucketBits == 12 && Record::parent_slot_bits == 14 &&
                Record::word_count == 5) {
    record.words[0] = (fields[0] & 0xfffu) | (fields[1] << 12);
    record.words[1] = (fields[1] >> 20) | (fields[2] << 4) | (fields[3] << 28);
    record.words[2] = (fields[3] >> 4) | (fields[4] << 20);
    record.words[3] = (fields[4] >> 12) | (parent_bucket << 12) | (left_slot << 24);
    record.words[4] = (left_slot >> 8) | (right_slot << 6);
    return;
  }
  for (unsigned i = 0; i < Record::word_count; ++i) {
    record.words[i] = 0;
  }
  packed_store(record.words, Record::word_count, 0, LocalBits, fields[0]);
  if constexpr (Record::compact_trailing_field) {
    for (unsigned i = 1; i + 1 < FieldCount; ++i)
      packed_store(record.words, Record::word_count,
                   LocalBits + (i - 1) * Spec::collision_bits,
                   Spec::collision_bits, fields[i]);
    packed_store(record.words, Record::word_count,
                 LocalBits + (FieldCount - 2) * Spec::collision_bits,
                 Spec::collision_bits - Record::trailing_bits_omitted,
                 fields[FieldCount - 1]);
  } else {
    for (unsigned i = 1; i < FieldCount; ++i)
      packed_store(record.words, Record::word_count,
                   LocalBits + (i - 1) * Spec::collision_bits,
                   Spec::collision_bits, fields[i]);
  }
  packed_store(record.words, Record::word_count, Record::parent_bit,
               ParentBucketBits, parent_bucket);
  packed_store(record.words, Record::word_count, Record::parent_bit + ParentBucketBits,
               Record::parent_slot_bits, left_slot);
  packed_store(record.words, Record::word_count,
               Record::parent_bit + ParentBucketBits + Record::parent_slot_bits,
               Record::parent_slot_bits, right_slot);
}

// Read collision field `field`; field 0 is rebuilt from the record's bucket and its local key bits,
// and a compact trailing field returns only its retained high bits.
template <typename Spec, unsigned FieldCount, unsigned LocalBits, unsigned ParentBucketBits,
          unsigned ParentSlots>
inline std::uint32_t load_bucket_collision_field(
    const PackedBucketCollisionRecord<Spec, FieldCount, LocalBits, ParentBucketBits, ParentSlots>&
        record,
    const std::uint32_t bucket, const unsigned field) {
  using Record = PackedBucketCollisionRecord<Spec, FieldCount, LocalBits, ParentBucketBits,
                                             ParentSlots>;
  if (field == 0)
    return (bucket << LocalBits) | packed_load(record.words, Record::word_count, 0, LocalBits);
  if constexpr (Record::compact_trailing_field) {
    if (field + 1 == FieldCount)
      return packed_load(record.words, Record::word_count,
                         LocalBits + (field - 1) * Spec::collision_bits,
                         Spec::collision_bits - Record::trailing_bits_omitted);
  }
  return packed_load(record.words, Record::word_count,
                     LocalBits + (field - 1) * Spec::collision_bits, Spec::collision_bits);
}

// Return the absolute parent arena slot (left or right) from its bucket-relative reference.
template <typename Spec, unsigned FieldCount, unsigned LocalBits, unsigned ParentBucketBits,
          unsigned ParentSlots>
inline std::uint32_t load_bucket_collision_parent(
    const PackedBucketCollisionRecord<Spec, FieldCount, LocalBits, ParentBucketBits, ParentSlots>&
        record,
    const bool right) {
  using Record = PackedBucketCollisionRecord<Spec, FieldCount, LocalBits, ParentBucketBits,
                                             ParentSlots>;
  const std::uint32_t bucket =
      packed_load(record.words, Record::word_count, Record::parent_bit, ParentBucketBits);
  const unsigned slot_bit =
      Record::parent_bit + ParentBucketBits + right * Record::parent_slot_bits;
  return bucket * ParentSlots +
         packed_load(record.words, Record::word_count, slot_bit, Record::parent_slot_bits);
}

template <typename Spec, unsigned LocalBits>
// Pack bucket-local generation fields and leaf provenance in the selected backend layout.
inline void store_bucket_level0(PackedBucketLevel0Record<Spec, LocalBits>& record,
                                const std::uint32_t fields[Spec::rounds],
                                const std::uint32_t leaf) {
  using Record = PackedBucketLevel0Record<Spec, LocalBits>;
  for (unsigned i = 0; i < Record::word_count; ++i) {
    record.words[i] = 0;
  }
  if constexpr (Record::compact_selection_head) {
    constexpr unsigned final_field_bit =
        LocalBits + (Spec::rounds - 2) * Spec::collision_bits;
    constexpr unsigned retained_final_bits = 128 - final_field_bit;
    packed_store(record.words, Record::word_count, 0, LocalBits, fields[0]);
    for (unsigned i = 1; i + 1 < Spec::rounds; ++i)
      packed_store(record.words, Record::word_count,
                   LocalBits + (i - 1) * Spec::collision_bits,
                   Spec::collision_bits, fields[i]);
    packed_store(record.words, Record::word_count, final_field_bit,
                 retained_final_bits, fields[Spec::rounds - 1]);
    packed_store(record.words, Record::word_count, 128, Spec::index_bits, leaf);
    return;
  }
  // Keep the five fields consumed by the first collision in the 16-byte split head. The local
  // key and leaf live in the tail, which is read sequentially for bucketing and only during rare
  // proof recovery—not for every random collision pair.
  for (unsigned i = 1; i < Spec::rounds; ++i)
    packed_store(record.words, Record::word_count,
                 (i - 1) * Spec::collision_bits, Spec::collision_bits, fields[i]);
  constexpr unsigned local_bit = (Spec::rounds - 1) * Spec::collision_bits;
  packed_store(record.words, Record::word_count, local_bit, LocalBits, fields[0]);
  packed_store(record.words, Record::word_count,
               local_bit + LocalBits, Spec::index_bits, leaf);
}

// Read generated field `field` from either level-zero layout; field 0 is rebuilt from the bucket.
template <typename Spec, unsigned LocalBits>
inline std::uint32_t load_bucket_level0_field(
    const PackedBucketLevel0Record<Spec, LocalBits>& record, const std::uint32_t bucket,
    const unsigned field) {
  using Record = PackedBucketLevel0Record<Spec, LocalBits>;
  if constexpr (Record::compact_selection_head) {
    if (field == 0)
      return (bucket << LocalBits) |
             packed_load(record.words, Record::word_count, 0, LocalBits);
    constexpr unsigned final_field_bit =
        LocalBits + (Spec::rounds - 2) * Spec::collision_bits;
    const unsigned bit = LocalBits + (field - 1) * Spec::collision_bits;
    const unsigned width = field + 1 == Spec::rounds
                               ? 128 - final_field_bit : Spec::collision_bits;
    return packed_load(record.words, Record::word_count, bit, width);
  }
  if (field == 0)
    return (bucket << LocalBits) |
           packed_load(record.words, Record::word_count,
                       (Spec::rounds - 1) * Spec::collision_bits, LocalBits);
  return packed_load(record.words, Record::word_count,
                     (field - 1) * Spec::collision_bits, Spec::collision_bits);
}

// Read the original leaf index retained by a level-zero record.
template <typename Spec, unsigned LocalBits>
inline std::uint32_t load_bucket_level0_leaf(
    const PackedBucketLevel0Record<Spec, LocalBits>& record) {
  using Record = PackedBucketLevel0Record<Spec, LocalBits>;
  if constexpr (Record::compact_selection_head)
    return packed_load(record.words, Record::word_count, 128, Spec::index_bits);
  return packed_load(record.words, Record::word_count,
                     (Spec::rounds - 1) * Spec::collision_bits + LocalBits, Spec::index_bits);
}

using mom_blake2b::load64_le;

inline std::uint32_t load32_le(const std::uint8_t* const p) {
  return static_cast<std::uint32_t>(p[0]) |
         (static_cast<std::uint32_t>(p[1]) << 8) |
         (static_cast<std::uint32_t>(p[2]) << 16) |
         (static_cast<std::uint32_t>(p[3]) << 24);
}

template <unsigned Index>
inline std::uint64_t b2_tail_word(const std::uint64_t first, const std::uint64_t second) {
  if constexpr (Index == 0) return first;
  if constexpr (Index == 1) return second;
  return 0;
}

template <unsigned Round>
// Specialize one BLAKE2b round for a two-word final block; all other message words are zero.
inline void b2_tail_round(std::uint64_t v[16], const std::uint64_t first,
                          const std::uint64_t second) {
  mom_blake2b::g(v[0], v[4], v[8], v[12],
                 b2_tail_word<mom_blake2b::SIGMA[Round][0]>(first, second),
                 b2_tail_word<mom_blake2b::SIGMA[Round][1]>(first, second));
  mom_blake2b::g(v[1], v[5], v[9], v[13],
                 b2_tail_word<mom_blake2b::SIGMA[Round][2]>(first, second),
                 b2_tail_word<mom_blake2b::SIGMA[Round][3]>(first, second));
  mom_blake2b::g(v[2], v[6], v[10], v[14],
                 b2_tail_word<mom_blake2b::SIGMA[Round][4]>(first, second),
                 b2_tail_word<mom_blake2b::SIGMA[Round][5]>(first, second));
  mom_blake2b::g(v[3], v[7], v[11], v[15],
                 b2_tail_word<mom_blake2b::SIGMA[Round][6]>(first, second),
                 b2_tail_word<mom_blake2b::SIGMA[Round][7]>(first, second));
  mom_blake2b::g(v[0], v[5], v[10], v[15],
                 b2_tail_word<mom_blake2b::SIGMA[Round][8]>(first, second),
                 b2_tail_word<mom_blake2b::SIGMA[Round][9]>(first, second));
  mom_blake2b::g(v[1], v[6], v[11], v[12],
                 b2_tail_word<mom_blake2b::SIGMA[Round][10]>(first, second),
                 b2_tail_word<mom_blake2b::SIGMA[Round][11]>(first, second));
  mom_blake2b::g(v[2], v[7], v[8], v[13],
                 b2_tail_word<mom_blake2b::SIGMA[Round][12]>(first, second),
                 b2_tail_word<mom_blake2b::SIGMA[Round][13]>(first, second));
  mom_blake2b::g(v[3], v[4], v[9], v[14],
                 b2_tail_word<mom_blake2b::SIGMA[Round][14]>(first, second),
                 b2_tail_word<mom_blake2b::SIGMA[Round][15]>(first, second));
}

// IGC expands the hot BLAKE2b uint64_t add/rotate chain into long Xe2 instruction sequences.
// Explicit halves expose the same bit-exact work as native 32-bit ALU operations while the
// backend-neutral scalar implementation below remains the fallback for every other compiler. The
// pair primitive itself is compiler-neutral so host tests can compare the two implementations.
using Blake2bPair = mom_blake2b::Word;

inline Blake2bPair b2_pair(const std::uint64_t value) {
  return {static_cast<std::uint32_t>(value), static_cast<std::uint32_t>(value >> 32)};
}

inline std::uint64_t b2_join(const Blake2bPair value) {
  return static_cast<std::uint64_t>(value.lo) |
         (static_cast<std::uint64_t>(value.hi) << 32);
}

template <unsigned Index>
inline Blake2bPair b2_tail_pair_word(const Blake2bPair first, const Blake2bPair second) {
  if constexpr (Index == 0) return first;
  if constexpr (Index == 1) return second;
  return {0u, 0u};
}

template <unsigned Round>
// Apply the same specialized final-block round using split words for the paired ALU path.
inline void b2_tail_pair_round(Blake2bPair v[16], const Blake2bPair first,
                               const Blake2bPair second) {
  mom_blake2b::mix(v[0], v[4], v[8], v[12],
                   b2_tail_pair_word<mom_blake2b::SIGMA[Round][0]>(first, second),
                   b2_tail_pair_word<mom_blake2b::SIGMA[Round][1]>(first, second));
  mom_blake2b::mix(v[1], v[5], v[9], v[13],
                   b2_tail_pair_word<mom_blake2b::SIGMA[Round][2]>(first, second),
                   b2_tail_pair_word<mom_blake2b::SIGMA[Round][3]>(first, second));
  mom_blake2b::mix(v[2], v[6], v[10], v[14],
                   b2_tail_pair_word<mom_blake2b::SIGMA[Round][4]>(first, second),
                   b2_tail_pair_word<mom_blake2b::SIGMA[Round][5]>(first, second));
  mom_blake2b::mix(v[3], v[7], v[11], v[15],
                   b2_tail_pair_word<mom_blake2b::SIGMA[Round][6]>(first, second),
                   b2_tail_pair_word<mom_blake2b::SIGMA[Round][7]>(first, second));
  mom_blake2b::mix(v[0], v[5], v[10], v[15],
                   b2_tail_pair_word<mom_blake2b::SIGMA[Round][8]>(first, second),
                   b2_tail_pair_word<mom_blake2b::SIGMA[Round][9]>(first, second));
  mom_blake2b::mix(v[1], v[6], v[11], v[12],
                   b2_tail_pair_word<mom_blake2b::SIGMA[Round][10]>(first, second),
                   b2_tail_pair_word<mom_blake2b::SIGMA[Round][11]>(first, second));
  mom_blake2b::mix(v[2], v[7], v[8], v[13],
                   b2_tail_pair_word<mom_blake2b::SIGMA[Round][12]>(first, second),
                   b2_tail_pair_word<mom_blake2b::SIGMA[Round][13]>(first, second));
  mom_blake2b::mix(v[3], v[4], v[9], v[14],
                   b2_tail_pair_word<mom_blake2b::SIGMA[Round][14]>(first, second),
                   b2_tail_pair_word<mom_blake2b::SIGMA[Round][15]>(first, second));
}

// Finalize the 140-byte header plus hash index using bit-exact paired BLAKE2b arithmetic.
inline void hash_words_from_tail_pair(const std::uint64_t midstate[8],
                                      const std::uint64_t first,
                                      const std::uint32_t second_low,
                                      const std::uint32_t index,
                                      std::uint64_t digest[8]) {
  const Blake2bPair message_first = b2_pair(first);
  const Blake2bPair message_second{second_low, index};
  Blake2bPair state[16];
  for (unsigned i = 0; i < 8; ++i) {
    state[i] = b2_pair(midstate[i]);
    state[i + 8] = b2_pair(mom_blake2b::IV[i]);
  }
  state[12].lo ^= 144u;
  state[14].lo = ~state[14].lo;
  state[14].hi = ~state[14].hi;
  b2_tail_pair_round<0>(state, message_first, message_second);
  b2_tail_pair_round<1>(state, message_first, message_second);
  b2_tail_pair_round<2>(state, message_first, message_second);
  b2_tail_pair_round<3>(state, message_first, message_second);
  b2_tail_pair_round<4>(state, message_first, message_second);
  b2_tail_pair_round<5>(state, message_first, message_second);
  b2_tail_pair_round<6>(state, message_first, message_second);
  b2_tail_pair_round<7>(state, message_first, message_second);
  b2_tail_pair_round<8>(state, message_first, message_second);
  b2_tail_pair_round<9>(state, message_first, message_second);
  b2_tail_pair_round<10>(state, message_first, message_second);
  b2_tail_pair_round<11>(state, message_first, message_second);
  for (unsigned i = 0; i < 8; ++i) {
    const Blake2bPair initial = b2_pair(midstate[i]);
    digest[i] = b2_join({initial.lo ^ state[i].lo ^ state[i + 8].lo,
                         initial.hi ^ state[i].hi ^ state[i + 8].hi});
  }
}

template <typename Spec>
// Finalize the header tail and index. The SPIR-V paired path depends on the device target,
// not Intel's compiler-brand macro; upstream DPC++ must retain the same optimization.
inline void hash_words_from_tail(const std::uint64_t midstate[8], const std::uint64_t first,
                                 const std::uint32_t second_low, const std::uint32_t index,
                                 std::uint64_t digest[8]) {
#if defined(__SYCL_DEVICE_ONLY__) && !defined(__NVPTX__) && \
    (defined(__INTEL_LLVM_COMPILER) || (defined(__SPIR__) && !defined(MOM_SYCL_PORTABLE_OPENCL)))
  hash_words_from_tail_pair(midstate, first, second_low, index, digest);
#else
  const std::uint64_t second = second_low | (static_cast<std::uint64_t>(index) << 32);
  std::uint64_t v[16];
  for (unsigned i = 0; i < 8; ++i) {
    v[i] = midstate[i];
    v[i + 8] = mom_blake2b::IV[i];
  }
  v[12] ^= 144;
  v[14] = ~v[14];
  b2_tail_round<0>(v, first, second);
  b2_tail_round<1>(v, first, second);
  b2_tail_round<2>(v, first, second);
  b2_tail_round<3>(v, first, second);
  b2_tail_round<4>(v, first, second);
  b2_tail_round<5>(v, first, second);
  b2_tail_round<6>(v, first, second);
  b2_tail_round<7>(v, first, second);
  b2_tail_round<8>(v, first, second);
  b2_tail_round<9>(v, first, second);
  b2_tail_round<10>(v, first, second);
  b2_tail_round<11>(v, first, second);
  for (unsigned i = 0; i < 8; ++i) {
    digest[i] = midstate[i] ^ v[i] ^ v[i + 8];
  }
#endif
}

template <typename Spec>
inline void hash_words_from_midstate(const std::uint64_t midstate[8],
                                     const std::uint8_t header[Spec::header_length],
                                     const std::uint32_t index, std::uint64_t digest[8]) {
  hash_words_from_tail<Spec>(midstate, load64_le(header + 128),
                             load32_le(header + 136), index, digest);
}

// Extract one little-endian 24-bit collision field, including a possible 64-bit word crossing.
inline std::uint32_t digest24(const std::uint64_t digest[8], const unsigned byte) {
  const unsigned word = byte / 8, shift = 8 * (byte % 8);
  std::uint64_t value = digest[word] >> shift;
  if (shift > 40) {
    value |= digest[word + 1] << (64 - shift);
  }
  return static_cast<std::uint32_t>(value) & 0xffffffu;
}

template <typename Spec>
// Personalize BLAKE2b for the Equihash parameters and hash the reusable first 128 header bytes.
inline void hash_header_midstate(const std::uint8_t header[Spec::header_length],
                                 std::uint64_t h[8]) {
  std::uint8_t personal[16];
  for (unsigned i = 0; i < 16; ++i)
    personal[i] = Spec::personal_byte(i);
  mom_blake2b::init(h, Spec::hash_length, personal);
  mom_blake2b::compress_bytes(h, header, 128, false);
}

template <typename Spec>
// Serialize the indexed digest from a prehashed header, retaining the specification's output length.
inline void hash_index_from_midstate(const std::uint64_t midstate[8],
                                     const std::uint8_t header[Spec::header_length],
                                     const std::uint32_t index,
                                     std::uint8_t digest[Spec::hash_length]) {
  std::uint64_t h[8];
  hash_words_from_midstate<Spec>(midstate, header, index, h);
  for (unsigned i = 0; i < Spec::hash_length; ++i)
    digest[i] = static_cast<std::uint8_t>(h[i / 8] >> (8 * (i % 8)));
}

template <typename Spec>
// Hash one indexed header from scratch as the independent generation/verification path.
inline void hash_index(const std::uint8_t header[Spec::header_length], const std::uint32_t index,
                       std::uint8_t digest[Spec::hash_length]) {
  std::uint64_t midstate[8];
  hash_header_midstate<Spec>(header, midstate);
  hash_index_from_midstate<Spec>(midstate, header, index, digest);
}

template <typename Spec>
// Select the indexed digest segment and expand its collision fields while retaining the leaf index.
inline void row_from_index_midstate(const std::uint64_t midstate[8],
                                    const std::uint8_t header[Spec::header_length],
                                    const std::uint32_t index, Row<Spec>& row) {
  std::uint64_t digest[8];
  hash_words_from_midstate<Spec>(midstate, header, index / Spec::indices_per_hash, digest);
  const unsigned offset = (index % Spec::indices_per_hash) * Spec::segment_bytes;
  row.first_index = index;
  for (unsigned i = 0; i < Spec::rounds; ++i) {
    const unsigned o = offset + i * Spec::collision_bytes;
    row.fields[i] = digest24(digest, o);
  }
}

template <typename Spec>
// Construct an independently hashed leaf row without relying on cached solver state.
inline void row_from_index(const std::uint8_t header[Spec::header_length], const std::uint32_t index,
                           Row<Spec>& row) {
  std::uint64_t midstate[8];
  hash_header_midstate<Spec>(header, midstate);
  row_from_index_midstate<Spec>(midstate, header, index, row);
}

template <typename Spec>
// Merge canonically ordered children after their collision prefix has matched.
inline void merge_ordered(const Node<Spec>& left, const Node<Spec>& right, Node<Spec>& out) {
  out.first_index = left.first_index;
  for (unsigned i = 0; i < Spec::rounds - 1; ++i) {
    out.fields[i] = left.fields[i + 1] ^ right.fields[i + 1];
  }
  out.fields[Spec::rounds - 1] = 0;
}

template <typename Spec>
// Decode fixed-width big-endian minimal-solution bits into the canonical leaf sequence.
inline void decode_indices(const std::uint8_t solution[Spec::solution_length],
                           std::uint32_t indices[Spec::proof_indices]) {
  for (unsigned i = 0; i < Spec::proof_indices; ++i) {
    std::uint32_t value = 0;
    for (unsigned bit = 0; bit < Spec::index_bits; ++bit) {
      const unsigned p = i * Spec::index_bits + bit;
      value = (value << 1) | ((solution[p / 8] >> (7 - p % 8)) & 1u);
    }
    indices[i] = value;
  }
}

// Pack already validated leaf indices into fixed-width big-endian minimal-solution bits.
template <typename Spec>
inline void encode_indices(const std::uint32_t indices[Spec::proof_indices],
                           std::uint8_t solution[Spec::solution_length]) {
  for (unsigned i = 0; i < Spec::solution_length; ++i) {
    solution[i] = 0;
  }
  for (unsigned i = 0; i < Spec::proof_indices; ++i)
    for (unsigned bit = 0; bit < Spec::index_bits; ++bit) {
      const unsigned p = i * Spec::index_bits + bit;
      solution[p / 8] |= static_cast<std::uint8_t>(((indices[i] >> (Spec::index_bits - bit - 1)) & 1u) <<
                                                   (7 - p % 8));
    }
}

template <typename Spec>
// Recursively enforce every collision and lower-minimum-leaf child order in the supplied proof tree.
inline bool verify_subtree(const Node<Spec> leaves[Spec::proof_indices], const unsigned begin,
                           const unsigned level, Node<Spec>& result) {
  if (level == 0) {
    result = leaves[begin];
    return true;
  }
  const unsigned half = 1u << (level - 1);
  Node<Spec> left{}, right{};
  if (!verify_subtree<Spec>(leaves, begin, level - 1, left) ||
      !verify_subtree<Spec>(leaves, begin + half, level - 1, right))
    return false;
  if (left.fields[0] != right.fields[0] || left.first_index >= right.first_index)
    return false;
  merge_ordered<Spec>(left, right, result);
  return true;
}

template <typename Spec>
// Rehash all distinct leaves and verify the complete canonical tree reaches zero XOR.
inline bool verify_solution(const std::uint8_t header[Spec::header_length],
                            const std::uint8_t solution[Spec::solution_length]) {
  std::uint32_t indices[Spec::proof_indices];
  decode_indices<Spec>(solution, indices);
  for (unsigned i = 0; i < Spec::proof_indices; ++i)
    for (unsigned j = 0; j < i; ++j)
      if (indices[i] == indices[j]) return false;

  Node<Spec> leaves[Spec::proof_indices];
  for (unsigned i = 0; i < Spec::proof_indices; ++i) {
    Row<Spec> row{};
    row_from_index<Spec>(header, indices[i], row);
    leaves[i].first_index = row.first_index;
    for (unsigned j = 0; j < Spec::rounds; ++j) {
      leaves[i].fields[j] = row.fields[j];
    }
  }
  Node<Spec> root{};
  return verify_subtree<Spec>(leaves, 0, Spec::k, root) && root.fields[0] == 0;
}

}  // namespace mom_equihash
