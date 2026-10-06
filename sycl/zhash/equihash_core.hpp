#pragma once

// Portable, backend-neutral Equihash primitives shared by the ZHash and Equihash(192,7)
// adapters. Keeping hashing, packed row layout, bucket/scatter contracts, and proof validation
// here prevents the two algorithms from drifting apart while leaving backend-specific occupancy
// decisions to their existing solvers.

#include <cstddef>
#include <cstdint>

#include "../blake2b_pair.hpp"

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

template <typename Spec>
struct Row {
  std::uint32_t fields[Spec::rounds];
  std::uint32_t first_index;
};

template <typename Spec>
using Node = Row<Spec>;

// Packed records are intentionally word-aligned: six 24-bit fields plus the 25-bit leaf index
// occupy 169 bits, while a first collision's five fields plus two parent references occupy 172
// bits.  Both therefore fit in six words, versus seven words for the expanded level-0 Row.
template <typename Spec>
struct alignas(8) PackedLevel0Record {
  std::uint32_t words[Spec::level0_words];
};

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

constexpr unsigned packed_bucket_collision_alignment(const unsigned n,
                                                       const unsigned field_count) {
#if defined(MOM_SYCL_HAS_HIP) || defined(__INTEL_LLVM_COMPILER)
  constexpr unsigned wide_fields = 7;
#else
  constexpr unsigned wide_fields = 6;
#endif
#if defined(MOM_SYCL_HAS_HIP)
  if (n == 144 && (field_count == 2 || field_count == 3)) return 4;
#endif
  if (n == 192) {
#if defined(__INTEL_LLVM_COMPILER)
    if (field_count == 7) return 4;
#endif
    if (field_count >= wide_fields) return 32;
    if (field_count >= 5) return 8;
#if defined(MOM_SYCL_HAS_HIP) || defined(__INTEL_LLVM_COMPILER)
    if (field_count == 2) return 4;
#endif
  }
  return field_count >= 5 ? 4 : 16;
}

constexpr unsigned bits_for_capacity(std::uint32_t capacity) {
  unsigned bits = 0;
  for (--capacity; capacity != 0; capacity >>= 1) {
    ++bits;
  }
  return bits;
}

constexpr unsigned packed_bucket_level0_alignment([[maybe_unused]] const unsigned n,
                                                   [[maybe_unused]] const unsigned k,
                                                   const unsigned word_count) {
#if defined(__INTEL_LLVM_COMPILER)
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
  static constexpr unsigned parent_slots = ParentSlots;
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
using ZHashRootRecord = CollisionRoundRecord<Spec, Spec::rounds - 1>;

inline std::uint64_t packed_mask(const unsigned bits) {
  return bits == 32 ? 0xffffffffull : ((1ull << bits) - 1ull);
}

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

template <typename Spec>
inline void clear_packed(PackedLevel0Record<Spec>& record) {
  for (unsigned i = 0; i < Spec::level0_words; ++i) {
    record.words[i] = 0;
  }
}

template <typename Spec>
inline void store_level0(PackedLevel0Record<Spec>& record,
                         const std::uint32_t fields[Spec::rounds], const std::uint32_t leaf) {
  clear_packed<Spec>(record);
  for (unsigned i = 0; i < Spec::rounds; ++i)
    packed_store(record.words, Spec::level0_words, i * Spec::collision_bits,
                 Spec::collision_bits, fields[i]);
  packed_store(record.words, Spec::level0_words, Spec::rounds * Spec::collision_bits,
               Spec::index_bits, leaf);
}

template <typename Spec>
inline std::uint32_t load_level0_field(const PackedLevel0Record<Spec>& record, const unsigned field) {
  return packed_load(record.words, Spec::level0_words, field * Spec::collision_bits,
                     Spec::collision_bits);
}

template <typename Spec>
inline std::uint32_t load_level0_leaf(const PackedLevel0Record<Spec>& record) {
  return packed_load(record.words, Spec::level0_words, Spec::rounds * Spec::collision_bits,
                     Spec::index_bits);
}

template <typename Spec, unsigned FieldCount>
inline void clear_packed(PackedCollisionRecord<Spec, FieldCount>& record) {
  for (unsigned i = 0; i < PackedCollisionRecord<Spec, FieldCount>::word_count; ++i)
    record.words[i] = 0;
}

template <typename Spec, unsigned FieldCount>
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

template <typename Spec, unsigned LocalBits>
inline std::uint32_t load_bucket_level0_leaf(
    const PackedBucketLevel0Record<Spec, LocalBits>& record) {
  using Record = PackedBucketLevel0Record<Spec, LocalBits>;
  if constexpr (Record::compact_selection_head)
    return packed_load(record.words, Record::word_count, 128, Spec::index_bits);
  return packed_load(record.words, Record::word_count,
                     (Spec::rounds - 1) * Spec::collision_bits + LocalBits, Spec::index_bits);
}

// Parent references are deliberately kept compact in device records.  This view supplies the
// round-specific arenas needed to walk those references back to level zero on the host (or in a
// later checked solver pass).  ZHash has five collision rounds, so round five is the root record
// itself and only rounds one through four need to be retained as child arenas.
template <typename Spec>
struct ProvenanceView {
  static_assert(Spec::rounds == 6, "the checked provenance view currently targets ZHash144_5");
  using Level0 = PackedLevel0Record<Spec>;
  using Round1 = CollisionRoundRecord<Spec, 1>;
  using Round2 = CollisionRoundRecord<Spec, 2>;
  using Round3 = CollisionRoundRecord<Spec, 3>;
  using Round4 = CollisionRoundRecord<Spec, 4>;

  const Level0* level0 = nullptr;
  std::uint32_t level0_capacity = 0;
  const Round1* round1 = nullptr;
  std::uint32_t round1_capacity = 0;
  const Round2* round2 = nullptr;
  std::uint32_t round2_capacity = 0;
  const Round3* round3 = nullptr;
  std::uint32_t round3_capacity = 0;
  const Round4* round4 = nullptr;
  std::uint32_t round4_capacity = 0;
};

template <typename Spec>
struct LeafSet {
  std::uint32_t values[Spec::proof_indices];
  std::uint32_t count;
};

template <typename Spec>
inline bool leaf_sets_disjoint(const LeafSet<Spec>& left, const LeafSet<Spec>& right) {
  for (std::uint32_t i = 0; i < left.count; ++i)
    for (std::uint32_t j = 0; j < right.count; ++j)
      if (left.values[i] == right.values[j]) return false;
  return true;
}

template <typename Spec>
inline bool leaf_set_append(const LeafSet<Spec>& left, const LeafSet<Spec>& right,
                            LeafSet<Spec>& out) {
  if (left.count + right.count > Spec::proof_indices || !leaf_sets_disjoint(left, right))
    return false;
  out.count = 0;
  for (std::uint32_t i = 0; i < left.count; ++i) {
    out.values[out.count++] = left.values[i];
  }
  for (std::uint32_t i = 0; i < right.count; ++i) {
    out.values[out.count++] = right.values[i];
  }
  return true;
}

template <typename Spec>
inline bool recover_node(const ProvenanceView<Spec>& view, const unsigned round,
                         const std::uint32_t ref, LeafSet<Spec>& out);

template <typename Spec>
inline bool load_node_field(const ProvenanceView<Spec>& view, const unsigned round,
                            const std::uint32_t ref, const unsigned field,
                            std::uint32_t& value) {
  if (round == 0) {
    if (view.level0 == nullptr || ref >= view.level0_capacity || field >= Spec::rounds)
      return false;
    value = load_level0_field<Spec>(view.level0[ref], field);
    return true;
  }
  if (round == 1) {
    if (view.round1 == nullptr || ref >= view.round1_capacity || field >= Spec::rounds - 1)
      return false;
    value = load_collision_field<Spec, Spec::rounds - 1>(view.round1[ref], field);
    return true;
  }
  if (round == 2) {
    if (view.round2 == nullptr || ref >= view.round2_capacity || field >= Spec::rounds - 2)
      return false;
    value = load_collision_field<Spec, Spec::rounds - 2>(view.round2[ref], field);
    return true;
  }
  if (round == 3) {
    if (view.round3 == nullptr || ref >= view.round3_capacity || field >= Spec::rounds - 3)
      return false;
    value = load_collision_field<Spec, Spec::rounds - 3>(view.round3[ref], field);
    return true;
  }
  if (round == 4) {
    if (view.round4 == nullptr || ref >= view.round4_capacity || field >= Spec::rounds - 4)
      return false;
    value = load_collision_field<Spec, Spec::rounds - 4>(view.round4[ref], field);
    return true;
  }
  return false;
}

template <typename Spec, unsigned FieldCount>
inline bool recover_collision_record(const ProvenanceView<Spec>& view,
                                     const PackedCollisionRecord<Spec, FieldCount>& record,
                                     const unsigned parent_round, LeafSet<Spec>& out) {
  static_assert(FieldCount > 0, "a provenance record must retain a parent pair");
  const std::uint32_t left_ref = load_collision_left<Spec, FieldCount>(record);
  const std::uint32_t right_ref = load_collision_right<Spec, FieldCount>(record);
  if (left_ref == right_ref) return false;

  LeafSet<Spec> left{};
  LeafSet<Spec> right{};
  if (!recover_node<Spec>(view, parent_round, left_ref, left) ||
      !recover_node<Spec>(view, parent_round, right_ref, right))
    return false;
  std::uint32_t left_field = 0, right_field = 0;
  if (!load_node_field<Spec>(view, parent_round, left_ref, 0, left_field) ||
      !load_node_field<Spec>(view, parent_round, right_ref, 0, right_field) ||
      left_field != right_field)
    return false;
  for (unsigned field = 0; field < FieldCount; ++field) {
    if (!load_node_field<Spec>(view, parent_round, left_ref, field + 1, left_field) ||
        !load_node_field<Spec>(view, parent_round, right_ref, field + 1, right_field) ||
        load_collision_field<Spec, FieldCount>(record, field) != (left_field ^ right_field))
      return false;
  }
  // Every record is stored with its lower minimum-leaf child first.  Checking this while walking
  // the tree makes the encoded solution deterministic and catches malformed parent references.
  if (left.count == 0 || right.count == 0 || left.values[0] >= right.values[0]) return false;
  return leaf_set_append<Spec>(left, right, out);
}

template <typename Spec>
inline bool recover_node(const ProvenanceView<Spec>& view, const unsigned round,
                         const std::uint32_t ref, LeafSet<Spec>& out) {
  out.count = 0;
  if (round == 0) {
    if (view.level0 == nullptr || ref >= view.level0_capacity) return false;
    out.values[0] = load_level0_leaf<Spec>(view.level0[ref]);
    out.count = 1;
    return true;
  }
  if (round == 1) {
    if (view.round1 == nullptr || ref >= view.round1_capacity) return false;
    return recover_collision_record<Spec, Spec::rounds - 1>(view, view.round1[ref], 0, out);
  }
  if (round == 2) {
    if (view.round2 == nullptr || ref >= view.round2_capacity) return false;
    return recover_collision_record<Spec, Spec::rounds - 2>(view, view.round2[ref], 1, out);
  }
  if (round == 3) {
    if (view.round3 == nullptr || ref >= view.round3_capacity) return false;
    return recover_collision_record<Spec, Spec::rounds - 3>(view, view.round3[ref], 2, out);
  }
  if (round == 4) {
    if (view.round4 == nullptr || ref >= view.round4_capacity) return false;
    return recover_collision_record<Spec, Spec::rounds - 4>(view, view.round4[ref], 3, out);
  }
  return false;
}

template <typename Spec>
inline bool is_zero_root(const ZHashRootRecord<Spec>& root) {
  static_assert(Spec::rounds == 6, "the ZHash root has one final active field");
  return load_collision_field<Spec, 1>(root, 0) == 0;
}

template <typename Spec>
inline bool recover_root(const ProvenanceView<Spec>& view, const ZHashRootRecord<Spec>& root,
                         std::uint32_t leaves[Spec::proof_indices]) {
  if (!is_zero_root<Spec>(root)) return false;
  LeafSet<Spec> recovered{};
  if (!recover_collision_record<Spec, 1>(view, root, Spec::rounds - 2, recovered) ||
      recovered.count != Spec::proof_indices)
    return false;
  for (unsigned i = 0; i < Spec::proof_indices; ++i) {
    leaves[i] = recovered.values[i];
  }
  return true;
}

template <typename Spec>
inline void encode_indices(const std::uint32_t indices[Spec::proof_indices],
                           std::uint8_t solution[Spec::solution_length]);

template <typename Spec>
inline bool recover_solution(const ProvenanceView<Spec>& view, const ZHashRootRecord<Spec>& root,
                             std::uint32_t leaves[Spec::proof_indices],
                             std::uint8_t solution[Spec::solution_length]) {
  if (!recover_root<Spec>(view, root, leaves)) return false;
  encode_indices<Spec>(leaves, solution);
  return true;
}

template <typename Spec>
inline bool make_collision(const PackedLevel0Record<Spec>& a, const PackedLevel0Record<Spec>& b,
                           const std::uint32_t a_ref, const std::uint32_t b_ref,
                           PackedCollisionRecord<Spec, Spec::rounds - 1>& out) {
  if (load_level0_field<Spec>(a, 0) != load_level0_field<Spec>(b, 0)) return false;
  const std::uint32_t a_leaf = load_level0_leaf<Spec>(a);
  const std::uint32_t b_leaf = load_level0_leaf<Spec>(b);
  if (a_leaf == b_leaf) return false;
  std::uint32_t fields[Spec::rounds - 1];
  for (unsigned i = 0; i < Spec::rounds - 1; ++i)
    fields[i] = load_level0_field<Spec>(a, i + 1) ^ load_level0_field<Spec>(b, i + 1);
  if (a_leaf < b_leaf) {
    store_collision<Spec>(out, fields, a_ref, b_ref);
  } else {
    store_collision<Spec>(out, fields, b_ref, a_ref);
  }
  return true;
}

// A later collision consumes a compact record from the preceding round.  `a_order` and `b_order`
// are the minimum original leaf indices of those records; keeping that small parallel array lets
// the compact record retain only two parent references while still imposing canonical child order.
template <typename Spec, unsigned FieldCount>
inline bool make_collision(
    const PackedCollisionRecord<Spec, FieldCount>& a,
    const PackedCollisionRecord<Spec, FieldCount>& b, const std::uint32_t a_ref,
    const std::uint32_t b_ref, const std::uint32_t a_order, const std::uint32_t b_order,
    PackedCollisionRecord<Spec, FieldCount - 1>& out) {
  static_assert(FieldCount > 0, "a collision must retain an active field");
  if (load_collision_field<Spec, FieldCount>(a, 0) !=
      load_collision_field<Spec, FieldCount>(b, 0)) return false;
  if (a_order == b_order) return false;
  const std::uint32_t a_left = load_collision_left<Spec, FieldCount>(a);
  const std::uint32_t a_right = load_collision_right<Spec, FieldCount>(a);
  const std::uint32_t b_left = load_collision_left<Spec, FieldCount>(b);
  const std::uint32_t b_right = load_collision_right<Spec, FieldCount>(b);
  if (a_left == b_left || a_left == b_right || a_right == b_left || a_right == b_right)
    return false;
  std::uint32_t fields[FieldCount - 1];
  for (unsigned i = 0; i < FieldCount - 1; ++i)
    fields[i] = load_collision_field<Spec, FieldCount>(a, i + 1) ^
                load_collision_field<Spec, FieldCount>(b, i + 1);
  if (a_order < b_order) {
    store_collision<Spec, FieldCount - 1>(out, fields, a_ref, b_ref);
  } else {
    store_collision<Spec, FieldCount - 1>(out, fields, b_ref, a_ref);
  }
  return true;
}

// Checked variants use the recursive arena view instead of trusting only the parallel minimum
// index.  They are the host/reference contract for rejecting a pair whose subtrees share any leaf;
// the compact device kernels retain the same parent references for a later checked pass.
template <typename Spec>
inline bool make_collision_checked(
    const PackedLevel0Record<Spec>& a, const PackedLevel0Record<Spec>& b,
    const std::uint32_t a_ref, const std::uint32_t b_ref, const ProvenanceView<Spec>& view,
    CollisionRoundRecord<Spec, 1>& out) {
  if (load_level0_field<Spec>(a, 0) != load_level0_field<Spec>(b, 0)) return false;
  LeafSet<Spec> left{};
  LeafSet<Spec> right{};
  if (!recover_node<Spec>(view, 0, a_ref, left) || !recover_node<Spec>(view, 0, b_ref, right) ||
      left.count != 1 || right.count != 1 ||
      left.values[0] != load_level0_leaf<Spec>(a) ||
      right.values[0] != load_level0_leaf<Spec>(b) ||
      !leaf_sets_disjoint<Spec>(left, right))
    return false;
  std::uint32_t fields[Spec::rounds - 1];
  for (unsigned i = 0; i < Spec::rounds - 1; ++i)
    fields[i] = load_level0_field<Spec>(a, i + 1) ^ load_level0_field<Spec>(b, i + 1);
  if (left.values[0] < right.values[0]) {
    store_collision<Spec, Spec::rounds - 1>(out, fields, a_ref, b_ref);
  } else {
    store_collision<Spec, Spec::rounds - 1>(out, fields, b_ref, a_ref);
  }
  return true;
}

template <typename Spec, unsigned FieldCount>
inline bool make_collision_checked(
    const PackedCollisionRecord<Spec, FieldCount>& a,
    const PackedCollisionRecord<Spec, FieldCount>& b, const std::uint32_t a_ref,
    const std::uint32_t b_ref, const ProvenanceView<Spec>& view,
    PackedCollisionRecord<Spec, FieldCount - 1>& out) {
  static_assert(FieldCount > 1, "the five-round ZHash solver stops at one root field");
  const unsigned parent_round = Spec::rounds - 1 - FieldCount;
  std::uint32_t left_field = 0;
  std::uint32_t right_field = 0;
  if (!load_node_field<Spec>(view, parent_round, a_ref, 0, left_field) ||
      !load_node_field<Spec>(view, parent_round, b_ref, 0, right_field) ||
      left_field != right_field)
    return false;

  LeafSet<Spec> left{};
  LeafSet<Spec> right{};
  if (!recover_node<Spec>(view, parent_round, a_ref, left) ||
      !recover_node<Spec>(view, parent_round, b_ref, right) ||
      !leaf_sets_disjoint<Spec>(left, right) || left.values[0] == right.values[0])
    return false;
  std::uint32_t fields[FieldCount - 1];
  for (unsigned i = 0; i < FieldCount - 1; ++i)
    fields[i] = load_collision_field<Spec, FieldCount>(a, i + 1) ^
                load_collision_field<Spec, FieldCount>(b, i + 1);
  if (left.values[0] < right.values[0])
    store_collision<Spec, FieldCount - 1>(out, fields, a_ref, b_ref);
  else
    store_collision<Spec, FieldCount - 1>(out, fields, b_ref, a_ref);
  return true;
}

template <typename Spec, typename Layout>
inline std::uint32_t bucket_round_reference(
    const PackedLevel0Record<Spec>* const input, const std::uint32_t* const bucket_counts,
    PackedCollisionRecord<Spec>* const output, const std::uint32_t output_capacity) {
  std::uint32_t emitted = 0;
  std::uint32_t bin_head[Layout::local_bins];
  std::uint32_t bin_next[Layout::bucket_slot_capacity];
  for (unsigned bucket = 0; bucket < Layout::bucket_count; ++bucket) {
    const std::uint32_t count = bucket_counts[bucket] < Layout::bucket_slot_capacity
                                     ? bucket_counts[bucket] : Layout::bucket_slot_capacity;
    for (unsigned i = 0; i < Layout::local_bins; ++i) {
      bin_head[i] = 0;
    }
    const std::size_t base = static_cast<std::size_t>(bucket) * Layout::bucket_slot_capacity;
    for (std::uint32_t i = 0; i < count; ++i) {
      const auto key = load_level0_field<Spec>(input[base + i], 0);
      const unsigned bin = Layout::local_bin(key);
      bin_next[i] = bin_head[bin];
      bin_head[bin] = i + 1u;
    }
    for (std::uint32_t a = 0; a < count; ++a) {
      const auto& left = input[base + a];
      unsigned link = bin_head[Layout::local_bin(load_level0_field<Spec>(left, 0))];
      while (link != 0) {
        const std::uint32_t b = link - 1u;
        link = bin_next[b];
        if (b >= a) continue;
        const auto& right = input[base + b];
        PackedCollisionRecord<Spec> collision{};
        if (!make_collision<Spec>(left, right,
                                  static_cast<std::uint32_t>(base + a),
                                  static_cast<std::uint32_t>(base + b),
                                  collision)) continue;
        if (emitted < output_capacity) {
          output[emitted] = collision;
        }
        ++emitted;
      }
    }
  }
  return emitted;
}

template <typename Spec, typename Layout>
inline std::uint32_t bucket_round_reference_checked(
    const PackedLevel0Record<Spec>* const input, const std::uint32_t* const bucket_counts,
    PackedCollisionRecord<Spec>* const output, std::uint32_t* const output_min_indices,
    const std::uint32_t output_capacity, const ProvenanceView<Spec>& view) {
  std::uint32_t emitted = 0;
  std::uint32_t bin_head[Layout::local_bins];
  std::uint32_t bin_next[Layout::bucket_slot_capacity];
  for (unsigned bucket = 0; bucket < Layout::bucket_count; ++bucket) {
    const std::uint32_t count = bucket_counts[bucket] < Layout::bucket_slot_capacity
                                     ? bucket_counts[bucket] : Layout::bucket_slot_capacity;
    for (unsigned i = 0; i < Layout::local_bins; ++i) {
      bin_head[i] = 0;
    }
    const std::size_t base = static_cast<std::size_t>(bucket) * Layout::bucket_slot_capacity;
    for (std::uint32_t i = 0; i < count; ++i) {
      const unsigned bin = Layout::local_bin(load_level0_field<Spec>(input[base + i], 0));
      bin_next[i] = bin_head[bin];
      bin_head[bin] = i + 1u;
    }
    for (std::uint32_t a = 0; a < count; ++a) {
      unsigned link = bin_head[Layout::local_bin(load_level0_field<Spec>(input[base + a], 0))];
      while (link != 0) {
        const std::uint32_t b = link - 1u;
        link = bin_next[b];
        if (b >= a) continue;
        PackedCollisionRecord<Spec> collision{};
        if (!make_collision_checked<Spec>(input[base + a], input[base + b],
                                          static_cast<std::uint32_t>(base + a),
                                          static_cast<std::uint32_t>(base + b), view, collision))
          continue;
        if (emitted < output_capacity) {
          output[emitted] = collision;
          const std::uint32_t a_leaf = load_level0_leaf<Spec>(input[base + a]);
          const std::uint32_t b_leaf = load_level0_leaf<Spec>(input[base + b]);
          output_min_indices[emitted] = a_leaf < b_leaf ? a_leaf : b_leaf;
        }
        ++emitted;
      }
    }
  }
  return emitted;
}

// Scatter preserves bucket-major storage and counts every reservation, including one that misses
// the bounded arena.  Thus a caller can clamp each bucket to its capacity for the next round while
// the separate overflow counter still makes an incomplete solve visible to its scheduler.
template <typename Spec, typename Layout>
inline std::uint32_t scatter_level0_reference(
    const PackedLevel0Record<Spec>* const input, const std::uint32_t input_count,
    PackedLevel0Record<Spec>* const output, std::uint32_t* const bucket_counts,
    std::uint32_t* const overflow_count) {
  const std::uint32_t before = *overflow_count;
  for (std::uint32_t i = 0; i < input_count; ++i) {
    const unsigned bucket = Layout::bucket_for(load_level0_field<Spec>(input[i], 0));
    const std::uint32_t slot = bucket_counts[bucket]++;
    if (slot < Layout::bucket_slot_capacity)
      output[static_cast<std::size_t>(bucket) * Layout::bucket_slot_capacity + slot] = input[i];
    else
      ++*overflow_count;
  }
  return input_count - (*overflow_count - before);
}

template <typename Spec, unsigned FieldCount, typename Layout>
inline std::uint32_t scatter_collision_reference(
    const PackedCollisionRecord<Spec, FieldCount>* const input,
    const std::uint32_t* const input_min_indices, const std::uint32_t input_count,
    PackedCollisionRecord<Spec, FieldCount>* const output,
    std::uint32_t* const output_min_indices, std::uint32_t* const bucket_counts,
    std::uint32_t* const overflow_count) {
  const std::uint32_t before = *overflow_count;
  for (std::uint32_t i = 0; i < input_count; ++i) {
    const unsigned bucket = Layout::bucket_for(load_collision_field<Spec, FieldCount>(input[i], 0));
    const std::uint32_t slot = bucket_counts[bucket]++;
    if (slot < Layout::bucket_slot_capacity) {
      const std::size_t output_slot =
          static_cast<std::size_t>(bucket) * Layout::bucket_slot_capacity + slot;
      output[output_slot] = input[i];
      output_min_indices[output_slot] = input_min_indices[i];
    } else {
      ++*overflow_count;
    }
  }
  return input_count - (*overflow_count - before);
}

template <typename Spec, unsigned FieldCount, typename Layout>
inline std::uint32_t collision_round_reference(
    const PackedCollisionRecord<Spec, FieldCount>* const input,
    const std::uint32_t* const input_min_indices, const std::uint32_t* const bucket_counts,
    PackedCollisionRecord<Spec, FieldCount - 1>* const output,
    std::uint32_t* const output_min_indices, const std::uint32_t output_capacity) {
  static_assert(FieldCount > 0, "a collision round needs an active field");
  std::uint32_t emitted = 0;
  std::uint32_t bin_head[Layout::local_bins];
  std::uint32_t bin_next[Layout::bucket_slot_capacity];
  std::uint32_t keys[Layout::bucket_slot_capacity];
  for (unsigned bucket = 0; bucket < Layout::bucket_count; ++bucket) {
    const std::uint32_t count = bucket_counts[bucket] < Layout::bucket_slot_capacity
                                     ? bucket_counts[bucket] : Layout::bucket_slot_capacity;
    for (unsigned i = 0; i < Layout::local_bins; ++i) {
      bin_head[i] = 0;
    }
    const std::size_t base = static_cast<std::size_t>(bucket) * Layout::bucket_slot_capacity;
    for (std::uint32_t i = 0; i < count; ++i) {
      keys[i] = load_collision_field<Spec, FieldCount>(input[base + i], 0);
      const unsigned bin = Layout::local_bin(keys[i]);
      bin_next[i] = bin_head[bin];
      bin_head[bin] = i + 1u;
    }
    for (std::uint32_t a = 0; a < count; ++a) {
      unsigned link = bin_head[Layout::local_bin(keys[a])];
      while (link != 0) {
        const std::uint32_t b = link - 1u;
        link = bin_next[b];
        if (b >= a || keys[b] != keys[a]) continue;
        PackedCollisionRecord<Spec, FieldCount - 1> collision{};
        if (!make_collision<Spec, FieldCount>(
                input[base + a], input[base + b], static_cast<std::uint32_t>(base + a),
                static_cast<std::uint32_t>(base + b), input_min_indices[base + a],
                input_min_indices[base + b], collision)) continue;
        if (emitted < output_capacity) {
          output[emitted] = collision;
          output_min_indices[emitted] = input_min_indices[base + a] < input_min_indices[base + b]
                                             ? input_min_indices[base + a]
                                             : input_min_indices[base + b];
        }
        ++emitted;
      }
    }
  }
  return emitted;
}

template <typename Spec, unsigned FieldCount, typename Layout>
inline std::uint32_t collision_round_reference_checked(
    const PackedCollisionRecord<Spec, FieldCount>* const input,
    const std::uint32_t* const input_min_indices, const std::uint32_t* const bucket_counts,
    PackedCollisionRecord<Spec, FieldCount - 1>* const output,
    std::uint32_t* const output_min_indices, const std::uint32_t output_capacity,
    const ProvenanceView<Spec>& view) {
  static_assert(FieldCount > 1, "the five-round ZHash solver stops at one root field");
  std::uint32_t emitted = 0;
  std::uint32_t bin_head[Layout::local_bins];
  std::uint32_t bin_next[Layout::bucket_slot_capacity];
  std::uint32_t keys[Layout::bucket_slot_capacity];
  for (unsigned bucket = 0; bucket < Layout::bucket_count; ++bucket) {
    const std::uint32_t count = bucket_counts[bucket] < Layout::bucket_slot_capacity
                                     ? bucket_counts[bucket] : Layout::bucket_slot_capacity;
    for (unsigned i = 0; i < Layout::local_bins; ++i) {
      bin_head[i] = 0;
    }
    const std::size_t base = static_cast<std::size_t>(bucket) * Layout::bucket_slot_capacity;
    for (std::uint32_t i = 0; i < count; ++i) {
      keys[i] = load_collision_field<Spec, FieldCount>(input[base + i], 0);
      const unsigned bin = Layout::local_bin(keys[i]);
      bin_next[i] = bin_head[bin];
      bin_head[bin] = i + 1u;
    }
    for (std::uint32_t a = 0; a < count; ++a) {
      unsigned link = bin_head[Layout::local_bin(keys[a])];
      while (link != 0) {
        const std::uint32_t b = link - 1u;
        link = bin_next[b];
        if (b >= a || keys[b] != keys[a]) continue;
        PackedCollisionRecord<Spec, FieldCount - 1> collision{};
        if (!make_collision_checked<Spec, FieldCount>(
                input[base + a], input[base + b], static_cast<std::uint32_t>(base + a),
                static_cast<std::uint32_t>(base + b), view, collision))
          continue;
        if (emitted < output_capacity) {
          output[emitted] = collision;
          const std::uint32_t a_min = input_min_indices[base + a];
          const std::uint32_t b_min = input_min_indices[base + b];
          output_min_indices[emitted] = a_min < b_min ? a_min : b_min;
        }
        ++emitted;
      }
    }
  }
  return emitted;
}

static constexpr std::uint64_t BLAKE2B_IV[8] = {
  0x6a09e667f3bcc908ull, 0xbb67ae8584caa73bull,
  0x3c6ef372fe94f82bull, 0xa54ff53a5f1d36f1ull,
  0x510e527fade682d1ull, 0x9b05688c2b3e6c1full,
  0x1f83d9abfb41bd6bull, 0x5be0cd19137e2179ull,
};

static constexpr std::uint8_t BLAKE2B_SIGMA[12][16] = {
  {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15},
  {14,10,4,8,9,15,13,6,1,12,0,2,11,7,5,3},
  {11,8,12,0,5,2,15,13,10,14,3,6,7,1,9,4},
  {7,9,3,1,13,12,11,14,2,6,5,10,4,0,15,8},
  {9,0,5,7,2,4,10,15,14,1,11,12,6,8,3,13},
  {2,12,6,10,0,11,8,3,4,13,7,5,15,14,1,9},
  {12,5,1,15,14,13,4,10,0,7,6,3,9,2,8,11},
  {13,11,7,14,12,1,3,9,5,0,15,4,8,6,2,10},
  {6,15,14,9,11,3,0,8,12,2,13,7,1,4,10,5},
  {10,2,8,4,7,6,1,5,15,11,9,14,3,12,13,0},
  {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15},
  {14,10,4,8,9,15,13,6,1,12,0,2,11,7,5,3},
};

inline std::uint64_t rotr64(const std::uint64_t x, const unsigned n) {
  return (x >> n) | (x << (64 - n));
}

inline std::uint64_t load64_le(const std::uint8_t* const p) {
  return static_cast<std::uint64_t>(p[0]) |
         (static_cast<std::uint64_t>(p[1]) << 8) |
         (static_cast<std::uint64_t>(p[2]) << 16) |
         (static_cast<std::uint64_t>(p[3]) << 24) |
         (static_cast<std::uint64_t>(p[4]) << 32) |
         (static_cast<std::uint64_t>(p[5]) << 40) |
         (static_cast<std::uint64_t>(p[6]) << 48) |
         (static_cast<std::uint64_t>(p[7]) << 56);
}

inline std::uint32_t load32_le(const std::uint8_t* const p) {
  return static_cast<std::uint32_t>(p[0]) |
         (static_cast<std::uint32_t>(p[1]) << 8) |
         (static_cast<std::uint32_t>(p[2]) << 16) |
         (static_cast<std::uint32_t>(p[3]) << 24);
}

inline void b2_g(std::uint64_t& a, std::uint64_t& b, std::uint64_t& c, std::uint64_t& d,
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

inline void b2_compress(std::uint64_t h[8], const std::uint8_t block[128],
                        const std::uint64_t count, const bool last) {
  std::uint64_t m[16];
  std::uint64_t v[16];
  for (unsigned i = 0; i < 16; ++i) {
    m[i] = load64_le(block + 8 * i);
  }
  for (unsigned i = 0; i < 8; ++i) {
    v[i] = h[i];
    v[i + 8] = BLAKE2B_IV[i];
  }
  v[12] ^= count;
  if (last) {
    v[14] = ~v[14];
  }
  for (unsigned r = 0; r < 12; ++r) {
    const std::uint8_t* const s = BLAKE2B_SIGMA[r];
    b2_g(v[0],v[4],v[8],v[12],m[s[0]],m[s[1]]);
    b2_g(v[1],v[5],v[9],v[13],m[s[2]],m[s[3]]);
    b2_g(v[2],v[6],v[10],v[14],m[s[4]],m[s[5]]);
    b2_g(v[3],v[7],v[11],v[15],m[s[6]],m[s[7]]);
    b2_g(v[0],v[5],v[10],v[15],m[s[8]],m[s[9]]);
    b2_g(v[1],v[6],v[11],v[12],m[s[10]],m[s[11]]);
    b2_g(v[2],v[7],v[8],v[13],m[s[12]],m[s[13]]);
    b2_g(v[3],v[4],v[9],v[14],m[s[14]],m[s[15]]);
  }
  for (unsigned i = 0; i < 8; ++i) {
    h[i] ^= v[i] ^ v[i + 8];
  }
}

template <unsigned Index>
inline std::uint64_t b2_tail_word(const std::uint64_t first, const std::uint64_t second) {
  if constexpr (Index == 0) return first;
  if constexpr (Index == 1) return second;
  return 0;
}

template <unsigned Round>
inline void b2_tail_round(std::uint64_t v[16], const std::uint64_t first,
                          const std::uint64_t second) {
  b2_g(v[0], v[4], v[8], v[12],
       b2_tail_word<BLAKE2B_SIGMA[Round][0]>(first, second),
       b2_tail_word<BLAKE2B_SIGMA[Round][1]>(first, second));
  b2_g(v[1], v[5], v[9], v[13],
       b2_tail_word<BLAKE2B_SIGMA[Round][2]>(first, second),
       b2_tail_word<BLAKE2B_SIGMA[Round][3]>(first, second));
  b2_g(v[2], v[6], v[10], v[14],
       b2_tail_word<BLAKE2B_SIGMA[Round][4]>(first, second),
       b2_tail_word<BLAKE2B_SIGMA[Round][5]>(first, second));
  b2_g(v[3], v[7], v[11], v[15],
       b2_tail_word<BLAKE2B_SIGMA[Round][6]>(first, second),
       b2_tail_word<BLAKE2B_SIGMA[Round][7]>(first, second));
  b2_g(v[0], v[5], v[10], v[15],
       b2_tail_word<BLAKE2B_SIGMA[Round][8]>(first, second),
       b2_tail_word<BLAKE2B_SIGMA[Round][9]>(first, second));
  b2_g(v[1], v[6], v[11], v[12],
       b2_tail_word<BLAKE2B_SIGMA[Round][10]>(first, second),
       b2_tail_word<BLAKE2B_SIGMA[Round][11]>(first, second));
  b2_g(v[2], v[7], v[8], v[13],
       b2_tail_word<BLAKE2B_SIGMA[Round][12]>(first, second),
       b2_tail_word<BLAKE2B_SIGMA[Round][13]>(first, second));
  b2_g(v[3], v[4], v[9], v[14],
       b2_tail_word<BLAKE2B_SIGMA[Round][14]>(first, second),
       b2_tail_word<BLAKE2B_SIGMA[Round][15]>(first, second));
}

// IGC expands the hot BLAKE2b uint64_t add/rotate chain into long Xe2 instruction sequences.
// Explicit halves expose the same bit-exact work as native 32-bit ALU operations while the
// backend-neutral scalar implementation below remains the fallback for every other compiler. The
// pair primitive itself is compiler-neutral so host tests can compare the two implementations.
using Blake2bPair = mom_blake2b_pair::Word;

inline Blake2bPair b2_pair(const std::uint64_t value) {
  return {static_cast<std::uint32_t>(value), static_cast<std::uint32_t>(value >> 32)};
}

inline std::uint64_t b2_join(const Blake2bPair value) {
  return static_cast<std::uint64_t>(value.lo) |
         (static_cast<std::uint64_t>(value.hi) << 32);
}

inline void b2_g_pair(Blake2bPair& a, Blake2bPair& b, Blake2bPair& c, Blake2bPair& d,
                      const Blake2bPair x, const Blake2bPair y) {
  mom_blake2b_pair::mix(a, b, c, d, x, y);
}

template <unsigned Index>
inline Blake2bPair b2_tail_pair_word(const Blake2bPair first, const Blake2bPair second) {
  if constexpr (Index == 0) return first;
  if constexpr (Index == 1) return second;
  return {0u, 0u};
}

template <unsigned Round>
inline void b2_tail_pair_round(Blake2bPair v[16], const Blake2bPair first,
                               const Blake2bPair second) {
  b2_g_pair(v[0], v[4], v[8], v[12],
            b2_tail_pair_word<BLAKE2B_SIGMA[Round][0]>(first, second),
            b2_tail_pair_word<BLAKE2B_SIGMA[Round][1]>(first, second));
  b2_g_pair(v[1], v[5], v[9], v[13],
            b2_tail_pair_word<BLAKE2B_SIGMA[Round][2]>(first, second),
            b2_tail_pair_word<BLAKE2B_SIGMA[Round][3]>(first, second));
  b2_g_pair(v[2], v[6], v[10], v[14],
            b2_tail_pair_word<BLAKE2B_SIGMA[Round][4]>(first, second),
            b2_tail_pair_word<BLAKE2B_SIGMA[Round][5]>(first, second));
  b2_g_pair(v[3], v[7], v[11], v[15],
            b2_tail_pair_word<BLAKE2B_SIGMA[Round][6]>(first, second),
            b2_tail_pair_word<BLAKE2B_SIGMA[Round][7]>(first, second));
  b2_g_pair(v[0], v[5], v[10], v[15],
            b2_tail_pair_word<BLAKE2B_SIGMA[Round][8]>(first, second),
            b2_tail_pair_word<BLAKE2B_SIGMA[Round][9]>(first, second));
  b2_g_pair(v[1], v[6], v[11], v[12],
            b2_tail_pair_word<BLAKE2B_SIGMA[Round][10]>(first, second),
            b2_tail_pair_word<BLAKE2B_SIGMA[Round][11]>(first, second));
  b2_g_pair(v[2], v[7], v[8], v[13],
            b2_tail_pair_word<BLAKE2B_SIGMA[Round][12]>(first, second),
            b2_tail_pair_word<BLAKE2B_SIGMA[Round][13]>(first, second));
  b2_g_pair(v[3], v[4], v[9], v[14],
            b2_tail_pair_word<BLAKE2B_SIGMA[Round][14]>(first, second),
            b2_tail_pair_word<BLAKE2B_SIGMA[Round][15]>(first, second));
}

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
    state[i + 8] = b2_pair(BLAKE2B_IV[i]);
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
inline void hash_words_from_tail(const std::uint64_t midstate[8], const std::uint64_t first,
                                 const std::uint32_t second_low, const std::uint32_t index,
                                 std::uint64_t digest[8]) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__INTEL_LLVM_COMPILER) && !defined(__NVPTX__)
  hash_words_from_tail_pair(midstate, first, second_low, index, digest);
#else
  const std::uint64_t second = second_low | (static_cast<std::uint64_t>(index) << 32);
  std::uint64_t v[16];
  for (unsigned i = 0; i < 8; ++i) {
    v[i] = midstate[i];
    v[i + 8] = BLAKE2B_IV[i];
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

inline std::uint32_t digest24(const std::uint64_t digest[8], const unsigned byte) {
  const unsigned word = byte / 8, shift = 8 * (byte % 8);
  std::uint64_t value = digest[word] >> shift;
  if (shift > 40) {
    value |= digest[word + 1] << (64 - shift);
  }
  return static_cast<std::uint32_t>(value) & 0xffffffu;
}

template <typename Spec>
inline void hash_header_midstate(const std::uint8_t header[Spec::header_length],
                                 std::uint64_t h[8]) {
  std::uint8_t parameter[64] = {}, block[128];
  parameter[0] = static_cast<std::uint8_t>(Spec::hash_length);
  parameter[2] = parameter[3] = 1;
  for (unsigned i = 0; i < 16; ++i) {
    parameter[48 + i] = Spec::personal_byte(i);
  }
  for (unsigned i = 0; i < 8; ++i) {
    h[i] = BLAKE2B_IV[i] ^ load64_le(parameter + 8 * i);
  }
  for (unsigned i = 0; i < 128; ++i) {
    block[i] = header[i];
  }
  b2_compress(h, block, 128, false);
}

template <typename Spec>
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
inline void hash_index(const std::uint8_t header[Spec::header_length], const std::uint32_t index,
                       std::uint8_t digest[Spec::hash_length]) {
  std::uint64_t midstate[8];
  hash_header_midstate<Spec>(header, midstate);
  hash_index_from_midstate<Spec>(midstate, header, index, digest);
}

template <typename Spec>
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
inline void row_from_index(const std::uint8_t header[Spec::header_length], const std::uint32_t index,
                           Row<Spec>& row) {
  std::uint64_t midstate[8];
  hash_header_midstate<Spec>(header, midstate);
  row_from_index_midstate<Spec>(midstate, header, index, row);
}

template <typename Spec>
inline void merge_ordered(const Node<Spec>& left, const Node<Spec>& right, Node<Spec>& out) {
  out.first_index = left.first_index;
  for (unsigned i = 0; i < Spec::rounds - 1; ++i) {
    out.fields[i] = left.fields[i + 1] ^ right.fields[i + 1];
  }
  out.fields[Spec::rounds - 1] = 0;
}

template <typename Spec>
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
