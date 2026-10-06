#pragma once

#include <sycl/sycl.hpp>

#include "equihash_core.hpp"

namespace mom_equihash {

template <std::size_t Count>
inline bool has_nonzero_remainder(const std::uint32_t (&values)[Count]) {
  std::uint32_t combined = 0;
  for (const std::uint32_t value : values) combined |= value;
  return combined != 0;
}

struct alignas(16) SplitRecordHead {
  std::uint32_t words[4];
};

template <typename Record>
inline void store_split_record(SplitRecordHead* const heads, std::uint32_t* const tails,
                               const std::size_t index, const Record& record) {
  static_assert(Record::word_count == 5);
  heads[index].words[0] = record.words[0];
  heads[index].words[1] = record.words[1];
  heads[index].words[2] = record.words[2];
  heads[index].words[3] = record.words[3];
  tails[index] = record.words[4];
}

template <typename Record>
inline Record load_split_record(const SplitRecordHead* const heads,
                                const std::uint32_t* const tails, const std::size_t index) {
  static_assert(Record::word_count == 5);
  Record record{};
  record.words[0] = heads[index].words[0];
  record.words[1] = heads[index].words[1];
  record.words[2] = heads[index].words[2];
  record.words[3] = heads[index].words[3];
  record.words[4] = tails[index];
  return record;
}

template <typename Record>
inline void store_aligned_record(Record* const output, const Record& record) {
  if constexpr (sizeof(Record) == 32 && Record::word_count <= 7) {
    auto* const dst = reinterpret_cast<sycl::uint4*>(output);
    dst[0] = sycl::uint4(record.words[0], record.words[1], record.words[2], record.words[3]);
    dst[1] = sycl::uint4(record.words[4], Record::word_count > 5 ? record.words[5] : 0,
                         Record::word_count > 6 ? record.words[6] : 0, 0);
  } else if constexpr (sizeof(Record) == 16 && Record::word_count <= 4) {
    *reinterpret_cast<sycl::uint4*>(output) =
        sycl::uint4(record.words[0], record.words[1], record.words[2],
                    Record::word_count > 3 ? record.words[3] : 0);
  } else if constexpr (sizeof(Record) == 24 && alignof(Record) >= alignof(std::uint64_t)) {
    auto* const dst = reinterpret_cast<std::uint64_t*>(output);
    const auto* const src = reinterpret_cast<const std::uint64_t*>(&record);
    for (unsigned i = 0; i < 3; ++i) {
      dst[i] = src[i];
    }
  } else {
    *output = record;
  }
}

template <typename Spec, unsigned FieldCount>
inline std::uint32_t load_bucket_field(const PackedCollisionRecord<Spec, FieldCount>& record,
                                       const std::uint32_t, const unsigned field) {
  return load_collision_field<Spec, FieldCount>(record, field);
}

template <typename Spec, unsigned LocalBits>
inline std::uint32_t load_generated_field(const PackedBucketLevel0Record<Spec, LocalBits>& record,
                                          const std::uint32_t bucket, const unsigned field) {
  return load_bucket_level0_field(record, bucket, field);
}

template <typename Spec, unsigned FieldCount>
inline std::uint32_t load_bucket_parent(const PackedCollisionRecord<Spec, FieldCount>& record,
                                        const bool right) {
  return right ? load_collision_right<Spec, FieldCount>(record)
               : load_collision_left<Spec, FieldCount>(record);
}

template <typename Spec, unsigned FieldCount, unsigned LocalBits, unsigned ParentBucketBits,
          unsigned ParentSlots>
inline std::uint32_t load_bucket_field(
    const PackedBucketCollisionRecord<Spec, FieldCount, LocalBits, ParentBucketBits, ParentSlots>&
        record,
    const std::uint32_t bucket, const unsigned field) {
  return load_bucket_collision_field(record, bucket, field);
}

template <typename Spec, unsigned FieldCount, unsigned LocalBits, unsigned ParentBucketBits,
          unsigned ParentSlots>
inline std::uint32_t load_bucket_parent(
    const PackedBucketCollisionRecord<Spec, FieldCount, LocalBits, ParentBucketBits, ParentSlots>&
        record,
    const bool right) {
  return load_bucket_collision_parent(record, right);
}

template <typename Spec, unsigned FieldCount, unsigned LocalBits, unsigned ParentBucketBits,
          unsigned ParentSlots>
inline bool bucket_parents_overlap(
    const PackedBucketCollisionRecord<Spec, FieldCount, LocalBits, ParentBucketBits, ParentSlots>& a,
    const PackedBucketCollisionRecord<Spec, FieldCount, LocalBits, ParentBucketBits, ParentSlots>& b) {
  using Record = PackedBucketCollisionRecord<Spec, FieldCount, LocalBits, ParentBucketBits,
                                             ParentSlots>;
  constexpr unsigned parent = Record::parent_bit;
  const std::uint32_t a_bucket = packed_load(a.words, Record::word_count, parent, ParentBucketBits);
  const std::uint32_t b_bucket = packed_load(b.words, Record::word_count, parent, ParentBucketBits);
  if (a_bucket != b_bucket) return false;
  const std::uint32_t a_left = packed_load(
      a.words, Record::word_count, parent + ParentBucketBits, Record::parent_slot_bits);
  const std::uint32_t a_right = packed_load(
      a.words, Record::word_count, parent + ParentBucketBits + Record::parent_slot_bits,
      Record::parent_slot_bits);
  const std::uint32_t b_left = packed_load(
      b.words, Record::word_count, parent + ParentBucketBits, Record::parent_slot_bits);
  const std::uint32_t b_right = packed_load(
      b.words, Record::word_count, parent + ParentBucketBits + Record::parent_slot_bits,
      Record::parent_slot_bits);
  return a_left == b_left || a_left == b_right || a_right == b_left || a_right == b_right;
}

template <typename Spec, unsigned FieldCount>
inline void store_bucket_output(PackedCollisionRecord<Spec, FieldCount>& record,
                                const std::uint32_t fields[FieldCount],
                                const std::uint32_t bucket, const std::uint32_t left_slot,
                                const std::uint32_t right_slot, const std::uint32_t input_slots) {
  const std::uint32_t base = bucket * input_slots;
  store_collision(record, fields, base + left_slot, base + right_slot);
}

template <typename Spec, unsigned FieldCount, unsigned LocalBits, unsigned ParentBucketBits,
          unsigned ParentSlots>
inline void store_bucket_output(
    PackedBucketCollisionRecord<Spec, FieldCount, LocalBits, ParentBucketBits, ParentSlots>& record,
    const std::uint32_t fields[FieldCount], const std::uint32_t bucket,
    const std::uint32_t left_slot, const std::uint32_t right_slot, const std::uint32_t) {
  store_bucket_collision(record, fields, bucket, left_slot, right_slot);
}

template <typename Spec, typename InputRecord, typename OutputRecord>
inline bool make_generated_collision_unordered(
    const InputRecord& a, const InputRecord& b, const std::uint32_t bucket,
    const std::uint32_t a_slot, const std::uint32_t b_slot, const std::uint32_t input_slots,
    OutputRecord& out, std::uint32_t& next_field) {
  std::uint32_t values[Spec::rounds - 1];
  for (unsigned i = 0; i < Spec::rounds - 1; ++i)
    values[i] = load_generated_field(a, bucket, i + 1) ^
                load_generated_field(b, bucket, i + 1);
  // Optimized Equihash solvers prune a non-final zero remainder: it overwhelmingly comes from
  // duplicate provenance and otherwise multiplies into millions of unusable late-round trees.
  // Every emitted root is still reconstructed and independently verified before it is returned.
  if constexpr (!InputRecord::compact_selection_head) {
    if (!has_nonzero_remainder(values)) return false;
  }
  next_field = values[0];
  if (a_slot < b_slot) {
    store_bucket_output(out, values, bucket, a_slot, b_slot, input_slots);
  } else {
    store_bucket_output(out, values, bucket, b_slot, a_slot, input_slots);
  }
  return true;
}

template <typename Spec, unsigned LocalBits, typename OutputRecord>
inline bool make_split_generated_collision_unordered(
    const SplitRecordHead& a, const SplitRecordHead& b, const std::uint32_t bucket,
    const std::uint32_t a_slot, const std::uint32_t b_slot, const std::uint32_t input_slots,
    OutputRecord& out, std::uint32_t& next_field) {
  using InputRecord = PackedBucketLevel0Record<Spec, LocalBits>;
  std::uint32_t values[Spec::rounds - 1];
#ifdef MOM_SYCL_HAS_HIP
  if constexpr (Spec::n == 144 && Spec::k == 5) {
    const std::uint32_t x0 = a.words[0] ^ b.words[0];
    const std::uint32_t x1 = a.words[1] ^ b.words[1];
    const std::uint32_t x2 = a.words[2] ^ b.words[2];
    const std::uint32_t x3 = a.words[3] ^ b.words[3];
    values[0] = (x0 >> 12) | ((x1 & 0xfu) << 20);
    values[1] = (x1 >> 4) & 0xffffffu;
    values[2] = (x1 >> 28) | ((x2 & 0xfffffu) << 4);
    values[3] = (x2 >> 20) | ((x3 & 0xfffu) << 12);
    values[4] = (x3 >> 12) & 0xfffffu;
  } else
#endif
  {
    if constexpr (InputRecord::compact_selection_head) {
      constexpr unsigned final_field_bit =
          LocalBits + (Spec::rounds - 2) * Spec::collision_bits;
      for (unsigned i = 0; i < Spec::rounds - 1; ++i) {
        const unsigned bit = LocalBits + i * Spec::collision_bits;
        const unsigned width = i == Spec::rounds - 2 ? 128 - final_field_bit : Spec::collision_bits;
        values[i] = packed_load(a.words, 4, bit, width) ^
                    packed_load(b.words, 4, bit, width);
      }
    } else {
      for (unsigned i = 0; i < Spec::rounds - 1; ++i)
        values[i] = packed_load(a.words, 4, i * Spec::collision_bits, Spec::collision_bits) ^
                    packed_load(b.words, 4, i * Spec::collision_bits, Spec::collision_bits);
      if (!has_nonzero_remainder(values)) return false;
    }
  }
  next_field = values[0];
  if (a_slot < b_slot) {
    store_bucket_output(out, values, bucket, a_slot, b_slot, input_slots);
  } else {
    store_bucket_output(out, values, bucket, b_slot, a_slot, input_slots);
  }
  return true;
}

template <typename Spec, unsigned FieldCount, unsigned LocalBits, typename InputRecord,
          typename OutputRecord>
inline bool make_split_bucket_collision_unordered(
    const SplitRecordHead& a, const SplitRecordHead& b, const std::uint32_t bucket,
    const std::uint32_t* tails, const std::size_t a_index, const std::size_t b_index,
    const std::uint32_t a_slot, const std::uint32_t b_slot,
    const std::uint32_t input_slots, OutputRecord& out, std::uint32_t& next_field) {
  static_assert(InputRecord::word_count == 5);
  static_assert(LocalBits + (FieldCount - 1) * Spec::collision_bits +
                    InputRecord::parent_bucket_bits <=
                128);
  constexpr unsigned parent_bit = LocalBits + (FieldCount - 1) * Spec::collision_bits;
  const std::uint32_t a_bucket = packed_load(a.words, 4, parent_bit,
                                              InputRecord::parent_bucket_bits);
  const std::uint32_t b_bucket = packed_load(b.words, 4, parent_bit,
                                              InputRecord::parent_bucket_bits);
  if (a_bucket == b_bucket) {
    InputRecord a_record{}, b_record{};
    for (unsigned i = 0; i < 4; ++i) {
      a_record.words[i] = a.words[i];
      b_record.words[i] = b.words[i];
    }
    a_record.words[4] = tails[a_index];
    b_record.words[4] = tails[b_index];
    const std::uint32_t a_left = load_bucket_parent(a_record, false);
    const std::uint32_t a_right = load_bucket_parent(a_record, true);
    const std::uint32_t b_left = load_bucket_parent(b_record, false);
    const std::uint32_t b_right = load_bucket_parent(b_record, true);
    if (a_left == b_left || a_left == b_right || a_right == b_left || a_right == b_right)
      return false;
  }
  std::uint32_t values[FieldCount - 1];
  if constexpr (Spec::collision_bits == 24 && FieldCount == 5 && LocalBits == 12 &&
                InputRecord::parent_bucket_bits == 12) {
    values[0] = ((a.words[0] >> 12) | ((a.words[1] & 0xfu) << 20)) ^
                ((b.words[0] >> 12) | ((b.words[1] & 0xfu) << 20));
    values[1] = ((a.words[1] >> 4) & 0xffffffu) ^
                ((b.words[1] >> 4) & 0xffffffu);
    values[2] = ((a.words[1] >> 28) | ((a.words[2] & 0xfffffu) << 4)) ^
                ((b.words[1] >> 28) | ((b.words[2] & 0xfffffu) << 4));
    values[3] = ((a.words[2] >> 20) | ((a.words[3] & 0xfffu) << 12)) ^
                ((b.words[2] >> 20) | ((b.words[3] & 0xfffu) << 12));
  } else {
    for (unsigned i = 0; i < FieldCount - 1; ++i)
      values[i] = packed_load(a.words, 4, LocalBits + i * Spec::collision_bits,
                              Spec::collision_bits) ^
                  packed_load(b.words, 4, LocalBits + i * Spec::collision_bits,
                              Spec::collision_bits);
  }
  if (!has_nonzero_remainder(values)) return false;
  next_field = values[0];
  if (a_slot < b_slot) {
    store_bucket_output(out, values, bucket, a_slot, b_slot, input_slots);
  } else {
    store_bucket_output(out, values, bucket, b_slot, a_slot, input_slots);
  }
  return true;
}

template <typename Spec, typename InputRecord, typename OutputRecord>
inline bool make_bucket_collision_unordered(
    const InputRecord& a, const InputRecord& b, const std::uint32_t bucket,
    const std::uint32_t a_slot, const std::uint32_t b_slot, const std::uint32_t input_slots,
    OutputRecord& out, std::uint32_t& next_field) {
  constexpr unsigned fields = InputRecord::field_count;
  static_assert(OutputRecord::field_count + 1 == fields);
  if (bucket_parents_overlap(a, b)) return false;
  if constexpr (requires { InputRecord::local_bits; OutputRecord::local_bits; }) {
    if constexpr (Spec::n == 192 && fields == 7 && InputRecord::local_bits == 12 &&
                  InputRecord::parent_bucket_bits == 12 &&
                  InputRecord::parent_slot_bits == 14 && OutputRecord::local_bits == 12 &&
                  OutputRecord::parent_bucket_bits == 12 &&
                  OutputRecord::parent_slot_bits == 14) {
      std::uint32_t x[5];
      for (unsigned i = 0; i < 5; ++i) {
        x[i] = a.words[i] ^ b.words[i];
      }
      const std::uint32_t v1 = (x[1] >> 4) & 0xffffffu;
      const std::uint32_t v2 = (x[1] >> 28) | ((x[2] & 0xfffffu) << 4);
      const std::uint32_t v3 = (x[2] >> 20) | ((x[3] & 0xfffu) << 12);
      const std::uint32_t v4 = (x[3] >> 12) | ((x[4] & 0xfu) << 20);
      const std::uint32_t v5 = (x[4] >> 4) & 0xffffffu;
      next_field = (x[0] >> 12) | ((x[1] & 0xfu) << 20);
      if ((next_field | v1 | v2 | v3 | v4 | v5) == 0) return false;
      out.words[0] = (next_field & 0xfffu) | (v1 << 12);
      out.words[1] = (v1 >> 20) | (v2 << 4) | (v3 << 28);
      out.words[2] = (v3 >> 4) | (v4 << 20);
      out.words[3] = (v4 >> 12) | (v5 << 12);
      const std::uint32_t left = a_slot < b_slot ? a_slot : b_slot;
      const std::uint32_t right = a_slot < b_slot ? b_slot : a_slot;
      out.words[4] = (v5 >> 20) | (bucket << 4) | (left << 16) | (right << 30);
      out.words[5] = right >> 2;
      return true;
    }
  }
  std::uint32_t values[fields - 1];
  for (unsigned i = 0; i < fields - 1; ++i)
    values[i] = load_bucket_field(a, bucket, i + 1) ^ load_bucket_field(b, bucket, i + 1);
  if constexpr (!InputRecord::compact_trailing_field) {
    if (!has_nonzero_remainder(values)) return false;
  }
  next_field = values[0];
  if (a_slot < b_slot)
    store_bucket_output(out, values, bucket, a_slot, b_slot, input_slots);
  else
    store_bucket_output(out, values, bucket, b_slot, a_slot, input_slots);
  return true;
}

template <typename Spec, typename InputRecord>
inline bool make_bucket_root_unordered(
    const InputRecord& a, const InputRecord& b, const std::uint32_t bucket,
    const std::uint32_t a_slot, const std::uint32_t b_slot,
    const std::uint32_t input_slots, ZHashRootRecord<Spec>& root) {
  const std::uint32_t field =
      load_bucket_field(a, bucket, 1) ^ load_bucket_field(b, bucket, 1);
  if (field != 0) return false;
  const std::uint32_t a_left = load_bucket_parent(a, false), a_right = load_bucket_parent(a, true);
  const std::uint32_t b_left = load_bucket_parent(b, false), b_right = load_bucket_parent(b, true);
  if (a_left == b_left || a_left == b_right || a_right == b_left || a_right == b_right)
    return false;
  const std::uint32_t base = bucket * input_slots;
  if (a_slot < b_slot) {
    store_collision<Spec, 1>(root, &field, base + a_slot, base + b_slot);
  } else {
    store_collision<Spec, 1>(root, &field, base + b_slot, base + a_slot);
  }
  return true;
}

// Level zero is written directly into the selected bucket arena. Counts are reservation counts,
// not just stored counts: a slot at or beyond the bounded capacity is never written, and the
// separate overflow counter records every dropped row for a safe retry/scheduler decision. Session
// arenas are separate allocations and its counter pointers are disjoint slices, which permits the
// kernel argument restrict promise below.
template <typename Spec, typename Layout> class GenerateBucketedKernel;
template <typename Spec, typename Layout>
inline sycl::event submit_generation_bucketed(
    sycl::queue& queue, const std::uint64_t* device_midstate, const std::uint64_t tail_first,
    const std::uint32_t tail_second,
    SplitRecordHead* device_bucket_heads, std::uint32_t* device_bucket_tails,
    std::uint32_t* device_bucket_counts,
    std::uint32_t* device_overflow_count) {
  using atomic_u32 = sycl::atomic_ref<std::uint32_t, sycl::memory_order::relaxed,
                                      sycl::memory_scope::device,
                                      sycl::access::address_space::global_space>;
#if defined(MOM_SYCL_HAS_HIP)
  // WG64 cuts RX 9060 XT generation from 8.57 to 8.08 ms and the full solve by 1.6%; other
  // compiler targets retain the existing WG128 layout.
  constexpr std::size_t work_group = 64;
  // Persistent scheduling cuts RX 9060 XT generation from 8.07 to 7.72 ms while each work-item
  // advances through several hashes; one million workers also outperformed the 524288-worker case.
  constexpr std::size_t global = 1u << 20;
#else
  constexpr std::size_t work_group = 128;
  constexpr std::size_t global =
      (static_cast<std::size_t>(Spec::hash_count) + work_group - 1) / work_group * work_group;
#endif
  static_assert(global % work_group == 0);
  return queue.parallel_for<GenerateBucketedKernel<Spec, Layout>>(
      sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(work_group)),
      [=](sycl::nd_item<1> item) MOM_SYCL_KERNEL_ARGS_RESTRICT {
        for (std::uint32_t hash = static_cast<std::uint32_t>(item.get_global_id(0));
             hash < Spec::hash_count; hash += static_cast<std::uint32_t>(global)) {
          std::uint64_t digest[8];
          hash_words_from_tail<Spec>(device_midstate, tail_first, tail_second, hash, digest);
          for (unsigned segment = 0; segment < Spec::indices_per_hash; ++segment) {
            const unsigned row_index = hash * Spec::indices_per_hash + segment;
            if (row_index >= Spec::row_count) continue;
            std::uint32_t fields[Spec::rounds];
            const unsigned offset = segment * Spec::segment_bytes;
            for (unsigned field = 0; field < Spec::rounds; ++field) {
              const unsigned o = offset + field * Spec::collision_bytes;
              fields[field] = digest24(digest, o);
            }
            PackedBucketLevel0Record<Spec, Layout::local_bits> record{};
            store_bucket_level0(record, fields, row_index);
            const unsigned bucket = Layout::bucket_for(fields[0]);
            const std::uint32_t slot = atomic_u32(device_bucket_counts[bucket]).fetch_add(1u);
            if (slot < Layout::bucket_slot_capacity) {
              store_split_record(device_bucket_heads, device_bucket_tails,
                                 bucket * Layout::bucket_slot_capacity + slot,
                                 record);
            } else {
              atomic_u32(device_overflow_count[0]).fetch_add(1u);
            }
          }
        }
      });
}

template <typename Spec, typename Layout> class GenerateBucketedFullKernel;
template <typename Spec, typename Layout>
inline sycl::event submit_generation_bucketed_full(
    sycl::queue& queue, const std::uint64_t* device_midstate, const std::uint64_t* device_tail,
    PackedBucketLevel0Record<Spec, Layout::local_bits>* device_bucket,
    std::uint32_t* device_bucket_counts, std::uint32_t* device_overflow_count) {
  using atomic_u32 = sycl::atomic_ref<std::uint32_t, sycl::memory_order::relaxed,
                                      sycl::memory_scope::device,
                                      sycl::access::address_space::global_space>;
  constexpr std::size_t work_group = 512;
#ifdef MOM_EQUIHASH_GENERATION_WORKERS
  constexpr std::size_t global = MOM_EQUIHASH_GENERATION_WORKERS;
#else
  constexpr std::size_t global = 2097152;
#endif
  static_assert(global % work_group == 0);
  return queue.parallel_for<GenerateBucketedFullKernel<Spec, Layout>>(
      sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(work_group)),
      [=](sycl::nd_item<1> item) MOM_SYCL_KERNEL_ARGS_RESTRICT {
        for (std::uint32_t hash = static_cast<std::uint32_t>(item.get_global_id(0));
             hash < Spec::hash_count; hash += global) {
          std::uint64_t digest[8];
          hash_words_from_tail<Spec>(device_midstate, device_tail[0],
                                     static_cast<std::uint32_t>(device_tail[1]), hash, digest);
          for (unsigned segment = 0; segment < Spec::indices_per_hash; ++segment) {
            const unsigned row_index = hash * Spec::indices_per_hash + segment;
            if (row_index >= Spec::row_count) continue;
            std::uint32_t fields[Spec::rounds];
            const unsigned offset = segment * Spec::segment_bytes;
            for (unsigned field = 0; field < Spec::rounds; ++field)
              fields[field] = digest24(digest, offset + field * Spec::collision_bytes);
            PackedBucketLevel0Record<Spec, Layout::local_bits> record{};
            store_bucket_level0(record, fields, row_index);
            const unsigned bucket = Layout::bucket_for(fields[0]);
            const std::uint32_t slot = atomic_u32(device_bucket_counts[bucket]).fetch_add(1u);
            if (slot < Layout::bucket_slot_capacity)
              store_aligned_record(device_bucket +
                  bucket * Layout::bucket_slot_capacity + slot, record);
            else
              atomic_u32(device_overflow_count[0]).fetch_add(1u);
          }
        }
      });
}

// Join one level-zero bucket and place each collision directly in its next-round bucket. This
// preserves absolute parent slots while avoiding a full flat-arena write and scatter pass.
template <typename Spec, typename InputLayout, typename OutputLayout, typename OutputRecord>
class BucketCollisionDirectKernel;

template <typename Spec, typename InputLayout, typename OutputLayout, typename OutputRecord>
inline sycl::event submit_bucket_round_bucketed(
    sycl::queue& queue, const SplitRecordHead* device_input_heads,
    const std::uint32_t* device_input_tails, const std::uint32_t* device_input_counts,
    SplitRecordHead* device_output_heads, std::uint32_t* device_output_tails,
    std::uint32_t* device_output_counts, std::uint32_t* device_overflow_count) {
  static_assert(sizeof(OutputRecord) == 20);
  using global_atomic_u32 = sycl::atomic_ref<std::uint32_t, sycl::memory_order::relaxed,
                                             sycl::memory_scope::device,
                                             sycl::access::address_space::global_space>;
  using local_atomic_u32 = sycl::atomic_ref<std::uint32_t, sycl::memory_order::relaxed,
                                            sycl::memory_scope::work_group,
                                            sycl::access::address_space::local_space>;
  const size_t global =
      static_cast<size_t>(InputLayout::bucket_count) * InputLayout::collision_work_group;
  return queue.submit([&](sycl::handler& h) {
    sycl::local_accessor<std::uint32_t, 1> bin_head(sycl::range<1>(InputLayout::local_bins), h);
    sycl::local_accessor<std::uint16_t, 1> bin_next(
        sycl::range<1>(InputLayout::bucket_slot_capacity), h);
    h.parallel_for<BucketCollisionDirectKernel<Spec, InputLayout, OutputLayout, OutputRecord>>(
        sycl::nd_range<1>(sycl::range<1>(global),
                          sycl::range<1>(InputLayout::collision_work_group)),
        [=](sycl::nd_item<1> item) MOM_SYCL_KERNEL_ARGS_RESTRICT {
          const unsigned bucket = static_cast<unsigned>(item.get_group(0));
          const unsigned lane = static_cast<unsigned>(item.get_local_id(0));
          const unsigned width = static_cast<unsigned>(item.get_local_range(0));
          const std::uint32_t count =
              device_input_counts[bucket] < InputLayout::bucket_slot_capacity
                  ? device_input_counts[bucket] : InputLayout::bucket_slot_capacity;
          const std::uint32_t base = bucket * InputLayout::bucket_slot_capacity;
          for (unsigned i = lane; i < InputLayout::local_bins; i += width) {
            bin_head[i] = 0;
          }
          sycl::group_barrier(item.get_group());
          for (unsigned i = lane; i < count; i += width) {
            unsigned bin;
            if constexpr (PackedBucketLevel0Record<
                              Spec, InputLayout::local_bits>::compact_selection_head) {
              bin = device_input_heads[base + i].words[0] &
                    (InputLayout::local_bins - 1u);
            } else {
              bin = (device_input_heads[base + i].words[3] >> 24) |
                    ((device_input_tails[base + i] &
                      ((1u << (InputLayout::local_bits - 8)) - 1u)) << 8);
            }
            bin_next[i] = static_cast<std::uint16_t>(
                local_atomic_u32(bin_head[bin]).exchange(i + 1u));
          }
          sycl::group_barrier(item.get_group());
          for (unsigned a = lane; a < count; a += width) {
            unsigned link = bin_next[a];
            while (link != 0) {
              const unsigned b = link - 1u;
              link = bin_next[b];
              OutputRecord collision{};
              std::uint32_t next_field = 0;
              if (!make_split_generated_collision_unordered<Spec, InputLayout::local_bits>(
                      device_input_heads[base + a], device_input_heads[base + b], bucket, a, b,
                      InputLayout::bucket_slot_capacity, collision, next_field))
                continue;
              const unsigned output_bucket = OutputLayout::bucket_for(next_field);
              const std::uint32_t slot =
                  global_atomic_u32(device_output_counts[output_bucket]).fetch_add(1u);
              if (slot >= OutputLayout::bucket_slot_capacity) {
                global_atomic_u32(device_overflow_count[0]).fetch_add(1u);
                continue;
              }
              const std::uint32_t output =
                  output_bucket * OutputLayout::bucket_slot_capacity + slot;
              store_split_record(device_output_heads, device_output_tails, output, collision);
            }
          }
      });
  });
}

template <typename Spec, typename InputLayout, typename OutputLayout, typename OutputRecord,
          unsigned Partitions = 4>
class BucketCollisionDirectFullKernel;
template <typename Spec, typename InputLayout, typename OutputLayout, typename OutputRecord,
          unsigned Partitions = 4>
inline sycl::event submit_bucket_round_bucketed_full(
    sycl::queue& queue,
    const PackedBucketLevel0Record<Spec, InputLayout::local_bits>* device_input,
    const std::uint32_t* device_input_counts, OutputRecord* device_output,
    std::uint32_t* device_output_counts, std::uint32_t* device_overflow_count) {
  using InputRecord = PackedBucketLevel0Record<Spec, InputLayout::local_bits>;
  using global_atomic_u32 = sycl::atomic_ref<std::uint32_t, sycl::memory_order::relaxed,
                                             sycl::memory_scope::device,
                                             sycl::access::address_space::global_space>;
  using local_atomic_u32 = sycl::atomic_ref<std::uint32_t, sycl::memory_order::relaxed,
                                            sycl::memory_scope::work_group,
                                            sycl::access::address_space::local_space>;
  static_assert(InputLayout::local_bins % Partitions == 0);
  constexpr unsigned WorkGroup = InputLayout::collision_work_group;
  constexpr unsigned partition_bins = InputLayout::local_bins / Partitions;
  constexpr unsigned selected_capacity =
      (InputLayout::bucket_slot_capacity + Partitions - 1) / Partitions + 512;
  const size_t global = static_cast<size_t>(InputLayout::bucket_count) * Partitions * WorkGroup;
  return queue.submit([&](sycl::handler& h) {
    sycl::local_accessor<std::uint32_t, 1> bin_head(sycl::range<1>(partition_bins), h);
    sycl::local_accessor<std::uint16_t, 1> selected(sycl::range<1>(selected_capacity), h);
    // Cache each selected record's local bin here, then reuse the storage for collision links.
    sycl::local_accessor<std::uint16_t, 1> bin_next(sycl::range<1>(selected_capacity), h);
    sycl::local_accessor<std::uint32_t, 0> selected_count(h);
    h.parallel_for<BucketCollisionDirectFullKernel<Spec, InputLayout, OutputLayout, OutputRecord,
                                                    Partitions>>(
        sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(WorkGroup)),
        [=](sycl::nd_item<1> item) MOM_SYCL_KERNEL_ARGS_RESTRICT {
          const unsigned group = static_cast<unsigned>(item.get_group(0));
          const unsigned bucket = group / Partitions;
          const unsigned partition = group % Partitions;
          const unsigned lane = static_cast<unsigned>(item.get_local_id(0));
          const unsigned width = static_cast<unsigned>(item.get_local_range(0));
          const std::uint32_t count =
              device_input_counts[bucket] < InputLayout::bucket_slot_capacity
                  ? device_input_counts[bucket] : InputLayout::bucket_slot_capacity;
          const std::uint32_t base = bucket * InputLayout::bucket_slot_capacity;
          for (unsigned i = lane; i < partition_bins; i += width) {
            bin_head[i] = 0;
          }
          if (lane == 0) {
            selected_count = 0;
          }
          sycl::group_barrier(item.get_group());
          for (unsigned i = lane; i < count; i += width) {
            const unsigned bin = InputLayout::local_bin(
                load_bucket_level0_field(device_input[base + i], bucket, 0));
            if (bin / partition_bins != partition) continue;
            const unsigned slot = local_atomic_u32(selected_count).fetch_add(1u);
            if (slot < selected_capacity) {
              selected[slot] = static_cast<std::uint16_t>(i);
              bin_next[slot] = static_cast<std::uint16_t>(bin % partition_bins);
            }
          }
          sycl::group_barrier(item.get_group());
          const unsigned selected_size = selected_count;
          if (selected_size > selected_capacity) {
            if (lane == 0) {
              global_atomic_u32(device_overflow_count[0]).fetch_add(1u);
            }
            return;
          }
          for (unsigned i = lane; i < selected_size; i += width) {
            const unsigned bin = bin_next[i];
            bin_next[i] = static_cast<std::uint16_t>(
                local_atomic_u32(bin_head[bin]).exchange(i + 1u));
          }
          sycl::group_barrier(item.get_group());
          for (unsigned ai = lane; ai < selected_size; ai += width) {
            const unsigned a = selected[ai];
            const InputRecord a_record = device_input[base + a];
            unsigned link = bin_next[ai];
            while (link != 0) {
              const unsigned bi = link - 1u;
              link = bin_next[bi];
              const unsigned b = selected[bi];
              const InputRecord b_record = device_input[base + b];
              OutputRecord collision{};
              std::uint32_t next_field = 0;
              if (!make_generated_collision_unordered<Spec>(
                      a_record, b_record, bucket, a, b,
                      InputLayout::bucket_slot_capacity, collision, next_field))
                continue;
              const unsigned output_bucket = OutputLayout::bucket_for(next_field);
              const std::uint32_t slot =
                  global_atomic_u32(device_output_counts[output_bucket]).fetch_add(1u);
              if (slot >= OutputLayout::bucket_slot_capacity) {
                global_atomic_u32(device_overflow_count[0]).fetch_add(1u);
                continue;
              }
              store_aligned_record(device_output +
                  output_bucket * OutputLayout::bucket_slot_capacity + slot, collision);
            }
          }
        });
  });
}

template <typename InputLayout, unsigned Partitions>
inline constexpr unsigned partitioned_collision_selected_capacity =
    (InputLayout::bucket_slot_capacity + Partitions - 1) / Partitions + 512;

template <typename InputLayout, unsigned Partitions, bool CacheHead>
inline constexpr std::size_t partitioned_split_collision_local_bytes =
    (InputLayout::local_bins / Partitions) * sizeof(std::uint32_t) +
    2 * partitioned_collision_selected_capacity<InputLayout, Partitions> *
        sizeof(std::uint16_t) +
    (CacheHead ? 4 * partitioned_collision_selected_capacity<InputLayout, Partitions> : 1) *
        sizeof(std::uint32_t) +
    sizeof(std::uint32_t);

template <typename Spec, typename InputLayout, typename OutputLayout, typename OutputRecord,
          unsigned Partitions, bool CacheHead = false>
class BucketCollisionPartitionedKernel;

// Adapt the disjoint-bin partitioning used by the later full-record rounds to split records. Each
// collision key belongs to exactly one partition; a skewed partition reports overflow for retry.
template <typename Spec, typename InputLayout, typename OutputLayout, typename OutputRecord,
          unsigned Partitions, bool CacheHead = false>
inline sycl::event submit_bucket_round_bucketed_partitioned(
    sycl::queue& queue, const SplitRecordHead* device_input_heads,
    const std::uint32_t* device_input_tails, const std::uint32_t* device_input_counts,
    SplitRecordHead* device_output_heads, std::uint32_t* device_output_tails,
    std::uint32_t* device_output_counts, std::uint32_t* device_overflow_count) {
  static_assert(sizeof(OutputRecord) == 20 && Partitions > 1);
  static_assert(InputLayout::local_bins % Partitions == 0);
  using global_atomic_u32 = sycl::atomic_ref<std::uint32_t, sycl::memory_order::relaxed,
                                             sycl::memory_scope::device,
                                             sycl::access::address_space::global_space>;
  using local_atomic_u32 = sycl::atomic_ref<std::uint32_t, sycl::memory_order::relaxed,
                                            sycl::memory_scope::work_group,
                                            sycl::access::address_space::local_space>;
  constexpr unsigned work_group = InputLayout::collision_work_group;
  constexpr unsigned partition_bins = InputLayout::local_bins / Partitions;
  constexpr unsigned selected_capacity =
      partitioned_collision_selected_capacity<InputLayout, Partitions>;
  constexpr unsigned cached_words = CacheHead ? selected_capacity * 4 : 1;
  const std::size_t global =
      static_cast<std::size_t>(InputLayout::bucket_count) * Partitions * work_group;
  return queue.submit([&](sycl::handler& handler) {
    sycl::local_accessor<std::uint32_t, 1> bin_head(sycl::range<1>(partition_bins), handler);
    sycl::local_accessor<std::uint16_t, 1> selected(sycl::range<1>(selected_capacity), handler);
    // Cache each selected record's local bin here, then reuse the storage for collision links.
    sycl::local_accessor<std::uint16_t, 1> bin_next(sycl::range<1>(selected_capacity), handler);
    // Collision fields fit in the four-word head; tails remain global for selection/overlap checks.
    sycl::local_accessor<std::uint32_t, 1> cached_heads(sycl::range<1>(cached_words), handler);
    sycl::local_accessor<std::uint32_t, 0> selected_count(handler);
    handler.parallel_for<BucketCollisionPartitionedKernel<
        Spec, InputLayout, OutputLayout, OutputRecord, Partitions, CacheHead>>(
        sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(work_group)),
        [=](sycl::nd_item<1> item) MOM_SYCL_KERNEL_ARGS_RESTRICT {
          const unsigned group = static_cast<unsigned>(item.get_group(0));
          const unsigned bucket = group / Partitions;
          const unsigned partition = group % Partitions;
          const unsigned lane = static_cast<unsigned>(item.get_local_id(0));
          const unsigned width = static_cast<unsigned>(item.get_local_range(0));
          const std::uint32_t count =
              device_input_counts[bucket] < InputLayout::bucket_slot_capacity
                  ? device_input_counts[bucket] : InputLayout::bucket_slot_capacity;
          const std::uint32_t base = bucket * InputLayout::bucket_slot_capacity;
          for (unsigned i = lane; i < partition_bins; i += width) {
            bin_head[i] = 0;
          }
          if (lane == 0) {
            selected_count = 0;
          }
          sycl::group_barrier(item.get_group());
          for (unsigned i = lane; i < count; i += width) {
            unsigned bin;
            if constexpr (PackedBucketLevel0Record<
                              Spec, InputLayout::local_bits>::compact_selection_head) {
              bin = device_input_heads[base + i].words[0] &
                    (InputLayout::local_bins - 1u);
            } else {
              bin = (device_input_heads[base + i].words[3] >> 24) |
                    ((device_input_tails[base + i] &
                      ((1u << (InputLayout::local_bits - 8)) - 1u)) << 8);
            }
            if (bin / partition_bins != partition) continue;
            const unsigned slot = local_atomic_u32(selected_count).fetch_add(1u);
            if (slot < selected_capacity) {
              selected[slot] = static_cast<std::uint16_t>(i);
              bin_next[slot] = static_cast<std::uint16_t>(bin % partition_bins);
              if constexpr (CacheHead) {
                for (unsigned word = 0; word < 4; ++word)
                  cached_heads[word * selected_capacity + slot] =
                      device_input_heads[base + i].words[word];
              }
            }
          }
          sycl::group_barrier(item.get_group());
          const unsigned selected_size = selected_count;
          if (selected_size > selected_capacity) {
            if (lane == 0) {
              global_atomic_u32(device_overflow_count[0]).fetch_add(1u);
            }
            return;
          }
          for (unsigned i = lane; i < selected_size; i += width) {
            const unsigned bin = bin_next[i];
            bin_next[i] = static_cast<std::uint16_t>(
                local_atomic_u32(bin_head[bin]).exchange(i + 1u));
          }
          sycl::group_barrier(item.get_group());
          for (unsigned ai = lane; ai < selected_size; ai += width) {
            const unsigned a = selected[ai];
            SplitRecordHead a_head{};
            if constexpr (CacheHead) {
              for (unsigned word = 0; word < 4; ++word)
                a_head.words[word] = cached_heads[word * selected_capacity + ai];
            } else {
              a_head = device_input_heads[base + a];
            }
            unsigned link = bin_next[ai];
            while (link != 0) {
              const unsigned bi = link - 1u;
              link = bin_next[bi];
              const unsigned b = selected[bi];
              SplitRecordHead b_head{};
              if constexpr (CacheHead) {
                for (unsigned word = 0; word < 4; ++word)
                  b_head.words[word] = cached_heads[word * selected_capacity + bi];
              } else {
                b_head = device_input_heads[base + b];
              }
              OutputRecord collision{};
              std::uint32_t next_field = 0;
              if (!make_split_generated_collision_unordered<Spec, InputLayout::local_bits>(
                      a_head, b_head, bucket, a, b,
                      InputLayout::bucket_slot_capacity, collision, next_field))
                continue;
              const unsigned output_bucket = OutputLayout::bucket_for(next_field);
              const std::uint32_t slot =
                  global_atomic_u32(device_output_counts[output_bucket]).fetch_add(1u);
              if (slot >= OutputLayout::bucket_slot_capacity) {
                global_atomic_u32(device_overflow_count[0]).fetch_add(1u);
                continue;
              }
              const std::uint32_t output =
                  output_bucket * OutputLayout::bucket_slot_capacity + slot;
              store_split_record(device_output_heads, device_output_tails, output, collision);
            }
          }
        });
  });
}

#ifndef MOM_SYCL_HAS_HIP
template <typename Spec, unsigned FieldCount, typename InputLayout, typename OutputLayout,
          typename InputRecord, typename OutputRecord>
class CollisionBucketSplitOutputKernel;

template <typename Spec, unsigned FieldCount, typename InputLayout, typename OutputLayout,
          typename InputRecord, typename OutputRecord, unsigned Partitions>
class CollisionBucketSplitOutputPartitionedKernel;

template <typename Spec, unsigned FieldCount, typename InputLayout, typename OutputLayout,
          typename InputRecord, typename OutputRecord, unsigned Partitions>
inline sycl::event submit_collision_round_bucketed_split_output_partitioned(
    sycl::queue& queue, const InputRecord* device_input,
    const std::uint32_t* device_input_counts, SplitRecordHead* device_output_heads,
    std::uint32_t* device_output_tails, std::uint32_t* device_output_counts,
    std::uint32_t* device_overflow_count) {
  static_assert(FieldCount > 1 && Partitions > 1 && InputRecord::field_count == FieldCount);
  static_assert(OutputRecord::word_count == 5 && OutputRecord::field_count + 1 == FieldCount);
  static_assert(InputLayout::local_bins % Partitions == 0);
  using global_atomic_u32 = sycl::atomic_ref<std::uint32_t, sycl::memory_order::relaxed,
                                             sycl::memory_scope::device,
                                             sycl::access::address_space::global_space>;
  using local_atomic_u32 = sycl::atomic_ref<std::uint32_t, sycl::memory_order::relaxed,
                                            sycl::memory_scope::work_group,
                                            sycl::access::address_space::local_space>;
  constexpr unsigned work_group = InputLayout::collision_work_group;
  constexpr unsigned partition_bins = InputLayout::local_bins / Partitions;
  constexpr unsigned selected_capacity =
      partitioned_collision_selected_capacity<InputLayout, Partitions>;
  const std::size_t global =
      static_cast<std::size_t>(InputLayout::bucket_count) * Partitions * work_group;
  return queue.submit([&](sycl::handler& handler) {
    sycl::local_accessor<std::uint32_t, 1> bin_head(sycl::range<1>(partition_bins), handler);
    sycl::local_accessor<std::uint16_t, 1> selected(sycl::range<1>(selected_capacity), handler);
    sycl::local_accessor<std::uint16_t, 1> bin_next(sycl::range<1>(selected_capacity), handler);
    sycl::local_accessor<std::uint32_t, 0> selected_count(handler);
    handler.parallel_for<CollisionBucketSplitOutputPartitionedKernel<
        Spec, FieldCount, InputLayout, OutputLayout, InputRecord, OutputRecord, Partitions>>(
        sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(work_group)),
        [=](sycl::nd_item<1> item) MOM_SYCL_KERNEL_ARGS_RESTRICT {
          const unsigned group = static_cast<unsigned>(item.get_group(0));
          const unsigned bucket = group / Partitions;
          const unsigned partition = group % Partitions;
          const unsigned lane = static_cast<unsigned>(item.get_local_id(0));
          const unsigned width = static_cast<unsigned>(item.get_local_range(0));
          const std::uint32_t count =
              device_input_counts[bucket] < InputLayout::bucket_slot_capacity
                  ? device_input_counts[bucket] : InputLayout::bucket_slot_capacity;
          const std::uint32_t base = bucket * InputLayout::bucket_slot_capacity;
          for (unsigned i = lane; i < partition_bins; i += width) {
            bin_head[i] = 0;
          }
          if (lane == 0) {
            selected_count = 0;
          }
          sycl::group_barrier(item.get_group());
          for (unsigned i = lane; i < count; i += width) {
            const unsigned bin =
                InputLayout::local_bin(load_bucket_field(device_input[base + i], bucket, 0));
            if (bin / partition_bins != partition) continue;
            const unsigned slot = local_atomic_u32(selected_count).fetch_add(1u);
            if (slot < selected_capacity) {
              selected[slot] = static_cast<std::uint16_t>(i);
              bin_next[slot] = static_cast<std::uint16_t>(bin % partition_bins);
            }
          }
          sycl::group_barrier(item.get_group());
          const unsigned selected_size = selected_count;
          if (selected_size > selected_capacity) {
            if (lane == 0) {
              global_atomic_u32(device_overflow_count[0]).fetch_add(1u);
            }
            return;
          }
          for (unsigned i = lane; i < selected_size; i += width) {
            const unsigned bin = bin_next[i];
            bin_next[i] = static_cast<std::uint16_t>(
                local_atomic_u32(bin_head[bin]).exchange(i + 1u));
          }
          sycl::group_barrier(item.get_group());
          for (unsigned ai = lane; ai < selected_size; ai += width) {
            const unsigned a = selected[ai];
            const InputRecord a_record = device_input[base + a];
            unsigned link = bin_next[ai];
            while (link != 0) {
              const unsigned bi = link - 1u;
              link = bin_next[bi];
              const unsigned b = selected[bi];
              const InputRecord b_record = device_input[base + b];
              OutputRecord collision{};
              std::uint32_t next_field = 0;
              if (!make_bucket_collision_unordered<Spec>(a_record, b_record, bucket, a, b,
                                                          InputLayout::bucket_slot_capacity,
                                                          collision, next_field))
                continue;
              const unsigned output_bucket = OutputLayout::bucket_for(next_field);
              const std::uint32_t slot =
                  global_atomic_u32(device_output_counts[output_bucket]).fetch_add(1u);
              if (slot >= OutputLayout::bucket_slot_capacity) {
                global_atomic_u32(device_overflow_count[0]).fetch_add(1u);
                continue;
              }
              store_split_record(device_output_heads, device_output_tails,
                                 output_bucket * OutputLayout::bucket_slot_capacity + slot,
                                 collision);
            }
          }
        });
  });
}

template <typename Spec, unsigned FieldCount, typename InputLayout, typename OutputLayout,
          typename InputRecord, typename OutputRecord>
inline sycl::event submit_collision_round_bucketed_split_output(
    sycl::queue& queue, const InputRecord* device_input,
    const std::uint32_t* device_input_counts, SplitRecordHead* device_output_heads,
    std::uint32_t* device_output_tails, std::uint32_t* device_output_counts,
    std::uint32_t* device_overflow_count) {
  static_assert(InputRecord::field_count == FieldCount && OutputRecord::word_count == 5 &&
                OutputRecord::field_count + 1 == FieldCount);
  using global_atomic_u32 = sycl::atomic_ref<std::uint32_t, sycl::memory_order::relaxed,
                                             sycl::memory_scope::device,
                                             sycl::access::address_space::global_space>;
  using local_atomic_u32 = sycl::atomic_ref<std::uint32_t, sycl::memory_order::relaxed,
                                            sycl::memory_scope::work_group,
                                            sycl::access::address_space::local_space>;
  const size_t global =
      static_cast<size_t>(InputLayout::bucket_count) * InputLayout::collision_work_group;
  return queue.submit([&](sycl::handler& h) {
    sycl::local_accessor<std::uint32_t, 1> bin_head(sycl::range<1>(InputLayout::local_bins), h);
    sycl::local_accessor<std::uint16_t, 1> bin_next(
        sycl::range<1>(InputLayout::bucket_slot_capacity), h);
    h.parallel_for<CollisionBucketSplitOutputKernel<Spec, FieldCount, InputLayout, OutputLayout,
                                                     InputRecord, OutputRecord>>(
        sycl::nd_range<1>(sycl::range<1>(global),
                          sycl::range<1>(InputLayout::collision_work_group)),
        [=](sycl::nd_item<1> item) MOM_SYCL_KERNEL_ARGS_RESTRICT {
          const unsigned bucket = static_cast<unsigned>(item.get_group(0));
          const unsigned lane = static_cast<unsigned>(item.get_local_id(0));
          const unsigned width = static_cast<unsigned>(item.get_local_range(0));
          const std::uint32_t count =
              device_input_counts[bucket] < InputLayout::bucket_slot_capacity
                  ? device_input_counts[bucket] : InputLayout::bucket_slot_capacity;
          const std::uint32_t base = bucket * InputLayout::bucket_slot_capacity;
          for (unsigned i = lane; i < InputLayout::local_bins; i += width) {
            bin_head[i] = 0;
          }
          sycl::group_barrier(item.get_group());
          for (unsigned i = lane; i < count; i += width) {
            const unsigned bin =
                InputLayout::local_bin(load_bucket_field(device_input[base + i], bucket, 0));
            bin_next[i] = static_cast<std::uint16_t>(
                local_atomic_u32(bin_head[bin]).exchange(i + 1u));
          }
          sycl::group_barrier(item.get_group());
          for (unsigned a = lane; a < count; a += width) {
            const InputRecord a_record = device_input[base + a];
            unsigned link = bin_next[a];
            while (link != 0) {
              const unsigned b = link - 1u;
              link = bin_next[b];
              const InputRecord b_record = device_input[base + b];
              OutputRecord collision{};
              std::uint32_t next_field = 0;
              if (!make_bucket_collision_unordered<Spec>(a_record, b_record, bucket, a, b,
                                                          InputLayout::bucket_slot_capacity,
                                                          collision, next_field))
                continue;
              const unsigned output_bucket = OutputLayout::bucket_for(next_field);
              const std::uint32_t slot =
                  global_atomic_u32(device_output_counts[output_bucket]).fetch_add(1u);
              if (slot >= OutputLayout::bucket_slot_capacity) {
                global_atomic_u32(device_overflow_count[0]).fetch_add(1u);
                continue;
              }
              store_split_record(device_output_heads, device_output_tails,
                  output_bucket * OutputLayout::bucket_slot_capacity + slot, collision);
            }
          }
        });
  });
}
#endif

template <typename Spec, unsigned FieldCount, unsigned LocalBits, typename InputLayout,
          typename OutputLayout, typename InputRecord, typename OutputRecord>
class SplitCollisionBucketKernel;

template <typename Spec, unsigned FieldCount, unsigned LocalBits, typename InputLayout,
          typename OutputLayout, typename InputRecord, typename OutputRecord>
inline sycl::event submit_split_collision_round_bucketed(
    sycl::queue& queue, const SplitRecordHead* device_input_heads,
    const std::uint32_t* device_input_tails, const std::uint32_t* device_input_counts,
    OutputRecord* device_output,
    std::uint32_t* device_output_counts, std::uint32_t* device_overflow_count) {
  static_assert(InputRecord::word_count == 5 && InputRecord::field_count == FieldCount);
  static_assert(OutputRecord::field_count + 1 == FieldCount);
  using global_atomic_u32 = sycl::atomic_ref<std::uint32_t, sycl::memory_order::relaxed,
                                             sycl::memory_scope::device,
                                             sycl::access::address_space::global_space>;
  using local_atomic_u32 = sycl::atomic_ref<std::uint32_t, sycl::memory_order::relaxed,
                                            sycl::memory_scope::work_group,
                                            sycl::access::address_space::local_space>;
  const size_t global =
      static_cast<size_t>(InputLayout::bucket_count) * InputLayout::collision_work_group;
  return queue.submit([&](sycl::handler& h) {
    sycl::local_accessor<std::uint32_t, 1> bin_head(sycl::range<1>(InputLayout::local_bins), h);
    sycl::local_accessor<std::uint16_t, 1> bin_next(
        sycl::range<1>(InputLayout::bucket_slot_capacity), h);
    h.parallel_for<SplitCollisionBucketKernel<Spec, FieldCount, LocalBits, InputLayout,
                                               OutputLayout, InputRecord, OutputRecord>>(
        sycl::nd_range<1>(sycl::range<1>(global),
                          sycl::range<1>(InputLayout::collision_work_group)),
        [=](sycl::nd_item<1> item) MOM_SYCL_KERNEL_ARGS_RESTRICT {
          const unsigned bucket = static_cast<unsigned>(item.get_group(0));
          const unsigned lane = static_cast<unsigned>(item.get_local_id(0));
          const unsigned width = static_cast<unsigned>(item.get_local_range(0));
          const std::uint32_t count =
              device_input_counts[bucket] < InputLayout::bucket_slot_capacity
                  ? device_input_counts[bucket] : InputLayout::bucket_slot_capacity;
          const std::uint32_t base = bucket * InputLayout::bucket_slot_capacity;
          for (unsigned i = lane; i < InputLayout::local_bins; i += width) {
            bin_head[i] = 0;
          }
          sycl::group_barrier(item.get_group());
          for (unsigned i = lane; i < count; i += width) {
            const unsigned bin =
                packed_load(device_input_heads[base + i].words, 4, 0, LocalBits);
            bin_next[i] = static_cast<std::uint16_t>(
                local_atomic_u32(bin_head[bin]).exchange(i + 1u));
          }
          sycl::group_barrier(item.get_group());
          for (unsigned a = lane; a < count; a += width) {
            unsigned link = bin_next[a];
            while (link != 0) {
              const unsigned b = link - 1u;
              link = bin_next[b];
              OutputRecord collision{};
              std::uint32_t next_field = 0;
              if (!make_split_bucket_collision_unordered<Spec, FieldCount, LocalBits, InputRecord>(
                      device_input_heads[base + a], device_input_heads[base + b], bucket,
                      device_input_tails, base + a, base + b, a, b,
                      InputLayout::bucket_slot_capacity, collision, next_field))
                continue;
              const unsigned output_bucket = OutputLayout::bucket_for(next_field);
              const std::uint32_t slot =
                  global_atomic_u32(device_output_counts[output_bucket]).fetch_add(1u);
              if (slot >= OutputLayout::bucket_slot_capacity) {
                global_atomic_u32(device_overflow_count[0]).fetch_add(1u);
                continue;
              }
              const std::uint32_t output =
                  output_bucket * OutputLayout::bucket_slot_capacity + slot;
              store_aligned_record(device_output + output, collision);
            }
          }
        });
  });
}

template <typename Spec, unsigned FieldCount, unsigned LocalBits, typename InputLayout,
          typename OutputLayout, typename InputRecord, typename OutputRecord,
          unsigned Partitions, bool CacheHead = false>
class SplitCollisionBucketPartitionedKernel;

// Preserve the same partition and overflow contract while converting split round-one records to
// the aligned representation used by the remaining rounds.
template <typename Spec, unsigned FieldCount, unsigned LocalBits, typename InputLayout,
          typename OutputLayout, typename InputRecord, typename OutputRecord,
          unsigned Partitions, bool CacheHead = false>
inline sycl::event submit_split_collision_round_bucketed_partitioned(
    sycl::queue& queue, const SplitRecordHead* device_input_heads,
    const std::uint32_t* device_input_tails, const std::uint32_t* device_input_counts,
    OutputRecord* device_output, std::uint32_t* device_output_counts,
    std::uint32_t* device_overflow_count) {
  static_assert(InputRecord::word_count == 5 && InputRecord::field_count == FieldCount);
  static_assert(OutputRecord::field_count + 1 == FieldCount && Partitions > 1);
  static_assert(InputLayout::local_bins % Partitions == 0);
  using global_atomic_u32 = sycl::atomic_ref<std::uint32_t, sycl::memory_order::relaxed,
                                             sycl::memory_scope::device,
                                             sycl::access::address_space::global_space>;
  using local_atomic_u32 = sycl::atomic_ref<std::uint32_t, sycl::memory_order::relaxed,
                                            sycl::memory_scope::work_group,
                                            sycl::access::address_space::local_space>;
  constexpr unsigned work_group = InputLayout::collision_work_group;
  constexpr unsigned partition_bins = InputLayout::local_bins / Partitions;
  constexpr unsigned selected_capacity =
      partitioned_collision_selected_capacity<InputLayout, Partitions>;
  constexpr unsigned cached_words = CacheHead ? selected_capacity * 4 : 1;
  const std::size_t global =
      static_cast<std::size_t>(InputLayout::bucket_count) * Partitions * work_group;
  return queue.submit([&](sycl::handler& handler) {
    sycl::local_accessor<std::uint32_t, 1> bin_head(sycl::range<1>(partition_bins), handler);
    sycl::local_accessor<std::uint16_t, 1> selected(sycl::range<1>(selected_capacity), handler);
    // Cache each selected record's local bin here, then reuse the storage for collision links.
    sycl::local_accessor<std::uint16_t, 1> bin_next(sycl::range<1>(selected_capacity), handler);
    // Collision fields fit in the four-word head; tails remain global for selection/overlap checks.
    sycl::local_accessor<std::uint32_t, 1> cached_heads(sycl::range<1>(cached_words), handler);
    sycl::local_accessor<std::uint32_t, 0> selected_count(handler);
    handler.parallel_for<SplitCollisionBucketPartitionedKernel<
        Spec, FieldCount, LocalBits, InputLayout, OutputLayout, InputRecord, OutputRecord,
        Partitions, CacheHead>>(
        sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(work_group)),
        [=](sycl::nd_item<1> item) MOM_SYCL_KERNEL_ARGS_RESTRICT {
          const unsigned group = static_cast<unsigned>(item.get_group(0));
          const unsigned bucket = group / Partitions;
          const unsigned partition = group % Partitions;
          const unsigned lane = static_cast<unsigned>(item.get_local_id(0));
          const unsigned width = static_cast<unsigned>(item.get_local_range(0));
          const std::uint32_t count =
              device_input_counts[bucket] < InputLayout::bucket_slot_capacity
                  ? device_input_counts[bucket] : InputLayout::bucket_slot_capacity;
          const std::uint32_t base = bucket * InputLayout::bucket_slot_capacity;
          for (unsigned i = lane; i < partition_bins; i += width) {
            bin_head[i] = 0;
          }
          if (lane == 0) {
            selected_count = 0;
          }
          sycl::group_barrier(item.get_group());
          for (unsigned i = lane; i < count; i += width) {
            const unsigned bin =
                packed_load(device_input_heads[base + i].words, 4, 0, LocalBits);
            if (bin / partition_bins != partition) continue;
            const unsigned slot = local_atomic_u32(selected_count).fetch_add(1u);
            if (slot < selected_capacity) {
              selected[slot] = static_cast<std::uint16_t>(i);
              bin_next[slot] = static_cast<std::uint16_t>(bin % partition_bins);
              if constexpr (CacheHead) {
                for (unsigned word = 0; word < 4; ++word)
                  cached_heads[word * selected_capacity + slot] =
                      device_input_heads[base + i].words[word];
              }
            }
          }
          sycl::group_barrier(item.get_group());
          const unsigned selected_size = selected_count;
          if (selected_size > selected_capacity) {
            if (lane == 0) {
              global_atomic_u32(device_overflow_count[0]).fetch_add(1u);
            }
            return;
          }
          for (unsigned i = lane; i < selected_size; i += width) {
            const unsigned bin = bin_next[i];
            bin_next[i] = static_cast<std::uint16_t>(
                local_atomic_u32(bin_head[bin]).exchange(i + 1u));
          }
          sycl::group_barrier(item.get_group());
          for (unsigned ai = lane; ai < selected_size; ai += width) {
            const unsigned a = selected[ai];
            SplitRecordHead a_head{};
            if constexpr (CacheHead) {
              for (unsigned word = 0; word < 4; ++word)
                a_head.words[word] = cached_heads[word * selected_capacity + ai];
            } else {
              a_head = device_input_heads[base + a];
            }
            unsigned link = bin_next[ai];
            while (link != 0) {
              const unsigned bi = link - 1u;
              link = bin_next[bi];
              const unsigned b = selected[bi];
              SplitRecordHead b_head{};
              if constexpr (CacheHead) {
                for (unsigned word = 0; word < 4; ++word)
                  b_head.words[word] = cached_heads[word * selected_capacity + bi];
              } else {
                b_head = device_input_heads[base + b];
              }
              OutputRecord collision{};
              std::uint32_t next_field = 0;
              if (!make_split_bucket_collision_unordered<Spec, FieldCount, LocalBits, InputRecord>(
                      a_head, b_head, bucket, device_input_tails, base + a, base + b, a, b,
                      InputLayout::bucket_slot_capacity, collision, next_field))
                continue;
              const unsigned output_bucket = OutputLayout::bucket_for(next_field);
              const std::uint32_t slot =
                  global_atomic_u32(device_output_counts[output_bucket]).fetch_add(1u);
              if (slot >= OutputLayout::bucket_slot_capacity) {
                global_atomic_u32(device_overflow_count[0]).fetch_add(1u);
                continue;
              }
              store_aligned_record(
                  device_output + output_bucket * OutputLayout::bucket_slot_capacity + slot,
                  collision);
            }
          }
        });
  });
}

template <typename Spec, unsigned FieldCount, typename InputLayout, typename OutputLayout,
          typename InputRecord, typename OutputRecord>
class CollisionBucketDirectKernel;
template <typename Spec, unsigned FieldCount, typename InputLayout, typename OutputLayout,
          typename InputRecord, typename OutputRecord>
inline sycl::event submit_collision_round_bucketed(
    sycl::queue& queue, const InputRecord* device_input, const std::uint32_t* device_input_counts,
    OutputRecord* device_output,
    std::uint32_t* device_output_counts, std::uint32_t* device_overflow_count) {
  static_assert(FieldCount > 1, "a bucketed collision round needs a next-round field");
  static_assert(InputRecord::field_count == FieldCount &&
                OutputRecord::field_count + 1 == FieldCount);
  using global_atomic_u32 = sycl::atomic_ref<std::uint32_t, sycl::memory_order::relaxed,
                                             sycl::memory_scope::device,
                                             sycl::access::address_space::global_space>;
  using local_atomic_u32 = sycl::atomic_ref<std::uint32_t, sycl::memory_order::relaxed,
                                            sycl::memory_scope::work_group,
                                            sycl::access::address_space::local_space>;
  const size_t global =
      static_cast<size_t>(InputLayout::bucket_count) * InputLayout::collision_work_group;
  return queue.submit([&](sycl::handler& h) {
    sycl::local_accessor<std::uint32_t, 1> bin_head(sycl::range<1>(InputLayout::local_bins), h);
    sycl::local_accessor<std::uint16_t, 1> bin_next(
        sycl::range<1>(InputLayout::bucket_slot_capacity), h);
    h.parallel_for<CollisionBucketDirectKernel<Spec, FieldCount, InputLayout, OutputLayout,
                                               InputRecord, OutputRecord>>(
        sycl::nd_range<1>(sycl::range<1>(global),
                          sycl::range<1>(InputLayout::collision_work_group)),
        [=](sycl::nd_item<1> item) MOM_SYCL_KERNEL_ARGS_RESTRICT {
          const unsigned bucket = static_cast<unsigned>(item.get_group(0));
          const unsigned lane = static_cast<unsigned>(item.get_local_id(0));
          const unsigned width = static_cast<unsigned>(item.get_local_range(0));
          const std::uint32_t count =
              device_input_counts[bucket] < InputLayout::bucket_slot_capacity
                  ? device_input_counts[bucket] : InputLayout::bucket_slot_capacity;
          const std::uint32_t base = bucket * InputLayout::bucket_slot_capacity;
          for (unsigned i = lane; i < InputLayout::local_bins; i += width) {
            bin_head[i] = 0;
          }
          sycl::group_barrier(item.get_group());
          for (unsigned i = lane; i < count; i += width) {
            const unsigned bin =
                InputLayout::local_bin(load_bucket_field(device_input[base + i], bucket, 0));
            bin_next[i] = static_cast<std::uint16_t>(
                local_atomic_u32(bin_head[bin]).exchange(i + 1u));
          }
          sycl::group_barrier(item.get_group());
          for (unsigned a = lane; a < count; a += width) {
            const InputRecord a_record = device_input[base + a];
            unsigned link = bin_next[a];
            while (link != 0) {
              const unsigned b = link - 1u;
              link = bin_next[b];
              const InputRecord b_record = device_input[base + b];
              OutputRecord collision{};
              std::uint32_t next_field = 0;
              if (!make_bucket_collision_unordered<Spec>(a_record, b_record, bucket, a, b,
                                                          InputLayout::bucket_slot_capacity,
                                                          collision, next_field))
                continue;
              const unsigned output_bucket = OutputLayout::bucket_for(next_field);
              const std::uint32_t slot =
                  global_atomic_u32(device_output_counts[output_bucket]).fetch_add(1u);
              if (slot >= OutputLayout::bucket_slot_capacity) {
                global_atomic_u32(device_overflow_count[0]).fetch_add(1u);
                continue;
              }
              const std::uint32_t output =
                  output_bucket * OutputLayout::bucket_slot_capacity + slot;
              store_aligned_record(device_output + output, collision);
            }
          }
        });
  });
}

template <typename Spec, unsigned FieldCount, typename InputLayout, typename OutputLayout,
          typename InputRecord, typename OutputRecord, unsigned Partitions, bool CacheInput = false>
class CollisionBucketPartitionedKernel;

template <typename InputLayout, typename InputRecord, unsigned Partitions, bool CacheInput>
inline constexpr std::size_t partitioned_collision_local_bytes =
    (InputLayout::local_bins / Partitions) * sizeof(std::uint32_t) +
    2 * partitioned_collision_selected_capacity<InputLayout, Partitions> *
        sizeof(std::uint16_t) +
    (CacheInput
         ? partitioned_collision_selected_capacity<InputLayout, Partitions> *
               InputRecord::word_count
         : 1) *
        sizeof(std::uint32_t) +
    sizeof(std::uint32_t);

// Split one bucket across disjoint local-bin ranges. Equal fields always select the same range, so
// no collision pair crosses workgroups. Report an unusually skewed range as overflow instead of
// silently dropping records; the owning solver can then retry with its alternative bucket layout.
template <typename Spec, unsigned FieldCount, typename InputLayout, typename OutputLayout,
          typename InputRecord, typename OutputRecord, unsigned Partitions, bool CacheInput = false>
inline sycl::event submit_collision_round_bucketed_partitioned(
    sycl::queue& queue, const InputRecord* device_input,
    const std::uint32_t* device_input_counts, OutputRecord* device_output,
    std::uint32_t* device_output_counts, std::uint32_t* device_overflow_count) {
  static_assert(FieldCount > 1 && Partitions > 1);
  static_assert(InputRecord::field_count == FieldCount &&
                OutputRecord::field_count + 1 == FieldCount);
  static_assert(InputLayout::local_bins % Partitions == 0);
  using global_atomic_u32 = sycl::atomic_ref<std::uint32_t, sycl::memory_order::relaxed,
                                             sycl::memory_scope::device,
                                             sycl::access::address_space::global_space>;
  using local_atomic_u32 = sycl::atomic_ref<std::uint32_t, sycl::memory_order::relaxed,
                                            sycl::memory_scope::work_group,
                                            sycl::access::address_space::local_space>;
  constexpr unsigned work_group = InputLayout::collision_work_group;
  constexpr unsigned partition_bins = InputLayout::local_bins / Partitions;
  constexpr unsigned selected_capacity =
      partitioned_collision_selected_capacity<InputLayout, Partitions>;
  constexpr unsigned cached_words =
      CacheInput ? selected_capacity * InputRecord::word_count : 1;
  const std::size_t global =
      static_cast<std::size_t>(InputLayout::bucket_count) * Partitions * work_group;
  return queue.submit([&](sycl::handler& handler) {
    sycl::local_accessor<std::uint32_t, 1> bin_head(sycl::range<1>(partition_bins), handler);
    sycl::local_accessor<std::uint16_t, 1> selected(sycl::range<1>(selected_capacity), handler);
    // Cache each selected record's local bin here, then reuse the storage for collision links.
    sycl::local_accessor<std::uint16_t, 1> bin_next(sycl::range<1>(selected_capacity), handler);
    sycl::local_accessor<std::uint32_t, 1> cached_input(sycl::range<1>(cached_words), handler);
    sycl::local_accessor<std::uint32_t, 0> selected_count(handler);
    handler.parallel_for<CollisionBucketPartitionedKernel<
        Spec, FieldCount, InputLayout, OutputLayout, InputRecord, OutputRecord, Partitions,
        CacheInput>>(
        sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(work_group)),
        [=](sycl::nd_item<1> item) MOM_SYCL_KERNEL_ARGS_RESTRICT {
          const unsigned group = static_cast<unsigned>(item.get_group(0));
          const unsigned bucket = group / Partitions;
          const unsigned partition = group % Partitions;
          const unsigned lane = static_cast<unsigned>(item.get_local_id(0));
          const unsigned width = static_cast<unsigned>(item.get_local_range(0));
          const std::uint32_t count =
              device_input_counts[bucket] < InputLayout::bucket_slot_capacity
                  ? device_input_counts[bucket] : InputLayout::bucket_slot_capacity;
          const std::uint32_t base = bucket * InputLayout::bucket_slot_capacity;
          for (unsigned i = lane; i < partition_bins; i += width)
            bin_head[i] = 0;
          if (lane == 0)
            selected_count = 0;
          sycl::group_barrier(item.get_group());
          for (unsigned i = lane; i < count; i += width) {
            const unsigned bin =
                InputLayout::local_bin(load_bucket_field(device_input[base + i], bucket, 0));
            if (bin / partition_bins != partition) continue;
            const unsigned slot = local_atomic_u32(selected_count).fetch_add(1u);
            if (slot < selected_capacity) {
              selected[slot] = static_cast<std::uint16_t>(i);
              bin_next[slot] = static_cast<std::uint16_t>(bin % partition_bins);
              if constexpr (CacheInput) {
                for (unsigned word = 0; word < InputRecord::word_count; ++word)
                  cached_input[word * selected_capacity + slot] =
                      device_input[base + i].words[word];
              }
            }
          }
          sycl::group_barrier(item.get_group());
          const unsigned selected_size = selected_count;
          if (selected_size > selected_capacity) {
            if (lane == 0)
              global_atomic_u32(device_overflow_count[0]).fetch_add(1u);
            return;
          }
          for (unsigned i = lane; i < selected_size; i += width) {
            const unsigned bin = bin_next[i];
            bin_next[i] = static_cast<std::uint16_t>(
                local_atomic_u32(bin_head[bin]).exchange(i + 1u));
          }
          sycl::group_barrier(item.get_group());
          for (unsigned ai = lane; ai < selected_size; ai += width) {
            const unsigned a = selected[ai];
            InputRecord a_record{};
            if constexpr (CacheInput) {
              for (unsigned word = 0; word < InputRecord::word_count; ++word)
                a_record.words[word] = cached_input[word * selected_capacity + ai];
            } else {
              a_record = device_input[base + a];
            }
            unsigned link = bin_next[ai];
            while (link != 0) {
              const unsigned bi = link - 1u;
              link = bin_next[bi];
              const unsigned b = selected[bi];
              InputRecord b_record{};
              if constexpr (CacheInput) {
                for (unsigned word = 0; word < InputRecord::word_count; ++word)
                  b_record.words[word] = cached_input[word * selected_capacity + bi];
              } else {
                b_record = device_input[base + b];
              }
              OutputRecord collision{};
              std::uint32_t next_field = 0;
              if (!make_bucket_collision_unordered<Spec>(
                      a_record, b_record, bucket, a, b, InputLayout::bucket_slot_capacity,
                      collision, next_field))
                continue;
              const unsigned output_bucket = OutputLayout::bucket_for(next_field);
              const std::uint32_t slot =
                  global_atomic_u32(device_output_counts[output_bucket]).fetch_add(1u);
              if (slot >= OutputLayout::bucket_slot_capacity) {
                global_atomic_u32(device_overflow_count[0]).fetch_add(1u);
                continue;
              }
              store_aligned_record(
                  device_output + output_bucket * OutputLayout::bucket_slot_capacity + slot,
                  collision);
            }
          }
        });
  });
}

template <typename Spec, typename Layout, typename InputRecord> class ZeroRootBucketKernel;
template <typename Spec, typename Layout, typename InputRecord>
inline sycl::event submit_zero_root_round_bucketed(
    sycl::queue& queue, const InputRecord* device_input,
    const std::uint32_t* device_input_counts, ZHashRootRecord<Spec>* device_roots,
    std::uint32_t* device_root_count, const std::uint32_t root_capacity) {
  using global_atomic_u32 = sycl::atomic_ref<std::uint32_t, sycl::memory_order::relaxed,
                                             sycl::memory_scope::device,
                                             sycl::access::address_space::global_space>;
  using local_atomic_u32 = sycl::atomic_ref<std::uint32_t, sycl::memory_order::relaxed,
                                            sycl::memory_scope::work_group,
                                            sycl::access::address_space::local_space>;
  const size_t global = static_cast<size_t>(Layout::bucket_count) * Layout::collision_work_group;
  return queue.submit([&](sycl::handler& h) {
    sycl::local_accessor<std::uint32_t, 1> bin_head(sycl::range<1>(Layout::local_bins), h);
    sycl::local_accessor<std::uint16_t, 1> bin_next(
        sycl::range<1>(Layout::bucket_slot_capacity), h);
    h.parallel_for<ZeroRootBucketKernel<Spec, Layout, InputRecord>>(
        sycl::nd_range<1>(sycl::range<1>(global),
                          sycl::range<1>(Layout::collision_work_group)),
        [=](sycl::nd_item<1> item) MOM_SYCL_KERNEL_ARGS_RESTRICT {
          const unsigned bucket = static_cast<unsigned>(item.get_group(0));
          const unsigned lane = static_cast<unsigned>(item.get_local_id(0));
          const unsigned width = static_cast<unsigned>(item.get_local_range(0));
          const std::uint32_t count =
              device_input_counts[bucket] < Layout::bucket_slot_capacity
                  ? device_input_counts[bucket] : Layout::bucket_slot_capacity;
          const std::uint32_t base = bucket * Layout::bucket_slot_capacity;
          for (unsigned i = lane; i < Layout::local_bins; i += width) {
            bin_head[i] = 0;
          }
          sycl::group_barrier(item.get_group());
          for (unsigned i = lane; i < count; i += width) {
            const unsigned bin =
                Layout::local_bin(load_bucket_field(device_input[base + i], bucket, 0));
            bin_next[i] = static_cast<std::uint16_t>(
                local_atomic_u32(bin_head[bin]).exchange(i + 1u));
          }
          sycl::group_barrier(item.get_group());
          for (unsigned a = lane; a < count; a += width) {
            unsigned link = bin_next[a];
            while (link != 0) {
              const unsigned b = link - 1u;
              link = bin_next[b];
              ZHashRootRecord<Spec> root{};
              if (!make_bucket_root_unordered<Spec>(
                      device_input[base + a], device_input[base + b], bucket, a, b,
                      Layout::bucket_slot_capacity, root))
                continue;
              const std::uint32_t output = global_atomic_u32(device_root_count[0]).fetch_add(1u);
              if (output < root_capacity) {
                device_roots[output] = root;
              }
            }
          }
        });
  });
}

template <typename Layout> class SumBucketCountsKernel;
template <typename Layout>
inline sycl::event submit_sum_bucket_counts(sycl::queue& queue,
                                            const std::uint32_t* device_counts,
                                            std::uint32_t* device_total) {
  constexpr unsigned width = 256;
  return queue.submit([&](sycl::handler& h) {
    sycl::local_accessor<std::uint32_t, 1> sums(sycl::range<1>(width), h);
    h.parallel_for<SumBucketCountsKernel<Layout>>(
        sycl::nd_range<1>(sycl::range<1>(width), sycl::range<1>(width)),
        [=](sycl::nd_item<1> item) MOM_SYCL_KERNEL_ARGS_RESTRICT {
          const unsigned lane = static_cast<unsigned>(item.get_local_id(0));
          std::uint32_t sum = 0;
          for (unsigned bucket = lane; bucket < Layout::bucket_count; bucket += width)
            sum += device_counts[bucket] < Layout::bucket_slot_capacity
                       ? device_counts[bucket] : Layout::bucket_slot_capacity;
          sums[lane] = sum;
          for (unsigned stride = width / 2; stride != 0; stride /= 2) {
            sycl::group_barrier(item.get_group());
            if (lane < stride) {
              sums[lane] += sums[lane + stride];
            }
          }
          if (lane == 0) {
            device_total[0] = sums[0];
          }
        });
  });
}

template <typename Spec, typename Round2, typename Round3, typename Round4>
class RecoverRootLeavesKernel;
template <typename Spec, unsigned FieldCount>
inline bool expand_recovery_round(const PackedCollisionRecord<Spec, FieldCount>* records,
                                  std::uint32_t* current, std::uint32_t* next,
                                  unsigned& count, const std::uint32_t arena_capacity) {
  for (unsigned i = 0; i < count; ++i) {
    if (current[i] >= arena_capacity) return false;
    const auto& record = records[current[i]];
    next[2 * i] = load_collision_left<Spec, FieldCount>(record);
    next[2 * i + 1] = load_collision_right<Spec, FieldCount>(record);
  }
  count *= 2;
  for (unsigned i = 0; i < count; ++i) {
    current[i] = next[i];
  }
  return true;
}

template <typename Spec, unsigned FieldCount, unsigned LocalBits, unsigned ParentBucketBits,
          unsigned ParentSlots>
inline bool expand_recovery_round(
    const PackedBucketCollisionRecord<Spec, FieldCount, LocalBits, ParentBucketBits, ParentSlots>*
        records,
    std::uint32_t* current, std::uint32_t* next, unsigned& count,
    const std::uint32_t arena_capacity) {
  for (unsigned i = 0; i < count; ++i) {
    if (current[i] >= arena_capacity) return false;
    const auto& record = records[current[i]];
    next[2 * i] = load_bucket_collision_parent(record, false);
    next[2 * i + 1] = load_bucket_collision_parent(record, true);
  }
  count *= 2;
  for (unsigned i = 0; i < count; ++i) {
    current[i] = next[i];
  }
  return true;
}

template <typename Record>
inline bool expand_split_recovery_round(
    const SplitRecordHead* heads, const std::uint32_t* tails, std::uint32_t* current,
    std::uint32_t* next, unsigned& count, const std::uint32_t arena_capacity) {
  for (unsigned i = 0; i < count; ++i) {
    if (current[i] >= arena_capacity) return false;
    const auto record = load_split_record<Record>(heads, tails, current[i]);
    next[2 * i] = load_bucket_collision_parent(record, false);
    next[2 * i + 1] = load_bucket_collision_parent(record, true);
  }
  count *= 2;
  for (unsigned i = 0; i < count; ++i) {
    current[i] = next[i];
  }
  return true;
}

template <typename Spec, unsigned Level0LocalBits, typename Round1, typename Round2,
          typename Round3, typename Round4>
inline sycl::event submit_recover_root_leaves(
    sycl::queue& queue, const SplitRecordHead* level0_heads, const std::uint32_t* level0_tails,
    const SplitRecordHead* round1_heads, const std::uint32_t* round1_tails,
    const Round2* round2, const Round3* round3, const Round4* round4,
    const ZHashRootRecord<Spec>* roots,
    const std::uint32_t* candidate_count, std::uint32_t* leaves, std::uint8_t* valid,
    const std::uint32_t candidate_begin, const std::uint32_t candidate_size,
    const std::uint32_t initial_capacity,
    const std::uint32_t later_capacity) {
  return queue.parallel_for<RecoverRootLeavesKernel<Spec, Round2, Round3, Round4>>(
      sycl::range<1>(candidate_size), [=](sycl::id<1> id) MOM_SYCL_KERNEL_ARGS_RESTRICT {
        const unsigned candidate = candidate_begin + static_cast<unsigned>(id[0]);
        if (candidate >= candidate_count[0]) {
          valid[candidate] = 0;
          return;
        }
        std::uint32_t current[Spec::proof_indices], next[Spec::proof_indices];
        bool ok = true;
        const auto& root = roots[candidate];
        current[0] = load_collision_left<Spec, 1>(root);
        current[1] = load_collision_right<Spec, 1>(root);
        unsigned count = 2;
        ok = ok && expand_recovery_round(round4, current, next, count, later_capacity);
        ok = ok && expand_recovery_round(round3, current, next, count, later_capacity);
        ok = ok && expand_recovery_round(round2, current, next, count, later_capacity);
        ok = ok && expand_split_recovery_round<Round1>(
                       round1_heads, round1_tails, current, next, count, initial_capacity);
        for (unsigned i = 0; ok && i < count; ++i) {
          ok = current[i] < initial_capacity;
          if (ok) {
            const auto record = load_split_record<PackedBucketLevel0Record<Spec, Level0LocalBits>>(
                level0_heads, level0_tails, current[i]);
            leaves[candidate * Spec::proof_indices + i] = load_bucket_level0_leaf(record);
          }
        }
        const unsigned base = candidate * Spec::proof_indices;
        for (unsigned level = 0; ok && level < Spec::k; ++level) {
          const unsigned half = 1u << level;
          for (unsigned begin = 0; begin < Spec::proof_indices; begin += 2 * half) {
            if (leaves[base + begin] < leaves[base + begin + half]) continue;
            for (unsigned i = 0; i < half; ++i) {
              const std::uint32_t swap = leaves[base + begin + i];
              leaves[base + begin + i] = leaves[base + begin + half + i];
              leaves[base + begin + half + i] = swap;
            }
          }
        }
        valid[candidate] = ok;
      });
}

template <typename Spec> class HashRecoveredLeavesKernel;
template <typename Spec> class VerifyRecoveredLeavesKernel;
template <typename Spec>
inline sycl::event submit_verify_recovered_leaves(
    sycl::queue& queue, const std::uint8_t* header, const std::uint64_t* midstate,
    const std::uint32_t* candidate_count, const std::uint32_t* leaves,
    std::uint32_t* fields, std::uint8_t* valid, const std::uint32_t candidate_begin,
    const std::uint32_t candidate_size) {
  queue.parallel_for<HashRecoveredLeavesKernel<Spec>>(
      sycl::range<1>(candidate_size * Spec::proof_indices),
      [=](sycl::id<1> id) MOM_SYCL_KERNEL_ARGS_RESTRICT {
        const unsigned local_item = static_cast<unsigned>(id[0]);
        const unsigned candidate = candidate_begin + local_item / Spec::proof_indices;
        const unsigned item = candidate * Spec::proof_indices + local_item % Spec::proof_indices;
        if (candidate >= candidate_count[0] || !valid[candidate]) return;
        Row<Spec> row{};
        row_from_index_midstate<Spec>(midstate, header, leaves[item], row);
        for (unsigned field = 0; field < Spec::rounds; ++field)
          fields[item * Spec::rounds + field] = row.fields[field];
      });
  return queue.parallel_for<VerifyRecoveredLeavesKernel<Spec>>(
      sycl::range<1>(candidate_size), [=](sycl::id<1> id) MOM_SYCL_KERNEL_ARGS_RESTRICT {
        const unsigned candidate = candidate_begin + static_cast<unsigned>(id[0]);
        if (candidate >= candidate_count[0] || !valid[candidate]) return;
        const unsigned leaf_base = candidate * Spec::proof_indices;
        bool ok = true;
        for (unsigned i = 0; ok && i < Spec::proof_indices; ++i)
          for (unsigned j = 0; j < i; ++j)
            if (leaves[leaf_base + i] == leaves[leaf_base + j]) {
              ok = false;
            }
        for (unsigned level = 0; ok && level < Spec::k; ++level) {
          const unsigned half = 1u << level, step = 2u * half;
          const unsigned active_fields = Spec::rounds - level;
          for (unsigned begin = 0; ok && begin < Spec::proof_indices; begin += step) {
            const unsigned left = (leaf_base + begin) * Spec::rounds;
            const unsigned right = (leaf_base + begin + half) * Spec::rounds;
            if (fields[left] != fields[right] ||
                leaves[leaf_base + begin] >= leaves[leaf_base + begin + half]) {
              ok = false;
              break;
            }
            for (unsigned field = 0; field + 1 < active_fields; ++field)
              fields[left + field] = fields[left + field + 1] ^ fields[right + field + 1];
          }
        }
        if (ok) {
          ok = fields[leaf_base * Spec::rounds] == 0;
        }
        valid[candidate] = ok;
      });
}

}  // namespace mom_equihash
