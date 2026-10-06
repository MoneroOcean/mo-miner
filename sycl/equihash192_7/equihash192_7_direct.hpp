#pragma once

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>

#include "../zhash/equihash_sycl.hpp"

namespace mom_equihash192_7::direct {

using Spec = mom_equihash::Equihash192_7;

template <unsigned BucketBits, unsigned SlotsPerBucket>
struct BucketLayout {
  static_assert(BucketBits < Spec::collision_bits && SlotsPerBucket != 0);
  static constexpr unsigned bucket_bits = BucketBits;
  static constexpr unsigned bucket_count = 1u << bucket_bits;
  static constexpr unsigned local_bits = Spec::collision_bits - bucket_bits;
  static constexpr unsigned local_bins = 1u << local_bits;
  static constexpr unsigned collision_work_group = 64;
  static constexpr unsigned bucket_slot_capacity = SlotsPerBucket;
  static constexpr std::size_t slot_count =
      static_cast<std::size_t>(bucket_count) * bucket_slot_capacity;

  static constexpr unsigned bucket_for(const std::uint32_t field) {
    return field >> local_bits;
  }
  static constexpr unsigned local_bin(const std::uint32_t field) {
    return field & (local_bins - 1u);
  }
};

template <typename Level0, typename Round1, typename Round2, typename Round3, typename Round4,
          typename Round5, typename Round6>
struct StagedArenaLayout {
  using Layout = Level0;
  using Level0Layout = Level0;
  using Round1Layout = Round1;
  using Round2Layout = Round2;
  using Round3Layout = Round3;
  using Round4Layout = Round4;
  using Round5Layout = Round5;
  using Round6Layout = Round6;
  using Level0Record = mom_equihash::PackedBucketLevel0Record<Spec, Level0Layout::local_bits>;
  using Round1Record =
      mom_equihash::PackedBucketCollisionRecord<Spec, 7, Round1Layout::local_bits,
                                                Level0Layout::bucket_bits,
                                                Level0Layout::bucket_slot_capacity>;
  using Round2Record =
      mom_equihash::PackedBucketCollisionRecord<Spec, 6, Round2Layout::local_bits,
                                                Round1Layout::bucket_bits,
                                                Round1Layout::bucket_slot_capacity>;
  using Round3Record =
      mom_equihash::PackedBucketCollisionRecord<Spec, 5, Round3Layout::local_bits,
                                                Round2Layout::bucket_bits,
                                                Round2Layout::bucket_slot_capacity>;
  using Round4Record =
      mom_equihash::PackedBucketCollisionRecord<Spec, 4, Round4Layout::local_bits,
                                                Round3Layout::bucket_bits,
                                                Round3Layout::bucket_slot_capacity>;
  using Round5Record =
      mom_equihash::PackedBucketCollisionRecord<Spec, 3, Round5Layout::local_bits,
                                                Round4Layout::bucket_bits,
                                                Round4Layout::bucket_slot_capacity>;
  using Round6Record =
      mom_equihash::PackedBucketCollisionRecord<Spec, 2, Round6Layout::local_bits,
                                                Round5Layout::bucket_bits,
                                                Round5Layout::bucket_slot_capacity>;
  using RootRecord = mom_equihash::ZHashRootRecord<Spec>;

  static_assert(Level0Layout::slot_count == Round1Layout::slot_count &&
                Level0Layout::slot_count == Round2Layout::slot_count &&
                Level0Layout::slot_count == Round3Layout::slot_count &&
                Level0Layout::slot_count == Round4Layout::slot_count &&
                Level0Layout::slot_count == Round5Layout::slot_count &&
                Level0Layout::slot_count == Round6Layout::slot_count);
  static constexpr std::size_t arena_bytes = Layout::slot_count *
      (sizeof(Level0Record) + sizeof(Round1Record) + sizeof(Round2Record) +
       sizeof(Round3Record) + sizeof(Round4Record) + sizeof(Round5Record) +
       sizeof(Round6Record));

  static_assert(Layout::slot_count < (1ull << Spec::parent_ref_bits));
#if defined(__INTEL_LLVM_COMPILER)
  static constexpr std::size_t level0_record_bytes = 28;
#else
  static constexpr std::size_t level0_record_bytes = 32;
#endif
#if defined(__INTEL_LLVM_COMPILER)
  static constexpr std::size_t round1_record_bytes =
      Round1Record::word_count * sizeof(std::uint32_t);
#else
  static constexpr std::size_t round1_record_bytes = 32;
#endif
#if defined(MOM_SYCL_HAS_HIP) || defined(__INTEL_LLVM_COMPILER)
  static constexpr std::size_t round2_record_bytes = 24;
#else
  static constexpr std::size_t round2_record_bytes = 32;
#endif
#if defined(MOM_SYCL_HAS_HIP) || defined(__INTEL_LLVM_COMPILER)
  static constexpr std::size_t round6_record_bytes = 12;
#else
  static constexpr std::size_t round6_record_bytes = 16;
#endif
  static_assert(sizeof(Level0Record) == level0_record_bytes &&
                sizeof(Round1Record) == round1_record_bytes &&
                sizeof(Round2Record) == round2_record_bytes && sizeof(Round3Record) == 24 &&
                sizeof(Round4Record) == 16 && sizeof(Round5Record) == 16 &&
                sizeof(Round6Record) == round6_record_bytes && sizeof(RootRecord) == 12);
};

template <unsigned BucketBits, unsigned SlotsPerBucket>
using ArenaLayout = StagedArenaLayout<
    BucketLayout<BucketBits, SlotsPerBucket>, BucketLayout<BucketBits, SlotsPerBucket>,
    BucketLayout<BucketBits, SlotsPerBucket>, BucketLayout<BucketBits, SlotsPerBucket>,
    BucketLayout<BucketBits, SlotsPerBucket>, BucketLayout<BucketBits, SlotsPerBucket>,
    BucketLayout<BucketBits, SlotsPerBucket>>;

#if defined(MOM_SYCL_HAS_HIP) || defined(__INTEL_LLVM_COMPILER)
constexpr unsigned ProductionSlotsPerBucket = 9232;
#else
constexpr unsigned ProductionSlotsPerBucket = 9216;
#endif
using ProductionArena = ArenaLayout<12, ProductionSlotsPerBucket>;
using RetryArena = ArenaLayout<13, ProductionSlotsPerBucket / 2>;
using HybridArena =
    StagedArenaLayout<BucketLayout<14, ProductionSlotsPerBucket / 4>,
                      BucketLayout<14, ProductionSlotsPerBucket / 4>,
                      BucketLayout<14, ProductionSlotsPerBucket / 4>,
                      BucketLayout<12, ProductionSlotsPerBucket>,
                      BucketLayout<12, ProductionSlotsPerBucket>,
                      BucketLayout<12, ProductionSlotsPerBucket>,
                      BucketLayout<14, ProductionSlotsPerBucket / 4>>;

template <typename Storage, typename Active>
inline constexpr bool arena_storage_compatible =
    sizeof(typename Storage::Level0Record) >= sizeof(typename Active::Level0Record) &&
    alignof(typename Storage::Level0Record) >= alignof(typename Active::Level0Record) &&
    sizeof(typename Storage::Round1Record) >= sizeof(typename Active::Round1Record) &&
    alignof(typename Storage::Round1Record) >= alignof(typename Active::Round1Record) &&
    sizeof(typename Storage::Round2Record) >= sizeof(typename Active::Round2Record) &&
    alignof(typename Storage::Round2Record) >= alignof(typename Active::Round2Record) &&
    sizeof(typename Storage::Round3Record) >= sizeof(typename Active::Round3Record) &&
    alignof(typename Storage::Round3Record) >= alignof(typename Active::Round3Record) &&
    sizeof(typename Storage::Round4Record) >= sizeof(typename Active::Round4Record) &&
    alignof(typename Storage::Round4Record) >= alignof(typename Active::Round4Record) &&
    sizeof(typename Storage::Round5Record) >= sizeof(typename Active::Round5Record) &&
    alignof(typename Storage::Round5Record) >= alignof(typename Active::Round5Record) &&
    sizeof(typename Storage::Round6Record) >= sizeof(typename Active::Round6Record) &&
    alignof(typename Storage::Round6Record) >= alignof(typename Active::Round6Record);

static_assert(ProductionArena::Layout::slot_count == RetryArena::Layout::slot_count);
static_assert(arena_storage_compatible<ProductionArena, RetryArena>);
static_assert(arena_storage_compatible<ProductionArena, HybridArena>);

template <unsigned WorkGroup, typename Layout> struct LaunchLayout : Layout {
  static constexpr unsigned collision_work_group = WorkGroup;
};

template <typename Record>
inline bool expand_round(const Record* const records, std::uint32_t* const current,
                         std::uint32_t* const next, const unsigned count,
                         const std::uint32_t capacity) {
  for (unsigned i = 0; i < count; ++i) {
    if (current[i] >= capacity) return false;
    const Record& record = records[current[i]];
    next[2 * i] = mom_equihash::load_bucket_parent(record, false);
    next[2 * i + 1] = mom_equihash::load_bucket_parent(record, true);
  }
  for (unsigned i = 0; i < 2 * count; ++i) {
    current[i] = next[i];
  }
  return true;
}

#ifndef MOM_SYCL_HAS_HIP
template <typename Record>
inline bool expand_split_round(const mom_equihash::SplitRecordHead* const heads,
                               const std::uint32_t* const tails,
                               std::uint32_t* const current, std::uint32_t* const next,
                               const unsigned count, const std::uint32_t capacity) {
  static_assert(Record::word_count == 5);
  for (unsigned i = 0; i < count; ++i) {
    if (current[i] >= capacity) return false;
    const Record record = mom_equihash::load_split_record<Record>(heads, tails, current[i]);
    next[2 * i] = mom_equihash::load_bucket_parent(record, false);
    next[2 * i + 1] = mom_equihash::load_bucket_parent(record, true);
  }
  for (unsigned i = 0; i < 2 * count; ++i) {
    current[i] = next[i];
  }
  return true;
}
#endif

template <typename Arena>
class RecoverLeavesKernel;

template <typename Arena>
inline sycl::event submit_recover_leaves(
    sycl::queue& queue, const typename Arena::Level0Record* const level0,
    const typename Arena::Round1Record* const round1,
    const typename Arena::Round2Record* const round2,
#ifdef MOM_SYCL_HAS_HIP
    const typename Arena::Round3Record* const round3,
#else
    const mom_equihash::SplitRecordHead* const round3_heads,
    const std::uint32_t* const round3_tails,
#endif
    const typename Arena::Round4Record* const round4,
    const typename Arena::Round5Record* const round5,
    const typename Arena::Round6Record* const round6,
    const typename Arena::RootRecord* const roots,
    const std::uint32_t* const candidate_count, const std::uint32_t candidate_capacity,
    std::uint32_t* const leaves, std::uint8_t* const valid) {
  constexpr std::uint32_t slots = static_cast<std::uint32_t>(Arena::Layout::slot_count);
  return queue.parallel_for<RecoverLeavesKernel<Arena>>(
      sycl::range<1>(candidate_capacity), [=](sycl::id<1> id) MOM_SYCL_KERNEL_ARGS_RESTRICT {
        const unsigned candidate = static_cast<unsigned>(id[0]);
        if (candidate >= candidate_count[0]) return;
        std::uint32_t current[Spec::proof_indices], next[Spec::proof_indices];
        const auto& root = roots[candidate];
        current[0] = mom_equihash::load_collision_left<Spec, 1>(root);
        current[1] = mom_equihash::load_collision_right<Spec, 1>(root);
        unsigned count = 2;
        bool ok = expand_round(round6, current, next, count, slots);
        count *= 2;
        ok = ok && expand_round(round5, current, next, count, slots);
        count *= 2;
        ok = ok && expand_round(round4, current, next, count, slots);
        count *= 2;
#ifdef MOM_SYCL_HAS_HIP
        ok = ok && expand_round(round3, current, next, count, slots);
#else
        ok = ok && expand_split_round<typename Arena::Round3Record>(
            round3_heads, round3_tails, current, next, count, slots);
#endif
        count *= 2;
        ok = ok && expand_round(round2, current, next, count, slots);
        count *= 2;
        ok = ok && expand_round(round1, current, next, count, slots);
        count *= 2;
        const unsigned base = candidate * Spec::proof_indices;
        for (unsigned i = 0; i < count; ++i) {
          ok = ok && current[i] < slots;
          if (ok) {
            leaves[base + i] = mom_equihash::load_bucket_level0_leaf(level0[current[i]]);
          }
        }
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

}  // namespace mom_equihash192_7::direct
