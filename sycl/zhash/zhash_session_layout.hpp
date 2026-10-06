#pragma once

#include <cstddef>
#include <cstdint>

#include "equihash_core.hpp"

namespace mom_zhash {

using ZHashSpec = mom_equihash::ZHash144_5;

template <unsigned BucketBits, unsigned SlotsPerBucket>
struct ZHashBucketLayout {
  static_assert(BucketBits < ZHashSpec::collision_bits && SlotsPerBucket != 0,
                "ZHash bucket geometry must be non-zero");
  static constexpr unsigned bucket_bits = BucketBits;
  static constexpr unsigned bucket_count = 1u << bucket_bits;
  static constexpr unsigned local_bits = ZHashSpec::collision_bits - BucketBits;
  static constexpr unsigned local_bins = 1u << local_bits;
  static constexpr unsigned collision_work_group = ZHashSpec::collision_work_group;
  static constexpr unsigned bucket_slot_capacity = SlotsPerBucket;
  static constexpr std::size_t slot_count =
      static_cast<std::size_t>(bucket_count) * bucket_slot_capacity;
  static constexpr unsigned bucket_for(const std::uint32_t field) {
    return field >> local_bits;
  }
  static constexpr unsigned local_bin(const std::uint32_t field) {
    return field & (local_bins - 1u);
  }
  static constexpr std::size_t local_memory_bytes =
      static_cast<std::size_t>(local_bins) * sizeof(std::uint32_t) +
      static_cast<std::size_t>(bucket_slot_capacity) * sizeof(std::uint16_t);
};

// Initial and later rounds can tune their bucket geometry independently. This lets a backend
// reduce collision work without truncating the arena or changing the portable layout.
template <unsigned SlotsPerBucket, unsigned LaterSlotsPerBucket = SlotsPerBucket,
          unsigned LaterBucketBits = 12, unsigned InitialBucketBits = 12>
struct ZHashArenaLayout : ZHashBucketLayout<InitialBucketBits, SlotsPerBucket> {
  using InitialLayout = ZHashBucketLayout<InitialBucketBits, SlotsPerBucket>;
  using LaterLayout = ZHashBucketLayout<LaterBucketBits, LaterSlotsPerBucket>;
  static constexpr unsigned max_candidate_roots = 8192;

  using Level0Record =
      mom_equihash::PackedBucketLevel0Record<ZHashSpec, InitialLayout::local_bits>;
  using Round1Record = mom_equihash::PackedBucketCollisionRecord<
      ZHashSpec, 5, InitialLayout::local_bits, InitialLayout::bucket_bits,
      InitialLayout::bucket_slot_capacity>;
  using Round2Record = mom_equihash::PackedBucketCollisionRecord<
      ZHashSpec, 4, LaterLayout::local_bits, InitialLayout::bucket_bits,
      InitialLayout::bucket_slot_capacity>;
  using Round3Record = mom_equihash::PackedBucketCollisionRecord<
      ZHashSpec, 3, LaterLayout::local_bits, LaterLayout::bucket_bits,
      LaterLayout::bucket_slot_capacity>;
#ifdef MOM_SYCL_HAS_HIP
  // Keep bucket-relative provenance for one more round on HIP so this bandwidth-bound arena uses
  // three words per record. The generic path retains its established full-offset representation.
  using Round4Record = mom_equihash::PackedBucketCollisionRecord<
      ZHashSpec, 2, LaterLayout::local_bits, LaterLayout::bucket_bits,
      LaterLayout::bucket_slot_capacity>;
#else
  using Round4Record = mom_equihash::CollisionRoundRecord<ZHashSpec, 4>;
#endif
  using RootRecord = mom_equihash::ZHashRootRecord<ZHashSpec>;

  static constexpr std::size_t level0_bytes = InitialLayout::slot_count * sizeof(Level0Record);
  static constexpr std::size_t round1_bytes = InitialLayout::slot_count * sizeof(Round1Record);
  static constexpr std::size_t round2_bytes = LaterLayout::slot_count * sizeof(Round2Record);
  static constexpr std::size_t round3_bytes = LaterLayout::slot_count * sizeof(Round3Record);
  static constexpr std::size_t round4_bytes = LaterLayout::slot_count * sizeof(Round4Record);
  static constexpr std::size_t candidate_root_bytes =
      max_candidate_roots * sizeof(RootRecord);
  static constexpr std::size_t recovered_leaf_bytes =
      max_candidate_roots * ZHashSpec::proof_indices * sizeof(std::uint32_t);
  static constexpr std::size_t recovered_valid_bytes = max_candidate_roots;
  static constexpr std::size_t verified_field_bytes =
      max_candidate_roots * ZHashSpec::proof_indices * ZHashSpec::rounds * sizeof(std::uint32_t);
  static constexpr std::size_t local_memory_bytes =
      InitialLayout::local_memory_bytes > LaterLayout::local_memory_bytes
          ? InitialLayout::local_memory_bytes : LaterLayout::local_memory_bytes;

  // Level zero plus four retained provenance arenas. Canonical ordering is deferred until root
  // recovery, so hot collision rounds need no parallel minimum-index arenas or flat root scratch.
  static constexpr std::size_t provenance_bytes =
      level0_bytes + round1_bytes + round2_bytes + round3_bytes + round4_bytes;
  static constexpr std::size_t bucket_counts_bytes =
      (2u * InitialLayout::bucket_count + 3u * LaterLayout::bucket_count) *
      sizeof(std::uint32_t);
  static constexpr std::size_t counter_bytes =
      ((ZHashSpec::header_length + 7u) & ~std::size_t{7}) + 8u * sizeof(std::uint64_t) +
      11u * sizeof(std::uint32_t);  // five output/overflow counts and zero-root count

  // This is the exact sum of requested device allocations, excluding allocator/runtime metadata.
  static constexpr std::size_t device_bytes =
      provenance_bytes + bucket_counts_bytes + candidate_root_bytes +
      recovered_leaf_bytes + recovered_valid_bytes + verified_field_bytes + counter_bytes;

  static constexpr std::size_t logical_row_count = ZHashSpec::row_count;

  static constexpr bool is_valid() {
    return InitialLayout::local_bits <= 16 && LaterLayout::local_bits <= 16 &&
           InitialLayout::bucket_slot_capacity <= 65535 &&
           LaterLayout::bucket_slot_capacity <= 65535;
  }
  static constexpr bool holds_logical_rows() {
    return InitialLayout::slot_count >= logical_row_count;
  }
  static_assert(is_valid(), "ZHash layout has incompatible local bucket geometry");
  static_assert(sizeof(Level0Record) == 20 && sizeof(Round1Record) <= 20 &&
                    sizeof(Round2Record) == 16,
                "ZHash packed record ABI changed; update the footprint contract");
#if defined(MOM_SYCL_HAS_HIP)
  static_assert(sizeof(Round3Record) == 12,
#else
  static_assert(sizeof(Round3Record) == 16,
#endif
                "ZHash round-three record ABI changed; update the footprint contract");
#ifdef MOM_SYCL_HAS_HIP
  static_assert(sizeof(Round4Record) == 12,
#else
  static_assert(sizeof(Round4Record) == 16,
#endif
                "ZHash packed record ABI changed; update the footprint contract");
};

// The compact layout keeps Intel below a local-memory occupancy cliff. Mining retries any bounded
// overflow with the larger default layout, which also raises level-zero capacity while keeping the
// peak allocation below the documented 4 GiB device minimum.
#ifdef MOM_ZHASH_INTEL_LATE_BUCKETS
using FastZHashArenaLayout = ZHashArenaLayout<4480, 5632, 13, 13>;
using DefaultZHashArenaLayout = ZHashArenaLayout<4672, 5632, 13, 13>;
#elif defined(MOM_SYCL_HAS_HIP)
// Offset adjacent bucket heads within a page instead of repeating a strongly aligned stride.
using FastZHashArenaLayout = ZHashArenaLayout<8720>;
using DefaultZHashArenaLayout = ZHashArenaLayout<9216>;
#else
using FastZHashArenaLayout = ZHashArenaLayout<8704>;
using DefaultZHashArenaLayout = ZHashArenaLayout<9216>;
#endif

enum class SessionStatus : std::uint8_t {
  idle,
  running,
  solved,
  no_solution,
  capacity_overflow,
  device_error,
};

// Host-only state machine used by the SYCL controller and by injectable-layout tests.  It makes
// truncation a terminal condition for one attempt: callers must reset/recreate the session before
// trying a different layout, so stale counts can never be mistaken for a complete solve.
class ZHashSessionLifecycle {
public:
  explicit ZHashSessionLifecycle(const std::size_t capacity) : capacity_(capacity) {}

  void reset() {
    status_ = SessionStatus::idle;
    stage_ = 0;
    flat_count_ = 0;
    overflow_count_ = 0;
    root_count_ = 0;
    zero_root_count_ = 0;
  }

  bool begin() {
    if (status_ == SessionStatus::running) return false;
    reset();
    status_ = SessionStatus::running;
    return true;
  }

  bool generated(const std::uint32_t overflow_count) {
    if (!running() || stage_ != 0) return false;
    if (overflow_count != 0) return fail_capacity(overflow_count);
    stage_ = 1;
    return true;
  }

  bool flat_output(const std::uint32_t count) {
    if (!running() || stage_ == 0 || stage_ > 5) return false;
    flat_count_ = count;
    if (count > capacity_) return fail_capacity(count - static_cast<std::uint32_t>(capacity_));
    return true;
  }

  bool scattered(const std::uint32_t overflow_count) {
    if (!running() || stage_ == 0 || stage_ >= 5) return false;
    if (overflow_count != 0) return fail_capacity(overflow_count);
    ++stage_;
    return true;
  }

  bool roots(const std::uint32_t root_count, const std::uint32_t zero_root_count) {
    if (!running() || stage_ != 5 || root_count > capacity_ || zero_root_count > root_count) {
      return fail_capacity(root_count > capacity_ ? root_count - static_cast<std::uint32_t>(capacity_) : 1u);
    }
    root_count_ = root_count;
    zero_root_count_ = zero_root_count;
    stage_ = 6;
    return true;
  }

  void finish(const bool solved) {
    if (running()) {
      status_ = solved ? SessionStatus::solved : SessionStatus::no_solution;
    }
  }

  void device_error() {
    if (running()) {
      status_ = SessionStatus::device_error;
    }
  }

  bool running() const { return status_ == SessionStatus::running; }
  SessionStatus status() const { return status_; }
  unsigned stage() const { return stage_; }
  std::size_t capacity() const { return capacity_; }
  std::uint32_t flat_count() const { return flat_count_; }
  std::uint32_t overflow_count() const { return overflow_count_; }
  std::uint32_t root_count() const { return root_count_; }
  std::uint32_t zero_root_count() const { return zero_root_count_; }

private:
  bool fail_capacity(const std::uint32_t count) {
    overflow_count_ += count;
    status_ = SessionStatus::capacity_overflow;
    return false;
  }

  std::size_t capacity_;
  SessionStatus status_ = SessionStatus::idle;
  unsigned stage_ = 0;
  std::uint32_t flat_count_ = 0;
  std::uint32_t overflow_count_ = 0;
  std::uint32_t root_count_ = 0;
  std::uint32_t zero_root_count_ = 0;
};

}  // namespace mom_zhash
