#pragma once

#include <sycl/sycl.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <new>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "../lib-internal.h"
#include "equihash192_7_direct.hpp"

namespace mom_equihash192_7::direct {

#if defined(MOM_SYCL_HAS_HIP)
constexpr unsigned default_round0_partitions = 8;
#else
constexpr unsigned default_round0_partitions = 2;
#endif

struct Solution {
  std::array<std::uint32_t, Spec::proof_indices> indices{};
  std::array<std::uint8_t, Spec::solution_length> encoded{};
};

class BucketArenaOverflow : public std::overflow_error {
public:
  explicit BucketArenaOverflow(const std::string& detail = {})
      : std::overflow_error("Equihash(192,7) bucket arena overflow" + detail) {}
};

template <typename Arena = ProductionArena, unsigned Round0WorkGroup = 1024,
          unsigned Round1WorkGroup = 1024, unsigned Round2WorkGroup = 1024,
          unsigned Round3WorkGroup = 1024, unsigned Round4WorkGroup = 1024,
          unsigned Round5WorkGroup = 1024, unsigned RootWorkGroup = 1024,
          unsigned Round0Partitions = default_round0_partitions>
class Session {
public:
  using Layout = typename Arena::Layout;
  using Round0 = LaunchLayout<Round0WorkGroup, typename Arena::Level0Layout>;
  using Round1 = LaunchLayout<Round1WorkGroup, typename Arena::Round1Layout>;
  using Round2 = LaunchLayout<Round2WorkGroup, typename Arena::Round2Layout>;
  using Round3 = LaunchLayout<Round3WorkGroup, typename Arena::Round3Layout>;
  using Round4 = LaunchLayout<Round4WorkGroup, typename Arena::Round4Layout>;
  using Round5 = LaunchLayout<Round5WorkGroup, typename Arena::Round5Layout>;
  using Root = LaunchLayout<RootWorkGroup, typename Arena::Round6Layout>;
  // Root counts vary with the header. Keep enough result space for valid high-count jobs instead
  // of failing an otherwise healthy device when a header narrowly exceeds the former 4096 limit.
  static constexpr std::uint32_t root_capacity = 8192;

  explicit Session(const sycl::device& device)
      : queue_(make_queue(device, std::getenv("MOM_EQUIHASH192_7_PROFILE") != nullptr)),
        level_zero_(sycl_is_level_zero_gpu(device)) {
#ifdef MOM_SYCL_HAS_HIP
    cache_partition_inputs_ =
        device.get_info<sycl::info::device::local_mem_size>() >= required_cache_local_bytes;
#endif
    allocate();
  }

  ~Session() {
    try {
      queue_.wait_and_throw();
    } catch (...) {
    }
    release();
  }

  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;

  std::vector<Solution> run(const std::uint8_t (&header)[Spec::header_length]) {
#ifdef MOM_SYCL_HAS_HIP
    if (cache_partition_inputs_) return run_with_cache<true>(header);
#endif
    return run_with_cache<false>(header);
  }

private:
  template <bool CacheInput>
  std::vector<Solution> run_with_cache(const std::uint8_t (&header)[Spec::header_length]) {
    if (std::getenv("MOM_EQUIHASH192_7_HYBRID_ARENA"))
      return run_layout<HybridArena, CacheInput>(header);
    if (std::getenv("MOM_EQUIHASH192_7_RETRY_ARENA"))
      return run_layout<RetryArena, CacheInput>(header);
    if (level_zero_) {
      try {
        return run_layout<HybridArena, CacheInput>(header);
      } catch (const BucketArenaOverflow&) {
        try {
          return run_layout<RetryArena, CacheInput>(header);
        } catch (const BucketArenaOverflow&) {
          if constexpr (std::is_same_v<Arena, RetryArena>) throw;
          return run_layout<Arena, CacheInput>(header);
        }
      }
    }
    try {
      return run_layout<Arena, CacheInput>(header);
    } catch (const BucketArenaOverflow&) {
      if constexpr (std::is_same_v<Arena, RetryArena>) throw;
      return run_layout<RetryArena, CacheInput>(header);
    }
  }

  template <typename ActiveArena, bool CacheInput>
  std::vector<Solution> run_layout(const std::uint8_t (&header)[Spec::header_length]) {
    const auto started = std::chrono::steady_clock::now();
    const bool profile = std::getenv("MOM_EQUIHASH192_7_PROFILE") && profile_reports_ < 8;
    if (profile) {
      ++profile_reports_;
    }
    static_assert(ActiveArena::Layout::slot_count == Layout::slot_count);
    using ActiveRound0 = LaunchLayout<Round0WorkGroup, typename ActiveArena::Level0Layout>;
    using ActiveRound1 = LaunchLayout<Round1WorkGroup, typename ActiveArena::Round1Layout>;
    using ActiveRound2 = LaunchLayout<Round2WorkGroup, typename ActiveArena::Round2Layout>;
    using ActiveRound3 = LaunchLayout<Round3WorkGroup, typename ActiveArena::Round3Layout>;
    using ActiveRound4 = LaunchLayout<Round4WorkGroup, typename ActiveArena::Round4Layout>;
    using ActiveRound5 = LaunchLayout<Round5WorkGroup, typename ActiveArena::Round5Layout>;
    constexpr unsigned active_root_work_group =
        std::is_same_v<ActiveArena, HybridArena> ? 512 : RootWorkGroup;
    using ActiveRound6 =
        LaunchLayout<active_root_work_group, typename ActiveArena::Round6Layout>;
    auto* level0 = reinterpret_cast<typename ActiveArena::Level0Record*>(level0_);
    auto* round1 = reinterpret_cast<typename ActiveArena::Round1Record*>(round1_);
    auto* round2 = reinterpret_cast<typename ActiveArena::Round2Record*>(round2_);
#ifdef MOM_SYCL_HAS_HIP
    auto* round3 = reinterpret_cast<typename ActiveArena::Round3Record*>(round3_);
#endif
    auto* round4 = reinterpret_cast<typename ActiveArena::Round4Record*>(round4_);
    auto* round5 = reinterpret_cast<typename ActiveArena::Round5Record*>(round5_);
    auto* round6 = reinterpret_cast<typename ActiveArena::Round6Record*>(round6_);
    std::array<std::uint64_t, 10> input{};
    mom_equihash::hash_header_midstate<Spec>(header, input.data());
    input[8] = mom_equihash::load64_le(header + 128);
    input[9] = mom_equihash::load32_le(header + 136);
    std::uint32_t status[2]{};
    MomSyclHostTransferGuard input_transfer(queue_, "equihash192_7 input upload");
    queue_.memcpy(input_, input.data(), sizeof(input));
    clear();
    std::array<sycl::event, 9> stages;
    stages[0] = mom_equihash::submit_generation_bucketed_full<Spec, ActiveRound0>(
        queue_, input_, input_ + 8, level0, counts_[0], overflow_);
    constexpr unsigned round0_partitions =
        std::is_same_v<ActiveArena, HybridArena> ? 2 : Round0Partitions;
    stages[1] = mom_equihash::submit_bucket_round_bucketed_full<
        Spec, ActiveRound0, ActiveRound1, typename ActiveArena::Round1Record, round0_partitions>(
        queue_, level0, counts_[0], round1, counts_[1], overflow_);
#ifdef MOM_SYCL_HAS_HIP
    stages[2] = mom_equihash::submit_collision_round_bucketed_partitioned<
        Spec, 7, ActiveRound1, ActiveRound2, typename ActiveArena::Round1Record,
        typename ActiveArena::Round2Record, 8, CacheInput>(
        queue_, round1, counts_[1], round2, counts_[2], overflow_);
    stages[3] = mom_equihash::submit_collision_round_bucketed_partitioned<
        Spec, 6, ActiveRound2, ActiveRound3, typename ActiveArena::Round2Record,
        typename ActiveArena::Round3Record, 8, CacheInput>(
        queue_, round2, counts_[2], round3, counts_[3], overflow_);
    stages[4] = mom_equihash::submit_collision_round_bucketed_partitioned<
        Spec, 5, ActiveRound3, ActiveRound4, typename ActiveArena::Round3Record,
        typename ActiveArena::Round4Record, 8, CacheInput>(
        queue_, round3, counts_[3], round4, counts_[4], overflow_);
    stages[5] = mom_equihash::submit_collision_round_bucketed_partitioned<
        Spec, 4, ActiveRound4, ActiveRound5, typename ActiveArena::Round4Record,
        typename ActiveArena::Round5Record, 4, CacheInput>(
        queue_, round4, counts_[4], round5, counts_[5], overflow_);
    stages[6] = mom_equihash::submit_collision_round_bucketed_partitioned<
        Spec, 3, ActiveRound5, ActiveRound6, typename ActiveArena::Round5Record,
        typename ActiveArena::Round6Record, 4, CacheInput>(queue_, round5, counts_[5], round6,
                                                           counts_[6], overflow_);
#else
    if (level_zero_) {
      constexpr unsigned partitions = 1u << (15 - ActiveRound1::bucket_bits);
      stages[2] = mom_equihash::submit_collision_round_bucketed_partitioned<
          Spec, 7, ActiveRound1, ActiveRound2, typename ActiveArena::Round1Record,
          typename ActiveArena::Round2Record, partitions>(queue_, round1, counts_[1], round2,
                                                          counts_[2], overflow_);
    } else {
      stages[2] = mom_equihash::submit_collision_round_bucketed<
          Spec, 7, ActiveRound1, ActiveRound2, typename ActiveArena::Round1Record,
          typename ActiveArena::Round2Record>(
          queue_, round1, counts_[1], round2, counts_[2], overflow_);
    }
    if (level_zero_) {
      constexpr unsigned partitions =
          std::is_same_v<ActiveArena, HybridArena> ? 2 : 4;
      stages[3] = mom_equihash::submit_collision_round_bucketed_split_output_partitioned<
          Spec, 6, ActiveRound2, ActiveRound3, typename ActiveArena::Round2Record,
          typename ActiveArena::Round3Record, partitions>(
          queue_, round2, counts_[2], round3_heads_, round3_tails_, counts_[3], overflow_);
      stages[4] = mom_equihash::submit_split_collision_round_bucketed_partitioned<
          Spec, 5, ActiveRound3::local_bits, ActiveRound3, ActiveRound4,
          typename ActiveArena::Round3Record, typename ActiveArena::Round4Record, 4>(
          queue_, round3_heads_, round3_tails_, counts_[3], round4, counts_[4], overflow_);
    } else {
      stages[3] = mom_equihash::submit_collision_round_bucketed_split_output<
          Spec, 6, ActiveRound2, ActiveRound3, typename ActiveArena::Round2Record,
          typename ActiveArena::Round3Record>(
          queue_, round2, counts_[2], round3_heads_, round3_tails_, counts_[3], overflow_);
      stages[4] = mom_equihash::submit_split_collision_round_bucketed<
          Spec, 5, ActiveRound3::local_bits, ActiveRound3, ActiveRound4,
          typename ActiveArena::Round3Record, typename ActiveArena::Round4Record>(
          queue_, round3_heads_, round3_tails_, counts_[3], round4, counts_[4], overflow_);
    }
    if (level_zero_) {
      stages[5] = mom_equihash::submit_collision_round_bucketed_partitioned<
          Spec, 4, ActiveRound4, ActiveRound5, typename ActiveArena::Round4Record,
          typename ActiveArena::Round5Record, 2>(
          queue_, round4, counts_[4], round5, counts_[5], overflow_);
      stages[6] = mom_equihash::submit_collision_round_bucketed_partitioned<
          Spec, 3, ActiveRound5, ActiveRound6, typename ActiveArena::Round5Record,
          typename ActiveArena::Round6Record, 2>(queue_, round5, counts_[5], round6, counts_[6],
                                                 overflow_);
    } else {
      stages[5] = mom_equihash::submit_collision_round_bucketed<
          Spec, 4, ActiveRound4, ActiveRound5, typename ActiveArena::Round4Record,
          typename ActiveArena::Round5Record>(
          queue_, round4, counts_[4], round5, counts_[5], overflow_);
      stages[6] = mom_equihash::submit_collision_round_bucketed<Spec, 3, ActiveRound5, ActiveRound6,
                                                                typename ActiveArena::Round5Record,
                                                                typename ActiveArena::Round6Record>(
          queue_, round5, counts_[5], round6, counts_[6], overflow_);
    }
#endif
    stages[7] = mom_equihash::submit_zero_root_round_bucketed<Spec, ActiveRound6>(
        queue_, round6, counts_[6], roots_, root_count_, root_capacity);
    stages[8] = submit_recover_leaves<ActiveArena>(
#ifdef MOM_SYCL_HAS_HIP
        queue_, level0, round1, round2, round3, round4, round5, round6,
#else
        queue_, level0, round1, round2, round3_heads_, round3_tails_, round4, round5, round6,
#endif
        roots_, root_count_, root_capacity, leaves_, valid_);

    // A D2H submission behind live kernels can itself busy-spin. Complete recovery first so the
    // low-CPU result-copy wait starts after submission has become nonblocking.
    sycl_wait_and_throw(stages.back(), queue_.get_device());
    sycl_wait_and_throw(queue_.memcpy(status, overflow_, sizeof(status)), queue_.get_device());
    const std::uint32_t overflow = status[0], root_count = status[1];
    if (overflow != 0) {
      std::string detail;
      std::vector<std::uint32_t> bucket_counts(count_stride);
      MomSyclHostTransferGuard overflow_transfer(queue_, "equihash192_7 overflow readback");
      constexpr std::array<std::uint32_t, 7> capacities{
          ActiveRound0::bucket_slot_capacity, ActiveRound1::bucket_slot_capacity,
          ActiveRound2::bucket_slot_capacity, ActiveRound3::bucket_slot_capacity,
          ActiveRound4::bucket_slot_capacity, ActiveRound5::bucket_slot_capacity,
          ActiveRound6::bucket_slot_capacity};
      for (unsigned round = 0; round < counts_.size(); ++round) {
        sycl_wait_and_throw(
            queue_.memcpy(bucket_counts.data(), counts_[round],
                          bucket_counts.size() * sizeof(bucket_counts[0])),
            queue_.get_device());
        const auto maximum = *std::max_element(bucket_counts.begin(), bucket_counts.end());
        if (maximum > capacities[round])
          detail += " r" + std::to_string(round) + "=" + std::to_string(maximum);
      }
      throw BucketArenaOverflow(detail);
    }
    if (root_count > root_capacity)
      throw std::overflow_error("Equihash(192,7) root arena overflow: " +
                                std::to_string(root_count));
    const auto device_done = std::chrono::steady_clock::now();
    std::vector<std::uint32_t> leaves(
        static_cast<std::size_t>(root_count) * Spec::proof_indices);
    std::vector<std::uint8_t> valid(root_count);
    MomSyclHostTransferGuard result_transfer(queue_, "equihash192_7 readback");
    if (root_count != 0) {
      queue_.memcpy(valid.data(), valid_, valid.size());
      sycl_wait_and_throw(
          queue_.memcpy(leaves.data(), leaves_, leaves.size() * sizeof(leaves[0])),
          queue_.get_device());
    }
    std::vector<Solution> solutions;
    for (std::uint32_t candidate = 0; candidate < root_count; ++candidate) {
      if (!valid[candidate]) continue;
      Solution solution;
      std::copy_n(leaves.data() + candidate * Spec::proof_indices,
                  Spec::proof_indices, solution.indices.begin());
      mom_equihash::encode_indices<Spec>(solution.indices.data(), solution.encoded.data());
      if (mom_equihash::verify_solution<Spec>(header, solution.encoded.data()))
        solutions.push_back(solution);
    }
    std::sort(solutions.begin(), solutions.end(), [](const auto& left, const auto& right) {
      return left.encoded < right.encoded;
    });
    solutions.erase(
        std::unique(solutions.begin(), solutions.end(), [](const auto& left, const auto& right) {
          return left.encoded == right.encoded;
        }),
        solutions.end());
    const auto host_done = std::chrono::steady_clock::now();
    if (profile) {
      std::array<double, 9> stage_ms{};
      for (unsigned i = 0; i < stages.size(); ++i) {
        const auto start = stages[i].get_profiling_info<
            sycl::info::event_profiling::command_start>();
        const auto end = stages[i].get_profiling_info<
            sycl::info::event_profiling::command_end>();
        stage_ms[i] = static_cast<double>(end - start) / 1.0e6;
      }
      const auto elapsed_ms = [](const auto begin, const auto end) {
        return std::chrono::duration<double, std::milli>(end - begin).count();
      };
      std::fprintf(stderr,
                   "equihash192_7 profile layout=%ux%u/%ux%u/%ux%u/%ux%u/%ux%u/%ux%u/%ux%u "
                   "roots=%u unique=%zu wall_ms=%.3f host_ms=%.3f "
                   "stage_ms=gen:%.3f,r0:%.3f,r1:%.3f,r2:%.3f,r3:%.3f,r4:%.3f,r5:%.3f,root:%.3f,"
                   "recover:%.3f\n",
                   ActiveRound0::bucket_count, ActiveRound0::bucket_slot_capacity,
                   ActiveRound1::bucket_count, ActiveRound1::bucket_slot_capacity,
                   ActiveRound2::bucket_count, ActiveRound2::bucket_slot_capacity,
                   ActiveRound3::bucket_count, ActiveRound3::bucket_slot_capacity,
                   ActiveRound4::bucket_count, ActiveRound4::bucket_slot_capacity,
                   ActiveRound5::bucket_count, ActiveRound5::bucket_slot_capacity,
                   ActiveRound6::bucket_count, ActiveRound6::bucket_slot_capacity, root_count,
                   solutions.size(), elapsed_ms(started, device_done),
                   elapsed_ms(device_done, host_done), stage_ms[0], stage_ms[1], stage_ms[2],
                   stage_ms[3], stage_ms[4], stage_ms[5], stage_ms[6], stage_ms[7], stage_ms[8]);
    }
    return solutions;
  }

  static sycl::queue make_queue(const sycl::device& device, const bool profile) {
    if (profile)
      return sycl::queue(device, sycl::property_list{
          sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{}});
    return sycl::queue(device, sycl::property_list{sycl::property::queue::in_order{}});
  }

#ifdef MOM_SYCL_HAS_HIP
  template <typename ActiveArena> static constexpr std::size_t cache_local_bytes() {
    return std::max(
        {mom_equihash::partitioned_collision_local_bytes<
             typename ActiveArena::Round1Layout, typename ActiveArena::Round1Record, 8, true>,
         mom_equihash::partitioned_collision_local_bytes<
             typename ActiveArena::Round2Layout, typename ActiveArena::Round2Record, 8, true>,
         mom_equihash::partitioned_collision_local_bytes<
             typename ActiveArena::Round3Layout, typename ActiveArena::Round3Record, 8, true>,
         mom_equihash::partitioned_collision_local_bytes<
             typename ActiveArena::Round4Layout, typename ActiveArena::Round4Record, 4, true>,
         mom_equihash::partitioned_collision_local_bytes<
             typename ActiveArena::Round5Layout, typename ActiveArena::Round5Record, 4, true>});
  }

  static constexpr std::size_t required_cache_local_bytes =
      std::max({cache_local_bytes<Arena>(), cache_local_bytes<RetryArena>(),
                cache_local_bytes<HybridArena>()});
#endif

  static constexpr std::size_t count_stride =
      std::max({Layout::bucket_count, RetryArena::Layout::bucket_count,
                HybridArena::Layout::bucket_count});
  sycl::queue queue_;
  const bool level_zero_;
  std::uint64_t* input_ = nullptr;
  typename Arena::Level0Record* level0_ = nullptr;
  typename Arena::Round1Record* round1_ = nullptr;
  typename Arena::Round2Record* round2_ = nullptr;
#ifdef MOM_SYCL_HAS_HIP
  typename Arena::Round3Record* round3_ = nullptr;
#else
  mom_equihash::SplitRecordHead* round3_heads_ = nullptr;
  std::uint32_t* round3_tails_ = nullptr;
#endif
  typename Arena::Round4Record* round4_ = nullptr;
  typename Arena::Round5Record* round5_ = nullptr;
  typename Arena::Round6Record* round6_ = nullptr;
  std::uint32_t* counts_storage_ = nullptr;
  std::array<std::uint32_t*, 7> counts_{};
  unsigned profile_reports_ = 0;
#ifdef MOM_SYCL_HAS_HIP
  bool cache_partition_inputs_ = false;
#endif
  std::uint32_t* overflow_ = nullptr;
  typename Arena::RootRecord* roots_ = nullptr;
  std::uint32_t* root_count_ = nullptr;
  std::uint32_t* leaves_ = nullptr;
  std::uint8_t* valid_ = nullptr;

  template <typename T>
  T* allocate(const std::size_t count) {
    if (T* value = sycl::malloc_device<T>(count, queue_)) {
      return value;
    }
    throw std::bad_alloc();
  }

  void allocate() {
    const auto device = queue_.get_device();
    const std::uint64_t required = Arena::arena_bytes +
        7ull * count_stride * sizeof(std::uint32_t) + 512ull * 1024ull * 1024ull;
    if (!mom_has_usm_device(device) ||
        device.get_info<sycl::info::device::global_mem_size>() < required ||
        device.get_info<sycl::info::device::max_work_group_size>() < Root::collision_work_group ||
        device.get_info<sycl::info::device::local_mem_size>() <
            Layout::local_bins * sizeof(std::uint32_t) +
            Layout::bucket_slot_capacity * sizeof(std::uint16_t))
      throw std::runtime_error("Equihash(192,7) direct arena is unsupported on this device");
    try {
      input_ = allocate<std::uint64_t>(10);
      level0_ = allocate<typename Arena::Level0Record>(Layout::slot_count);
      round1_ = allocate<typename Arena::Round1Record>(Layout::slot_count);
      round2_ = allocate<typename Arena::Round2Record>(Layout::slot_count);
#ifdef MOM_SYCL_HAS_HIP
      round3_ = allocate<typename Arena::Round3Record>(Layout::slot_count);
#else
      round3_heads_ = allocate<mom_equihash::SplitRecordHead>(Layout::slot_count);
      round3_tails_ = allocate<std::uint32_t>(Layout::slot_count);
#endif
      round4_ = allocate<typename Arena::Round4Record>(Layout::slot_count);
      round5_ = allocate<typename Arena::Round5Record>(Layout::slot_count);
      round6_ = allocate<typename Arena::Round6Record>(Layout::slot_count);
      counts_storage_ = allocate<std::uint32_t>(7 * count_stride + 2);
      for (unsigned i = 0; i < counts_.size(); ++i)
        counts_[i] = counts_storage_ + i * count_stride;
      overflow_ = counts_storage_ + 7 * count_stride;
      roots_ = allocate<typename Arena::RootRecord>(root_capacity);
      root_count_ = overflow_ + 1;
      leaves_ = allocate<std::uint32_t>(root_capacity * Spec::proof_indices);
      valid_ = allocate<std::uint8_t>(root_capacity);
    } catch (...) {
      release();
      throw;
    }
  }

  void clear() {
    queue_.fill(counts_storage_, 0u, 7 * count_stride + 2);
  }

  template <typename T>
  void free(T*& value) noexcept {
    if (value) {
      try {
        sycl::free(value, queue_);
      } catch (...) {
      }
      value = nullptr;
    }
  }

  void release() noexcept {
    free(input_);
    free(level0_);
    free(round1_);
    free(round2_);
#ifdef MOM_SYCL_HAS_HIP
    free(round3_);
#else
    free(round3_heads_);
    free(round3_tails_);
#endif
    free(round4_);
    free(round5_);
    free(round6_);
    free(counts_storage_);
    counts_.fill(nullptr);
    overflow_ = root_count_ = nullptr;
    free(roots_);
    free(leaves_);
    free(valid_);
  }
};

}  // namespace mom_equihash192_7::direct
