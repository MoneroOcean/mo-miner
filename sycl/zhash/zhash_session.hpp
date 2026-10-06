#pragma once

#include <sycl/sycl.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

#include "equihash_sycl.hpp"
#include "zhash_session_layout.hpp"

namespace mom_zhash {

struct SessionOptions {
  // The session does not know about another algorithm's allocations.  Keep a reserve for the
  // backend/runtime unless the caller supplies a stricter absolute device-byte limit.
  std::uint64_t safety_reserve_bytes = 512ull * 1024ull * 1024ull;
  std::uint64_t max_device_bytes = 0;
  // Round totals are diagnostic only; dedicated overflow counters guard every bounded arena.
  bool collect_round_counts = false;
  bool collect_stage_times = false;
};

struct Solution {
  std::uint32_t root_slot = 0;
  std::array<std::uint32_t, ZHashSpec::proof_indices> indices{};
  std::array<std::uint8_t, ZHashSpec::solution_length> encoded{};
};

struct RunReport {
  SessionStatus status = SessionStatus::idle;
  std::uint64_t device_bytes = 0;
  std::array<std::uint32_t, 5> flat_counts{};
  std::array<std::uint32_t, 5> overflow_counts{};  // generation, then four bucket rounds
  std::array<double, 6> stage_ms{};  // generation, four collision rounds, zero-root round
  std::uint32_t root_count = 0;
  std::uint32_t zero_root_count = 0;
  std::uint32_t rejected_root_count = 0;
  std::string error;
  std::vector<Solution> solutions;
};

// A synchronous, one-header controller for one SYCL device.  It intentionally owns no miner ABI:
// callers provide a queue and receive validated canonical solutions.  All persistent records use
// device USM; only bounded result metadata and candidate leaves are copied to the host.
template <typename Layout = DefaultZHashArenaLayout>
class Session {
public:
  using Spec = ZHashSpec;
  using Level0Record = typename Layout::Level0Record;
  using Round1Record = typename Layout::Round1Record;
  using Round2Record = typename Layout::Round2Record;
  using Round3Record = typename Layout::Round3Record;
  using Round4Record = typename Layout::Round4Record;
  using RootRecord = typename Layout::RootRecord;
  using InitialLayout = typename Layout::InitialLayout;
  using LaterLayout = typename Layout::LaterLayout;
  // Most headers stay within this one-pass window; rare graph bursts recover only their tail.
  static constexpr unsigned fast_candidate_roots =
      Layout::max_candidate_roots < 256 ? Layout::max_candidate_roots : 256;

  explicit Session(const sycl::queue& source, const SessionOptions& options = {})
      : queue_(make_queue(source, options.collect_stage_times)),
        options_(options), lifecycle_(Layout::slot_count) {
#ifdef MOM_SYCL_HAS_HIP
    cache_partition_inputs_ = queue_.get_device().get_info<sycl::info::device::local_mem_size>() >=
                              required_cache_local_bytes;
#endif
    allocate();
  }

  ~Session() {
    try {
      queue_.wait_and_throw();
    } catch (...) {
    }
    free_all();
  }

  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;

  sycl::device device() const { return queue_.get_device(); }
  const sycl::queue& queue() const { return queue_; }
  std::size_t device_bytes() const { return Layout::device_bytes; }
  SessionStatus status() const { return lifecycle_.status(); }
  const std::string& last_error() const { return last_error_; }

  // Discards all counters and starts a clean attempt.  It is safe to call after an overflow or a
  // rejected root; a different capacity requires a different Session<Layout> instantiation.
  void reset() {
    queue_.wait_and_throw();
    clear_device_state();
    lifecycle_.reset();
    last_error_.clear();
  }

  RunReport run(const std::uint8_t (&header)[Spec::header_length]) {
    RunReport report;
    report.device_bytes = Layout::device_bytes;
    if (!lifecycle_.begin()) {
      report.status = SessionStatus::device_error;
      report.error = "ZHash session is already running";
      return report;
    }
    last_error_.clear();

    try {
      clear_device_state();
      std::uint64_t midstate[8];
      mom_equihash::hash_header_midstate<Spec>(header, midstate);
      alignas(std::uint64_t) std::uint8_t input[input_storage_bytes]{};
      std::memcpy(input, header, Spec::header_length);
      std::memcpy(input + midstate_offset, midstate, sizeof(midstate));
      queue_.memcpy(input_storage_, input, sizeof(input));

      std::array<sycl::event, 6> stages;
      stages[0] = mom_equihash::submit_generation_bucketed<Spec, InitialLayout>(
          queue_, device_midstate_, mom_equihash::load64_le(header + 128),
          mom_equihash::load32_le(header + 136),
          level0_heads_, level0_tails_,
          bucket_counts_[0], overflow_counts_);
      stages[1] = submit_first_round();
      stages[2] = submit_second_round();
      stages[3] = submit_third_round();
      stages[4] = submit_fourth_round();

      stages[5] = mom_equihash::submit_zero_root_round_bucketed<Spec, LaterLayout>(
          queue_, round4_, bucket_counts_[4], candidate_roots_, zero_root_count_,
          Layout::max_candidate_roots);
      if (options_.collect_round_counts) {
        mom_equihash::submit_sum_bucket_counts<InitialLayout>(
            queue_, bucket_counts_[1], flat_counts_);
        for (unsigned round = 1; round < 4; ++round)
          mom_equihash::submit_sum_bucket_counts<LaterLayout>(
              queue_, bucket_counts_[round + 1], flat_counts_ + round);
      }

      mom_equihash::submit_recover_root_leaves<Spec, InitialLayout::local_bits, Round1Record,
                                               Round2Record>(
          queue_, level0_heads_, level0_tails_, round1_heads_, round1_tails_, round2_, round3_,
          round4_, candidate_roots_, zero_root_count_, recovered_leaves_, recovered_valid_, 0,
          fast_candidate_roots,
          static_cast<std::uint32_t>(InitialLayout::slot_count),
          static_cast<std::uint32_t>(LaterLayout::slot_count));
      const sycl::event verified = mom_equihash::submit_verify_recovered_leaves<Spec>(
          queue_, device_header_, device_midstate_, zero_root_count_, recovered_leaves_,
          verified_fields_, recovered_valid_, 0, fast_candidate_roots);
      // A D2H submission behind live kernels can itself busy-spin. Complete verification first so
      // the low-CPU result-copy wait starts after submission has become nonblocking.
      sycl_wait_and_throw(verified, queue_.get_device());

      std::array<std::uint32_t, fast_candidate_roots * Spec::proof_indices> fast_leaves{};
      std::array<std::uint8_t, fast_candidate_roots> fast_valid{};
      std::array<std::uint32_t, 6> result_counts{};
      if (options_.collect_round_counts)
        queue_.memcpy(report.flat_counts.data(), flat_counts_, sizeof(report.flat_counts));
      queue_.memcpy(result_counts.data(), result_counts_storage_, sizeof(result_counts));
      queue_.memcpy(fast_leaves.data(), recovered_leaves_, sizeof(fast_leaves));
      sycl_wait_and_throw(queue_.memcpy(fast_valid.data(), recovered_valid_, sizeof(fast_valid)),
                          queue_.get_device());
      std::copy_n(result_counts.begin(), 5, report.overflow_counts.begin());
      report.zero_root_count = result_counts[5];
      if (options_.collect_stage_times)
        for (unsigned i = 0; i < stages.size(); ++i) {
          const auto start = stages[i].get_profiling_info<
              sycl::info::event_profiling::command_start>();
          const auto end = stages[i].get_profiling_info<
              sycl::info::event_profiling::command_end>();
          report.stage_ms[i] = static_cast<double>(end - start) / 1.0e6;
        }

      if (!lifecycle_.generated(report.overflow_counts[0]))
        return capacity_report(report, "level-zero bucket arena overflow");
      for (unsigned round = 0; round < 4; ++round) {
        if (!lifecycle_.flat_output(report.flat_counts[round]))
          return capacity_report(report, "flat collision output exceeded arena capacity");
        if (!lifecycle_.scattered(report.overflow_counts[round + 1]))
          return capacity_report(report, "collision bucket arena overflow");
      }
      report.root_count = report.zero_root_count;
      report.flat_counts[4] = report.root_count;
      if (!lifecycle_.flat_output(report.root_count) ||
          !lifecycle_.roots(report.root_count, report.zero_root_count))
        return capacity_report(report, "root output exceeded arena capacity");
      if (report.zero_root_count > Layout::max_candidate_roots)
        return capacity_report(report, "zero-root candidate capacity exceeded");

      if (report.zero_root_count <= fast_candidate_roots) {
        recover_roots(report, fast_leaves.data(), fast_valid.data());
      } else {
        mom_equihash::submit_recover_root_leaves<Spec, InitialLayout::local_bits, Round1Record,
                                                 Round2Record>(
            queue_, level0_heads_, level0_tails_, round1_heads_, round1_tails_, round2_, round3_,
            round4_, candidate_roots_, zero_root_count_, recovered_leaves_, recovered_valid_,
            fast_candidate_roots, report.zero_root_count - fast_candidate_roots,
            static_cast<std::uint32_t>(InitialLayout::slot_count),
            static_cast<std::uint32_t>(LaterLayout::slot_count));
        const sycl::event tail_verified = mom_equihash::submit_verify_recovered_leaves<Spec>(
            queue_, device_header_, device_midstate_, zero_root_count_, recovered_leaves_,
            verified_fields_, recovered_valid_, fast_candidate_roots,
            report.zero_root_count - fast_candidate_roots);
        sycl_wait_and_throw(tail_verified, queue_.get_device());
        std::vector<std::uint32_t> leaves(
            static_cast<std::size_t>(report.zero_root_count) * Spec::proof_indices);
        std::vector<std::uint8_t> valid(report.zero_root_count);
        std::copy(fast_leaves.begin(), fast_leaves.end(), leaves.begin());
        std::copy(fast_valid.begin(), fast_valid.end(), valid.begin());
        const std::size_t tail = report.zero_root_count - fast_candidate_roots;
        queue_.memcpy(leaves.data() + fast_candidate_roots * Spec::proof_indices,
                      recovered_leaves_ + fast_candidate_roots * Spec::proof_indices,
                      tail * Spec::proof_indices * sizeof(leaves[0]));
        sycl_wait_and_throw(
            queue_.memcpy(valid.data() + fast_candidate_roots,
                          recovered_valid_ + fast_candidate_roots, tail),
            queue_.get_device());
        recover_roots(report, leaves.data(), valid.data());
      }
      lifecycle_.finish(!report.solutions.empty());
      report.status = lifecycle_.status();
      return report;
    } catch (const sycl::exception& error) {
      return device_error_report(report, error.what());
    } catch (const std::exception& error) {
      return device_error_report(report, error.what());
    }
  }

private:
  static sycl::queue make_queue(const sycl::queue& source, const bool profile) {
    if (profile)
      return sycl::queue(source.get_context(), source.get_device(),
                         sycl::property_list{sycl::property::queue::in_order{},
                                             sycl::property::queue::enable_profiling{}});
    return sycl::queue(source.get_context(), source.get_device(),
                       sycl::property_list{sycl::property::queue::in_order{}});
  }

  sycl::queue queue_;
  SessionOptions options_;
  ZHashSessionLifecycle lifecycle_;
  std::string last_error_;

  std::uint8_t* device_header_ = nullptr;
  std::uint64_t* device_midstate_ = nullptr;
  std::uint8_t* input_storage_ = nullptr;
  mom_equihash::SplitRecordHead* level0_heads_ = nullptr;
  std::uint32_t* level0_tails_ = nullptr;
  mom_equihash::SplitRecordHead* round1_heads_ = nullptr;
  std::uint32_t* round1_tails_ = nullptr;
  Round2Record* round2_ = nullptr;
  Round3Record* round3_ = nullptr;
  Round4Record* round4_ = nullptr;
  std::array<std::uint32_t*, 5> bucket_counts_{};
  std::uint32_t* flat_counts_ = nullptr;
  std::uint32_t* overflow_counts_ = nullptr;
  std::uint32_t* zero_root_count_ = nullptr;
  RootRecord* candidate_roots_ = nullptr;
  std::uint32_t* recovered_leaves_ = nullptr;
  std::uint8_t* recovered_valid_ = nullptr;
  std::uint32_t* verified_fields_ = nullptr;
  std::uint32_t* counts_storage_ = nullptr;
  std::uint32_t* bucket_counts_storage_ = nullptr;
  std::uint32_t* result_counts_storage_ = nullptr;

  static constexpr std::size_t midstate_offset = (Spec::header_length + 7u) & ~std::size_t{7};
  static constexpr std::size_t input_storage_bytes = midstate_offset + 8u * sizeof(std::uint64_t);
  static constexpr std::size_t bucket_count_size =
      2u * InitialLayout::bucket_count + 3u * LaterLayout::bucket_count;
#ifdef MOM_SYCL_HAS_HIP
  static constexpr std::size_t required_cache_local_bytes = std::max({
      mom_equihash::partitioned_split_collision_local_bytes<InitialLayout, 8, true>,
      mom_equihash::partitioned_split_collision_local_bytes<InitialLayout, 4, true>,
      mom_equihash::partitioned_collision_local_bytes<LaterLayout, Round2Record, 4, true>,
      mom_equihash::partitioned_collision_local_bytes<LaterLayout, Round3Record, 4, true>});
  bool cache_partition_inputs_ = false;
#endif

  template <typename T>
  T* allocate_objects(const std::size_t count) {
    T* const result = sycl::malloc_device<T>(count, queue_);
    if (result == nullptr) throw std::bad_alloc();
    return result;
  }

  void allocate() {
    const sycl::device device = queue_.get_device();
    if (!mom_has_usm_device(device))
      throw std::runtime_error("ZHash session requires device USM");
    const std::uint64_t global_bytes =
        device.get_info<sycl::info::device::global_mem_size>();
    const std::uint64_t max_alloc =
        device.get_info<sycl::info::device::max_mem_alloc_size>();
    const std::uint64_t limit = options_.max_device_bytes != 0
                                    ? options_.max_device_bytes
                                    : global_bytes -
                                          (global_bytes > options_.safety_reserve_bytes
                                               ? options_.safety_reserve_bytes
                                               : global_bytes);
    if (Layout::device_bytes > limit)
      throw std::runtime_error("ZHash arena plan exceeds the device memory budget");
    if (!Layout::holds_logical_rows())
      throw std::runtime_error("ZHash session layout does not hold the complete logical row space");
    if (Layout::level0_bytes > max_alloc || Layout::round1_bytes > max_alloc ||
        Layout::round2_bytes > max_alloc || Layout::round3_bytes > max_alloc ||
        Layout::round4_bytes > max_alloc || Layout::candidate_root_bytes > max_alloc)
      throw std::runtime_error("ZHash arena plan exceeds the device allocation limit");
    if (Layout::collision_work_group >
            device.get_info<sycl::info::device::max_work_group_size>() ||
        Layout::local_memory_bytes > device.get_info<sycl::info::device::local_mem_size>())
      throw std::runtime_error("ZHash bucket layout exceeds the device work-group/local-memory limits");
    if (InitialLayout::slot_count >= std::numeric_limits<std::uint32_t>::max() ||
        LaterLayout::slot_count >= std::numeric_limits<std::uint32_t>::max())
      throw std::runtime_error("ZHash arena slot count does not fit kernel counters");

    try {
      input_storage_ = allocate_objects<std::uint8_t>(input_storage_bytes);
      device_header_ = input_storage_;
      device_midstate_ = reinterpret_cast<std::uint64_t*>(input_storage_ + midstate_offset);
      level0_heads_ = allocate_objects<mom_equihash::SplitRecordHead>(InitialLayout::slot_count);
      level0_tails_ = allocate_objects<std::uint32_t>(InitialLayout::slot_count);
      round1_heads_ = allocate_objects<mom_equihash::SplitRecordHead>(InitialLayout::slot_count);
      round1_tails_ = allocate_objects<std::uint32_t>(InitialLayout::slot_count);
      round2_ = allocate_objects<Round2Record>(LaterLayout::slot_count);
      round3_ = allocate_objects<Round3Record>(LaterLayout::slot_count);
      round4_ = allocate_objects<Round4Record>(LaterLayout::slot_count);
      counts_storage_ = allocate_objects<std::uint32_t>(bucket_count_size + 6u);
      bucket_counts_storage_ = counts_storage_;
      bucket_counts_[0] = bucket_counts_storage_;
      bucket_counts_[1] = bucket_counts_[0] + InitialLayout::bucket_count;
      bucket_counts_[2] = bucket_counts_[1] + InitialLayout::bucket_count;
      for (unsigned round = 3; round < 5; ++round)
        bucket_counts_[round] = bucket_counts_[round - 1] + LaterLayout::bucket_count;
      flat_counts_ = allocate_objects<std::uint32_t>(5);
      result_counts_storage_ = counts_storage_ + bucket_count_size;
      overflow_counts_ = result_counts_storage_;
      zero_root_count_ = result_counts_storage_ + 5;
      candidate_roots_ = allocate_objects<RootRecord>(Layout::max_candidate_roots);
      recovered_leaves_ = allocate_objects<std::uint32_t>(
          Layout::max_candidate_roots * Spec::proof_indices);
      recovered_valid_ = allocate_objects<std::uint8_t>(Layout::max_candidate_roots);
      verified_fields_ = allocate_objects<std::uint32_t>(
          Layout::max_candidate_roots * Spec::proof_indices * Spec::rounds);
    } catch (...) {
      free_all();
      throw;
    }
  }

  template <typename T>
  void free_ptr(T*& pointer) noexcept {
    if (pointer != nullptr) {
      try {
        sycl::free(pointer, queue_);
      } catch (...) {
      }
      pointer = nullptr;
    }
  }

  void free_all() noexcept {
    free_ptr(input_storage_);
    device_header_ = nullptr;
    device_midstate_ = nullptr;
    free_ptr(level0_heads_);
    free_ptr(level0_tails_);
    free_ptr(round1_heads_);
    free_ptr(round1_tails_);
    free_ptr(round2_);
    free_ptr(round3_);
    free_ptr(round4_);
    free_ptr(counts_storage_);
    bucket_counts_storage_ = result_counts_storage_ = nullptr;
    bucket_counts_.fill(nullptr);
    free_ptr(flat_counts_);
    overflow_counts_ = zero_root_count_ = nullptr;
    free_ptr(candidate_roots_);
    free_ptr(recovered_leaves_);
    free_ptr(recovered_valid_);
    free_ptr(verified_fields_);
  }

  void clear_device_state() {
    if (options_.collect_round_counts) {
      queue_.fill(flat_counts_, 0u, 5);
    }
    queue_.fill(counts_storage_, 0u, bucket_count_size + 6u);
  }

  sycl::event submit_first_round() {
#ifdef MOM_SYCL_HAS_HIP
    if (cache_partition_inputs_)
      return mom_equihash::submit_bucket_round_bucketed_partitioned<
          Spec, InitialLayout, InitialLayout, Round1Record, 8, true>(
          queue_, level0_heads_, level0_tails_, bucket_counts_[0], round1_heads_, round1_tails_,
          bucket_counts_[1], overflow_counts_ + 1);
    return mom_equihash::submit_bucket_round_bucketed_partitioned<
        Spec, InitialLayout, InitialLayout, Round1Record, 8>(
        queue_, level0_heads_, level0_tails_, bucket_counts_[0], round1_heads_, round1_tails_,
        bucket_counts_[1], overflow_counts_ + 1);
#else
    return mom_equihash::submit_bucket_round_bucketed<Spec, InitialLayout, InitialLayout,
                                                       Round1Record>(
        queue_, level0_heads_, level0_tails_, bucket_counts_[0], round1_heads_, round1_tails_,
        bucket_counts_[1], overflow_counts_ + 1);
#endif
  }

  sycl::event submit_second_round() {
#ifdef MOM_SYCL_HAS_HIP
    if (cache_partition_inputs_)
      return mom_equihash::submit_split_collision_round_bucketed_partitioned<
          Spec, 5, InitialLayout::local_bits, InitialLayout, LaterLayout, Round1Record,
          Round2Record, 4, true>(queue_, round1_heads_, round1_tails_, bucket_counts_[1], round2_,
                                bucket_counts_[2], overflow_counts_ + 2);
    return mom_equihash::submit_split_collision_round_bucketed_partitioned<
        Spec, 5, InitialLayout::local_bits, InitialLayout, LaterLayout, Round1Record,
        Round2Record, 4>(queue_, round1_heads_, round1_tails_, bucket_counts_[1], round2_,
                         bucket_counts_[2], overflow_counts_ + 2);
#else
    return mom_equihash::submit_split_collision_round_bucketed<
        Spec, 5, InitialLayout::local_bits, InitialLayout, LaterLayout, Round1Record,
        Round2Record>(queue_, round1_heads_, round1_tails_, bucket_counts_[1], round2_,
                      bucket_counts_[2], overflow_counts_ + 2);
#endif
  }

  sycl::event submit_third_round() {
#ifdef MOM_SYCL_HAS_HIP
    if (cache_partition_inputs_)
      return mom_equihash::submit_collision_round_bucketed_partitioned<
          Spec, 4, LaterLayout, LaterLayout, Round2Record, Round3Record, 4, true>(
          queue_, round2_, bucket_counts_[2], round3_, bucket_counts_[3], overflow_counts_ + 3);
    return mom_equihash::submit_collision_round_bucketed_partitioned<
        Spec, 4, LaterLayout, LaterLayout, Round2Record, Round3Record, 4>(
        queue_, round2_, bucket_counts_[2], round3_, bucket_counts_[3], overflow_counts_ + 3);
#else
    return mom_equihash::submit_collision_round_bucketed<Spec, 4, LaterLayout, LaterLayout>(
        queue_, round2_, bucket_counts_[2], round3_, bucket_counts_[3], overflow_counts_ + 3);
#endif
  }

  sycl::event submit_fourth_round() {
#ifdef MOM_SYCL_HAS_HIP
    if (cache_partition_inputs_)
      return mom_equihash::submit_collision_round_bucketed_partitioned<
          Spec, 3, LaterLayout, LaterLayout, Round3Record, Round4Record, 4, true>(
          queue_, round3_, bucket_counts_[3], round4_, bucket_counts_[4], overflow_counts_ + 4);
    return mom_equihash::submit_collision_round_bucketed_partitioned<
        Spec, 3, LaterLayout, LaterLayout, Round3Record, Round4Record, 4>(
        queue_, round3_, bucket_counts_[3], round4_, bucket_counts_[4], overflow_counts_ + 4);
#else
    return mom_equihash::submit_collision_round_bucketed<Spec, 3, LaterLayout, LaterLayout>(
        queue_, round3_, bucket_counts_[3], round4_, bucket_counts_[4], overflow_counts_ + 4);
#endif
  }

  RunReport& capacity_report(RunReport& report, const char* const message) {
    report.status = SessionStatus::capacity_overflow;
    report.error = message;
    last_error_ = report.error;
    return report;
  }

  RunReport& device_error_report(RunReport& report, const char* const message) {
    lifecycle_.device_error();
    report.status = SessionStatus::device_error;
    report.error = message;
    last_error_ = report.error;
    return report;
  }

  void recover_roots(RunReport& report, const std::uint32_t* leaves,
                     const std::uint8_t* valid) {
    for (std::size_t candidate = 0; candidate < report.zero_root_count; ++candidate) {
      Solution solution{};
      solution.root_slot = static_cast<std::uint32_t>(candidate);
      for (unsigned i = 0; i < Spec::proof_indices; ++i)
        solution.indices[i] = leaves[candidate * Spec::proof_indices + i];
      mom_equihash::encode_indices<Spec>(solution.indices.data(), solution.encoded.data());
      if (!valid[candidate]) {
        ++report.rejected_root_count;
        continue;
      }
      bool duplicate = false;
      for (const Solution& previous : report.solutions)
        if (std::memcmp(previous.encoded.data(), solution.encoded.data(), solution.encoded.size()) == 0) {
          duplicate = true;
          break;
        }
      if (!duplicate) {
        report.solutions.push_back(solution);
      }
    }
  }
};

}  // namespace mom_zhash
