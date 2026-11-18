// Copyright GNU GPLv3 (c) 2023-2026 MoneroOcean <support@moneroocean.stream>

#pragma once

#include <algorithm>
#include <list>
#include <string>

namespace mom::cpu {

// Append one entry to a comma-separated device list.
inline void add_result_dev(std::string& result_dev, const std::string& add_str) {
  if (!result_dev.empty()) {
    result_dev += ",";
  }
  result_dev += add_str;
}

// Plan CPU worker processes for one algorithm: the batch (hashes per process) of each process,
// filling the L3 cache with batch_mem bytes per hash. RandomX runs one process per socket that
// shares its dataset; Argon2 keeps single-hash processes; GhostRider always hashes 8 at a time.
inline std::list<unsigned> cpu_thread_batches(
  const std::string& algo, const unsigned max_cpu_batch, const unsigned socket_count,
  const unsigned thread_count, const unsigned l3cache, const unsigned batch_mem
) {
  unsigned used_l3cache = 0, used_threads = 0;
  std::list<unsigned> threads;
  if (algo.starts_with("rx/")) {
    // One process per socket shares its RandomX dataset across batch threads.
    const unsigned batch = std::max(1u, std::min(thread_count, l3cache / batch_mem) / socket_count);
    for (unsigned i = 0; i != socket_count; ++i) {
      threads.push_back(batch);
    }
    return threads;
  }

  // fill threads list with single batch
  while (++used_threads <= thread_count && (used_l3cache += batch_mem) <= l3cache) {
    threads.push_back(algo == "ghostrider" ? 8 : 1);
  }
  if (!algo.starts_with("argon2/")) {
    // increase batch size until we hit L3 cache limit
    while (used_l3cache < l3cache) {
      bool updated = false;
      for (auto& i : threads) {
        if (i < max_cpu_batch && (used_l3cache += batch_mem) <= l3cache) {
          ++i;
          updated = true;
        }
      }
      if (!updated)
        break; // in case we hit all max_cpu_batch and not L3 cache
    }
  }
  if (threads.empty()) {
    threads.push_back(1);
  }
  return threads;
}

// Append the planned processes as "cpu[*batch][^count]" entries, merging runs of equal batches.
inline void append_grouped_cpu_devs(
  std::string& result_dev, const std::list<unsigned>& threads
) {
  unsigned prev_batch = 0, same_batch_threads = 0;
  auto add_last_dev = [&]() {
    if (!same_batch_threads || !prev_batch) {
      return;
    }
    add_result_dev(result_dev, "cpu" + (prev_batch != 1 ? "*" + std::to_string(prev_batch) : ""));
    if (same_batch_threads != 1) {
      result_dev += "^" + std::to_string(same_batch_threads);
    }
    same_batch_threads = 0;
  };
  for (const unsigned batch : threads) {
    if (same_batch_threads && prev_batch != batch) {
      add_last_dev();
    }
    prev_batch = batch;
    ++same_batch_threads;
  }
  add_last_dev();
}

// Controller-reported topology with scheduling defaults: at least one socket and thread, and 2 MiB
// of L3 per thread when the platform exposes none, so every thread still gets a worker.
struct CpuTopology {
  unsigned sockets;
  unsigned threads;
  unsigned l3cache;
};

inline CpuTopology normalize_cpu_topology(const unsigned sockets, const unsigned threads,
                                          const unsigned l3cache) {
  const unsigned thread_count = std::max(1u, threads);
  return {std::max(1u, sockets), thread_count,
          l3cache ? l3cache : thread_count * 2u * 1024u * 1024u};
}

// CPU device list for one algorithm that needs batch_mem bytes of L3 cache per hash.
inline std::string cpu_algo_devs(const std::string& algo, const unsigned max_cpu_batch,
                                 const CpuTopology& cpu, const unsigned batch_mem) {
  std::string result;
  append_grouped_cpu_devs(result, cpu_thread_batches(
    algo, max_cpu_batch, cpu.sockets, cpu.threads, cpu.l3cache, batch_mem));
  return result;
}

} // namespace mom::cpu
