// Copyright GNU GPLv3 (c) 2023-2025 MoneroOcean <support@moneroocean.stream>

#pragma once

#include <cstdint>
#include <map>
#include <set>
#include <string>

#if defined(_WIN32)
#if defined(MOMINER_SYCL_BUILD)
#define MOMINER_SYCL_API __declspec(dllexport)
#else
#define MOMINER_SYCL_API __declspec(dllimport)
#endif
#else
#define MOMINER_SYCL_API
#endif

MOMINER_SYCL_API std::map<std::string, std::string> algo_params(
  unsigned max_cpu_batch, unsigned cpu_sockets, unsigned cpu_threads, unsigned cpu_l3cache,
  const std::map<std::string, unsigned>& algo2mem,
  const std::set<std::string>& cpu_algos,
  const std::set<std::string>& gpu_cn_algos,
  const std::set<std::string>& gpu_c29_algos
);

MOMINER_SYCL_API void cn_gpu(
  const uint8_t* inputs, unsigned input_size, uint8_t* output,
  void* Spads, unsigned batch, const std::string& dev_str
);

MOMINER_SYCL_API int c29(
  unsigned job_id, unsigned c29_proof_size,
  const uint8_t* inputs, unsigned input_size, uint8_t* output,
  uint32_t* output_edges, uint64_t* pnonce, const std::string& dev_str
);
