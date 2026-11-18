// Copyright GNU GPLv3 (c) 2023-2026 MoneroOcean <support@moneroocean.stream>

#include "core.h"
#include "cpu-scheduling.h"
#include "../sycl/lib.h"

#include "backend/cpu/Cpu.h"
#include "crypto/cn/CnCtx.h"
#include "crypto/cn/CryptoNight.h"
#include "crypto/ghostrider/ghostrider.h"
#include "crypto/randomx/configuration.h"
#include "crypto/randomx/aes_hash.hpp"
#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <ranges>
#include <set>
#include <thread>

const constexpr unsigned MAX_BLOB_LEN    = 512;
#include "job/algorithms.inc"

#include "job/setup.inc"

// Publish the first GPU job's SYCL program build options from the loop thread before the compute
// thread starts; compute threads never write the environment (other threads read it concurrently).
// DPC++ applies the options at the process's first program build, so later jobs cannot change them.
void Core::prepare(const Message& first) {
  if (first.name != "job" && first.name != "bench" && first.name != "test")
    return;
  const auto algo = first.values.find("algo");
  const auto dev = first.values.find("dev");
  if (algo == first.values.end() || dev == first.values.end() || !gpu_algos.contains(algo->second))
    return;
  try {
    sycl_set_compile_options(algo->second, dev->second.substr(0, dev->second.find('*')));
  } catch (...) {
    // The job itself reports an unusable device or missing GPU runtime.
  }
}

// Answer "algo_params": every supported algorithm with its default devices and tuning for this
// host, from the controller's CPU topology plus SYCL GPU discovery (skipped for CPU-only runs).
void Core::get_algo_params(const MessageValues& v) {
  if (!v.contains("cpu_sockets")) throw std::string("Missing cpu_sockets algo_params key");
  if (!v.contains("cpu_threads")) throw std::string("Missing cpu_threads algo_params key");
  if (!v.contains("cpu_l3cache")) throw std::string("Missing cpu_l3cache algo_params key");
  unsigned cpu_sockets = 0;
  unsigned cpu_threads = 0;
  unsigned cpu_l3cache = 0;
  if (!mom::job_boundary::parse_unsigned(v.at("cpu_sockets"), cpu_sockets) ||
      !mom::job_boundary::parse_unsigned(v.at("cpu_threads"), cpu_threads) ||
      !mom::job_boundary::parse_unsigned(v.at("cpu_l3cache"), cpu_l3cache)) {
    throw std::string("Invalid unsigned algo_params topology");
  }
  const auto keys2set = [](const auto& map) {
    const auto keys = std::views::keys(map);
    return std::set<std::string>(keys.begin(), keys.end());
  };
  // SYCL/GPU algo params can be skipped (e.g. for CPU-only builds/tests)
  const bool skip_sycl = std::getenv("MOM_SKIP_SYCL_ALGO_PARAMS");
  if (skip_sycl) {
    send_msg("algo_params", cpu_only_algo_params(
      MAX_CN_CPU_WAYS, cpu_sockets, cpu_threads, cpu_l3cache, keys2set(cpu_name2algo)
    ));
    return;
  }
  // algo_params returns std::map<std::string,std::string>, which is exactly MessageValues
  send_msg("algo_params", algo_params(
    MAX_CN_CPU_WAYS, cpu_sockets, cpu_threads, cpu_l3cache, algo2mem, keys2set(cpu_name2algo),
    gpu_algos_with(&FN::gpu_cn), gpu_algos_with(&FN::gpu_c29), gpu_algos_with(&FN::gpu_kawpow),
    gpu_algos_with(&FN::gpu_etchash), gpu_algos_with(&FN::gpu_autolykos2),
    gpu_algos_with(&FN::gpu_pearlhash), gpu_algos_with(&FN::gpu_fishhash),
    gpu_algos_with(&FN::gpu_karlsenhashv2), gpu_algos_with(&FN::gpu_misc),
    gpu_algos_with(&FN::gpu_verthash), gpu_algos_with(&FN::gpu_zelhash),
    gpu_algos_with(&FN::gpu_beamhash3)
  ));
}
