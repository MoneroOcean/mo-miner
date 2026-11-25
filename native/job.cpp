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
#include <initializer_list>
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
  if (algo == first.values.end() || dev == first.values.end() || !gpu_algos.contains(algo->second) ||
      sycl_compile_options_for(algo->second) == SyclCompileOptions::none)
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
  const auto topology = [&](const char* const key) {
    const auto it = v.find(key);
    if (it == v.end()) throw std::string("Missing ") + key + " algo_params key";
    unsigned value = 0;
    if (!mom::job_boundary::parse_unsigned(it->second, value))
      throw std::string("Invalid unsigned algo_params topology");
    return value;
  };
  const unsigned cpu_sockets = topology("cpu_sockets");
  const unsigned cpu_threads = topology("cpu_threads");
  const unsigned cpu_l3cache = topology("cpu_l3cache");
  const auto keys2set = [](const auto& map) {
    const auto keys = std::views::keys(map);
    return std::set<std::string>(keys.begin(), keys.end());
  };
  // SYCL/GPU algo params can be skipped (e.g. for CPU-only builds/tests)
  const bool skip_sycl = mom_native_env_enabled("MOM_SKIP_SYCL_ALGO_PARAMS");
  if (skip_sycl) {
    send_msg("algo_params", cpu_only_algo_params(
      MAX_CN_CPU_WAYS, cpu_sockets, cpu_threads, cpu_l3cache, keys2set(cpu_name2algo)
    ));
    return;
  }
  // algo_params returns std::map<std::string,std::string>, which is exactly MessageValues
  send_msg("algo_params", algo_params(
    MAX_CN_CPU_WAYS, cpu_sockets, cpu_threads, cpu_l3cache, algo2mem, keys2set(cpu_name2algo),
    gpu_algos_with({DEV::CN_GPU}), gpu_algos_with({DEV::C29_GPU, DEV::C30_GPU}),
    gpu_algos_with({DEV::PROGPOW_GPU}), gpu_algos_with({DEV::ETHASH_FAMILY_GPU}),
    gpu_algos_with({DEV::AUTOLYKOS2_GPU}), gpu_algos_with({DEV::PEARLHASH_GPU}),
    gpu_algos_with({DEV::FISHHASH_GPU}), gpu_algos_with({DEV::KARLSENHASHV2_GPU}),
    gpu_algos_with({DEV::MISC_GPU}), gpu_algos_with({DEV::VERTHASH_GPU}),
    gpu_algos_with({DEV::ZCASH_EQUIHASH_GPU}), gpu_algos_with({DEV::BEAMHASH3_GPU})
  ));
}
