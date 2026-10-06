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

#include "job/execution.inc"

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
    keys2set(gpu_cn_algo2fn), keys2set(gpu_c29_algo2fn), keys2set(gpu_kawpow_algo2fn),
    keys2set(gpu_etchash_algo2fn), keys2set(gpu_autolykos2_algo2fn),
    keys2set(gpu_pearlhash_algo2fn), keys2set(gpu_fishhash_algo2fn),
    keys2set(gpu_karlsenhashv2_algo2fn), keys2set(gpu_misc_algo2fn),
    keys2set(gpu_verthash_algo2fn), keys2set(gpu_zelhash_algo2fn),
    keys2set(gpu_beamhash3_algo2fn)
  ));
}
