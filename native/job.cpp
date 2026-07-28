// Copyright GNU GPLv3 (c) 2023-2025 MoneroOcean <support@moneroocean.stream>

#include "core.h"
#include "../sycl/lib.h"

#include "backend/cpu/Cpu.h"
#include "crypto/cn/CnCtx.h"
#include "crypto/cn/CryptoNight.h"
#include "crypto/ghostrider/ghostrider.h"
#include "crypto/randomx/configuration.h"
#include "crypto/randomx/aes_hash.hpp"
#include "base/tools/bswap_64.h"

#include <algorithm>
#include <ranges>
#include <list>
#include <set>
#include <thread>
#include <cstdlib>
#include <cstring>

const constexpr unsigned MAX_BLOB_LEN    = 512;
#include "job/algorithms.inc"

#include "job/execution.inc"

void Core::get_algo_params(const MessageValues& v) {
  if (!v.contains("cpu_sockets")) throw std::string("Missing cpu_sockets algo_params key");
  if (!v.contains("cpu_threads")) throw std::string("Missing cpu_threads algo_params key");
  if (!v.contains("cpu_l3cache")) throw std::string("Missing cpu_l3cache algo_params key");
  const unsigned cpu_sockets = atoi(v.at("cpu_sockets").c_str()),
                 cpu_threads = atoi(v.at("cpu_threads").c_str()),
                 cpu_l3cache = atoi(v.at("cpu_l3cache").c_str());
  const auto keys2set = [](const auto& map) {
    const auto keys = std::views::keys(map);
    return std::set<std::string>(keys.begin(), keys.end());
  };
  // SYCL/GPU algo params can be skipped (e.g. for CPU-only builds/tests)
  const bool skip_sycl = std::getenv("MOM_SKIP_SYCL_ALGO_PARAMS");
  const auto gpu_set = [&](const auto& map) {
    return skip_sycl ? std::set<std::string>{} : keys2set(map);
  };
  if (skip_sycl) {
    send_msg("algo_params", cpu_only_algo_params(
      MAX_CN_CPU_WAYS, cpu_sockets, cpu_threads, cpu_l3cache, keys2set(cpu_name2algo)
    ));
    return;
  }
  // algo_params returns std::map<std::string,std::string>, which is exactly MessageValues
  send_msg("algo_params", algo_params(
    MAX_CN_CPU_WAYS, cpu_sockets, cpu_threads, cpu_l3cache, algo2mem, keys2set(cpu_name2algo),
    gpu_set(gpu_cn_algo2fn), gpu_set(gpu_c29_algo2fn), gpu_set(gpu_kawpow_algo2fn),
    gpu_set(gpu_etchash_algo2fn), gpu_set(gpu_autolykos2_algo2fn), gpu_set(gpu_pearlhash_algo2fn),
    gpu_set(gpu_fishhash_algo2fn), gpu_set(gpu_karlsenhashv2_algo2fn),
    gpu_set(gpu_zelhash_algo2fn), gpu_set(gpu_beamhash3_algo2fn)
  ));
}
