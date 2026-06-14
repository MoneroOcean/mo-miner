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
const constexpr unsigned SPAD_LEN        = 200;
const constexpr unsigned MAX_CN_CPU_WAYS = 5;

static const xmrig::ICpuInfo& cpu_info() { return *xmrig::Cpu::info(); }
#define ci cpu_info()

static const std::map<std::string, xmrig::Algorithm::Id> cpu_name2algo = {
  { "cn/0",            xmrig::Algorithm::CN_0           },
  { "cn/1",            xmrig::Algorithm::CN_1           },
  { "cn/2",            xmrig::Algorithm::CN_2           },
  { "cn/r",            xmrig::Algorithm::CN_R           },
  { "cn/fast",         xmrig::Algorithm::CN_FAST        },
  { "cn/half",         xmrig::Algorithm::CN_HALF        },
  { "cn/xao",          xmrig::Algorithm::CN_XAO         },
  { "cn/rto",          xmrig::Algorithm::CN_RTO         },
  { "cn/rwz",          xmrig::Algorithm::CN_RWZ         },
  { "cn/zls",          xmrig::Algorithm::CN_ZLS         },
  { "cn/double",       xmrig::Algorithm::CN_DOUBLE      },
  { "cn/ccx",          xmrig::Algorithm::CN_CCX         },
  { "cn/upx2",         xmrig::Algorithm::CN_UPX2        },
  { "cn-pico/0",       xmrig::Algorithm::CN_PICO_0      },
  { "cn-pico/tlo",     xmrig::Algorithm::CN_PICO_TLO    },
  { "cn-lite/0",       xmrig::Algorithm::CN_LITE_0      },
  { "cn-lite/1",       xmrig::Algorithm::CN_LITE_1      },
  { "cn-heavy/0",      xmrig::Algorithm::CN_HEAVY_0     },
  { "cn-heavy/xhv",    xmrig::Algorithm::CN_HEAVY_XHV   },
  { "cn-heavy/tube",   xmrig::Algorithm::CN_HEAVY_TUBE  },
  { "ghostrider",      xmrig::Algorithm::GHOSTRIDER_RTM },
  { "argon2/chukwa",   xmrig::Algorithm::AR2_CHUKWA     },
  { "argon2/chukwav2", xmrig::Algorithm::AR2_CHUKWA_V2  },
  { "argon2/wrkz",     xmrig::Algorithm::AR2_WRKZ       },
  { "panthera",        xmrig::Algorithm::RX_XLA         },
  { "rx/0",            xmrig::Algorithm::RX_0           },
  { "rx/2",            xmrig::Algorithm::RX_V2          },
  { "rx/wow",          xmrig::Algorithm::RX_WOW         },
  { "rx/arq",          xmrig::Algorithm::RX_ARQ         },
  { "rx/graft",        xmrig::Algorithm::RX_GRAFT       },
  { "rx/sfx",          xmrig::Algorithm::RX_SFX         },
  { "rx/yada",         xmrig::Algorithm::RX_YADA        },
};

static const std::map<std::string, RandomX_ConfigurationBase*> rx_cpu_name2config = {
  { "panthera",        &RandomX_ScalaConfig   },
  { "rx/0",            &RandomX_MoneroConfig  },
  { "rx/2",            &RandomX_MoneroConfigV2 },
  { "rx/wow",          &RandomX_WowneroConfig },
  { "rx/arq",          &RandomX_ArqmaConfig   },
  { "rx/graft",        &RandomX_GraftConfig   },
  { "rx/sfx",          &RandomX_SafexConfig   },
  { "rx/yada",         &RandomX_YadaConfig    },
};

static const xmrig::CnHash::AlgoVariant cpu_params2variant[MAX_CN_CPU_WAYS][2] = {
  { xmrig::CnHash::AV_SINGLE, xmrig::CnHash::AV_SINGLE_SOFT },
  { xmrig::CnHash::AV_DOUBLE, xmrig::CnHash::AV_DOUBLE_SOFT },
  { xmrig::CnHash::AV_TRIPLE, xmrig::CnHash::AV_TRIPLE_SOFT },
  { xmrig::CnHash::AV_QUAD,   xmrig::CnHash::AV_QUAD_SOFT   },
  { xmrig::CnHash::AV_PENTA,  xmrig::CnHash::AV_PENTA_SOFT  }
};

static const std::map<std::string, gpu_cn_hash_fun> gpu_cn_algo2fn = {
  { "cn/gpu", cn_gpu },
};

static const std::map<std::string, gpu_c29_hash_fun> gpu_c29_algo2fn = {
  { "c29", c29 }
};

static const std::map<std::string, gpu_kawpow_hash_fun> gpu_kawpow_algo2fn = {
  { "kawpow", kawpow }
};

static const std::map<std::string, gpu_etchash_hash_fun> gpu_etchash_algo2fn = {
  { "etchash", etchash }
};

static const std::map<std::string, gpu_autolykos2_hash_fun> gpu_autolykos2_algo2fn = {
  { "autolykos2", autolykos2 }
};

static const std::map<std::string, gpu_pearl_hash_fun> gpu_pearl_algo2fn = {
  { "pearl", pearl }
};

static const std::map<std::string, unsigned> algo2mem = [](){
  std::map<std::string, unsigned> result = {
    { "cn/gpu", 2*1024*1024 }, // host memory is not really used (number used only for algo_params calcs)
    { "c29",    0 },           // host memory is not used even for algo_params calcs
    { "kawpow", 0 },
    { "etchash", 0 },
    { "autolykos2", 0 },
    { "pearl", 0 }
  };
  for (const auto& i : cpu_name2algo) result[i.first] = xmrig::Algorithm(i.second).l3();
  return result;
}();

static xmrig::VirtualMemory* alloc_huge_mem(const unsigned size) {
  xmrig::VirtualMemory* const mem = new xmrig::VirtualMemory(size, true, false, false);
  if (mem->raw()) return mem;
  throw std::string("Can't allocate " + std::to_string(size) + " bytes of memory");
}

static void* alloc_mem(const unsigned size) {
  void* const mem = _mm_malloc(size, 4096);
  if (mem) return mem;
  throw std::string("Can't allocate " + std::to_string(size) + " bytes of memory");
}

void ghostrider(
  const uint8_t* input, const size_t input_size, uint8_t* const output,
  cryptonight_ctx** const ctx, const uint64_t height
) {
  xmrig::ghostrider::hash_octa(input, input_size, output, ctx, nullptr);
}

static void init_rx_dataset_thread(
  randomx_dataset* const dataset, randomx_cache* const cache,
  const unsigned start, const unsigned count
) {
  if (ci.hasAVX2() && (count % 5)) {
    randomx_init_dataset(dataset, cache, start, count - (count % 5));
    randomx_init_dataset(dataset, cache, start + count - 5, 5);
  } else randomx_init_dataset(dataset, cache, start, count);
}

static randomx_flags get_rx_vm_flags(
  const std::string& algo, const bool is_rx_jit, const randomx_dataset* const m_rx_dataset,
  const xmrig::VirtualMemory* const m_rx_dataset_mem
) {
  unsigned rx_flags = RANDOMX_FLAG_DEFAULT;
  if (algo != "panthera" && m_rx_dataset_mem->isHugePages()) rx_flags |= RANDOMX_FLAG_LARGE_PAGES;
  if (ci.hasAES()) rx_flags |= RANDOMX_FLAG_HARD_AES;
  if (m_rx_dataset) rx_flags |= RANDOMX_FLAG_FULL_MEM;
  if (is_rx_jit) rx_flags |= RANDOMX_FLAG_JIT;
  const auto assembly = ci.assembly();
  if (assembly == xmrig::Assembly::RYZEN || assembly == xmrig::Assembly::BULLDOZER)
    rx_flags |= RANDOMX_FLAG_AMD;
  return static_cast<randomx_flags>(rx_flags);
}

void Core::set_job(
  const bool is_set_nonce, const bool is_no_same_input, const MessageValues& v,
  std::function<void(void)> fn_extra_setup
) {
  if (!v.contains("dev"))      throw std::string("Missing dev job key");
  if (!v.contains("algo"))     throw std::string("Missing algo job key");
  if (!v.contains("blob_hex")) throw std::string("Missing blob_hex job key");

  const std::string new_dev_str        = v.at("dev"),
                    new_algo_str       = v.at("algo"),
                    new_input_hex      = v.at("blob_hex"),
                    new_seed_hex       = v.contains("seed_hex") ? v.at("seed_hex") : std::string();
  const unsigned    new_height         = v.contains("height") ? atoi(v.at("height").c_str()) : 0,
                    new_thread_id      = v.contains("thread_id") ?
                                         atoi(v.at("thread_id").c_str()) : 0,
                    new_thread_num     = v.contains("thread_num") ?
                                         atoi(v.at("thread_num").c_str()) : 1,
                    new_nonce_bytes    = v.contains("noncebytes") ?
                                         atoi(v.at("noncebytes").c_str()) : 4,
		    new_nonce_offset   = v.contains("nonceoffset") ?
                                         atoi(v.at("nonceoffset").c_str()) : 39,
		    new_c29_proof_size = v.contains("proofsize") ?
                                         atoi(v.at("proofsize").c_str()) : 32;
  const uint64_t    new_nonce          = v.contains("nonce") ? strtoull(v.at("nonce").c_str(), NULL, 16) : 0,
                    new_nicehash_mask  = v.contains("nicehash_mask") ? strtoull(v.at("nicehash_mask").c_str(), NULL, 16) : 0;

  auto batch_parts = tokenize(new_dev_str, '*');
  if (batch_parts.size() == 0 || batch_parts.size() > 2)
    throw std::string("Invalid dev specification");
  const std::string new_dev_str2 = batch_parts[0];
  const unsigned new_batch = batch_parts.size() == 2 ? atoi(batch_parts[1].c_str()) : 1;
  const DEV new_dev =
    new_dev_str2 == "cpu" ? (rx_cpu_name2config.contains(new_algo_str) ? DEV::RX_CPU : DEV::CPU) :
    new_algo_str == "kawpow" ? DEV::KAWPOW_GPU :
    new_algo_str == "etchash" ? DEV::ETCHASH_GPU :
    new_algo_str == "autolykos2" ? DEV::AUTOLYKOS2_GPU :
    new_algo_str == "pearl" ? DEV::PEARL_GPU :
    new_algo_str.starts_with("c29") ? DEV::C29_GPU : DEV::GPU;

  if (new_dev == DEV::C29_GPU && new_batch != 1)
    throw std::string("Invalid batch size for c29s algo. Should be 1.");
  if (new_dev == DEV::KAWPOW_GPU && new_batch == 0)
    throw std::string("Invalid kawpow intensity");
  if (new_dev == DEV::ETCHASH_GPU && new_batch == 0)
    throw std::string("Invalid etchash intensity");
  if (new_dev == DEV::AUTOLYKOS2_GPU && new_batch == 0)
    throw std::string("Invalid autolykos2 intensity");
  if (new_dev == DEV::PEARL_GPU && new_batch == 0)
    throw std::string("Invalid pearl intensity");
  if (new_nonce_bytes != 4 && new_nonce_bytes != 8)
    throw std::string("Only support 4 or 8 bytes long nonces");
  if (new_dev == DEV::KAWPOW_GPU && (new_nonce_bytes != 8 || new_nonce_offset != 32))
    throw std::string("KawPow requires an 8-byte nonce at offset 32");
  if (new_dev == DEV::ETCHASH_GPU && (new_nonce_bytes != 8 || new_nonce_offset != 32))
    throw std::string("Etchash requires an 8-byte nonce at offset 32");
  if (new_dev == DEV::AUTOLYKOS2_GPU && (new_nonce_bytes != 8 || new_nonce_offset != 32))
    throw std::string("Autolykos2 requires an 8-byte nonce at offset 32");

  FN new_fn;
  uint8_t new_seed[MAX_BLOB_LEN]{};
  unsigned new_seed_len = HASH_LEN;
  const RandomX_ConfigurationBase* new_rx_config;
  switch (new_dev) {
    case DEV::CPU: {
      const auto pi = cpu_name2algo.find(new_algo_str);
      if (pi == cpu_name2algo.end()) throw std::string("Unsupported algo");
      const auto new_algo = pi->second;
      if (new_algo == xmrig::Algorithm::GHOSTRIDER_RTM) {
        if (new_batch != 8) throw std::string("Bad CPU batch");
        new_fn.cpu = ghostrider;
      } else {
        if (new_batch > MAX_CN_CPU_WAYS) throw std::string("Bad CPU batch");
        new_fn.cpu = xmrig::CnHash::fn(
          new_algo,
          cpu_params2variant[new_batch - 1][ci.hasAES() ? 0 : 1],
          xmrig::Assembly::AUTO
        );
      }
      break;
    }

    case DEV::RX_CPU: {
      if (new_seed_hex.empty()) throw std::string("No seed_hex job key");
      if ((new_seed_hex.size() & 1) || new_seed_hex.size() > MAX_BLOB_LEN * 2)
        throw std::string("Bad seed length");
      new_seed_len = new_seed_hex.size() / 2;
      if (!hex2bin(new_seed_hex.c_str(), new_seed_len, new_seed)) throw std::string("Bad seed hex");
      const auto pi = rx_cpu_name2config.find(new_algo_str);
      if (pi == rx_cpu_name2config.end()) throw std::string("Unsupported algo");
      new_rx_config = pi->second;
      new_fn.any = nullptr; // all work is done in the m_thread_pool
      break;
    }

    case DEV::GPU: {
      const auto pi = gpu_cn_algo2fn.find(new_algo_str);
      if (pi == gpu_cn_algo2fn.end()) throw std::string("Unsupported algo");
      new_fn.gpu_cn = pi->second;
      break;
    }

    case DEV::C29_GPU: {
      const auto pi = gpu_c29_algo2fn.find(new_algo_str);
      if (pi == gpu_c29_algo2fn.end()) throw std::string("Unsupported algo");
      new_fn.gpu_c29 = pi->second;
      break;
    }

    case DEV::KAWPOW_GPU: {
      const auto pi = gpu_kawpow_algo2fn.find(new_algo_str);
      if (pi == gpu_kawpow_algo2fn.end()) throw std::string("Unsupported algo");
      new_fn.gpu_kawpow = pi->second;
      break;
    }

    case DEV::ETCHASH_GPU: {
      if (!new_seed_hex.empty()) {
        if (new_seed_hex.size() != HASH_LEN * 2) throw std::string("Bad seed length");
        if (!hex2bin(new_seed_hex.c_str(), HASH_LEN, new_seed)) throw std::string("Bad seed hex");
      }
      const auto pi = gpu_etchash_algo2fn.find(new_algo_str);
      if (pi == gpu_etchash_algo2fn.end()) throw std::string("Unsupported algo");
      new_fn.gpu_etchash = pi->second;
      break;
    }

    case DEV::AUTOLYKOS2_GPU: {
      const auto pi = gpu_autolykos2_algo2fn.find(new_algo_str);
      if (pi == gpu_autolykos2_algo2fn.end()) throw std::string("Unsupported algo");
      new_fn.gpu_autolykos2 = pi->second;
      break;
    }

    case DEV::PEARL_GPU: {
      const auto pi = gpu_pearl_algo2fn.find(new_algo_str);
      if (pi == gpu_pearl_algo2fn.end()) throw std::string("Unsupported algo");
      new_fn.gpu_pearl = pi->second;
      break;
    }
  }

  uint8_t new_input[MAX_BLOB_LEN];
  const unsigned new_input_len = new_input_hex.size() >> 1;
  if ((new_input_hex.size() & 1) || new_input_len > MAX_BLOB_LEN)
    throw std::string("Bad input length");
  if (!hex2bin(new_input_hex.c_str(), new_input_len, new_input))
    throw std::string("Bad input hex");
  if (new_dev == DEV::KAWPOW_GPU && new_input_len < 40)
    throw std::string("Bad kawpow input length");
  if (new_dev == DEV::ETCHASH_GPU && new_input_len < 40)
    throw std::string("Bad etchash input length");
  if (new_dev == DEV::AUTOLYKOS2_GPU && new_input_len < 40)
    throw std::string("Bad autolykos2 input length");
  if (new_dev == DEV::PEARL_GPU && new_input_len < 76)
    throw std::string("Bad pearl input length");

  const unsigned new_mem_size = algo2mem.at(new_algo_str);
  const bool same_compute_input =
    is_no_same_input && new_input_hex == m_input_hex &&
    m_batch == new_batch && m_mem_size == new_mem_size &&
    m_seed_hex == new_seed_hex && m_algo_str == new_algo_str &&
    m_dev == new_dev && m_dev_str == new_dev_str2 &&
    m_height == new_height && m_nonce_bytes == new_nonce_bytes &&
    m_nonce_offset == new_nonce_offset && m_c29_proof_size == new_c29_proof_size &&
    m_input_len == new_input_len && m_nicehash_mask == new_nicehash_mask;
  if (same_compute_input) {
    // Pools can retarget c29 by sending the same work with a new target/job id.
    // Keep the existing nonce/search state and only refresh submit metadata.
    fn_extra_setup();
    return;
  }

  // new hashing setup (all errors were checked above)
  ++ m_job_ref; // used to stop old m_thread_pool jobs
  if (m_batch != new_batch || m_mem_size != new_mem_size ||
      m_seed_hex != new_seed_hex || m_algo_str != new_algo_str) {
    // free previous memory
    const bool is_ethlike_change = is_small_blob_gpu_dev(m_dev) || is_small_blob_gpu_dev(new_dev);
    free_memory(
      m_batch != new_batch || is_ethlike_change,
      m_mem_size != new_mem_size,
      (m_seed_hex.empty() && !new_seed_hex.empty()) || is_ethlike_change,
      !m_seed_hex.empty() && new_seed_hex.empty()
    );

    if (!is_small_blob_gpu_dev(new_dev) && m_lpads == nullptr)
      m_lpads = alloc_huge_mem(new_batch * new_mem_size);

    if (new_dev == DEV::RX_CPU) {
      randomx_apply_config(*new_rx_config);
      // setup rx cache, dataset and thread_pool
      if (m_rx_cache_mem == nullptr)
        m_rx_cache_mem = alloc_huge_mem(RANDOMX_CACHE_MAX_SIZE);
      if (m_rx_dataset_mem == nullptr)
        m_rx_dataset_mem = alloc_huge_mem(RANDOMX_DATASET_MAX_SIZE);
      if (m_rx_cache == nullptr) {
        m_rx_cache = randomx_create_cache(new_algo_str == "panthera" ? RANDOMX_FLAG_DEFAULT : RANDOMX_FLAG_JIT, m_rx_cache_mem->raw());
        if (m_rx_cache == nullptr) {
          m_is_rx_jit = false;
          m_rx_cache = randomx_create_cache(RANDOMX_FLAG_DEFAULT, m_rx_cache_mem->raw());
        }
      }
      if (m_rx_dataset == nullptr)
        m_rx_dataset = randomx_create_dataset(m_rx_dataset_mem->raw());
      if (m_thread_pool == nullptr) {
        m_thread_pool = new ctpl::thread_pool(new_batch);
        if (!ci.hasAES()) SelectSoftAESImpl(new_batch);
      }

      // recompute cache, dataset for new seed
      if (m_seed_hex != new_seed_hex || m_algo_str != new_algo_str) {
        randomx_init_cache(m_rx_cache, new_seed, new_seed_len);
        // init dataset in parallel threads
        const unsigned rx_dataset_item_count = randomx_dataset_item_count(),
                       thread_count          = std::thread::hardware_concurrency();
        if (new_algo_str == "panthera") {
          randomx_init_dataset(m_rx_dataset, m_rx_cache, 0, rx_dataset_item_count);
        } else if (thread_count > 1) {
          std::list<std::thread> threads;
          for (unsigned i = 0; i < thread_count; ++i) {
            const unsigned a = (rx_dataset_item_count * i) / thread_count,
                           b = (rx_dataset_item_count * (i + 1)) / thread_count;
            threads.emplace_back(init_rx_dataset_thread, m_rx_dataset, m_rx_cache, a, b - a);
          }
          for (auto& thread : threads) thread.join();
        } else init_rx_dataset_thread(m_rx_dataset, m_rx_cache, 0, rx_dataset_item_count);
      }

      // recreate vms
      if (m_vm == nullptr) {
        m_vm = new randomx_vm*[new_batch];
        for (unsigned i = 0; i != new_batch; ++ i) {
          m_vm[i] = randomx_create_vm(
            get_rx_vm_flags(new_algo_str, m_is_rx_jit, m_rx_dataset, m_rx_dataset_mem), nullptr, m_rx_dataset,
            m_lpads->scratchpad() + i * new_mem_size, 0
          );
        }
      }
    } else if (is_small_blob_gpu_dev(new_dev)) {
      if (m_input == nullptr) m_input = static_cast<uint8_t*>(alloc_mem(MAX_BLOB_LEN));
      if (m_output == nullptr) m_output = static_cast<uint8_t*>(alloc_mem(HASH_LEN));
      if (m_spads == nullptr) m_spads = alloc_mem(HASH_LEN);
    } else { // setup cn/c29 stuff
      if (m_input == nullptr) m_input = static_cast<uint8_t*>(alloc_mem(new_batch * MAX_BLOB_LEN));
      if (m_output == nullptr) m_output = static_cast<uint8_t*>(alloc_mem(new_batch * HASH_LEN));
      if (m_spads == nullptr) m_spads = alloc_mem(new_batch * SPAD_LEN);
      if (m_ctx == nullptr) {
        m_ctx = new cryptonight_ctx*[new_batch];
        xmrig::CnCtx::create(m_ctx, m_lpads->scratchpad(), new_mem_size, new_batch);
      }
    }
    if (m_algo_str != new_algo_str) set_fn(new_fn.any);
    m_batch    = new_batch;
    m_mem_size = new_mem_size;
    m_seed_hex = new_seed_hex;
    m_algo_str = new_algo_str;
  }

  m_input_hex      = new_input_hex;
  m_dev            = new_dev;
  m_dev_str        = new_dev_str2;
  m_height         = new_height;
  m_nonce_bytes    = new_nonce_bytes;
  m_nonce_offset   = new_nonce_offset;
  m_c29_proof_size = new_c29_proof_size;
  m_input_len      = new_input_len;
  m_nicehash_mask  = new_nicehash_mask;
  std::memcpy(m_seed, new_seed, HASH_LEN);
  fn_extra_setup();

  // start rx job compute threads
  if (new_dev == DEV::RX_CPU) {
    // need static copy here so it will be alive in rx threads
    static uint8_t new_input2[MAX_BLOB_LEN];
    memcpy(new_input2, new_input, m_input_len);
    const unsigned job_ref = m_job_ref;
    const bool is_rx_v2 = new_algo_str == "rx/2";
    const xmrig::Algorithm rx_algo(cpu_name2algo.at(new_algo_str));
    for (unsigned batch_id = 0; batch_id != m_batch; ++batch_id) m_thread_pool->push(
      [=, this, &m_job_ref = m_job_ref, &m_hash_count = m_hash_count](int) {
        const unsigned thread_id = batch_id;
        try {
          alignas(16) uint8_t  input[MAX_BLOB_LEN];
          alignas(16) uint8_t  output[HASH_LEN];
          alignas(16) uint8_t  raw_hash[HASH_LEN];
          alignas(16) uint8_t  prev_input[MAX_BLOB_LEN];
          alignas(16) uint64_t temp_hash[8];
          uint32_t nonce = new_nonce + new_thread_id * m_batch + batch_id;
          if (m_nicehash_mask) nonce |= bswap_32(*get_nonce32(new_input2, 0)) & static_cast<uint32_t>(m_nicehash_mask);
          const unsigned nonce_step = new_thread_num * m_batch;
          unsigned hashrate_update_counter = HASHRATE_COUNTER_INTERVAL;
          memcpy(input, new_input2, m_input_len);
          if (is_set_nonce) { *get_nonce32(input, 0) = bswap_32(nonce); nonce += nonce_step; }
          if (is_rx_v2) memcpy(prev_input, input, m_input_len);
          randomx_calculate_hash_first(m_vm[thread_id], temp_hash, input, m_input_len, rx_algo);
          while (job_ref == m_job_ref) { // continue until we get a new job
            uint32_t* const pnonce = get_nonce32(input, 0);
            const uint32_t prev_nonce = nonce;
            *pnonce = bswap_32(nonce += nonce_step);
	    // check that current nonce is greater than previous one and nince hash protected nonce part is not changed
            if (m_target && ( m_nicehash_mask ? (prev_nonce & static_cast<uint32_t>(m_nicehash_mask)) != (nonce & static_cast<uint32_t>(m_nicehash_mask)) :
                              prev_nonce > nonce )
            ) {
              send_error("Nonce overflow");
              break; // will also effectively stops this thread
            }
            randomx_calculate_hash_next(m_vm[thread_id], temp_hash, input, m_input_len, output, rx_algo);
            const uint8_t* commitment = nullptr;
            if (is_rx_v2) {
              memcpy(raw_hash, output, HASH_LEN);
              randomx_calculate_commitment(prev_input, m_input_len, raw_hash, output);
              memcpy(prev_input, input, m_input_len);
              commitment = raw_hash;
            }
            if (!is_set_nonce) { // test job
              char hash[HASH_LEN*2+1];
              send_msg("test", "result", hash_bin2hex(output, hash));
              break;
            }
            if (--hashrate_update_counter == 0) {
              hashrate_update_counter = HASHRATE_COUNTER_INTERVAL;
              m_mutex_hashrate.lock();
              m_hash_count += HASHRATE_COUNTER_INTERVAL;
              m_mutex_hashrate.unlock();
            }
            if (m_target && *get_result(output, 0) < m_target)
              send_result(nonce - nonce_step, 4, output, nullptr, 32, commitment);
          }
          // only send for mine jobs
          if (m_target) send_last_nonce(nonce, 4, m_pool_id);
        } catch(const std::string& err) {
          send_error(std::string("Compute function thread exception: ") + err);
        } catch(...) {
          send_error("Compute function thread exception");
        }
      }
    );
  } else {
    if (new_dev == DEV::PEARL_GPU) {
      // pearl has a single header blob with NO embedded nonce; the search variable is an internal
      // seed counter (one seeded attempt per Execute call), so step by worker count, not by batch.
      memcpy(m_input, new_input, m_input_len);
      m_nonce_step = new_thread_num;
      // keep the seed non-zero in mine mode (m_nonce64==0 is the test-job sentinel in Execute)
      m_nonce64 = is_set_nonce ? new_nonce + new_thread_id + m_nonce_step : 0;
      m_nonce32 = 0;
      return;
    }
    if (is_nonce_at_32_gpu_dev(new_dev)) {
      memcpy(m_input, new_input, m_input_len);
      const uint64_t current_nonce = new_nonce + static_cast<uint64_t>(new_thread_id) * m_batch;
      m_nonce_step = new_thread_num * m_batch;
      if (is_set_nonce) {
        std::memcpy(m_input + m_nonce_offset, &current_nonce, sizeof(current_nonce));
        m_nonce64 = current_nonce + m_nonce_step;
      } else {
        m_nonce64 = 0;
      }
      m_nonce32 = 0;
      return;
    }

    m_nonce_step = new_thread_num;
    for (unsigned i = 0; i != m_batch; ++i)
      memcpy(m_input + m_input_len*i, new_input, m_input_len);
    if (m_nonce_bytes == 4) {
      m_nonce32 = new_nonce + new_thread_id;
      if (m_nicehash_mask) m_nonce32 |= bswap_32(*get_nonce32(new_input, 0)) & static_cast<uint32_t>(m_nicehash_mask);
      if (is_set_nonce) for (unsigned i = 0; i != m_batch; ++i) {
        *get_nonce32(i) = bswap_32(m_nonce32); m_nonce32 += m_nonce_step;
      }
    } else {
      m_nonce64 = new_nonce + new_thread_id;
      if (m_nicehash_mask) m_nonce64 |= bswap_64(*get_nonce64(new_input, 0)) & m_nicehash_mask;
      if (is_set_nonce) for (unsigned i = 0; i != m_batch; ++i) {
        *get_nonce64(i) = bswap_64(m_nonce64); m_nonce64 += m_nonce_step;
      }
    }
  }
}

void Core::get_algo_params(const MessageValues& v) {
  if (!v.contains("cpu_sockets")) throw std::string("Missing cpu_sockets algo_params key");
  if (!v.contains("cpu_threads")) throw std::string("Missing cpu_threads algo_params key");
  if (!v.contains("cpu_l3cache")) throw std::string("Missing cpu_l3cache algo_params key");
  const unsigned cpu_sockets = atoi(v.at("cpu_sockets").c_str()),
                 cpu_threads = atoi(v.at("cpu_threads").c_str()),
                 cpu_l3cache = atoi(v.at("cpu_l3cache").c_str());
  const auto& cpu_algo_keys = std::views::keys(cpu_name2algo);
  const auto& gpu_cn_algo_keys = std::views::keys(gpu_cn_algo2fn);
  const auto& gpu_c29_algo_keys = std::views::keys(gpu_c29_algo2fn);
  const auto& gpu_kawpow_algo_keys = std::views::keys(gpu_kawpow_algo2fn);
  const auto& gpu_etchash_algo_keys = std::views::keys(gpu_etchash_algo2fn);
  const auto& gpu_autolykos2_algo_keys = std::views::keys(gpu_autolykos2_algo2fn);
  const auto& gpu_pearl_algo_keys = std::views::keys(gpu_pearl_algo2fn);
  const bool skip_sycl_algos = std::getenv("MOMINER_SKIP_SYCL_ALGO_PARAMS");
  const auto sycl_algo_set = [skip_sycl_algos](const auto& keys) {
    return skip_sycl_algos ? std::set<std::string>{} : std::set<std::string>(keys.begin(), keys.end());
  };
  const std::set<std::string> cpu_algos(cpu_algo_keys.begin(), cpu_algo_keys.end()),
                              gpu_cn_algos = sycl_algo_set(gpu_cn_algo_keys),
                              gpu_c29_algos = sycl_algo_set(gpu_c29_algo_keys),
                              gpu_kawpow_algos = sycl_algo_set(gpu_kawpow_algo_keys),
                              gpu_etchash_algos = sycl_algo_set(gpu_etchash_algo_keys),
                              gpu_autolykos2_algos = sycl_algo_set(gpu_autolykos2_algo_keys),
                              gpu_pearl_algos = sycl_algo_set(gpu_pearl_algo_keys);
  const std::map<std::string, std::string>& result_map = algo_params(
    MAX_CN_CPU_WAYS, cpu_sockets, cpu_threads, cpu_l3cache, algo2mem, cpu_algos,
    gpu_cn_algos, gpu_c29_algos, gpu_kawpow_algos, gpu_etchash_algos, gpu_autolykos2_algos,
    gpu_pearl_algos
  );
  MessageValues result;
  for (const auto& i : result_map) result[i.first] = i.second;
  send_msg("algo_params", result);
}
