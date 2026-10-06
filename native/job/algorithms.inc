const constexpr unsigned SPAD_LEN        = 200;
const constexpr unsigned MAX_CN_CPU_WAYS = 5;

static const xmrig::ICpuInfo& cpu_info() { return *xmrig::Cpu::info(); }
#define ci cpu_info()

// Look up algo->value in a name map, throwing the standard error if absent.
template<class Map> static const typename Map::mapped_type& algo_lookup(
  const Map& map, const std::string& algo
) {
  const auto it = map.find(algo);
  if (it == map.end()) throw std::string("Unsupported algo");
  return it->second;
}

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
  { "c29", c29 },
  { "c30", c30 }
};

static const std::map<std::string, gpu_kawpow_hash_fun> gpu_kawpow_algo2fn = {
  { "kawpow", kawpow },
  { "firopow", firopow },
  { "evrprogpow", evrprogpow },
  { "meowpow", meowpow }
};

static const std::map<std::string, gpu_etchash_hash_fun> gpu_etchash_algo2fn = {
  { "etchash", etchash },
  { "octopus", octopus }
};

static const std::map<std::string, gpu_autolykos2_hash_fun> gpu_autolykos2_algo2fn = {
  { "autolykos2", autolykos2 }
};

static const std::map<std::string, gpu_pearlhash_hash_fun> gpu_pearlhash_algo2fn = {
  { "pearlhash", pearlhash }
};

static const std::map<std::string, gpu_fishhash_hash_fun> gpu_fishhash_algo2fn = {
  { "fishhash", fishhash }
};

static const std::map<std::string, gpu_karlsenhashv2_hash_fun> gpu_karlsenhashv2_algo2fn = {
  { "karlsenhashv2", karlsenhashv2 }
};

static const std::map<std::string, gpu_etchash_hash_fun> gpu_misc_algo2fn = {
  { "hoohash", hoohash },
  { "walahash", walahash },
  { "nexapow", nexapow }
};

static const std::map<std::string, gpu_verthash_hash_fun> gpu_verthash_algo2fn = {
  { "verthash", verthash }
};

static const std::map<std::string, gpu_zelhash_hash_fun> gpu_zelhash_algo2fn = {
  { "zelhash", zelhash },
  { "zhash", zhash },
  { "equihash192_7", equihash192_7 }
};

static const std::map<std::string, gpu_beamhash3_hash_fun> gpu_beamhash3_algo2fn = {
  { "beamhash3", beamhash3 }
};

static const std::map<std::string, unsigned> algo2mem = [](){
  std::map<std::string, unsigned> result = {
    { "cn/gpu", 2*1024*1024 }, // host memory is not really used (number used only for algo_params calcs)
    { "c29",    0 },
    { "c30",    0 },           // host memory is not used even for algo_params calcs
    { "kawpow", 0 },
    { "firopow", 0 },
    { "evrprogpow", 0 },
    { "meowpow", 0 },
    { "etchash", 0 },
    { "octopus", 0 },
    { "autolykos2", 0 },
    { "pearlhash", 0 },
    { "fishhash", 0 },
    { "karlsenhashv2", 0 },
    { "hoohash", 0 },
    { "walahash", 0 },
    { "nexapow", 0 },
    { "verthash", 0 },
    { "zelhash", 0 },
    { "zhash", 0 },
    { "equihash192_7", 0 },
    { "beamhash3", 0 }
  };
  for (const auto& i : cpu_name2algo) result[i.first] = xmrig::Algorithm(i.second).l3();
  return result;
}();

static void add_cpu_algo_dev(
  std::map<std::string, std::string>& result, const std::string& algo,
  const unsigned max_cpu_batch, const unsigned socket_count,
  const unsigned thread_count, const unsigned l3cache
) {
  const auto mem = algo2mem.find(algo);
  if (mem == algo2mem.end()) return;
  std::string result_dev;
  mom::cpu::append_grouped_cpu_devs(result_dev, mom::cpu::cpu_thread_batches(
    algo, max_cpu_batch, socket_count, thread_count, l3cache, mem->second
  ));
  result[algo] = result_dev;
}

static std::map<std::string, std::string> cpu_only_algo_params(
  const unsigned max_cpu_batch,
  const unsigned cpu_sockets, const unsigned cpu_threads, const unsigned cpu_l3cache,
  const std::set<std::string>& cpu_algos
) {
  const unsigned socket_count = std::max(1u, cpu_sockets);
  const unsigned thread_count = std::max(1u, cpu_threads);
  const unsigned l3cache = cpu_l3cache ? cpu_l3cache : thread_count * 2u * 1024u * 1024u;
  std::map<std::string, std::string> result;
  for (const auto& algo : cpu_algos)
    add_cpu_algo_dev(result, algo, max_cpu_batch, socket_count, thread_count, l3cache);
  return result;
}

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
