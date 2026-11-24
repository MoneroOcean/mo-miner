const constexpr unsigned SPAD_LEN        = 200;
const constexpr unsigned MAX_CN_CPU_WAYS = 5;

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

// GPU algorithms with their dispatch kind and native entry point; exactly one FN slot is set.
struct GpuAlgo {
  DEV dev;
  FN fn;
};
static const std::map<std::string, GpuAlgo> gpu_algos = {
  { "cn/gpu",       { DEV::CN_GPU,             { .gpu_cn = cn_gpu } } },
  { "c29",          { DEV::C29_GPU,            { .gpu_cuckaroo = c29 } } },
  { "c30",          { DEV::C30_GPU,            { .gpu_cuckaroo = c30 } } },
  { "kawpow",       { DEV::PROGPOW_GPU,        { .gpu_progpow = kawpow } } },
  { "firopow",      { DEV::PROGPOW_GPU,        { .gpu_progpow = firopow } } },
  { "evrprogpow",   { DEV::PROGPOW_GPU,        { .gpu_progpow = evrprogpow } } },
  { "meowpow",      { DEV::PROGPOW_GPU,        { .gpu_progpow = meowpow } } },
  { "etchash",      { DEV::ETHASH_FAMILY_GPU,  { .gpu_hash = etchash } } },
  { "octopus",      { DEV::ETHASH_FAMILY_GPU,  { .gpu_hash = octopus } } },
  { "autolykos2",   { DEV::AUTOLYKOS2_GPU,     { .gpu_autolykos2 = autolykos2 } } },
  { "pearlhash",    { DEV::PEARLHASH_GPU,      { .gpu_pearlhash = pearlhash } } },
  { "fishhash",     { DEV::FISHHASH_GPU,       { .gpu_hash = fishhash } } },
  { "karlsenhashv2", { DEV::KARLSENHASHV2_GPU,  { .gpu_hash = karlsenhashv2 } } },
  { "hoohash",      { DEV::MISC_GPU,           { .gpu_hash = hoohash } } },
  { "walahash",     { DEV::MISC_GPU,           { .gpu_hash = walahash } } },
  { "xelishashv3",  { DEV::MISC_GPU,           { .gpu_hash = xelishashv3 } } },
  { "nexapow",      { DEV::MISC_GPU,           { .gpu_hash = nexapow } } },
  { "verthash",     { DEV::VERTHASH_GPU,       { .gpu_hash = verthash } } },
  { "equihash192_7", { DEV::ZCASH_EQUIHASH_GPU, { .gpu_equihash = equihash192_7 } } },
  { "zelhash",      { DEV::ZCASH_EQUIHASH_GPU, { .gpu_equihash = zelhash } } },
  { "zhash",        { DEV::ZCASH_EQUIHASH_GPU, { .gpu_equihash = zhash } } },
  { "beamhash3",    { DEV::BEAMHASH3_GPU,      { .gpu_equihash = beamhash3 } } },
};

// Tuning follows the dispatch kind even when multiple kinds use the same function signature.
static std::set<std::string> gpu_algos_with(std::initializer_list<DEV> kinds) {
  std::set<std::string> result;
  for (const auto& [name, algo] : gpu_algos) {
    if (std::find(kinds.begin(), kinds.end(), algo.dev) != kinds.end())
      result.insert(name);
  }
  return result;
}

// Host scratch bytes per hash. GPU algorithms keep their memory on the device; cn/gpu's value only
// sizes algo_params batches.
static const std::map<std::string, unsigned> algo2mem = [](){
  std::map<std::string, unsigned> result;
  for (const auto& i : gpu_algos) {
    result[i.first] = 0;
  }
  result["cn/gpu"] = 2*1024*1024;
  for (const auto& i : cpu_name2algo) {
    result[i.first] = xmrig::Algorithm(i.second).l3();
  }
  return result;
}();

// CPU-only algo_params answer (MOM_SKIP_SYCL_ALGO_PARAMS): each CPU algorithm's device list.
static std::map<std::string, std::string> cpu_only_algo_params(
  const unsigned max_cpu_batch,
  const unsigned cpu_sockets, const unsigned cpu_threads, const unsigned cpu_l3cache,
  const std::set<std::string>& cpu_algos
) {
  const mom::cpu::CpuTopology cpu =
    mom::cpu::normalize_cpu_topology(cpu_sockets, cpu_threads, cpu_l3cache);
  std::map<std::string, std::string> result;
  for (const auto& algo : cpu_algos) {
    const auto mem = algo2mem.find(algo);
    if (mem != algo2mem.end())
      result[algo] = mom::cpu::cpu_algo_devs(algo, max_cpu_batch, cpu, mem->second);
  }
  return result;
}

static xmrig::VirtualMemory* alloc_huge_mem(const unsigned size) {
  xmrig::VirtualMemory* const mem = new xmrig::VirtualMemory(size, true, false, false);
  if (mem->raw()) return mem;
  delete mem;
  throw std::string("Can't allocate " + std::to_string(size) + " bytes of memory");
}

static unsigned allocation_size(const unsigned count, const unsigned item_size) {
  if (count && item_size > std::numeric_limits<unsigned>::max() / count)
    throw std::string("Requested allocation is too large");
  return count * item_size;
}

static void* alloc_mem(const unsigned size) {
  void* const mem = _mm_malloc(size, 4096);
  if (mem) return mem;
  throw std::string("Can't allocate " + std::to_string(size) + " bytes of memory");
}
