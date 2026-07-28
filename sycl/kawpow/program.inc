namespace mom_kawpow {

constexpr uint32_t KAWPOW_EPOCH_LENGTH  = 7500;
constexpr uint32_t KAWPOW_PERIOD_LENGTH = 3;
// FiroPow: DAG epoch every 1300 blocks, ProgPoW program changes every block (period == height).
constexpr uint32_t FIROPOW_EPOCH_LENGTH  = 1300;
constexpr uint32_t FIROPOW_PERIOD_LENGTH = 1;
// EvrProgPow: same seal structure as KawPoW (different magic), DAG epoch every 12000 blocks.
constexpr uint32_t EVRPROGPOW_EPOCH_LENGTH  = 12000;
constexpr uint32_t EVRPROGPOW_PERIOD_LENGTH = 3;
// MeowPow (Meowcoin): KawPoW with classic-Ethereum DAG sizing but a SHORTER ProgPoW inner loop
// (REGS 16, CNT_CACHE 6, CNT_MATH 9) and a longer period (6). Same DAG epoch (7500) as KawPoW.
constexpr uint32_t MEOWPOW_EPOCH_LENGTH  = 7500;
constexpr uint32_t MEOWPOW_PERIOD_LENGTH = 6;
constexpr uint32_t KAWPOW_LANES         = 16;
constexpr uint32_t KAWPOW_REGS          = 32;
constexpr uint32_t KAWPOW_DAG_LOADS     = 4;
constexpr uint32_t KAWPOW_CACHE_WORDS   = 4096;
constexpr uint32_t KAWPOW_CNT_DAG       = 64;
constexpr uint32_t KAWPOW_CNT_CACHE     = 11;
constexpr uint32_t KAWPOW_CNT_MATH      = 18;
// MeowPow's shorter inner loop (the only ProgPoW variant here that changes these): HALF the registers,
// ~half the cache/math ops. REGS<=KAWPOW_REGS so the shared 32-wide mix[]/KawpowProgram layout fits.
constexpr uint32_t MEOWPOW_REGS         = 16;
constexpr uint32_t MEOWPOW_CNT_CACHE    = 6;
constexpr uint32_t MEOWPOW_CNT_MATH     = 9;
constexpr uint32_t KAWPOW_DATA_LOADS    = 256 / (sizeof(uint32_t) * KAWPOW_LANES);
constexpr uint32_t KAWPOW_DATASET_PARENTS = 512;
constexpr uint32_t NODE_WORDS           = 16;
constexpr uint32_t FNV_PRIME            = 0x01000193U;
constexpr uint32_t FNV_OFFSET_BASIS     = 0x811c9dc5U;
constexpr uint32_t MAX_KAWPOW_OUTPUTS   = 15;

constexpr uint32_t RAVENCOIN_KAWPOW[15] = {
  0x00000072, 0x00000041, 0x00000056, 0x00000045, 0x0000004E,
  0x00000043, 0x0000004F, 0x00000049, 0x0000004E, 0x0000004B,
  0x00000041, 0x00000057, 0x00000050, 0x0000004F, 0x00000057
};

// EvrProgPow seal magic: ASCII "EVRMORE-PROGPOW" (hyphen 0x2D at index 7). Same 15-word placement
// (st[10..24] initial / st[16..24] final) and seal structure as KawPoW -- only the data differs.
constexpr uint32_t EVRMORE_EVRPROGPOW[15] = {
  0x00000045, 0x00000056, 0x00000052, 0x0000004D, 0x0000004F,  // E V R M O
  0x00000052, 0x00000045, 0x0000002D, 0x00000050, 0x00000052,  // R E - P R
  0x0000004F, 0x00000047, 0x00000050, 0x0000004F, 0x00000057   // O G P O W
};

// MeowPow seal magic: ASCII "MEOWCOINMEOWPOW" (Meowcoin consensus kawpow_hash domain string). Same
// 15-word placement / seal structure as KawPoW; only the magic data differs.
constexpr uint32_t MEOWCOIN_MEOWPOW[15] = {
  0x0000004D, 0x00000045, 0x0000004F, 0x00000057, 0x00000043,  // M E O W C
  0x0000004F, 0x00000049, 0x0000004E, 0x0000004D, 0x00000045,  // O I N M E
  0x0000004F, 0x00000057, 0x00000050, 0x0000004F, 0x00000057   // O W P O W
};

// FiroPow uses the chfast/ethash (EIP-1057) dataset sizing, NOT the classic Ethereum sizing baked
// into data_sizes.h (which KawPoW/Ravencoin uses). The light CACHE sizing is identical between the
// two (so cache_sizes[epoch] + ethash_get_seedhash + ethash_compute_cache_nodes are reused), but the
// full DATASET is larger: init 1.5 GiB (vs 1 GiB), growth 8 MiB/epoch, num_items = largest prime
// <= init/128 + epoch*growth/128, then *128 bytes. The DAG-gen kernel and search are otherwise
// identical -- only the node count (and thus the search modulus) changes.
constexpr int FIRO_FULL_DATASET_INIT_ITEMS = ((1 << 30) + (1 << 29)) / 128;  // 12582912 (1.5 GiB)
constexpr int FIRO_FULL_DATASET_GROWTH_ITEMS = (1 << 23) / 128;              // 65536 per epoch
// EvrProgPow uses the SAME chfast/EIP-1057 prime sizing but a 3 GiB init (confirmed against the
// EvrmoreOrg/cpp-evrprogpow reference: full_dataset_init=(1<<30)*3, growth=1<<23, num_items=
// largest_prime(init/128 + epoch*growth/128)*128; epoch 0 = 25165813 items = 3.0000 GiB).
constexpr int EVR_FULL_DATASET_INIT_ITEMS = ((1u << 30) * 3) / 128;          // 25165824 (3 GiB)
constexpr int EVR_FULL_DATASET_GROWTH_ITEMS = (1 << 23) / 128;               // 65536 per epoch
// MeowPow uses the STANDARD ethash chfast sizing (init 1 GiB, growth 8 MiB/epoch) -- which equals the
// classic data_sizes.h table -- EXCEPT for the epoch-110 "dag change" hard fork (Meowcoin core
// create_epoch_context): for epoch >= 110 the dataset/cache SIZES are computed with meow_epoch =
// epoch*4 (a one-time 4x DAG-size jump at block 960000 / epoch 110), while the keccak SEED still uses
// the real epoch. So below 110 it matches KawPoW sizing; at/above 110 the DAG is ~4.4 GiB+.
constexpr int MEOW_FULL_DATASET_INIT_ITEMS   = (1 << 30) / 128;              // 8388608 (1 GiB)
constexpr int MEOW_FULL_DATASET_GROWTH_ITEMS = (1 << 23) / 128;             // 65536 per epoch
constexpr int MEOW_LIGHT_CACHE_INIT_ITEMS    = (1 << 24) / 64;              // 262144 (16 MiB)
constexpr int MEOW_LIGHT_CACHE_GROWTH_ITEMS  = (1 << 17) / 64;             // 2048 per epoch
constexpr uint32_t MEOWPOW_DAGCHANGE_EPOCH   = 110;                        // block 960000 / 7500 -> *4 sizing

static bool firo_is_odd_prime(const int number) {
  for (int64_t d = 3; d * d <= static_cast<int64_t>(number); d += 2)
    if (number % d == 0) return false;
  return true;
}

static int firo_find_largest_prime(int n) {
  if (n < 2) return 0;
  if (n == 2) return 2;
  if (n % 2 == 0) --n;
  while (!firo_is_odd_prime(n)) n -= 2;
  return n;
}

// chfast/EIP-1057 full-dataset byte size for an epoch: largest_prime(init + epoch*growth) * 128.
// Shared by FiroPow (1.5 GiB init) and EvrProgPow (3 GiB init); a multiple of 128 (so also of 64).
static uint64_t chfast_dag_bytes(const uint32_t epoch, const int init_items, const int growth_items) {
  const int upper = init_items + static_cast<int>(epoch) * growth_items;
  return static_cast<uint64_t>(firo_find_largest_prime(upper)) * 128ull;
}

// chfast light-cache byte size: largest_prime(init + epoch*growth) * 64. Only MeowPow needs this
// (its epoch-110 dag-change scales the cache too); the other variants reuse the classic cache_sizes[]
// table. A multiple of 64 (NODE_WORDS*4) so it tiles cleanly into 64-byte ethash nodes.
static uint64_t chfast_cache_bytes(const uint32_t epoch, const int init_items, const int growth_items) {
  const int upper = init_items + static_cast<int>(epoch) * growth_items;
  return static_cast<uint64_t>(firo_find_largest_prime(upper)) * 64ull;
}

// Used by MOM_LOOP_STATS to break a dispatch call into host-side phases.
static uint64_t kawpow_now_us() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
    std::chrono::steady_clock::now().time_since_epoch()
  ).count();
}

static bool kawpow_loop_stats() {
  static const bool enabled = std::getenv("MOM_LOOP_STATS") != nullptr;
  return enabled;
}

static void format_duration_ms(char* out, size_t out_size, uint64_t ms) {
  if (ms < 1000) {
    std::snprintf(out, out_size, "%" PRIu64 "ms", ms);
  } else if (ms < 60000) {
    std::snprintf(out, out_size, "%.2f s", static_cast<double>(ms) / 1000.0);
  } else {
    std::snprintf(out, out_size, "%.2f min", static_cast<double>(ms) / 60000.0);
  }
}

struct KawpowCacheOp {
  uint32_t src;
  uint32_t dst;
  uint32_t merge_mode;
  uint32_t merge_shift;
};

struct KawpowMathOp {
  uint32_t src1;
  uint32_t src2;
  uint32_t dst;
  uint32_t math_selector;
  uint32_t merge_mode;
  uint32_t merge_shift;
};

struct KawpowDataOp {
  uint32_t dst;
  uint32_t merge_mode;
  uint32_t merge_shift;
};

struct KawpowProgram {
  KawpowCacheOp cache[KAWPOW_CNT_CACHE];
  KawpowMathOp math[KAWPOW_CNT_MATH];
  KawpowDataOp data[KAWPOW_DATA_LOADS];
};

// Per-period spec constants for the Intel spec-const path (the CUDA build folds the program via the
// runtime source JIT instead, but these stay declared for the shared SLM/CPU code below).
#if !defined(MOM_SYCL_ADAPTIVECPP) && !defined(MOM_SYCL_PORTABLE_OPENCL)
constexpr sycl::specialization_id<KawpowProgram> kawpow_program_id;
constexpr sycl::specialization_id<FastModData> kawpow_dag_mod_id;
constexpr sycl::specialization_id<bool> kawpow_cpu_offset_barrier_id;
#endif

struct KawpowResult {
  uint32_t count;
  uint64_t nonce[MAX_KAWPOW_OUTPUTS];
  uint32_t output[MAX_KAWPOW_OUTPUTS][8];
  uint32_t mix_hash[MAX_KAWPOW_OUTPUTS][8];
};

struct Uint2 {
  uint32_t x;
  uint32_t y;
};

// alignas(16): the per-loop dag read (device.inc) is the hottest global op (64/nonce); 16-byte
// alignment lets it lower to one 128-bit vector load (ld.global.v4.u32 / block_load) instead of 4
// scalar loads. The DAG base is 256B-aligned USM and offsets step whole DagLoads, so it is already
// 16B-aligned at runtime -- only the declared alignment was blocking the wide load. Size stays 16
// bytes, so this stays byte-compatible with the jit.inc mirror (which gets the same alignas).
struct alignas(16) DagLoad {
  uint32_t s[KAWPOW_DAG_LOADS];
};

struct Kiss99 {
  uint32_t z;
  uint32_t w;
  uint32_t jsr;
  uint32_t jcong;
};

#include "device.inc"
#include "jit.inc"

// Per-algo ProgPoW variant: epoch/period divisors (host-side DAG/program selection) plus the keccak
// seal mode + magic words baked into the kernel. KawPoW and EvrProgPow share SealMode::KAWPOW (magic
// differs); FiroPow uses SealMode::FIRO (no magic). SealMode is defined in device.inc.
struct KawpowVariant {
  uint32_t epoch_length;
  uint32_t period_length;
  SealMode seal;
  const uint32_t* magic;  // 15 words; ignored for SealMode::FIRO
  // Dataset sizing: chfast_init_items == 0 means classic Ethereum sizing (data_sizes.h table, KawPoW);
  // otherwise chfast/EIP-1057 sizing largest_prime(init + epoch*growth)*128 -- FiroPow 1.5 GiB init,
  // EvrProgPow 3 GiB init. Light cache sizing is identical across all three variants.
  int chfast_init_items;
  int chfast_growth_items;
  // Light-cache chfast sizing (init/growth in 64-byte items). 0 == use the classic cache_sizes[] table
  // (KawPoW/FiroPow/EvrProgPow). MeowPow sets these because its epoch-110 dag-change scales the cache.
  int chfast_cache_init_items;
  int chfast_cache_growth_items;
  // MeowPow "dag change" hard fork (Meowcoin core): for epoch >= this threshold, the DATASET AND CACHE
  // SIZES are computed with a scaled epoch (epoch*4) while the keccak SEED keeps the real epoch. 0
  // disables it (KawPoW/FiroPow/EvrProgPow). MeowPow uses MEOWPOW_DAGCHANGE_EPOCH (110, block 960000).
  uint32_t dagchange_epoch;
  // Inner-loop ProgPoW shape. KawPoW/FiroPow/EvrProgPow = 32/11/18; MeowPow = 16/6/9. These drive both
  // make_program() (host) and the search-kernel template instantiation (REGS/CNT_CACHE/CNT_MATH).
  uint32_t regs;
  uint32_t cnt_cache;
  uint32_t cnt_math;
};

constexpr KawpowVariant KAWPOW_VARIANT{
  KAWPOW_EPOCH_LENGTH, KAWPOW_PERIOD_LENGTH, SealMode::KAWPOW, RAVENCOIN_KAWPOW, 0, 0, 0, 0, 0,
  KAWPOW_REGS, KAWPOW_CNT_CACHE, KAWPOW_CNT_MATH};
constexpr KawpowVariant FIROPOW_VARIANT{
  FIROPOW_EPOCH_LENGTH, FIROPOW_PERIOD_LENGTH, SealMode::FIRO, RAVENCOIN_KAWPOW,
  FIRO_FULL_DATASET_INIT_ITEMS, FIRO_FULL_DATASET_GROWTH_ITEMS, 0, 0, 0,
  KAWPOW_REGS, KAWPOW_CNT_CACHE, KAWPOW_CNT_MATH};
constexpr KawpowVariant EVRPROGPOW_VARIANT{
  EVRPROGPOW_EPOCH_LENGTH, EVRPROGPOW_PERIOD_LENGTH, SealMode::KAWPOW, EVRMORE_EVRPROGPOW,
  EVR_FULL_DATASET_INIT_ITEMS, EVR_FULL_DATASET_GROWTH_ITEMS, 0, 0, 0,
  KAWPOW_REGS, KAWPOW_CNT_CACHE, KAWPOW_CNT_MATH};
constexpr KawpowVariant MEOWPOW_VARIANT{
  MEOWPOW_EPOCH_LENGTH, MEOWPOW_PERIOD_LENGTH, SealMode::KAWPOW, MEOWCOIN_MEOWPOW,
  MEOW_FULL_DATASET_INIT_ITEMS, MEOW_FULL_DATASET_GROWTH_ITEMS,
  MEOW_LIGHT_CACHE_INIT_ITEMS, MEOW_LIGHT_CACHE_GROWTH_ITEMS, MEOWPOW_DAGCHANGE_EPOCH,
  MEOWPOW_REGS, MEOWPOW_CNT_CACHE, MEOWPOW_CNT_MATH};

// regs/cnt_cache/cnt_math are per-variant (KawPoW 32/11/18; meowpow 16/6/9). They size the dst/src
// permutation, the kiss99 draw count, and the math src index modulus (REGS*(REGS-1): %992 KawPoW,
// %240 meowpow). The output arrays stay sized at the 18/11 max; meowpow fills only the active prefix.
KawpowProgram make_program(const uint64_t period, const uint32_t regs = KAWPOW_REGS,
                           const uint32_t cnt_cache = KAWPOW_CNT_CACHE,
                           const uint32_t cnt_math = KAWPOW_CNT_MATH) {
  KawpowProgram program{};
  uint32_t dst_seq[KAWPOW_REGS];
  uint32_t src_seq[KAWPOW_REGS];

  for (uint32_t i = 0; i < regs; ++i) {
    dst_seq[i] = i;
    src_seq[i] = i;
  }

  uint32_t fnv_hash = FNV_OFFSET_BASIS;
  Kiss99 rnd{
    fnv1a(fnv_hash, static_cast<uint32_t>(period)),
    fnv1a(fnv_hash, static_cast<uint32_t>(period >> 32)),
    fnv1a(fnv_hash, static_cast<uint32_t>(period)),
    fnv1a(fnv_hash, static_cast<uint32_t>(period >> 32))
  };

  for (uint32_t i = regs; i > 1; --i) {
    std::swap(dst_seq[i - 1], dst_seq[kiss99(rnd) % i]);
    std::swap(src_seq[i - 1], src_seq[kiss99(rnd) % i]);
  }

  uint32_t dst_counter = 0;
  uint32_t src_counter = 0;
  uint32_t cache_counter = 0;
  uint32_t math_counter = 0;

  for (uint32_t i = 0; i < cnt_cache || i < cnt_math; ++i) {
    if (i < cnt_cache) {
      // Order matters: the src/dst sequence reads must precede the kiss99() selector draw to match
      // the reference ProgPoW program (the test vector guards this).
      const uint32_t src = src_seq[(src_counter++) % regs];
      const uint32_t dst = dst_seq[(dst_counter++) % regs];
      const uint32_t selector = kiss99(rnd);
      program.cache[cache_counter++] = {src, dst, selector % 4, merge_shift(selector)};
    }

    if (i < cnt_math) {
      const uint32_t src_rnd = kiss99(rnd) % (regs * (regs - 1));
      const uint32_t src1 = src_rnd % regs;
      uint32_t src2 = src_rnd / regs;
      if (src2 >= src1) ++src2;

      const uint32_t math_selector = kiss99(rnd);
      const uint32_t merge_selector = kiss99(rnd);
      program.math[math_counter++] = {
        src1,
        src2,
        dst_seq[(dst_counter++) % regs],
        math_selector % 11,
        merge_selector % 4,
        merge_shift(merge_selector)
      };
    }
  }

  for (uint32_t i = 0; i < KAWPOW_DATA_LOADS; ++i) {
    // The first data load always targets register 0; the rest pull from the shuffled dst sequence.
    const uint32_t dst = i == 0 ? 0 : dst_seq[(dst_counter++) % regs];
    const uint32_t selector = kiss99(rnd);
    program.data[i] = {dst, selector % 4, merge_shift(selector)};
  }

  return program;
}

// cache_bytes sizes the cache (classic cache_sizes[] or MeowPow's chfast/scaled size); seed_epoch is
// the epoch the keccak seedhash is derived from (== epoch except for MeowPow >= dag-change, where the
// SIZE uses epoch*4 but the SEED keeps the real epoch).
void compute_light_cache(std::vector<uint32_t>& cache, const uint64_t cache_bytes, const uint32_t seed_epoch) {
  cache.assign(cache_bytes / sizeof(uint32_t), 0);

  const ethash_h256_t seed = ethash_get_seedhash(seed_epoch);
  if (!ethash_compute_cache_nodes(cache.data(), cache_bytes, &seed)) {
    throw std::string("Can't calculate kawpow light cache");
  }
}

// Named so the per-period specialized bundle can be built ahead of time.
