// DPC++ CUDA-backend kawpow JIT: build the per-period search kernel from SYCL *source* at runtime
// with the ProgPoW program baked in as a `const` struct, so the random-math interpreter folds to
// straight-line constant ops (spec-constants do NOT fold on this backend; baking matches the Intel
// spec-const speedup). The device code is the SAME sycl/kawpow_device.inc the AOT build uses; read at
// runtime and appended after a self-contained prelude (constants/structs/FastModData) plus the baked
// program/dag-mod and a free-function-kernel wrapper. Only built for MOM_SYCL_CUDA (Linux/NVIDIA).
#if defined(MOM_SYCL_CUDA)
#include <dlfcn.h>  // dladdr: locate the loaded module so kawpow_device.inc can ship beside it

// Prelude: everything sycl/kawpow_device.inc relies on (it assumes constants/structs/FastModData are
// already defined). Kept byte-compatible with the host KawpowProgram/FastModData so the emitted
// literal initializers below bind correctly; the kawpow test vector guards against drift.
static const char* const KAWPOW_JIT_PRELUDE = R"KJP(
#define MO_KAW_JIT_FOLD 1   // make kawpow_device.inc force-inline so the baked program const-folds
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/work_group_static.hpp>
#include <cstdint>
using std::uint8_t; using std::uint32_t; using std::uint64_t;
namespace syclexp = sycl::ext::oneapi::experimental;

constexpr uint32_t KAWPOW_LANES         = 16;
constexpr uint32_t KAWPOW_REGS          = 32;
constexpr uint32_t KAWPOW_DAG_LOADS     = 4;
constexpr uint32_t KAWPOW_CACHE_WORDS   = 4096;
constexpr uint32_t KAWPOW_CNT_DAG       = 64;
constexpr uint32_t KAWPOW_CNT_CACHE     = 11;
constexpr uint32_t KAWPOW_CNT_MATH      = 18;
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

struct KawpowCacheOp { uint32_t src, dst, merge_mode, merge_shift; };
struct KawpowMathOp { uint32_t src1, src2, dst, math_selector, merge_mode, merge_shift; };
struct KawpowDataOp { uint32_t dst, merge_mode, merge_shift; };
struct KawpowProgram {
  KawpowCacheOp cache[KAWPOW_CNT_CACHE];
  KawpowMathOp math[KAWPOW_CNT_MATH];
  KawpowDataOp data[KAWPOW_DATA_LOADS];
};
struct KawpowResult {
  uint32_t count;
  uint64_t nonce[MAX_KAWPOW_OUTPUTS];
  uint32_t output[MAX_KAWPOW_OUTPUTS][8];
  uint32_t mix_hash[MAX_KAWPOW_OUTPUTS][8];
};
struct Uint2 { uint32_t x, y; };
struct DagLoad { uint32_t s[KAWPOW_DAG_LOADS]; };
struct Kiss99 { uint32_t z, w, jsr, jcong; };

// FastModData / fast_mod_dev mirror sycl/lib-internal.h (the AOT build pulls them from there).
struct FastModData { uint32_t reciprocal, increment, shift, divisor; };
inline uint32_t fast_mod_dev(const uint32_t a, const FastModData d) {
  const uint64_t t = a;
  const uint32_t q = static_cast<uint32_t>(((t + d.increment) * d.reciprocal) >> d.shift);
  return a - q * d.divisor;
}
)KJP";

// Free-function-kernel wrapper. Mirrors the AOT MOM_SYCL_CUDA search lambda, but reads the baked
// BAKED_PROGRAM/BAKED_DAGMOD consts (which fold) and gets work-group local memory from
// work_group_static instead of a local_accessor. Appended after the device .inc + the baked consts.
static const char* const KAWPOW_JIT_WRAPPER = R"KJW(
extern "C"
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void kawpow_search_jit(const uint8_t* d_input, const DagLoad* __restrict__ d_dag_load, KawpowResult* d_result,
                       uint32_t intensity, uint64_t start_nonce, uint64_t target, int is_test) {
  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
  static syclexp::work_group_static<uint32_t[KAWPOW_CACHE_WORDS]> c_dag;
  const uint32_t lid = item.get_local_id(0);
  const uint32_t gid = item.get_global_id(0);
  const bool active = gid < intensity;
  const uint64_t full_nonce = start_nonce + gid;

  for (uint32_t word = lid * KAWPOW_DAG_LOADS; word < KAWPOW_CACHE_WORDS;
       word += item.get_local_range(0) * KAWPOW_DAG_LOADS) {
    const DagLoad load = d_dag_load[word / KAWPOW_DAG_LOADS];
    for (unsigned i = 0; i < KAWPOW_DAG_LOADS; ++i) c_dag[word + i] = load.s[i];
  }
  item.barrier(sycl::access::fence_space::local_space);

  const auto sg = item.get_sub_group();
  const uint32_t lane_id = static_cast<uint32_t>(sg.get_local_id()[0]) & (KAWPOW_LANES - 1);

  uint32_t state2[8];
  kawpow_initial_dev(d_input, static_cast<uint32_t>(full_nonce),
                     static_cast<uint32_t>(full_nonce >> 32), state2);

  uint32_t digest[8];
  const KawpowSgExchange ex{sg, static_cast<uint32_t>(sg.get_local_id()[0]) & ~(KAWPOW_LANES - 1u)};
  uint32_t* const c_dag_ptr = &c_dag[0];
  kawpow_search_dev(ex, lane_id, state2, c_dag_ptr, d_dag_load, BAKED_PROGRAM, BAKED_DAGMOD, digest);

  uint32_t final_state[25];
  kawpow_final_dev(state2, digest, final_state);
  if ((is_test && gid == 0) || (active && target && kawpow_meets_target_words(final_state, target))) {
    store_kawpow_result(d_result, full_nonce, final_state, digest);
  }
}
)KJW";

static std::string kawpow_read_file(const std::string& path) {
  if (path.empty()) return {};
  std::ifstream f(path, std::ios::binary);
  if (!f) return {};
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

// Directory of the loaded mom.node (so kawpow_device.inc can ship beside it), via dladdr.
static std::string kawpow_module_dir() {
  Dl_info info{};
  if (dladdr(reinterpret_cast<const void*>(&kawpow_read_file), &info) && info.dli_fname) {
    const std::string p = info.dli_fname;
    const size_t slash = p.find_last_of('/');
    if (slash != std::string::npos) return p.substr(0, slash);
  }
  return {};
}

// kawpow_device.inc holds the device code the CUDA JIT compiles at runtime. It ships as a separate
// file: look beside the loaded module (release: libs/kawpow_device.inc next to mom.node), then in
// the source tree (dev: sycl/kawpow_device.inc), with $MOM_KAWPOW_DEVICE_INC as an override. Read
// once and cached.
static const std::string& kawpow_device_src() {
  static const std::string src = [] {
    const char* const env = std::getenv("MOM_KAWPOW_DEVICE_INC");
    const std::string dir = kawpow_module_dir();
    const std::string candidates[] = {
      env ? std::string(env) : std::string(),
      dir.empty() ? std::string() : dir + "/kawpow_device.inc",
      dir.empty() ? std::string() : dir + "/sycl/kawpow_device.inc",
      dir.empty() ? std::string() : dir + "/../../sycl/kawpow_device.inc",  // build/Release -> repo/sycl
      "sycl/kawpow_device.inc",
    };
    for (const std::string& p : candidates) {
      std::string c = kawpow_read_file(p);
      if (!c.empty()) return c;
    }
    throw std::string("kawpow JIT: cannot find kawpow_device.inc beside mom.node or in "
                      "sycl/ (set MOM_KAWPOW_DEVICE_INC)");
  }();
  return src;
}

static std::string kawpow_emit_baked(const KawpowProgram& p, const FastModData& m) {
  // Emit one struct's brace-init "{a,b,c,...},"; pass fields in declaration order (no field names,
  // so order is the only thing binding the literal to the struct layout).
  const auto braces = [](std::initializer_list<uint32_t> fields) {
    std::string r = "{";
    const char* sep = "";
    for (const uint32_t v : fields) { r += sep + std::to_string(v); sep = ","; }
    return r + "},";
  };
  std::string s = "static const KawpowProgram BAKED_PROGRAM = {{";
  for (const auto& o : p.cache) s += braces({o.src, o.dst, o.merge_mode, o.merge_shift});
  s += "},{";
  for (const auto& o : p.math) s += braces({o.src1, o.src2, o.dst, o.math_selector, o.merge_mode, o.merge_shift});
  s += "},{";
  for (const auto& o : p.data) s += braces({o.dst, o.merge_mode, o.merge_shift});
  s += "}};\n";
  s += "static const FastModData BAKED_DAGMOD = {" + std::to_string(m.reciprocal) + "," +
       std::to_string(m.increment) + "," + std::to_string(m.shift) + "," + std::to_string(m.divisor) + "};\n";
  return s;
}

#endif // MOM_SYCL_CUDA
