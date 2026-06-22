// Copyright GNU GPLv3 (c) 2026 MoneroOcean <support@moneroocean.stream>
//
// kHeavyHash (Kaspa) GPU search kernel. Compute-bound, no DAG/dataset/epoch.
// Per nonce: cSHAKE256("ProofOfWorkHash") keccak -> 64x64 nibble matrix x 64 nibble vector heavy step
// (>>10 + 4-bit pack) -> XOR with the pre-pow hash -> cSHAKE256("HeavyHash") keccak -> LE-U256 vs target.
// The per-job 64x64 matrix (xoshiro256++ seeded by the block seed, regenerated to full rank 64) is built
// on the HOST and uploaded (8 KiB); the kernel just reads it from SLM. Ported bit-exact from rusty-kaspa
// consensus/pow/src/{matrix.rs,xoshiro.rs} + crypto/hashes/src/pow_hashers.rs (validated offline against
// js-sha3 cshake256). Uses the SCALAR u64 Keccak (the IGC Uint2-pair miscompile does not affect it).

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

#include "lib-internal.h"
#include "../native/consts.h"

namespace mom_kheavyhash {

constexpr uint32_t MAX_OUTPUTS = 15;

struct KHeavyResult {
  uint32_t count;
  uint64_t nonce[MAX_OUTPUTS];
  uint8_t output[MAX_OUTPUTS][HASH_LEN];
};

// rusty-kaspa baked cSHAKE256 initial states (crypto/hashes/src/pow_hashers.rs).
static constexpr uint64_t POW_INIT[25] = {
  1242148031264380989ULL, 3008272977830772284ULL, 2188519011337848018ULL, 1992179434288343456ULL, 8876506674959887717ULL,
  5399642050693751366ULL, 1745875063082670864ULL, 8605242046444978844ULL, 17936695144567157056ULL, 3343109343542796272ULL,
  1123092876221303306ULL, 4963925045340115282ULL, 17037383077651887893ULL, 16629644495023626889ULL, 12833675776649114147ULL,
  3784524041015224902ULL, 1082795874807940378ULL, 13952716920571277634ULL, 13411128033953605860ULL, 15060696040649351053ULL,
  9928834659948351306ULL, 5237849264682708699ULL, 12825353012139217522ULL, 6706187291358897596ULL, 196324915476054915ULL
};
static constexpr uint64_t HEAVY_INIT[25] = {
  4239941492252378377ULL, 8746723911537738262ULL, 8796936657246353646ULL, 1272090201925444760ULL, 16654558671554924250ULL,
  8270816933120786537ULL, 13907396207649043898ULL, 6782861118970774626ULL, 9239690602118867528ULL, 11582319943599406348ULL,
  17596056728278508070ULL, 15212962468105129023ULL, 7812475424661425213ULL, 3370482334374859748ULL, 5690099369266491460ULL,
  8596393687355028144ULL, 570094237299545110ULL, 9119540418498120711ULL, 16901969272480492857ULL, 13372017233735502424ULL,
  14372891883993151831ULL, 5171152063242093102ULL, 10573107899694386186ULL, 6096431547456407061ULL, 1592359455985097269ULL
};

// ---- device little-endian load/store + scalar Keccak-f[1600] (copied from etchash; IGC-safe) ----
inline uint64_t rotl64_dev(const uint64_t v, const unsigned s) { return (v << s) | (v >> (64 - s)); }
inline uint64_t load64_le_dev(const uint8_t* const p) {
  uint64_t v = 0;
  for (unsigned i = 0; i < 8; ++i) v |= static_cast<uint64_t>(p[i]) << (8 * i);
  return v;
}
inline void store64_le_dev(uint8_t* const p, const uint64_t v) {
  for (unsigned i = 0; i < 8; ++i) p[i] = static_cast<uint8_t>(v >> (8 * i));
}

// chi step over one 5-lane row, reading from the rho+pi temporaries t[] and writing back to st[].
inline void keccakf1600_u64_chi_dev(uint64_t st[25], const unsigned row, const uint64_t t[25]) {
  st[row + 0] = t[row + 0] ^ ((~t[row + 1]) & t[row + 2]);
  st[row + 1] = t[row + 1] ^ ((~t[row + 2]) & t[row + 3]);
  st[row + 2] = t[row + 2] ^ ((~t[row + 3]) & t[row + 4]);
  st[row + 3] = t[row + 3] ^ ((~t[row + 4]) & t[row + 0]);
  st[row + 4] = t[row + 4] ^ ((~t[row + 0]) & t[row + 1]);
}

// Explicit-lane Keccak-f[1600] round (theta into 5 column parities, rho+pi as straight-line constant
// rotates into named t[] lanes, chi per row). This is the canonical high-throughput form -- it removes
// the serial pi rotation chain and the piln/rotc array indexing of the compact loop variant, letting
// IGC keep all 25 lanes in registers and schedule freely. Bit-identical to the standard permutation
// (mirrors the validated keccakf1600_pair_round_dev in kawpow_device.inc, just scalar u64).
inline void keccakf1600_u64_round_dev(uint64_t st[25], const unsigned round) {
  static constexpr uint64_t rndc[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL, 0x8000000080008000ULL,
    0x000000000000808bULL, 0x0000000080000001ULL, 0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008aULL, 0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL, 0x000000000000800aULL, 0x800000008000000aULL,
    0x8000000080008081ULL, 0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
  };

  uint64_t t[25];
  uint64_t u;

  // theta: column parities
  t[0] = st[0] ^ st[5] ^ st[10] ^ st[15] ^ st[20];
  t[1] = st[1] ^ st[6] ^ st[11] ^ st[16] ^ st[21];
  t[2] = st[2] ^ st[7] ^ st[12] ^ st[17] ^ st[22];
  t[3] = st[3] ^ st[8] ^ st[13] ^ st[18] ^ st[23];
  t[4] = st[4] ^ st[9] ^ st[14] ^ st[19] ^ st[24];

  u = t[4] ^ rotl64_dev(t[1], 1);
  st[0] ^= u; st[5] ^= u; st[10] ^= u; st[15] ^= u; st[20] ^= u;
  u = t[0] ^ rotl64_dev(t[2], 1);
  st[1] ^= u; st[6] ^= u; st[11] ^= u; st[16] ^= u; st[21] ^= u;
  u = t[1] ^ rotl64_dev(t[3], 1);
  st[2] ^= u; st[7] ^= u; st[12] ^= u; st[17] ^= u; st[22] ^= u;
  u = t[2] ^ rotl64_dev(t[4], 1);
  st[3] ^= u; st[8] ^= u; st[13] ^= u; st[18] ^= u; st[23] ^= u;
  u = t[3] ^ rotl64_dev(t[0], 1);
  st[4] ^= u; st[9] ^= u; st[14] ^= u; st[19] ^= u; st[24] ^= u;

  // rho + pi: rotate each lane by its constant offset into its permuted slot
  t[0]  = st[0];
  t[10] = rotl64_dev(st[1], 1);
  t[20] = rotl64_dev(st[2], 62);
  t[5]  = rotl64_dev(st[3], 28);
  t[15] = rotl64_dev(st[4], 27);

  t[16] = rotl64_dev(st[5], 36);
  t[1]  = rotl64_dev(st[6], 44);
  t[11] = rotl64_dev(st[7], 6);
  t[21] = rotl64_dev(st[8], 55);
  t[6]  = rotl64_dev(st[9], 20);

  t[7]  = rotl64_dev(st[10], 3);
  t[17] = rotl64_dev(st[11], 10);
  t[2]  = rotl64_dev(st[12], 43);
  t[12] = rotl64_dev(st[13], 25);
  t[22] = rotl64_dev(st[14], 39);

  t[23] = rotl64_dev(st[15], 41);
  t[8]  = rotl64_dev(st[16], 45);
  t[18] = rotl64_dev(st[17], 15);
  t[3]  = rotl64_dev(st[18], 21);
  t[13] = rotl64_dev(st[19], 8);

  t[14] = rotl64_dev(st[20], 18);
  t[24] = rotl64_dev(st[21], 2);
  t[9]  = rotl64_dev(st[22], 61);
  t[19] = rotl64_dev(st[23], 56);
  t[4]  = rotl64_dev(st[24], 14);

  // chi (+ iota folded into row 0)
  keccakf1600_u64_chi_dev(st, 0, t);
  st[0] ^= rndc[round];
  keccakf1600_u64_chi_dev(st, 5, t);
  keccakf1600_u64_chi_dev(st, 10, t);
  keccakf1600_u64_chi_dev(st, 15, t);
  keccakf1600_u64_chi_dev(st, 20, t);
}
inline void keccakf1600_u64_dev(uint64_t st[25]) {
#pragma unroll
  for (unsigned round = 0; round < 24; ++round) keccakf1600_u64_round_dev(st, round);
}

// final hash (LE U256) <= target (LE-byte 32-byte target; byte 31 is most significant)
inline bool meets_target_le_dev(const uint8_t out[32], const uint8_t* const target) {
  for (int i = 31; i >= 0; --i) {
    if (out[i] == target[i]) continue;
    return out[i] < target[i];
  }
  return true;
}

inline void store_kheavy_result(KHeavyResult* const result, const uint64_t nonce, const uint8_t output[32]) {
  using atomic_ref = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                       sycl::access::address_space::global_space>;
  const uint32_t index = atomic_ref(result->count).fetch_add(1);
  if (index >= MAX_OUTPUTS) return;
  result->nonce[index] = nonce;
  for (unsigned i = 0; i < 32; ++i) result->output[index][i] = output[i];
}

#include "blake3_device.inc"  // BLAKE3 single-chunk device hash (for PyrinHashV2's plain-blake3 PoW)

class KHeavyKernel;  // distinct kernel name

static sycl::event submit_kheavyhash_search_gpu(
  sycl::queue& q, MOM_BUNDLE_T& kb, const uint8_t* const d_input, const uint64_t start_nonce,
  const uint16_t* const __restrict__ d_matrix, const uint32_t intensity, const uint8_t* const d_target,
  KHeavyResult* const d_result, const bool is_test
) {
  constexpr unsigned WG = 256;
  const size_t global = (static_cast<size_t>(intensity) + WG - 1) / WG * WG;
  return q.submit([&](sycl::handler& h) {
    MOM_USE_BUNDLE(h, kb);
    sycl::local_accessor<uint16_t, 1> smat(sycl::range<1>(4096), h);
    h.parallel_for<KHeavyKernel>(
      sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(WG)),
      [=](sycl::nd_item<1> item) {
        const uint32_t lid = static_cast<uint32_t>(item.get_local_id(0));
        for (uint32_t i = lid; i < 4096; i += WG) smat[i] = d_matrix[i];
        item.barrier(sycl::access::fence_space::local_space);

        const uint32_t gid = static_cast<uint32_t>(item.get_global_id(0));
        if (gid >= intensity) return;
        const uint64_t nonce = start_nonce + gid;

        // STEP1: PrePowHash = cSHAKE256("ProofOfWorkHash") of the 80B header (words 0..9).
        uint64_t st[25];
#pragma unroll
        for (unsigned i = 0; i < 25; ++i) st[i] = POW_INIT[i];
        st[0] ^= load64_le_dev(d_input + 0);
        st[1] ^= load64_le_dev(d_input + 8);
        st[2] ^= load64_le_dev(d_input + 16);
        st[3] ^= load64_le_dev(d_input + 24);
        st[4] ^= load64_le_dev(d_input + 32);  // timestamp (word 4)
        st[9] ^= nonce;                        // nonce (word 9; overrides the header's nonce field)
        keccakf1600_u64_dev(st);
        uint8_t pph[32];
#pragma unroll
        for (unsigned i = 0; i < 4; ++i) store64_le_dev(pph + i * 8, st[i]);

        // STEP3: expand to 64 nibbles (hi then lo per byte) and do the heavy matvec + 4-bit pack.
        uint8_t vec[64];
#pragma unroll
        for (unsigned i = 0; i < 32; ++i) { vec[2 * i] = pph[i] >> 4; vec[2 * i + 1] = pph[i] & 0x0F; }
        uint8_t product[32];
        for (unsigned i = 0; i < 32; ++i) {
          uint32_t s1 = 0, s2 = 0;
          const uint32_t r1 = (2 * i) * 64;
          const uint32_t r2 = (2 * i + 1) * 64;
#pragma unroll
          for (unsigned j = 0; j < 64; ++j) {
            s1 += static_cast<uint32_t>(smat[r1 + j]) * vec[j];
            s2 += static_cast<uint32_t>(smat[r2 + j]) * vec[j];
          }
          product[i] = static_cast<uint8_t>(((s1 >> 10) << 4) | (s2 >> 10));
        }
        // STEP4: XOR with the pre-pow hash, then HeavyHash.
#pragma unroll
        for (unsigned k = 0; k < 32; ++k) product[k] ^= pph[k];
        uint64_t st2[25];
#pragma unroll
        for (unsigned i = 0; i < 25; ++i) st2[i] = HEAVY_INIT[i];
#pragma unroll
        for (unsigned i = 0; i < 4; ++i) st2[i] ^= load64_le_dev(product + i * 8);
        keccakf1600_u64_dev(st2);
        uint8_t final_hash[32];
#pragma unroll
        for (unsigned i = 0; i < 4; ++i) store64_le_dev(final_hash + i * 8, st2[i]);

        if ((is_test && gid == 0) || meets_target_le_dev(final_hash, d_target))
          store_kheavy_result(d_result, nonce, final_hash);
      }
    );
  });
}

// ---- PyrinHashV2 (Pyrin PYI): same kHeavyHash matrix, but plain-BLAKE3 powHash/final + V2 nibble-XOR
// reduction (no keccak, no DAG). powHash = BLAKE3_32(prePow(32)||ts(8)||32 zeros||nonce(8), 80B), nonce@72.
class PyrinKernel;
static sycl::event submit_pyrinhashv2_search_gpu(
  sycl::queue& q, MOM_BUNDLE_T& kb, const uint8_t* const d_input, const uint64_t start_nonce,
  const uint16_t* const __restrict__ d_matrix, const uint32_t intensity, const uint8_t* const d_target,
  KHeavyResult* const d_result, const bool is_test
) {
  constexpr unsigned WG = 256;
  const size_t global = (static_cast<size_t>(intensity) + WG - 1) / WG * WG;
  return q.submit([&](sycl::handler& h) {
    MOM_USE_BUNDLE(h, kb);
    sycl::local_accessor<uint16_t, 1> smat(sycl::range<1>(4096), h);
    h.parallel_for<PyrinKernel>(
      sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(WG)),
      [=](sycl::nd_item<1> item) {
        const uint32_t lid = static_cast<uint32_t>(item.get_local_id(0));
        for (uint32_t i = lid; i < 4096; i += WG) smat[i] = d_matrix[i];
        item.barrier(sycl::access::fence_space::local_space);
        const uint32_t gid = static_cast<uint32_t>(item.get_global_id(0));
        if (gid >= intensity) return;
        const uint64_t nonce = start_nonce + gid;

        uint8_t pre[80];
        for (unsigned i = 0; i < 80; ++i) pre[i] = d_input[i];
        for (unsigned i = 0; i < 8; ++i) pre[72 + i] = static_cast<uint8_t>(nonce >> (8 * i));  // nonce@72 LE
        uint8_t pph[32]; blake3_dev(pph, 32, pre, 80);

        uint8_t vec[64];
        for (unsigned i = 0; i < 32; ++i) { vec[2 * i] = pph[i] >> 4; vec[2 * i + 1] = pph[i] & 0x0F; }
        uint8_t product[32];
        for (unsigned i = 0; i < 32; ++i) {
          uint32_t s1 = 0, s2 = 0;
          const uint32_t r1 = (2 * i) * 64, r2 = (2 * i + 1) * 64;
          for (unsigned j = 0; j < 64; ++j) { s1 += static_cast<uint32_t>(smat[r1 + j]) * vec[j]; s2 += static_cast<uint32_t>(smat[r2 + j]) * vec[j]; }
          const uint32_t hi = (s1 & 0xF) ^ ((s1 >> 4) & 0xF) ^ ((s1 >> 8) & 0xF);  // V2 nibble-XOR reduce
          const uint32_t lo = (s2 & 0xF) ^ ((s2 >> 4) & 0xF) ^ ((s2 >> 8) & 0xF);
          product[i] = static_cast<uint8_t>((hi << 4) | lo);
        }
        for (unsigned k = 0; k < 32; ++k) product[k] ^= pph[k];
        uint8_t final_hash[32]; blake3_dev(final_hash, 32, product, 32);

        if ((is_test && gid == 0) || meets_target_le_dev(final_hash, d_target))
          store_kheavy_result(d_result, nonce, final_hash);
      }
    );
  });
}

// ---- HOST per-job matrix generation: xoshiro256++ seeded by the 32B block seed, regen to full rank 64 ----
struct XoshiroHost {
  uint64_t s0, s1, s2, s3;
  explicit XoshiroHost(const uint8_t seed[32]) {
    s0 = load64_le_dev(seed + 0); s1 = load64_le_dev(seed + 8);
    s2 = load64_le_dev(seed + 16); s3 = load64_le_dev(seed + 24);
  }
  uint64_t next() {
    const uint64_t res = s0 + rotl64_dev(s0 + s3, 23);
    const uint64_t t = s1 << 17;
    s2 ^= s0; s3 ^= s1; s1 ^= s2; s0 ^= s3;
    s2 ^= t; s3 = rotl64_dev(s3, 45);
    return res;
  }
};

static unsigned compute_rank_host(const uint16_t* const mat) {
  static thread_local double a[64][64];
  for (int i = 0; i < 64; ++i) for (int j = 0; j < 64; ++j) a[i][j] = static_cast<double>(mat[i * 64 + j]);
  constexpr double EPS = 1e-9;
  unsigned rank = 0;
  bool sel[64] = { false };
  for (int i = 0; i < 64; ++i) {
    int j = 0;
    for (; j < 64; ++j) if (!sel[j] && std::fabs(a[j][i]) > EPS) break;
    if (j != 64) {
      ++rank; sel[j] = true;
      for (int p = i + 1; p < 64; ++p) a[j][p] /= a[j][i];
      for (int k = 0; k < 64; ++k)
        if (k != j && std::fabs(a[k][i]) > EPS)
          for (int p = i + 1; p < 64; ++p) a[k][p] -= a[j][p] * a[k][i];
    }
  }
  return rank;
}

static void gen_matrix_host(const uint8_t seed[32], uint16_t* const mat) {
  XoshiroHost gen(seed);
  // Real seeds reach full rank in a handful of rejection-sampling iterations; cap the attempts so a
  // degenerate (e.g. all-zero) seed -> all-zero xoshiro256++ state can't spin forever. On the cap we
  // accept the last matrix rather than hang; non-degenerate seeds never come close to MAX_ATTEMPTS.
  constexpr int MAX_ATTEMPTS = 1024;
  for (int attempt = 0; attempt < MAX_ATTEMPTS; ++attempt) {
    for (int i = 0; i < 64; ++i)
      for (int j = 0; j < 64; j += 16) {
        const uint64_t val = gen.next();
        for (int sh = 0; sh < 16; ++sh) mat[i * 64 + j + sh] = static_cast<uint16_t>((val >> (4 * sh)) & 0x0F);
      }
    if (compute_rank_host(mat) == 64) break;
  }
}

// ---- per-device state ----
class KHeavyState {
public:
  sycl::device device;
  sycl::queue queue;
  std::unique_ptr<MOM_BUNDLE_T> bundle;
  bool shared_io;
  uint8_t* input = nullptr;     // 80B header
  uint8_t* target = nullptr;    // 32B LE target
  uint16_t* matrix = nullptr;   // 64x64 = 4096 u16
  KHeavyResult* result = nullptr;
  uint8_t cached_seed[32];
  bool has_matrix = false;
  std::mutex mutex;

  explicit KHeavyState(const std::string& dev_str)
    : device(get_dev(dev_str)),
      queue(device, sycl::property_list{sycl::property::queue::in_order{}}),
      shared_io(device.is_cpu() || !device.has(sycl::aspect::usm_device_allocations)) {
    if (!device.has(sycl::aspect::usm_shared_allocations) ||
        (!device.is_cpu() && !device.has(sycl::aspect::usm_device_allocations)))
      throw std::string("kheavyhash SYCL device does not support required allocations");
    bundle = std::make_unique<MOM_BUNDLE_T>(MOM_GET_EXEC_BUNDLE(queue.get_context()));
    std::memset(cached_seed, 0, sizeof(cached_seed));
  }
  ~KHeavyState() { queue.wait_and_throw(); free_all(); }

  template <typename T> T* alloc(const size_t n) {
    return shared_io ? sycl::malloc_shared<T>(n, queue) : sycl::malloc_device<T>(n, queue);
  }
  void free_ptr(auto*& p) { if (p) sycl::free(p, queue); p = nullptr; }
  void free_all() { free_ptr(input); free_ptr(target); free_ptr(matrix); free_ptr(result); }

  void ensure_input() {
    if (input && target && matrix && result) return;
    queue.wait_and_throw();
    free_all();
    input = alloc<uint8_t>(80);
    target = alloc<uint8_t>(HASH_LEN);
    matrix = alloc<uint16_t>(4096);
    result = sycl::malloc_shared<KHeavyResult>(1, queue);
    if (!input || !target || !matrix || !result) throw std::string("Can't allocate kheavyhash SYCL buffers");
    has_matrix = false;
  }

  // Regenerate the per-job matrix only when the 32B block seed (header[0..31]) changes.
  void ensure_matrix(const uint8_t* const header) {
    if (has_matrix && std::memcmp(cached_seed, header, 32) == 0) return;
    std::vector<uint16_t> hmat(4096);
    gen_matrix_host(header, hmat.data());
    if (shared_io) std::memcpy(matrix, hmat.data(), 4096 * sizeof(uint16_t));
    else sycl_wait_and_throw(queue.memcpy(matrix, hmat.data(), 4096 * sizeof(uint16_t)), device);
    std::memcpy(cached_seed, header, 32);
    has_matrix = true;
  }
};

static KHeavyState& kheavyhash_state(const std::string& dev_str) {
  static std::mutex states_mutex;
  static std::map<std::string, std::unique_ptr<KHeavyState>> states;
  std::lock_guard<std::mutex> lock(states_mutex);
  auto& state = states[dev_str];
  if (!state) state = std::make_unique<KHeavyState>(dev_str);
  return *state;
}

} // namespace mom_kheavyhash

using namespace mom_kheavyhash;

int kheavyhash(
  const unsigned, const uint32_t, const uint8_t* const input, const unsigned input_size, uint8_t* const output,
  uint8_t* const mix_hash, uint64_t* const pnonce, const uint8_t* const target, const uint8_t* const /*seed_hash*/,
  const unsigned intensity, const bool is_test, const bool /*is_benchmark*/, const std::string& dev_str
) {
  if (input_size < 80) throw std::string("Bad kheavyhash input length (need 80B header)");

  KHeavyState& state = kheavyhash_state(dev_str);
  std::lock_guard<std::mutex> state_lock(state.mutex);
  state.ensure_input();
  state.ensure_matrix(input);  // matrix seeded by header[0..31] (pre_pow_hash)

  uint64_t start_nonce = 0;
  std::memcpy(&start_nonce, input + 72, sizeof(start_nonce));

  if (state.shared_io) {
    std::memcpy(state.input, input, 80);
    std::memcpy(state.target, target, HASH_LEN);
  } else {
    state.queue.memcpy(state.input, input, 80);
    state.queue.memcpy(state.target, target, HASH_LEN);
  }
  std::memset(state.result, 0, sizeof(KHeavyResult));

  sycl_wait_and_throw(
    submit_kheavyhash_search_gpu(state.queue, *state.bundle, state.input, start_nonce, state.matrix,
                                 intensity, state.target, state.result, is_test),
    state.device);

  // Optional unbuffered perf log (MOM_KHEAVYHASH_PERF): the mom.js bench hashrate report is stdout-
  // buffered and can be lost on non-graceful exit; this prints the measured rate straight to stderr.
  static const bool perf_log = std::getenv("MOM_KHEAVYHASH_PERF") != nullptr;
  if (perf_log && !is_test) {
    static uint64_t acc = 0;
    static auto t0 = std::chrono::steady_clock::now();
    acc += intensity;
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (sec >= 2.0) {
      std::fprintf(stderr, "[kheavyhash] %.2f MH/s (%s)\n", static_cast<double>(acc) / sec / 1e6, dev_str.c_str());
      acc = 0; t0 = std::chrono::steady_clock::now();
    }
  }

  const uint32_t count = state.result->count;
  if (count == 0) return 0;
  const uint32_t index = std::min(count, MAX_OUTPUTS) - 1;
  *pnonce = state.result->nonce[index];
  std::memcpy(output, state.result->output[index], HASH_LEN);
  if (mix_hash) std::memset(mix_hash, 0, HASH_LEN);  // kHeavyHash has no separate mix hash
  return 1;
}

int pyrinhashv2(
  const unsigned, const uint32_t, const uint8_t* const input, const unsigned input_size, uint8_t* const output,
  uint8_t* const mix_hash, uint64_t* const pnonce, const uint8_t* const target, const uint8_t* const /*seed_hash*/,
  const unsigned intensity, const bool is_test, const bool /*is_benchmark*/, const std::string& dev_str
) {
  if (input_size < 80) throw std::string("Bad pyrinhashv2 input length (need 80B header)");
  KHeavyState& state = kheavyhash_state(dev_str);  // shares the kHeavyHash matrix infrastructure
  std::lock_guard<std::mutex> state_lock(state.mutex);
  state.ensure_input();
  state.ensure_matrix(input);  // 64x64 matrix seeded by header[0..31] (pre_pow_hash), same Kaspa xoshiro gen
  uint64_t start_nonce = 0;
  std::memcpy(&start_nonce, input + 72, sizeof(start_nonce));
  if (state.shared_io) { std::memcpy(state.input, input, 80); std::memcpy(state.target, target, HASH_LEN); }
  else { state.queue.memcpy(state.input, input, 80); state.queue.memcpy(state.target, target, HASH_LEN); }
  std::memset(state.result, 0, sizeof(KHeavyResult));
  sycl_wait_and_throw(
    submit_pyrinhashv2_search_gpu(state.queue, *state.bundle, state.input, start_nonce, state.matrix,
                                  intensity, state.target, state.result, is_test),
    state.device);
  static const bool perf_log = std::getenv("MOM_PYRIN_PERF") != nullptr;
  if (perf_log && !is_test) {
    static uint64_t acc = 0; static auto t0 = std::chrono::steady_clock::now();
    acc += intensity;
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (sec >= 2.0) { std::fprintf(stderr, "[pyrinhashv2] %.2f MH/s (%s)\n", static_cast<double>(acc) / sec / 1e6, dev_str.c_str()); acc = 0; t0 = std::chrono::steady_clock::now(); }
  }
  const uint32_t count = state.result->count;
  if (count == 0) return 0;
  const uint32_t index = std::min(count, MAX_OUTPUTS) - 1;
  *pnonce = state.result->nonce[index];
  std::memcpy(output, state.result->output[index], HASH_LEN);
  if (mix_hash) std::memset(mix_hash, 0, HASH_LEN);
  return 1;
}
