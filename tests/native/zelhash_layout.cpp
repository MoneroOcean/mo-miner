// Copyright GNU GPLv3 (c) 2026 MoneroOcean <support@moneroocean.stream>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <vector>

#include "../../sycl/blake2b_pair.hpp"

constexpr unsigned EQUIHASH_ROW_LEN = 20;
// Source fragments deliberately follow their producer dependency order.
// clang-format off
#include "zelhash_layout_math.inc"
#include "zelhash_blake_constants.inc"
#include "../../sycl/zelhash/blake2b.inc"
#include "zelhash_recovery_follower.inc"
// clang-format on

#if defined(MOM_EQ_ALIGNED_GPU_RECORDS)
// Record arithmetic uses the exact source body; this scalar boundary does not prove vector
// lowering.
inline void store_aligned_u32x4(uint32_t *out, uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
  out[0] = a;
  out[1] = b;
  out[2] = c;
  out[3] = d;
}
#include "zelhash_compact_store.inc"
#endif
} // namespace mom_zelhash

using namespace mom_zelhash;
static unsigned assertions = 0;
static void require(bool condition, const char *message) {
  ++assertions;
  if (condition) return;
  std::fprintf(stderr, "ZelHash layout regression: %s\n", message);
  std::abort();
}
static uint32_t random_word() {
  static uint32_t state = 0xa3c59ac3u;
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return state;
}
template <bool GPU_COMPACT>
static uint32_t recover_leaf(const std::vector<uint32_t> &records,
                             const std::vector<uint32_t> &index, uint32_t bucket, uint32_t slot,
                             unsigned slot_capacity, bool valid_input = true) {
  constexpr unsigned PROOF = 1u << K;
  const unsigned lid = 0;
  const uint32_t valid[] = {valid_input};
  const uint32_t node_bucket[] = {bucket}, node_slot[] = {slot};
  uint32_t next_slot[] = {0xabcdef12u};
  struct Levels {
    const uint32_t *level[1];
    const uint32_t *l0_index;
  };
  const Levels lv = {{records.data()}, index.data()};
#include "zelhash_recovery_leaf.inc"
  return next_slot[0];
}
static void test_layout() {
  require(N == 125 && K == 4 && NUM_ENTRIES == (1ull << 26), "consensus constants");
  require(MAX_SLOTS < (1u << 14) && MAX_SLOTS % 16 == 0, "encoded slot width");
  require(NBUCKETS < (1u << 16), "encoded bucket width");
#if defined(MOM_SYCL_ADAPTIVECPP_CUDA)
  require(MAX_SLOTS == 2368 && NBUCKETS == 32768, "ACPP CUDA six-deviation capacity");
#elif defined(MOM_SYCL_HAS_CUDA)
  require(MAX_SLOTS == 8192 && NBUCKETS == 8759, "native CUDA geometry preserved");
#elif defined(MOM_SYCL_HAS_HIP)
  require(MAX_SLOTS == 8464 && NBUCKETS == 8192, "HIP geometry preserved");
#elif defined(MOM_SYCL_ADAPTIVECPP)
  require(MAX_SLOTS == 8704 && NBUCKETS == 8192, "ACPP geometry preserved");
#else
  require(MAX_SLOTS == 4480 && NBUCKETS == 16384, "native/portable geometry preserved");
#endif
  for (uint32_t field : {0u, 1u, NBUCKETS - 1u, NBUCKETS, 0x1ffffffu}) {
    require(field_bucket(field) < NBUCKETS && field_rest(field) < NRESTBINS, "field bounds");
    require(field_from_parts(field_bucket(field), field_rest(field)) == field, "field roundtrip");
  }
  for (unsigned sample = 0; sample < 10000; ++sample) {
    uint32_t fields[5];
    for (auto &field : fields)
      field = random_word() & 0x1ffffffu;
    const uint32_t leaf = random_word() & 0x3ffffffu;
    uint32_t record[5];
    dense_l0_store(record, fields, leaf);
    require((record[0] & 0x1ffffffu) == fields[0], "dense active field");
    require(dense_l0_load_index(record) == leaf, "dense 26-bit leaf roundtrip");
    for (unsigned k = 0; k < 4; ++k) {
      require(load_follower<0, false>(record, k) == fields[k + 1], "dense followers");
      require(recovery_follower<false>(record, 0, k) == fields[k + 1], "recovery followers");
    }
#if defined(MOM_EQ_ALIGNED_GPU_RECORDS)
    uint32_t compact[4];
    dense_l0_store_compact(compact, fields);
    require(collision_rest<0, true>(compact) == field_rest(fields[0]), "compact rest");
    for (unsigned k = 0; k < 4; ++k)
      require(recovery_follower<true>(compact, 0, k) == fields[k + 1], "compact followers");
#endif
  }
  for (uint32_t leaf : {0u, 7u, 8u, 0x3ffffffu}) {
    const uint32_t fields[5] = {0x1ffffffu, 0x1ffffffu, 0x1ffffffu, 0x1ffffffu, 0x1ffffffu};
    uint32_t record[5];
    dense_l0_store(record, fields, leaf);
    require(dense_l0_load_index(record) == leaf, "boundary leaf roundtrip");
  }
#if defined(MOM_SYCL_HAS_CUDA)
  std::vector<bool> seen(static_cast<size_t>(64) * NBUCKETS);
  for (uint32_t slot = 0; slot < 64; ++slot)
    for (uint32_t bucket = 0; bucket < NBUCKETS; ++bucket) {
      const size_t physical = cuda_record_pos(bucket, slot);
      require(physical < seen.size() && !seen[physical], "CUDA tile bijection");
      seen[physical] = true;
    }
  require(cuda_record_pos(NBUCKETS - 1, MAX_SLOTS - 1) + 1 ==
              static_cast<size_t>(NBUCKETS) * MAX_SLOTS,
          "last CUDA slot extent");
#endif
  const unsigned capacity = 16;
  const unsigned bucket = 2, slot = 7;
  const size_t position = static_cast<size_t>(bucket) * capacity + slot;
  std::vector<uint32_t> records((position + 1) * level_u32(0, false));
  std::vector<uint32_t> indices(position + 1, 123456u);
  const uint32_t fields[5] = {};
  dense_l0_store(records.data() + position * level_u32(0, false), fields, 7654321u);
  require(recover_leaf<false>(records, indices, bucket, slot, capacity) == 7654321u,
          "noncompact recovery reads embedded index");
  require(recover_leaf<false>(records, indices, bucket, slot, capacity, false) == 0xabcdef12u,
          "invalid node does not touch leaf");
#if defined(MOM_EQ_ALIGNED_GPU_RECORDS)
#if defined(MOM_EQ_FORWARD_MAP)
  require(recover_leaf<true>(records, indices, bucket, slot, capacity) == (bucket << 14 | slot),
          "CUDA recovery produces forward inversion target");
#else
  require(recover_leaf<true>(records, indices, bucket, slot, capacity) == indices[position],
          "ACPP compact recovery reads separate index");
#endif
#endif
}
static void test_hash() {
  for (unsigned header_case = 0; header_case < 8; ++header_case) {
    uint8_t header[140];
    for (auto &byte : header)
      byte = static_cast<uint8_t>(random_word());
    const BaseState base = make_base_state(header);
    B2bPair paired[8];
    for (unsigned i = 0; i < 8; ++i)
      paired[i] = {static_cast<uint32_t>(base.h[i]), static_cast<uint32_t>(base.h[i] >> 32)};
    const B2bPair m0 = {b2b_load32le(base.pending), b2b_load32le(base.pending + 4)};
    const B2bPair m1 = {b2b_load32le(base.pending + 8), 0};
    for (uint32_t hash_index : {0u, 1u, 15u, 16u, 31u, 0xffffffu}) {
      uint32_t native[16], pairs[16];
      stock_hash_words(base.h, base.pending, hash_index, native);
      stock_hash_words_pair(paired, m0, m1, hash_index, pairs);
      require(std::equal(native, native + 16, pairs), "pair hash equals u64 reference");
    }
    for (unsigned block = 0; block < 4; ++block) {
      uint32_t accumulated[16] = {};
      for (unsigned lane = 0; lane < 16; ++lane) {
        const uint32_t g = block * 16 + lane;
        uint32_t stock[16], naive[16], masked[16];
        stock_hash_words_pair(paired, m0, m1, g, stock);
        for (unsigned word = 0; word < 16; ++word)
          accumulated[word] += stock[word];
        std::copy(accumulated, accumulated + 16, masked);
        for (unsigned word : {3u, 7u, 11u, 15u})
          masked[word] &= 0xf8ffffffu;
        zelhash_twist_naive(base.h, base.pending, g, naive);
        require(std::equal(masked, masked + 16, naive), "16-hash u32 prefix equals naive twist");
        for (unsigned sub = 0; sub < 4; ++sub) {
          uint32_t fields[5], reference_fields[5];
          uint8_t row[20];
          words_to_fields_25(masked + 4 * sub, fields);
          entry_row_naive(base.h, base.pending, g * 4 + sub, row);
          row_to_fields(row, reference_fields);
          require(std::equal(fields, fields + 5, reference_fields),
                  "word slicer equals expanded row");
        }
      }
    }
  }
}
int main() {
  test_layout();
  test_hash();
  std::printf("ZelHash source-bound CPU layout/recovery/hash passed: %u assertions\n", assertions);
}
