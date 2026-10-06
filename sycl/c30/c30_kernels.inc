// Device-side c30 primitives and exact-bucket kernels.
// Included only by c30.cpp; no shared SYCL helper is modified.

namespace {

constexpr std::uint32_t C30_EDGE_MASK = (1u << mom::c30::kEdgeBits) - 1u;
constexpr std::uint32_t C30_EDGE_BLOCK_MASK = 63u;
constexpr std::uint32_t C30_EDGE_BLOCK_SIZE = C30_EDGE_BLOCK_MASK + 1u;
#ifndef MOM_C30_BUCKET_BITS
#ifdef MOM_SYCL_HAS_HIP
#define MOM_C30_BUCKET_BITS 12
#else
#define MOM_C30_BUCKET_BITS 13
#endif
#endif
#ifndef MOM_C30_TRIM_LOCAL_SIZE
#define MOM_C30_TRIM_LOCAL_SIZE 512
#endif
constexpr std::uint32_t C30_SEED_LOCAL_SIZE = 128u;
// DPC++ Intel benefits from maximum independent SipHash chains; keep the established
// coarser launch for AdaptiveCpp's CUDA/HIP lanes.
#if defined(MOM_SYCL_ADAPTIVECPP) && defined(MOM_SYCL_HAS_HIP)
constexpr std::uint32_t C30_SEED_BLOCKS_PER_ITEM = 4u;
#elif defined(MOM_SYCL_ADAPTIVECPP)
constexpr std::uint32_t C30_SEED_BLOCKS_PER_ITEM = 16u;
#else
constexpr std::uint32_t C30_SEED_BLOCKS_PER_ITEM = 1u;
#endif
constexpr std::uint32_t C30_BUCKET_BITS = MOM_C30_BUCKET_BITS;
constexpr std::uint32_t C30_BUCKET_COUNT = 1u << C30_BUCKET_BITS;
constexpr std::uint32_t C30_COUNTER_WORDS = 1u << (mom::c30::kEdgeBits - C30_BUCKET_BITS - 5u);
constexpr std::uint32_t C30_COUNTER_WORDS_2 = C30_COUNTER_WORDS * 2u;
constexpr std::uint32_t C30_SPARSE_EDGE_CAPACITY = 512u;
constexpr std::uint32_t C30_SPARSE_TABLE_SIZE = 1024u;
constexpr std::uint32_t C30_SPARSE_TABLE_MASK = C30_SPARSE_TABLE_SIZE - 1u;
constexpr std::uint32_t C30_SPARSE_KEY_MASK = (1u << 19u) - 1u;
constexpr std::uint32_t C30_SPARSE_DUPLICATE = 1u << 31u;
#if defined(MOM_SYCL_ADAPTIVECPP) && !defined(MOM_SYCL_HAS_HIP)
constexpr std::size_t C30_SPARSE_LOCAL_SIZE = 64u;
#else
constexpr std::size_t C30_SPARSE_LOCAL_SIZE = 256u;
#endif
constexpr std::uint32_t C30_RECOVER_TABLE_SIZE = 128u;
constexpr std::uint32_t C30_RECOVER_TABLE_MASK = C30_RECOVER_TABLE_SIZE - 1u;

struct C30PackedEdge {
  std::uint16_t u_low, v_low, high;
};
static_assert(sizeof(C30PackedEdge) == 6);

inline C30PackedEdge c30_pack_edge(const std::uint32_t u, const std::uint32_t v) {
  const std::uint32_t u_high = u >> C30_BUCKET_BITS;
  return {static_cast<std::uint16_t>(u_high), static_cast<std::uint16_t>(v),
          static_cast<std::uint16_t>((v >> 16u) | ((u_high >> 16u) << 14u))};
}

inline std::uint32_t c30_unpack_u(const C30PackedEdge edge, const std::uint32_t bucket) {
  const std::uint32_t high = edge.u_low | (static_cast<std::uint32_t>(edge.high >> 14u) << 16u);
  return (high << C30_BUCKET_BITS) | bucket;
}

inline std::uint32_t c30_unpack_v(const C30PackedEdge edge) {
  return edge.v_low | (static_cast<std::uint32_t>(edge.high & 0x3fffu) << 16u);
}

struct C30DeviceKey {
  std::uint64_t k0;
  std::uint64_t k1;
  std::uint64_t k2;
  std::uint64_t k3;
};

struct C30U64 {
  std::uint32_t lo, hi;
};

inline C30U64 c30_pair(const std::uint64_t value) {
  return {static_cast<std::uint32_t>(value), static_cast<std::uint32_t>(value >> 32u)};
}
inline C30U64 c30_xor(const C30U64 a, const C30U64 b) {
  return {a.lo ^ b.lo, a.hi ^ b.hi};
}
inline C30U64 c30_add(const C30U64 a, const C30U64 b) {
  const std::uint32_t lo = a.lo + b.lo;
  return {lo, a.hi + b.hi + (lo < a.lo)};
}
inline C30U64 c30_rotate_left(const C30U64 value, const unsigned bits) {
  if (bits == 32u)
    return {value.hi, value.lo};
  return {(value.lo << bits) | (value.hi >> (32u - bits)),
          (value.hi << bits) | (value.lo >> (32u - bits))};
}

inline std::uint64_t c30_rotate_left(const std::uint64_t value, const unsigned bits) {
  return (value << bits) | (value >> (64u - bits));
}

inline void c30_sip_round(std::uint64_t& v0, std::uint64_t& v1, std::uint64_t& v2,
                          std::uint64_t& v3) {
  v0 += v1;
  v2 += v3;
  v1 = c30_rotate_left(v1, 13);
  v3 = c30_rotate_left(v3, 16);
  v1 ^= v0;
  v3 ^= v2;
  v0 = c30_rotate_left(v0, 32);
  v2 += v1;
  v0 += v3;
  v1 = c30_rotate_left(v1, 17);
  v3 = c30_rotate_left(v3, 21);
  v1 ^= v2;
  v3 ^= v0;
  v2 = c30_rotate_left(v2, 32);
}

inline void c30_sip_round(C30U64& v0, C30U64& v1, C30U64& v2, C30U64& v3) {
  v0 = c30_add(v0, v1);
  v2 = c30_add(v2, v3);
  v1 = c30_xor(c30_rotate_left(v1, 13), v0);
  v3 = c30_xor(c30_rotate_left(v3, 16), v2);
  v0 = c30_rotate_left(v0, 32);
  v2 = c30_add(v2, v1);
  v0 = c30_add(v0, v3);
  v1 = c30_xor(c30_rotate_left(v1, 17), v2);
  v3 = c30_xor(c30_rotate_left(v3, 21), v0);
  v2 = c30_rotate_left(v2, 32);
}

inline std::uint64_t c30_edge_hash(const C30DeviceKey key, const std::uint32_t nonce) {
  const std::uint32_t target = nonce & C30_EDGE_BLOCK_MASK;
  const std::uint64_t first = static_cast<std::uint64_t>(nonce & ~C30_EDGE_BLOCK_MASK);
  std::uint64_t v0 = key.k0;
  std::uint64_t v1 = key.k1;
  std::uint64_t v2 = key.k2;
  std::uint64_t v3 = key.k3;
  std::uint64_t selected = 0;
  std::uint64_t last = 0;
  for (std::uint32_t i = 0; i < 64; ++i) {
    const std::uint64_t value_nonce = first + i;
    v3 ^= value_nonce;
    for (unsigned round = 0; round < 4; ++round)
      c30_sip_round(v0, v1, v2, v3);
    v0 ^= value_nonce;
    v2 ^= 0xff;
    for (unsigned round = 0; round < 8; ++round)
      c30_sip_round(v0, v1, v2, v3);
    const std::uint64_t value = (v0 ^ v1) ^ (v2 ^ v3);
    if (i == target)
      selected = value;
    if (i == C30_EDGE_BLOCK_MASK)
      last = value;
  }
  return target == C30_EDGE_BLOCK_MASK ? last : selected ^ last;
}

inline std::uint32_t c30_endpoint(const std::uint64_t hash, const bool u_side) {
  return static_cast<std::uint32_t>(u_side ? hash : hash >> 32) & C30_EDGE_MASK;
}

inline std::uint32_t c30_other_endpoint(const std::uint64_t hash, const bool u_side) {
  return static_cast<std::uint32_t>(u_side ? hash >> 32 : hash) & C30_EDGE_MASK;
}

using C30GlobalAtomic =
    sycl::atomic_ref<std::uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                     sycl::access::address_space::global_space>;
using C30LocalAtomic =
    sycl::atomic_ref<std::uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
                     sycl::access::address_space::local_space>;

inline void c30_degree_add(sycl::local_accessor<std::uint32_t, 1> counters,
                           const std::uint32_t endpoint) {
  const std::uint32_t high = endpoint >> C30_BUCKET_BITS;
  const std::uint32_t bit = 1u << (high & 31u);
  const std::uint32_t word = high >> 5;
  const std::uint32_t old = C30LocalAtomic(counters[word]).fetch_or(bit);
  if (old & bit)
    C30LocalAtomic(counters[word + C30_COUNTER_WORDS]).fetch_or(bit);
}

inline bool c30_degree_two(const sycl::local_accessor<std::uint32_t, 1>& counters,
                           const std::uint32_t endpoint) {
  const std::uint32_t high = endpoint >> C30_BUCKET_BITS;
  return (counters[(high >> 5) + C30_COUNTER_WORDS] & (1u << (high & 31u))) != 0;
}

inline std::uint32_t c30_bucket(const std::uint32_t endpoint) {
  return endpoint & (C30_BUCKET_COUNT - 1u);
}

inline std::uint32_t c30_sparse_key(const std::uint32_t endpoint) {
  return (endpoint >> C30_BUCKET_BITS) + 1u;
}

inline std::uint32_t c30_sparse_slot(const std::uint32_t key) {
  return (key * 0x9e3779b1u) & C30_SPARSE_TABLE_MASK;
}

inline void c30_sparse_add(sycl::local_accessor<std::uint32_t, 1> table, const std::uint32_t key) {
  std::uint32_t slot = c30_sparse_slot(key);
  for (std::uint32_t probe = 0; probe < C30_SPARSE_TABLE_SIZE; ++probe) {
    C30LocalAtomic cell(table[slot]);
    std::uint32_t value = cell.load();
    if ((value & C30_SPARSE_KEY_MASK) == key) {
      cell.fetch_or(C30_SPARSE_DUPLICATE);
      return;
    }
    if (value == 0) {
      if (cell.compare_exchange_strong(value, key))
        return;
      if ((value & C30_SPARSE_KEY_MASK) == key) {
        cell.fetch_or(C30_SPARSE_DUPLICATE);
        return;
      }
    }
    slot = (slot + 1u) & C30_SPARSE_TABLE_MASK;
  }
}

inline bool c30_sparse_degree_two(const sycl::local_accessor<std::uint32_t, 1>& table,
                                  const std::uint32_t key) {
  std::uint32_t slot = c30_sparse_slot(key);
  for (std::uint32_t probe = 0; probe < C30_SPARSE_TABLE_SIZE; ++probe) {
    const std::uint32_t value = table[slot];
    if ((value & C30_SPARSE_KEY_MASK) == key)
      return (value & C30_SPARSE_DUPLICATE) != 0;
    if (value == 0)
      return false;
    slot = (slot + 1u) & C30_SPARSE_TABLE_MASK;
  }
  return false;
}

inline std::uint32_t c30_recover_slot(const std::uint32_t u, const std::uint32_t v) {
  std::uint32_t hash = u * 0x9e3779b1u;
  hash ^= v + 0x85ebca6bu + (hash << 6u) + (hash >> 2u);
  return hash & C30_RECOVER_TABLE_MASK;
}

inline void c30_recover_insert(sycl::local_accessor<std::uint32_t, 1> table,
                               const std::uint32_t* target_u, const std::uint32_t* target_v,
                               const std::uint32_t index) {
  std::uint32_t slot = c30_recover_slot(target_u[index], target_v[index]);
  for (std::uint32_t probe = 0; probe < C30_RECOVER_TABLE_SIZE; ++probe) {
    C30LocalAtomic cell(table[slot]);
    std::uint32_t empty = 0;
    if (cell.compare_exchange_strong(empty, index + 1u))
      return;
    slot = (slot + 1u) & C30_RECOVER_TABLE_MASK;
  }
}

inline void c30_recover_matches(const sycl::local_accessor<std::uint32_t, 1>& table,
                                const std::uint32_t* target_u, const std::uint32_t* target_v,
                                std::uint32_t* recovered, const std::uint32_t u,
                                const std::uint32_t v, const std::uint32_t nonce) {
  std::uint32_t slot = c30_recover_slot(u, v);
  for (std::uint32_t probe = 0; probe < C30_RECOVER_TABLE_SIZE; ++probe) {
    const std::uint32_t entry = table[slot];
    if (entry == 0)
      return;
    const std::uint32_t index = entry - 1u;
    if (u == target_u[index] && v == target_v[index])
      C30GlobalAtomic(recovered[index]).store(nonce);
    slot = (slot + 1u) & C30_RECOVER_TABLE_MASK;
  }
}

// Cortex derives all 64 edges in a block from one stateful SipHash-4/8 chain.  Keep this
// sequential dependency in one work-item, but publish the complete block to the other 63 lanes.
// The old per-nonce kernel called this chain once for every nonce, repeating the same 64-state
// walk roughly 64 times.
inline void c30_sip_block(const C30DeviceKey key, const std::uint32_t first,
                          std::uint64_t* values) {
  std::uint64_t v0 = key.k0;
  std::uint64_t v1 = key.k1;
  std::uint64_t v2 = key.k2;
  std::uint64_t v3 = key.k3;
  for (std::uint32_t i = 0; i < C30_EDGE_BLOCK_SIZE; ++i) {
    const std::uint64_t value_nonce = static_cast<std::uint64_t>(first) + i;
    v3 ^= value_nonce;
    for (unsigned round = 0; round < 4; ++round)
      c30_sip_round(v0, v1, v2, v3);
    v0 ^= value_nonce;
    v2 ^= 0xff;
    for (unsigned round = 0; round < 8; ++round)
      c30_sip_round(v0, v1, v2, v3);
    values[i] = (v0 ^ v1) ^ (v2 ^ v3);
  }
}

inline void c30_sip_block32(const C30DeviceKey key, const std::uint32_t first, C30U64* values) {
  C30U64 v0 = c30_pair(key.k0), v1 = c30_pair(key.k1);
  C30U64 v2 = c30_pair(key.k2), v3 = c30_pair(key.k3);
  for (std::uint32_t i = 0; i < C30_EDGE_BLOCK_SIZE; ++i) {
    const C30U64 nonce{first + i, 0};
    v3 = c30_xor(v3, nonce);
    for (unsigned round = 0; round < 4; ++round)
      c30_sip_round(v0, v1, v2, v3);
    v0 = c30_xor(v0, nonce);
    v2.lo ^= 0xffu;
    for (unsigned round = 0; round < 8; ++round)
      c30_sip_round(v0, v1, v2, v3);
    values[i] = c30_xor(c30_xor(v0, v1), c30_xor(v2, v3));
  }
}

sycl::event c30_seed_fixed(sycl::queue& queue, const C30DeviceKey key, std::uint32_t* counts,
                           const std::uint32_t output_stride, sycl::uint2* output_edges,
                           std::uint32_t* overflow) {
  constexpr std::size_t local_size = C30_SEED_LOCAL_SIZE;
  constexpr std::size_t work_items =
      (1ull << mom::c30::kEdgeBits) / (C30_EDGE_BLOCK_SIZE * C30_SEED_BLOCKS_PER_ITEM);
  return queue.submit([&](sycl::handler& handler) {
    handler.parallel_for(
        sycl::nd_range<1>{sycl::range<1>{work_items}, sycl::range<1>{local_size}},
        [=](sycl::nd_item<1> item) MOM_SYCL_KERNEL_ARGS_RESTRICT {
          const std::uint32_t work_item = item.get_global_id(0);
          for (std::uint32_t block = 0; block < C30_SEED_BLOCKS_PER_ITEM; ++block) {
            const std::uint32_t first =
                (work_item * C30_SEED_BLOCKS_PER_ITEM + block) * C30_EDGE_BLOCK_SIZE;
            std::uint64_t values[C30_EDGE_BLOCK_SIZE];
            c30_sip_block(key, first, values);
            const std::uint64_t last = values[C30_EDGE_BLOCK_MASK];
            for (std::uint32_t i = 0; i < C30_EDGE_BLOCK_SIZE; ++i) {
              const std::uint64_t hash = i == C30_EDGE_BLOCK_MASK ? last : values[i] ^ last;
              const std::uint32_t u = c30_endpoint(hash, true);
              const std::uint32_t bucket = c30_bucket(u);
              const std::uint32_t slot = C30GlobalAtomic(counts[bucket]).fetch_add(1u);
              if (slot >= output_stride) {
                C30GlobalAtomic(*overflow).fetch_add(1u);
                continue;
              }
              const std::uint32_t output = bucket * output_stride + slot;
              output_edges[output] = sycl::uint2{u, c30_other_endpoint(hash, true)};
            }
          }
        });
  });
}

sycl::event c30_seed_packed(sycl::queue& queue, const C30DeviceKey key, std::uint32_t* counts,
                            const std::uint32_t output_stride, C30PackedEdge* output_edges,
                            std::uint32_t* overflow) {
  constexpr std::size_t local_size = C30_SEED_LOCAL_SIZE;
  constexpr std::size_t work_items =
      (1ull << mom::c30::kEdgeBits) / (C30_EDGE_BLOCK_SIZE * C30_SEED_BLOCKS_PER_ITEM);
  return queue.submit([&](sycl::handler& handler) {
    handler.parallel_for(
        sycl::nd_range<1>{sycl::range<1>{work_items}, sycl::range<1>{local_size}},
        [=](sycl::nd_item<1> item) MOM_SYCL_KERNEL_ARGS_RESTRICT {
          const std::uint32_t work_item = item.get_global_id(0);
          for (std::uint32_t block = 0; block < C30_SEED_BLOCKS_PER_ITEM; ++block) {
            const std::uint32_t first =
                (work_item * C30_SEED_BLOCKS_PER_ITEM + block) * C30_EDGE_BLOCK_SIZE;
            std::uint64_t values[C30_EDGE_BLOCK_SIZE];
            c30_sip_block(key, first, values);
            const std::uint64_t last = values[C30_EDGE_BLOCK_MASK];
            for (std::uint32_t i = 0; i < C30_EDGE_BLOCK_SIZE; ++i) {
              const std::uint64_t hash = i == C30_EDGE_BLOCK_MASK ? last : values[i] ^ last;
              const std::uint32_t u = c30_endpoint(hash, true), v = c30_other_endpoint(hash, true);
              const std::uint32_t bucket = c30_bucket(u);
              const std::uint32_t slot = C30GlobalAtomic(counts[bucket]).fetch_add(1u);
              if (slot >= output_stride) {
                C30GlobalAtomic(*overflow).fetch_add(1u);
                continue;
              }
              output_edges[bucket * output_stride + slot] = c30_pack_edge(u, v);
            }
          }
        });
  });
}

sycl::event c30_trim_packed(sycl::queue& queue, const C30PackedEdge* input_edges,
                            const std::uint32_t* input_counts,
                            const std::uint32_t input_stride, std::uint32_t* output_counts,
                            const std::uint32_t output_stride, C30PackedEdge* output_edges,
                            std::uint32_t* overflow) {
  const std::size_t local_size = MOM_C30_TRIM_LOCAL_SIZE;
  return queue.submit([&](sycl::handler& handler) {
    sycl::local_accessor<std::uint32_t, 1> counters{sycl::range<1>{C30_COUNTER_WORDS_2}, handler};
    handler.parallel_for(
        sycl::nd_range<1>{sycl::range<1>{static_cast<std::size_t>(C30_BUCKET_COUNT) * local_size},
                          sycl::range<1>{local_size}},
        [=](sycl::nd_item<1> item) MOM_SYCL_KERNEL_ARGS_RESTRICT {
          const std::uint32_t bucket = item.get_group(0), lane = item.get_local_id(0);
          for (std::uint32_t i = lane; i < C30_COUNTER_WORDS_2; i += local_size)
            counters[i] = 0;
          sycl::group_barrier(item.get_group());
          const std::uint32_t begin = bucket * input_stride;
          const std::uint32_t end = begin + sycl::min(input_counts[bucket], input_stride);
          for (std::uint32_t i = begin + lane; i < end; i += local_size)
            c30_degree_add(counters, c30_unpack_u(input_edges[i], bucket));
          sycl::group_barrier(item.get_group());
          for (std::uint32_t i = begin + lane; i < end; i += local_size) {
            const C30PackedEdge edge = input_edges[i];
            const std::uint32_t u = c30_unpack_u(edge, bucket);
            if (!c30_degree_two(counters, u))
              continue;
            const std::uint32_t v = c30_unpack_v(edge);
            const std::uint32_t next_bucket = c30_bucket(v);
            const std::uint32_t slot = C30GlobalAtomic(output_counts[next_bucket]).fetch_add(1u);
            if (slot >= output_stride) {
              C30GlobalAtomic(*overflow).fetch_add(1u);
              continue;
            }
            output_edges[next_bucket * output_stride + slot] = c30_pack_edge(v, u);
          }
        });
  });
}

sycl::event c30_trim_sparse_packed(sycl::queue& queue, const C30PackedEdge* input_edges,
                                   const std::uint32_t* input_counts,
                                   const std::uint32_t input_stride, std::uint32_t* output_counts,
                                   const std::uint32_t output_stride,
                                   C30PackedEdge* output_edges, std::uint32_t* overflow) {
  constexpr std::size_t local_size = C30_SPARSE_LOCAL_SIZE;
  return queue.submit([&](sycl::handler& handler) {
    sycl::local_accessor<std::uint32_t, 1> table{sycl::range<1>{C30_SPARSE_TABLE_SIZE}, handler};
    handler.parallel_for(
        sycl::nd_range<1>{sycl::range<1>{static_cast<std::size_t>(C30_BUCKET_COUNT) * local_size},
                          sycl::range<1>{local_size}},
        [=](sycl::nd_item<1> item) MOM_SYCL_KERNEL_ARGS_RESTRICT {
          const std::uint32_t bucket = item.get_group(0), lane = item.get_local_id(0);
          const std::uint32_t begin = bucket * input_stride;
          const std::uint32_t count = sycl::min(input_counts[bucket], input_stride);
          const bool use_table = count <= C30_SPARSE_EDGE_CAPACITY;
          if (use_table) {
            for (std::uint32_t slot = lane; slot < C30_SPARSE_TABLE_SIZE; slot += local_size)
              table[slot] = 0;
          }
          sycl::group_barrier(item.get_group());
          if (use_table)
            for (std::uint32_t slot = lane; slot < count; slot += local_size)
              c30_sparse_add(table,
                             c30_sparse_key(c30_unpack_u(input_edges[begin + slot], bucket)));
          sycl::group_barrier(item.get_group());
          for (std::uint32_t slot = lane; slot < count; slot += local_size) {
            const C30PackedEdge edge = input_edges[begin + slot];
            const std::uint32_t u = c30_unpack_u(edge, bucket);
            bool degree_two = use_table && c30_sparse_degree_two(table, c30_sparse_key(u));
            if (!use_table) {
              for (std::uint32_t other = 0; other < count; ++other) {
                if (other == slot)
                  continue;
                if (c30_unpack_u(input_edges[begin + other], bucket) == u) {
                  degree_two = true;
                  break;
                }
              }
            }
            if (!degree_two)
              continue;
            const std::uint32_t v = c30_unpack_v(edge);
            const std::uint32_t next_bucket = c30_bucket(v);
            const std::uint32_t output_slot =
                C30GlobalAtomic(output_counts[next_bucket]).fetch_add(1u);
            if (output_slot >= output_stride) {
              C30GlobalAtomic(*overflow).fetch_add(1u);
              continue;
            }
            output_edges[next_bucket * output_stride + output_slot] = c30_pack_edge(v, u);
          }
        });
  });
}

sycl::event c30_compact_packed(sycl::queue& queue, const C30PackedEdge* input_edges,
                               const std::uint32_t* input_counts,
                               const std::uint32_t input_stride, sycl::uint2* output_edges,
                               std::uint32_t* output_count) {
  constexpr std::size_t local_size = 256;
  return queue.submit([&](sycl::handler& handler) {
    handler.parallel_for(
        sycl::nd_range<1>{sycl::range<1>{static_cast<std::size_t>(C30_BUCKET_COUNT) * local_size},
                          sycl::range<1>{local_size}},
        [=](sycl::nd_item<1> item) MOM_SYCL_KERNEL_ARGS_RESTRICT {
          const std::uint32_t bucket = item.get_group(0);
          const std::uint32_t begin = bucket * input_stride;
          const std::uint32_t count = sycl::min(input_counts[bucket], input_stride);
          for (std::uint32_t slot = item.get_local_id(0); slot < count;
               slot += item.get_local_range(0)) {
            const C30PackedEdge edge = input_edges[begin + slot];
            const std::uint32_t output = C30GlobalAtomic(*output_count).fetch_add(1u);
            output_edges[output] = sycl::uint2{c30_unpack_u(edge, bucket), c30_unpack_v(edge)};
          }
        });
  });
}

sycl::event c30_trim_fixed(sycl::queue& queue, const sycl::uint2* input_edges,
                           const std::uint32_t* input_counts, const std::uint32_t input_stride,
                           std::uint32_t* output_counts, const std::uint32_t output_stride,
                           sycl::uint2* output_edges, std::uint32_t* overflow) {
  const std::size_t local_size = MOM_C30_TRIM_LOCAL_SIZE;
  return queue.submit([&](sycl::handler& handler) {
    sycl::local_accessor<std::uint32_t, 1> counters{sycl::range<1>{C30_COUNTER_WORDS_2}, handler};
    handler.parallel_for(
        sycl::nd_range<1>{sycl::range<1>{static_cast<std::size_t>(C30_BUCKET_COUNT) * local_size},
                          sycl::range<1>{local_size}},
        [=](sycl::nd_item<1> item) MOM_SYCL_KERNEL_ARGS_RESTRICT {
          const std::uint32_t bucket = item.get_group(0);
          const std::uint32_t lane = item.get_local_id(0);
          for (std::uint32_t i = lane; i < C30_COUNTER_WORDS_2; i += local_size)
            counters[i] = 0;
          sycl::group_barrier(item.get_group());
          const std::uint32_t begin = bucket * input_stride;
          const std::uint32_t end = begin + sycl::min(input_counts[bucket], input_stride);
          for (std::uint32_t i = begin + lane; i < end; i += local_size)
            c30_degree_add(counters, input_edges[i].x());
          sycl::group_barrier(item.get_group());
          for (std::uint32_t i = begin + lane; i < end; i += local_size) {
            const sycl::uint2 edge = input_edges[i];
            const std::uint32_t endpoint = edge.x();
            if (!c30_degree_two(counters, endpoint))
              continue;
            const std::uint32_t next_bucket = c30_bucket(edge.y());
            const std::uint32_t slot = C30GlobalAtomic(output_counts[next_bucket]).fetch_add(1u);
            if (slot >= output_stride) {
              C30GlobalAtomic(*overflow).fetch_add(1u);
              continue;
            }
            const std::uint32_t output = next_bucket * output_stride + slot;
            output_edges[output] = sycl::uint2{edge.y(), edge.x()};
          }
        });
  });
}

sycl::event c30_trim_sparse_fixed(sycl::queue& queue, const sycl::uint2* input_edges,
                                  const std::uint32_t* input_counts,
                                  const std::uint32_t input_stride,
                                  std::uint32_t* output_counts,
                                  const std::uint32_t output_stride, sycl::uint2* output_edges,
                                  std::uint32_t* overflow) {
  constexpr std::size_t local_size = C30_SPARSE_LOCAL_SIZE;
  return queue.submit([&](sycl::handler& handler) {
    sycl::local_accessor<std::uint32_t, 1> table{sycl::range<1>{C30_SPARSE_TABLE_SIZE}, handler};
    handler.parallel_for(
        sycl::nd_range<1>{sycl::range<1>{static_cast<std::size_t>(C30_BUCKET_COUNT) * local_size},
                          sycl::range<1>{local_size}},
        [=](sycl::nd_item<1> item) MOM_SYCL_KERNEL_ARGS_RESTRICT {
          const std::uint32_t bucket = item.get_group(0), lane = item.get_local_id(0);
          const std::uint32_t begin = bucket * input_stride;
          const std::uint32_t count = sycl::min(input_counts[bucket], input_stride);
          const bool use_table = count <= C30_SPARSE_EDGE_CAPACITY;
          if (use_table) {
            for (std::uint32_t slot = lane; slot < C30_SPARSE_TABLE_SIZE; slot += local_size)
              table[slot] = 0;
          }
          sycl::group_barrier(item.get_group());
          if (use_table)
            for (std::uint32_t slot = lane; slot < count; slot += local_size)
              c30_sparse_add(table, c30_sparse_key(input_edges[begin + slot].x()));
          sycl::group_barrier(item.get_group());
          for (std::uint32_t slot = lane; slot < count; slot += local_size) {
            const sycl::uint2 edge = input_edges[begin + slot];
            const std::uint32_t u = edge.x();
            bool degree_two = use_table && c30_sparse_degree_two(table, c30_sparse_key(u));
            if (!use_table) {
              for (std::uint32_t other = 0; other < count; ++other) {
                if (other == slot)
                  continue;
                if (input_edges[begin + other].x() == u) {
                  degree_two = true;
                  break;
                }
              }
            }
            if (!degree_two)
              continue;
            const std::uint32_t next_bucket = c30_bucket(edge.y());
            const std::uint32_t output_slot =
                C30GlobalAtomic(output_counts[next_bucket]).fetch_add(1u);
            if (output_slot >= output_stride) {
              C30GlobalAtomic(*overflow).fetch_add(1u);
              continue;
            }
            output_edges[next_bucket * output_stride + output_slot] =
                sycl::uint2{edge.y(), edge.x()};
          }
        });
  });
}

sycl::event c30_compact(sycl::queue& queue, const sycl::uint2* input_edges,
                        const std::uint32_t* input_counts, const std::uint32_t input_stride,
                        sycl::uint2* output_edges, std::uint32_t* output_count) {
  const std::size_t local_size = 256;
  return queue.submit([&](sycl::handler& handler) {
    handler.parallel_for(
        sycl::nd_range<1>{sycl::range<1>{static_cast<std::size_t>(C30_BUCKET_COUNT) * local_size},
                          sycl::range<1>{local_size}},
        [=](sycl::nd_item<1> item) MOM_SYCL_KERNEL_ARGS_RESTRICT {
          const std::uint32_t bucket = item.get_group(0);
          const std::uint32_t begin = bucket * input_stride;
          const std::uint32_t count = sycl::min(input_counts[bucket], input_stride);
          for (std::uint32_t slot = item.get_local_id(0); slot < count;
               slot += item.get_local_range(0)) {
            const std::uint32_t output = C30GlobalAtomic(*output_count).fetch_add(1u);
            output_edges[output] = input_edges[begin + slot];
          }
        });
  });
}

sycl::event c30_recover(sycl::queue& queue, const C30DeviceKey key,
                        const std::uint32_t* target_u, const std::uint32_t* target_v,
                        std::uint32_t* recovered) {
  constexpr std::size_t local_size = C30_SEED_LOCAL_SIZE;
  constexpr std::size_t block_count = 1ull << (mom::c30::kEdgeBits - 6u);
  return queue.submit([&](sycl::handler& handler) {
    sycl::local_accessor<std::uint32_t, 1> table{sycl::range<1>{C30_RECOVER_TABLE_SIZE}, handler};
    handler.parallel_for(
        sycl::nd_range<1>{sycl::range<1>{block_count}, sycl::range<1>{local_size}},
        [=](sycl::nd_item<1> item) MOM_SYCL_KERNEL_ARGS_RESTRICT {
          const std::uint32_t lane = item.get_local_id(0);
          for (std::uint32_t slot = lane; slot < C30_RECOVER_TABLE_SIZE; slot += local_size)
            table[slot] = 0;
          sycl::group_barrier(item.get_group());
          for (std::uint32_t index = lane; index < mom::c30::kProofSize; index += local_size)
            c30_recover_insert(table, target_u, target_v, index);
          sycl::group_barrier(item.get_group());
          const std::uint32_t block = item.get_global_id(0);
          std::uint64_t values[C30_EDGE_BLOCK_SIZE];
          c30_sip_block(key, block * C30_EDGE_BLOCK_SIZE, values);
          const std::uint64_t last = values[C30_EDGE_BLOCK_MASK];
          for (std::uint32_t edge = 0; edge < C30_EDGE_BLOCK_SIZE; ++edge) {
            const std::uint64_t hash = edge == C30_EDGE_BLOCK_MASK ? last : values[edge] ^ last;
            const std::uint32_t u = c30_endpoint(hash, true);
            const std::uint32_t v = c30_other_endpoint(hash, true);
            c30_recover_matches(table, target_u, target_v, recovered, u, v,
                                block * C30_EDGE_BLOCK_SIZE + edge);
          }
        });
  });
}

sycl::event c30_recover32(sycl::queue& queue, const C30DeviceKey key,
                          const std::uint32_t* target_u, const std::uint32_t* target_v,
                          std::uint32_t* recovered) {
  constexpr std::size_t block_count = 1ull << (mom::c30::kEdgeBits - 6u);
  return queue.submit([&](sycl::handler& handler) {
    sycl::local_accessor<std::uint32_t, 1> table{sycl::range<1>{C30_RECOVER_TABLE_SIZE}, handler};
    handler.parallel_for(
        sycl::nd_range<1>{sycl::range<1>{block_count}, sycl::range<1>{C30_SEED_LOCAL_SIZE}},
        [=](sycl::nd_item<1> item) MOM_SYCL_KERNEL_ARGS_RESTRICT {
          const std::uint32_t lane = item.get_local_id(0);
          for (std::uint32_t slot = lane; slot < C30_RECOVER_TABLE_SIZE;
               slot += C30_SEED_LOCAL_SIZE)
            table[slot] = 0;
          sycl::group_barrier(item.get_group());
          for (std::uint32_t index = lane; index < mom::c30::kProofSize;
               index += C30_SEED_LOCAL_SIZE)
            c30_recover_insert(table, target_u, target_v, index);
          sycl::group_barrier(item.get_group());
          const std::uint32_t block = item.get_global_id(0);
          C30U64 values[C30_EDGE_BLOCK_SIZE];
          c30_sip_block32(key, block * C30_EDGE_BLOCK_SIZE, values);
          const C30U64 last = values[C30_EDGE_BLOCK_MASK];
          for (std::uint32_t edge = 0; edge < C30_EDGE_BLOCK_SIZE; ++edge) {
            const C30U64 hash = edge == C30_EDGE_BLOCK_MASK ? last : c30_xor(values[edge], last);
            const std::uint32_t u = hash.lo & C30_EDGE_MASK, v = hash.hi & C30_EDGE_MASK;
            c30_recover_matches(table, target_u, target_v, recovered, u, v,
                                block * C30_EDGE_BLOCK_SIZE + edge);
          }
        });
  });
}

} // namespace
