// Reuse the persistent GPU buffers across graphs, reinterpreted as uint2 edge pairs.
auto buffer_a1_u2 = c29_buffers.buffer_a1.reinterpret<sycl::uint2>(sycl::range(BUFFER_SIZE_A1 / 2));
auto buffer_a2_u2 = c29_buffers.buffer_a2.reinterpret<sycl::uint2>(sycl::range(BUFFER_SIZE_A2 / 2));
auto buffer_b_u2 =
    c29_buffers.buffer_b.reinterpret<sycl::uint2>(sycl::range(c29_buffers.buffer_b.size() / 2));
sycl::buffer<uint32_t, 1>& buffer_i1 = c29_buffers.buffer_i1;
sycl::buffer<uint32_t, 1>& buffer_i2 = c29_buffers.buffer_i2;
sycl::buffer<uint32_t, 1>& buffer_trimmed_edge_count = c29_buffers.buffer_trimmed_edge_count;
sycl::buffer<sycl::uint2, 1>& buffer_trimmed_edges_u2 = c29_buffers.buffer_trimmed_edges_u2;

profile.mark("buffers ready");

auto zero_buffer = [&](const char* name, sycl::buffer<uint32_t, 1>& buffer) {
  const sycl::event event = compute_queue.submit([&](sycl::handler& handler) {
    sycl::accessor accessor{buffer, handler, sycl::write_only, sycl::no_init};
    handler.fill(accessor, static_cast<uint32_t>(0));
  });
  profile.add_event(name, event);
};

zero_buffer("clear_i1_start", buffer_i1);
zero_buffer("clear_i2_start", buffer_i2);

const uint32_t duck_edges_b = static_cast<uint32_t>(c29_buffers.buffer_b.size() / (2 * 4096));

using local_atomic_ref =
    sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
                     sycl::access::address_space::local_space>;

// FluffySeed2A: Generate initial edges using SipHash cryptographic function
const uint32_t seed_local_size = c29_seed_local_size();
const uint32_t seed_blocks_per_item = c29_seed_blocks_per_item();
const uint32_t seed_work_items = NUM_EDGES / (EDGE_BLOCK_SIZE * seed_blocks_per_item);
#ifdef MOM_C29_INTEL
constexpr bool intel_gpu = true;
#else
const sycl::device c29_device = compute_queue.get_device();
const bool intel_gpu =
    c29_device.is_gpu() &&
    c29_device.get_info<sycl::info::device::vendor>().find("Intel") != std::string::npos;
#endif
const uint32_t initial_bucket_bits = intel_gpu ? 14u : 12u;
const uint32_t initial_bucket_count = 1u << initial_bucket_bits;
uint32_t initial_bucket_edges =
    static_cast<uint32_t>((BUFFER_SIZE_A1 + BUFFER_SIZE_A2) / (2 * initial_bucket_count));
uint32_t initial_a1_buckets;
// A bucket holds a whole number of 16-byte edge pairs, and its byte stride is 16 times an odd
// number: strides with many factors of two put every bucket's write position on the same cache sets
// (on AMD a 2^14-multiple stride costs 6 ms in every trim round).
for (;;) {
  initial_a1_buckets =
      initial_bucket_count - static_cast<uint32_t>((BUFFER_SIZE_A2 / 2) / initial_bucket_edges);
  if ((initial_bucket_edges & 3u) == 2u &&
      static_cast<uint64_t>(initial_a1_buckets) * initial_bucket_edges <= BUFFER_SIZE_A1 / 2)
    break;
  --initial_bucket_edges;
}
const uint32_t trim_bucket_bits = intel_gpu ? 14u : 12u;
const uint32_t trim_bucket_count = 1u << trim_bucket_bits;
const uint32_t trim_bucket_mask = trim_bucket_count - 1;
// Two edges fewer keeps the byte stride at 16 times an odd number.
const uint32_t trim_bucket_edges = (duck_edges_b >> (trim_bucket_bits - 12)) - 2u;
const uint32_t trim_counter_words = 1u << (EDGE_BITS - trim_bucket_bits - 4);
const bool cuda_gpu = mom_is_cuda(device), hip_gpu = mom_is_hip(device);
const uint32_t trim_threads = intel_gpu ? 256u : hip_gpu ? 512u : COMPUTE_THREADS;
const uint32_t round_no1_threads = COMPUTE_THREADS;

// Paired stores. A scattered global store costs the same whether it carries 4, 8 or 16 bytes, and
// on NVIDIA and AMD these stores, not hashing or the bucket counters, dominate seeding (44 of 63 ms
// on an RTX 5060 Ti). So a work group pairs edges bound for the same bucket and writes each pair
// with one 16-byte store. An edge waits for its partner in the bucket's parking slot in local
// memory:
//   tag word:  0 = empty, ~0 = held by a thread, otherwise v + 1 of the parked edge;
//   data word: u of the parked edge, read or written only by the thread holding the tag.
// A thread that finds the slot held never waits (a lane spinning on another lane's progress can
// stall its whole warp or wave); it writes its edge alone, padded with a zero edge. Edges still
// parked when the work group ends are flushed the same way. A zero edge is a hole that every reader
// skips, as the all-zero edge is never stored. Intel gains nothing from this: its seed is not bound
// by these stores, and its 16384 buckets would need 128 KiB of slots.
// The holes must fit in buffer A's spare. A work group leaves up to one unpaired edge per bucket,
// so it has to bring at least 32 edges per bucket: the selectable settings that do add 1.1% to 3.3%
// of slots, those with 16 edges per bucket up to 5.1%.
const uint32_t parking_words = 2u * initial_bucket_count;
const bool seed_parking = (cuda_gpu || hip_gpu) &&
    seed_local_size * seed_blocks_per_item * EDGE_BLOCK_SIZE >= 32u * initial_bucket_count &&
    device.get_info<sycl::info::device::local_mem_size>() >= parking_words * sizeof(uint32_t);
// Walk each SipHash block twice instead of keeping its 64 hashes in thread-private memory. On Intel
// reading them back costs more than the second walk (seed 49 -> 44 ms on an Arc B580); on AMD the
// second walk costs 11 ms. Like the rest of the Intel tuning this follows the build flag; a CUDA
// image in a combined build takes the same walk at no measured cost.
#ifdef MOM_C29_INTEL
constexpr bool seed_two_pass = true;
#else
constexpr bool seed_two_pass = false;
#endif
// The first trim round is store-bound in the same way on NVIDIA only; on AMD pairing costs what it
// saves. Its holes need the larger buffer B (see C29Buffers).
const bool round1_parking = cuda_gpu && seed_parking && trim_bucket_bits == 12 &&
    duck_edges_b >= DUCK_SIZE_B_PAIRED * 1024 &&
    device.get_info<sycl::info::device::local_mem_size>() >=
        (2u * trim_bucket_count + (1u << (EDGE_BITS - initial_bucket_bits - 4))) * sizeof(uint32_t);

const sycl::event seed_event = compute_queue.submit([&](sycl::handler& handler) {
  sycl::accessor acc_buffer_a1{buffer_a1_u2, handler, sycl::write_only, sycl::no_init};
  sycl::accessor acc_buffer_a2{buffer_a2_u2, handler, sycl::write_only, sycl::no_init};
  sycl::accessor acc_index_2{buffer_i2, handler, sycl::read_write};
  sycl::local_accessor<uint32_t, 1> parking{sycl::range<1>(seed_parking ? parking_words : 2u),
                                            handler};
  // Intel Xe2 otherwise chooses SIMD32 here, which is slower for the two-pass walk (47 against 44
  // ms on an Arc B580) and doubles the hardware thread's private allocation where the per-lane
  // 64-value block remains. Other GPU toolchains leave the width unconstrained.
  handler.parallel_for(
      sycl::nd_range<1>(sycl::range<1>(seed_work_items), sycl::range<1>(seed_local_size)),
      [=](sycl::nd_item<1> item) MOM_REQD_SG_16 MOM_SYCL_KERNEL_ARGS_RESTRICT {
        const uint32_t global_id = item.get_global_id(0);
        const uint32_t local_id = item.get_local_id(0);
        // Store two edges of bucket `bucket_id` (the second may be the zero edge) with one 16-byte
        // store.
        const auto store_pair = [&](const uint32_t bucket_id, const uint32_t u0, const uint32_t v0,
                                    const uint32_t u1, const uint32_t v1) {
          const uint32_t bucket_index = c29_global_atomic(acc_index_2, bucket_id).fetch_add(2);
          if (bucket_index + 1 < initial_bucket_edges) {
            const uint32_t a2_bucket = bucket_id - initial_a1_buckets;
            sycl::uint2* const slot = bucket_id < initial_a1_buckets
                ? &acc_buffer_a1[bucket_id * initial_bucket_edges + bucket_index]
                : &acc_buffer_a2[a2_bucket * initial_bucket_edges + bucket_index];
            *reinterpret_cast<sycl::uint4*>(slot) = sycl::uint4(u0, v0, u1, v1);
          }
        };
        if (seed_parking) {
          for (uint32_t word = local_id; word < parking_words; word += seed_local_size)
            parking[word] = 0;
          item.barrier(sycl::access::fence_space::local_space);
        }
        // Bucket one edge by u, then pair it or store it.
        const auto place_edge = [&](const uint64_t hash_lookup) {
          const uint32_t edge_u = static_cast<uint32_t>(hash_lookup & EDGE_MASK);
          const uint32_t edge_v = static_cast<uint32_t>((hash_lookup >> 32) & EDGE_MASK);
          if (!(edge_u || edge_v))
            return;

          const uint32_t low_mask = (1u << (initial_bucket_bits - 6)) - 1;
          const uint32_t bucket_id =
              ((edge_u & low_mask) << 6) | ((edge_u >> (initial_bucket_bits - 6)) & 63u);
          if (seed_parking) {
            const uint32_t tag = local_atomic_ref(parking[2 * bucket_id]).exchange(0xffffffffu);
            if (tag == 0) {   // empty: park this edge
              parking[2 * bucket_id + 1] = edge_u;
              local_atomic_ref(parking[2 * bucket_id]).store(edge_v + 1);
            } else if (tag == 0xffffffffu) {   // held by another thread: do not wait
              store_pair(bucket_id, edge_u, edge_v, 0, 0);
            } else {   // take the parked edge and write the pair
              const uint32_t parked_u = parking[2 * bucket_id + 1];
              local_atomic_ref(parking[2 * bucket_id]).store(0);
              store_pair(bucket_id, edge_u, edge_v, parked_u, tag - 1);
            }
            return;
          }
          const uint32_t bucket_index = c29_global_atomic(acc_index_2, bucket_id).fetch_add(1);
          if (bucket_index < initial_bucket_edges) {
            if (bucket_id < initial_a1_buckets)
              acc_buffer_a1[bucket_id * initial_bucket_edges + bucket_index] =
                  sycl::uint2(edge_u, edge_v);
            else
              acc_buffer_a2[(bucket_id - initial_a1_buckets) * initial_bucket_edges +
                            bucket_index] = sycl::uint2(edge_u, edge_v);
          }
        };
        for (uint32_t block_offset = 0; block_offset < seed_blocks_per_item * EDGE_BLOCK_SIZE;
             block_offset += EDGE_BLOCK_SIZE) {
          const uint64_t base_nonce =
              global_id * (seed_blocks_per_item * EDGE_BLOCK_SIZE) + block_offset;
          uint64_t sip_v0 = seed_k0, sip_v1 = seed_k1, sip_v2 = seed_k2, sip_v3 = seed_k3;
          if (seed_two_pass) {
            // Every edge of a block is XORed with the block's last hash. Walking the block twice
            // (first for that last hash) keeps the 64 hashes out of thread-private memory.
            uint64_t hash_last = 0;
            for (uint32_t step = 0; step < EDGE_BLOCK_SIZE; step++)
              hash_last = siphash_step(sip_v0, sip_v1, sip_v2, sip_v3, base_nonce + step);
            sip_v0 = seed_k0, sip_v1 = seed_k1, sip_v2 = seed_k2, sip_v3 = seed_k3;
            for (uint32_t step = 0; step < EDGE_BLOCK_SIZE; step++) {
              const uint64_t hash = siphash_step(sip_v0, sip_v1, sip_v2, sip_v3, base_nonce + step);
              place_edge(step == EDGE_BLOCK_MASK ? hash_last : hash ^ hash_last);
            }
            continue;
          }
          uint64_t hash_block[EDGE_BLOCK_SIZE];
          siphash_fill_block(sip_v0, sip_v1, sip_v2, sip_v3, base_nonce, hash_block);
          const uint64_t hash_last = hash_block[EDGE_BLOCK_MASK];

#if defined(_WIN32) && defined(MOM_SYCL_ADAPTIVECPP) && defined(MOM_SYCL_HAS_HIP)
          // Keep Windows HIP's 64-edge scatter compact instead of fully unrolling it.
#pragma clang loop unroll_count(2)
#endif
          for (uint32_t hash_index = 0; hash_index < EDGE_BLOCK_SIZE; hash_index++) {
            place_edge(hash_index == EDGE_BLOCK_MASK ? hash_last
                                                     : hash_block[hash_index] ^ hash_last);
          }
        }
        if (seed_parking) {
          // Edges left without a partner in this work group; the barrier makes every slot
          // quiescent.
          item.barrier(sycl::access::fence_space::local_space);
          for (uint32_t bucket_id = local_id; bucket_id < initial_bucket_count;
               bucket_id += seed_local_size) {
            const uint32_t tag = parking[2 * bucket_id];
            if (tag != 0)
              store_pair(bucket_id, parking[2 * bucket_id + 1], tag - 1, 0, 0);
          }
        }
      });
});
profile.add_event("seed2a", seed_event);
