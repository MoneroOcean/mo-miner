// Reuse the persistent GPU buffers across graphs, reinterpreted as uint2 edge pairs.
auto buffer_a1_u2 = c29_buffers.buffer_a1.reinterpret<sycl::uint2>(sycl::range(BUFFER_SIZE_A1 / 2));
auto buffer_a2_u2 = c29_buffers.buffer_a2.reinterpret<sycl::uint2>(sycl::range(BUFFER_SIZE_A2 / 2));
auto buffer_b_u2 = c29_buffers.buffer_b.reinterpret<sycl::uint2>(sycl::range(BUFFER_SIZE_B / 2));
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

const uint32_t duck_edges_b = static_cast<uint32_t>(DUCK_SIZE_B) * 1024;

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
for (;;) {
  initial_a1_buckets =
      initial_bucket_count - static_cast<uint32_t>((BUFFER_SIZE_A2 / 2) / initial_bucket_edges);
  if (static_cast<uint64_t>(initial_a1_buckets) * initial_bucket_edges <= BUFFER_SIZE_A1 / 2)
    break;
  --initial_bucket_edges;
}
const uint32_t trim_bucket_bits = intel_gpu ? 14u : 12u;
const uint32_t trim_bucket_count = 1u << trim_bucket_bits;
const uint32_t trim_bucket_mask = trim_bucket_count - 1;
const uint32_t trim_bucket_edges = duck_edges_b >> (trim_bucket_bits - 12);
const uint32_t trim_counter_words = 1u << (EDGE_BITS - trim_bucket_bits - 4);
const uint32_t trim_threads = intel_gpu ? 256u : COMPUTE_THREADS;
const uint32_t round_no1_threads = intel_gpu ? 1024u : trim_threads;

const sycl::event seed_event = compute_queue.submit([&](sycl::handler& handler) {
  sycl::accessor acc_buffer_a1{buffer_a1_u2, handler, sycl::write_only, sycl::no_init};
  sycl::accessor acc_buffer_a2{buffer_a2_u2, handler, sycl::write_only, sycl::no_init};
  sycl::accessor acc_index_2{buffer_i2, handler, sycl::read_write};
  // Intel Xe2 otherwise chooses SIMD32 here, doubling the hardware thread's private allocation
  // for the per-lane 64-value block. Other GPU toolchains leave the width unconstrained.
  handler.parallel_for(
      sycl::nd_range<1>(sycl::range<1>(seed_work_items), sycl::range<1>(seed_local_size)),
      [=](sycl::nd_item<1> item) MOM_REQD_SG_16 MOM_SYCL_KERNEL_ARGS_RESTRICT {
        const uint32_t global_id = item.get_global_id(0);
        for (uint32_t block_offset = 0; block_offset < seed_blocks_per_item * EDGE_BLOCK_SIZE;
             block_offset += EDGE_BLOCK_SIZE) {
          const uint64_t base_nonce =
              global_id * (seed_blocks_per_item * EDGE_BLOCK_SIZE) + block_offset;
          uint64_t sip_v0 = seed_k0, sip_v1 = seed_k1, sip_v2 = seed_k2, sip_v3 = seed_k3;
          uint64_t hash_block[EDGE_BLOCK_SIZE];
          siphash_fill_block(sip_v0, sip_v1, sip_v2, sip_v3, base_nonce, hash_block);
          const uint64_t hash_last = hash_block[EDGE_BLOCK_MASK];

#if defined(_WIN32) && defined(MOM_SYCL_ADAPTIVECPP) && defined(MOM_SYCL_HAS_HIP)
          // Keep Windows HIP's 64-edge scatter compact instead of fully unrolling it.
#pragma clang loop unroll_count(2)
#endif
          for (uint32_t hash_index = 0; hash_index < EDGE_BLOCK_SIZE; hash_index++) {
            const uint64_t hash_lookup =
                hash_index == EDGE_BLOCK_MASK ? hash_last : hash_block[hash_index] ^ hash_last;
            const uint32_t edge_u = static_cast<uint32_t>(hash_lookup & EDGE_MASK);
            const uint32_t edge_v = static_cast<uint32_t>((hash_lookup >> 32) & EDGE_MASK);
            if (!(edge_u || edge_v))
              continue;

            const uint32_t low_mask = (1u << (initial_bucket_bits - 6)) - 1;
            const uint32_t bucket_id =
                ((edge_u & low_mask) << 6) | ((edge_u >> (initial_bucket_bits - 6)) & 63u);
            const uint32_t bucket_index = c29_global_atomic(acc_index_2, bucket_id).fetch_add(1);
            if (bucket_index < initial_bucket_edges) {
              if (bucket_id < initial_a1_buckets)
                acc_buffer_a1[bucket_id * initial_bucket_edges + bucket_index] =
                    sycl::uint2(edge_u, edge_v);
              else
                acc_buffer_a2[(bucket_id - initial_a1_buckets) * initial_bucket_edges +
                              bucket_index] = sycl::uint2(edge_u, edge_v);
            }
          }
        }
      });
});
profile.add_event("seed2a", seed_event);
