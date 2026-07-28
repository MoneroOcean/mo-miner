    // Reuse the persistent GPU buffers across graphs, reinterpreting each as uint2 (edge pairs)
    // and ulong4 (8-word block writes) for the different kernel access patterns.
    auto buffer_a1_u2  = c29_buffers.buffer_a1.reinterpret<sycl::uint2>(sycl::range(BUFFER_SIZE_A1 / 2));
    auto buffer_a1_ul4 = c29_buffers.buffer_a1.reinterpret<sycl::ulong4>(sycl::range(BUFFER_SIZE_A1 / 8));
    auto buffer_a2_u2  = c29_buffers.buffer_a2.reinterpret<sycl::uint2>(sycl::range(BUFFER_SIZE_A2 / 2));
    auto buffer_a2_ul4 = c29_buffers.buffer_a2.reinterpret<sycl::ulong4>(sycl::range(BUFFER_SIZE_A2 / 8));
    auto buffer_b_u2   = c29_buffers.buffer_b.reinterpret<sycl::uint2>(sycl::range(BUFFER_SIZE_B / 2));
    auto buffer_b_ul4  = c29_buffers.buffer_b.reinterpret<sycl::ulong4>(sycl::range(BUFFER_SIZE_B / 8));
    sycl::buffer<uint32_t, 1>& buffer_i1 = c29_buffers.buffer_i1;
    sycl::buffer<uint32_t, 1>& buffer_i2 = c29_buffers.buffer_i2;
    sycl::buffer<uint32_t, 1>& buffer_trimmed_edge_count = c29_buffers.buffer_trimmed_edge_count;
    sycl::buffer<sycl::uint2, 1>& buffer_trimmed_edges_u2 = c29_buffers.buffer_trimmed_edges_u2;

    profile.mark("buffers ready");

    auto zero_buffer = [&](const char* name, sycl::buffer<uint32_t, 1>& buffer) {
      const sycl::event event = compute_queue.submit([&](sycl::handler& handler) {
        sycl::accessor accessor{buffer, handler, sycl::write_only, sycl::no_init};
        mom_use_bundle(handler, kernel_bundle);
        handler.fill(accessor, static_cast<uint32_t>(0));
      });
      profile.add_event(name, event);
    };

    zero_buffer("clear_i1_start", buffer_i1);
    zero_buffer("clear_i2_start", buffer_i2);

    const uint32_t duck_edges_a = static_cast<uint32_t>(DUCK_SIZE_A) * 1024;
    const uint32_t duck_edges_b = static_cast<uint32_t>(DUCK_SIZE_B) * 1024;

    using local_atomic_ref  = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group, sycl::access::address_space::local_space>;

    // FluffySeed2A: Generate initial edges using SipHash cryptographic function
    const uint32_t seed_local_size = c29_seed_local_size();
    const uint32_t seed_blocks_per_item = c29_seed_blocks_per_item();
    const uint32_t seed_work_items = NUM_EDGES / (EDGE_BLOCK_SIZE * seed_blocks_per_item);
    const bool direct_seed = c29_direct_seed_enabled();

    if (direct_seed) {
      const sycl::event direct_seed_event = compute_queue.submit([&](sycl::handler& handler) {
        sycl::accessor acc_buffer_a1{buffer_a1_u2, handler, sycl::write_only, sycl::no_init};
        sycl::accessor acc_buffer_a2{buffer_a2_u2, handler, sycl::write_only, sycl::no_init};
        sycl::accessor acc_index_2{buffer_i2, handler, sycl::read_write};
        mom_use_bundle(handler, kernel_bundle);
        handler.parallel_for(sycl::nd_range<1>(sycl::range<1>(seed_work_items), sycl::range<1>(seed_local_size)),
                            [=](sycl::nd_item<1> item) {
          const uint32_t global_id = item.get_global_id(0);

          for (uint32_t block_offset = 0; block_offset < seed_blocks_per_item * EDGE_BLOCK_SIZE; block_offset += EDGE_BLOCK_SIZE) {
            const uint64_t base_nonce = global_id * (seed_blocks_per_item * EDGE_BLOCK_SIZE) + block_offset;
            uint64_t sip_v0 = seed_k0, sip_v1 = seed_k1, sip_v2 = seed_k2, sip_v3 = seed_k3;
            uint64_t hash_block[EDGE_BLOCK_SIZE];

            siphash_fill_block(sip_v0, sip_v1, sip_v2, sip_v3, base_nonce, hash_block);

            const uint64_t hash_last = hash_block[EDGE_BLOCK_MASK];

            for (uint32_t hash_index = 0; hash_index < EDGE_BLOCK_SIZE; hash_index++) {
              const uint64_t hash_lookup = hash_index == EDGE_BLOCK_MASK ? hash_last : hash_block[hash_index] ^ hash_last;
              const uint32_t edge_u      = static_cast<uint32_t>(hash_lookup & EDGE_MASK);
              const uint32_t edge_v      = static_cast<uint32_t>((hash_lookup >> 32) & EDGE_MASK);
              if (!(edge_u || edge_v)) continue;

              const uint32_t bucket_id = ((edge_u & 63u) << 6) | ((edge_u >> 6) & 63u);
              const uint32_t bucket_index = sycl::min(c29_global_atomic(acc_index_2, bucket_id).fetch_add(1), duck_edges_a - 1);

              if (bucket_id < 62 * 64)
                acc_buffer_a1[bucket_id * duck_edges_a + bucket_index] = sycl::uint2(edge_u, edge_v);
              else
                acc_buffer_a2[(bucket_id - 62 * 64) * duck_edges_a + bucket_index] = sycl::uint2(edge_u, edge_v);
            }
          }
        });
      });
      profile.add_event("seed2a_direct", direct_seed_event);
    } else {
      const sycl::event seed_event = compute_queue.submit([&](sycl::handler& handler) {
        sycl::accessor acc_buffer_b{buffer_b_ul4, handler, sycl::write_only, sycl::no_init};
        sycl::accessor acc_buffer_a1{buffer_a1_ul4, handler, sycl::write_only, sycl::no_init};
        sycl::accessor acc_index_1{buffer_i1, handler, sycl::read_write};

        // Local memory for temporary edge storage and bucket counters
        sycl::local_accessor<uint64_t, 2> temp_storage{sycl::range<2>(64, 16), handler};
        sycl::local_accessor<uint32_t, 1> bucket_counters{sycl::range<1>(64), handler};
        mom_use_bundle(handler, kernel_bundle);
        handler.parallel_for(sycl::nd_range<1>(sycl::range<1>(seed_work_items), sycl::range<1>(seed_local_size)),
                            [=](sycl::nd_item<1> item) {
          const uint32_t global_id = item.get_global_id(0);
          const uint32_t local_id  = item.get_local_id(0);

          if (local_id < 64) bucket_counters[local_id] = 0;

          item.barrier(sycl::access::fence_space::local_space);

          // Process nonces in blocks to generate graph edges efficiently
          for (uint32_t block_offset = 0; block_offset < seed_blocks_per_item * EDGE_BLOCK_SIZE; block_offset += EDGE_BLOCK_SIZE) {
            const uint64_t base_nonce = global_id * (seed_blocks_per_item * EDGE_BLOCK_SIZE) + block_offset;
            uint64_t sip_v0 = seed_k0, sip_v1 = seed_k1, sip_v2 = seed_k2, sip_v3 = seed_k3;
            uint64_t hash_block[EDGE_BLOCK_SIZE];

            siphash_fill_block(sip_v0, sip_v1, sip_v2, sip_v3, base_nonce, hash_block);

            const uint64_t hash_last = hash_block[EDGE_BLOCK_MASK];

            // Extract graph edges from hash values and distribute to buckets
            for (uint32_t hash_index = 0; hash_index < EDGE_BLOCK_SIZE; hash_index++) {
              const uint64_t hash_lookup = hash_index == EDGE_BLOCK_MASK ? hash_last : hash_block[hash_index] ^ hash_last;
              const uint32_t edge_u      = static_cast<uint32_t>(hash_lookup & EDGE_MASK);
              const uint32_t edge_v      = static_cast<uint32_t>((hash_lookup >> 32) & EDGE_MASK);
              const uint32_t bucket_id   = edge_u & 63;
              const uint32_t counter       = local_atomic_ref(bucket_counters[bucket_id]).fetch_add(1);
              const uint32_t counter_local = counter % 16;
              temp_storage[bucket_id][counter_local] = edge_u | (static_cast<uint64_t>(edge_v) << 32);

              item.barrier(sycl::access::fence_space::local_space);

              // Write accumulated edges when buffer reaches threshold
              if ((counter > 0) && (counter_local == 0 || counter_local == 8)) {
                const uint32_t write_count = sycl::min(c29_global_atomic(acc_index_1, bucket_id).fetch_add(8), duck_edges_a * 64 - 8);
                const uint32_t write_index = ((bucket_id < 32 ? bucket_id : bucket_id - 32) * duck_edges_a * 64 + write_count) >> 2;
                auto* const dest_buffer    = bucket_id < 32 ? &acc_buffer_b : &acc_buffer_a1;
                c29_write8(*dest_buffer, write_index, temp_storage, bucket_id, 8 - counter_local);

                // Clear written entries for reuse
                for (uint32_t clear_index = 0; clear_index < 8; clear_index++) temp_storage[bucket_id][8 - counter_local + clear_index] = 0;
              }
            }
          }

          item.barrier(sycl::access::fence_space::local_space);

          // Final flush of remaining edges in local buffers
          if (local_id < 64) {
            const uint32_t final_counter     = bucket_counters[local_id];
            const uint32_t final_counter_base = (final_counter % 16) >= 8 ? 8 : 0;
            const uint32_t final_write_count  = sycl::min(c29_global_atomic(acc_index_1, local_id).fetch_add(8), duck_edges_a * 64 - 8);
            const uint32_t final_write_index  = ((local_id < 32 ? local_id : local_id - 32) * duck_edges_a * 64 + final_write_count) >> 2;
            auto* const final_dest_buffer     = local_id < 32 ? &acc_buffer_b : &acc_buffer_a1;
            c29_write8(*final_dest_buffer, final_write_index, temp_storage, local_id, final_counter_base);
          }
        });
      });

      profile.add_event("seed2a", seed_event);

      // FluffySeed2B: Redistribute edges to smaller buckets for better memory access patterns
      auto redistribute_edges = [](sycl::queue& queue, sycl::buffer<sycl::uint2, 1>& source_buffer,
                                   sycl::buffer<sycl::ulong4, 1>& dest_buffer_1, sycl::buffer<sycl::ulong4, 1>& dest_buffer_2,
                                   sycl::buffer<uint32_t, 1>& source_indexes, sycl::buffer<uint32_t, 1>& dest_indexes,
                                   const uint32_t start_block, const uint32_t duck_edges_a, auto& kernel_bundle) {
        return queue.submit([&](sycl::handler& handler) {
          sycl::accessor acc_source{source_buffer, handler, sycl::read_only};
          sycl::accessor acc_dest_1{dest_buffer_1, handler, sycl::write_only, sycl::no_init};
          sycl::accessor acc_dest_2{dest_buffer_2, handler, sycl::write_only, sycl::no_init};
          sycl::accessor acc_source_indexes{source_indexes, handler, sycl::read_only};
          sycl::accessor acc_dest_indexes{dest_indexes, handler, sycl::read_write};
          const constexpr uint32_t BUCKET_GRANULARITY = 32;
          sycl::local_accessor<uint64_t, 2> temp_storage{sycl::range<2>(64, 16), handler};
          sycl::local_accessor<uint32_t, 1> bucket_counters{sycl::range<1>(64), handler};
          mom_use_bundle(handler, kernel_bundle);
          handler.parallel_for(sycl::nd_range<1>(sycl::range<1>(1024 * 128), sycl::range<1>(128)),
                              [=](sycl::nd_item<1> item) {
            const uint32_t local_id      = item.get_local_id(0);
            const uint32_t work_group_id = item.get_group(0);
            if (local_id < 64) bucket_counters[local_id] = 0;

            item.barrier(sycl::access::fence_space::local_space);

            const uint32_t current_bucket         = work_group_id / BUCKET_GRANULARITY;
            const uint32_t micro_block_number     = work_group_id % BUCKET_GRANULARITY;
            const uint32_t bucket_edge_count      = sycl::min(acc_source_indexes[current_bucket + start_block], duck_edges_a * 64);
            const uint32_t micro_block_edge_count = duck_edges_a * 64 / BUCKET_GRANULARITY;
            const uint32_t processing_loops       = micro_block_edge_count / 128;
            const bool     use_dest_buffer_2      = start_block == 32 && current_bucket >= 30;
            const uint32_t memory_offset          = use_dest_buffer_2 ? 0 : start_block * duck_edges_a * 64;
            const uint32_t bucket_offset          = use_dest_buffer_2 ? 30 : 0;
            auto* const    destination_buffer     = use_dest_buffer_2 ? &acc_dest_2 : &acc_dest_1;

            for (uint32_t loop_index = 0; loop_index < processing_loops; loop_index++) {
              const uint32_t edge_index = micro_block_number * micro_block_edge_count + 128 * loop_index + local_id;
              const sycl::uint2 current_edge = edge_index < bucket_edge_count ? acc_source[(current_bucket * duck_edges_a * 64) + edge_index]
                                                                               : sycl::uint2(0, 0);
              const bool skip_edge      = (edge_index >= bucket_edge_count) || (current_edge.x() == 0 && current_edge.y() == 0);
              const uint32_t bucket_id  = (current_edge.x() >> 6) & (64 - 1);
              uint32_t edge_counter = 0, local_counter = 0;

              if (!skip_edge) {
                edge_counter  = local_atomic_ref(bucket_counters[bucket_id]).fetch_add(1);
                local_counter = edge_counter % 16;
                temp_storage[bucket_id][local_counter] = current_edge.x() | (static_cast<uint64_t>(current_edge.y()) << 32);
              }

              item.barrier(sycl::access::fence_space::local_space);

              if ((edge_counter > 0) && (local_counter == 0 || local_counter == 8)) {
                const uint32_t write_count = sycl::min(c29_global_atomic(acc_dest_indexes,
                  start_block * 64 + current_bucket * 64 + bucket_id).fetch_add(8), duck_edges_a - 8);
                const uint32_t write_index = (memory_offset + (((current_bucket - bucket_offset) * 64 + bucket_id) * duck_edges_a + write_count)) >> 2;
                c29_write8(*destination_buffer, write_index, temp_storage, bucket_id, 8 - local_counter);
                for (uint32_t clear_index = 0; clear_index < 8; clear_index++) temp_storage[bucket_id][8 - local_counter + clear_index] = 0;
              }
            }

            item.barrier(sycl::access::fence_space::local_space);

            // Final flush for remaining edges
            if (local_id < 64) {
              const uint32_t final_counter      = bucket_counters[local_id];
              const uint32_t final_counter_base = (final_counter % 16) >= 8 ? 8 : 0;
              const uint32_t final_write_count  = sycl::min(c29_global_atomic(acc_dest_indexes,
                start_block * 64 + current_bucket * 64 + local_id).fetch_add(8), duck_edges_a - 8);
              const uint32_t final_write_index  = (memory_offset + (((current_bucket - bucket_offset) * 64 + local_id) * duck_edges_a + final_write_count)) >> 2;
              c29_write8(*destination_buffer, final_write_index, temp_storage, local_id, final_counter_base);
            }
          });
        });
      };

      profile.add_event("redistribute_32",
        redistribute_edges(compute_queue, buffer_a1_u2, buffer_a1_ul4, buffer_a2_ul4, buffer_i1, buffer_i2, 32, duck_edges_a, kernel_bundle));
      profile.add_event("redistribute_0",
        redistribute_edges(compute_queue, buffer_b_u2, buffer_a1_ul4, buffer_a2_ul4, buffer_i1, buffer_i2, 0, duck_edges_a, kernel_bundle));
      zero_buffer("clear_i1_seed", buffer_i1);
    }
