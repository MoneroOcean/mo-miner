    // Helper functions for 2-bit edge counters used in trimming phases
    auto increment_2bit_counter = [](const uint32_t bucket_id, sycl::local_accessor<uint32_t, 1> counters) -> void {
      const uint32_t word_index = bucket_id >> 5;
      const uint8_t bit_index   = bucket_id & 0x1F;
      const uint32_t bit_mask   = 1 << bit_index;
      const uint32_t old_value  = local_atomic_ref(counters[word_index]).fetch_or(bit_mask) & bit_mask;
      if (old_value > 0) local_atomic_ref(counters[word_index + 4096]).fetch_or(bit_mask);
    };

    auto read_2bit_counter = [](const uint32_t bucket_id, const sycl::local_accessor<uint32_t, 1>& counters) -> bool {
      const uint32_t word_index = bucket_id >> 5;
      const uint8_t bit_index   = bucket_id & 0x1F;
      const uint32_t bit_mask   = 1 << bit_index;
      return (counters[word_index + 4096] & bit_mask) > 0;
    };

    // FluffyRound1: First trimming round using 2-bit counters to remove low-degree edges
    const sycl::event round1_event = compute_queue.submit([&](sycl::handler& handler) {
      sycl::accessor acc_a1{buffer_a1_u2, handler, sycl::read_only};
      sycl::accessor acc_a2{buffer_a2_u2, handler, sycl::read_only};
      sycl::accessor acc_b{buffer_b_u2, handler, sycl::write_only, sycl::no_init};
      sycl::accessor acc_i2{buffer_i2, handler, sycl::read_write};
      sycl::accessor acc_i1{buffer_i1, handler, sycl::read_write};
      sycl::local_accessor<uint32_t, 1> edge_counters{sycl::range<1>(EDGE_COUNTER_WORDS), handler};
      mom_use_bundle(handler, kernel_bundle);
      handler.parallel_for(sycl::nd_range<1>(sycl::range<1>(4096 * COMPUTE_THREADS), sycl::range<1>(COMPUTE_THREADS)),
                          [=](sycl::nd_item<1> item) {
        const uint32_t local_id      = item.get_local_id(0);
        const uint32_t work_group_id = item.get_group(0);
        const bool is_source_1    = work_group_id < (62 * 64);
        auto* const source_buffer = is_source_1 ? &acc_a1 : &acc_a2;
        const uint32_t read_group = is_source_1 ? work_group_id : work_group_id - (62 * 64);

        for (uint32_t counter_index = local_id; counter_index < EDGE_COUNTER_WORDS; counter_index += COMPUTE_THREADS)
          edge_counters[counter_index] = 0;

        item.barrier(sycl::access::fence_space::local_space);

        const uint32_t edges_in_bucket = sycl::min(acc_i2[work_group_id], duck_edges_a);
        const uint32_t processing_loops = (edges_in_bucket + COMPUTE_THREADS - 1) / COMPUTE_THREADS;

        // First pass: count edge occurrences to identify degree-2+ edges
        for (uint32_t loop_index = 0; loop_index < processing_loops; loop_index++) {
          const uint32_t edge_local_index = loop_index * COMPUTE_THREADS + local_id;
          if (edge_local_index < edges_in_bucket) {
            const sycl::uint2 current_edge = (*source_buffer)[duck_edges_a * read_group + edge_local_index];
            if (current_edge.x() || current_edge.y())
              increment_2bit_counter((current_edge.x() & EDGE_MASK) >> 12, edge_counters);
          }
        }

        item.barrier(sycl::access::fence_space::local_space);

        // Second pass: write edges that appear multiple times (degree >= 2)
        for (uint32_t loop_index = 0; loop_index < processing_loops; loop_index++) {
          const uint32_t edge_local_index = loop_index * COMPUTE_THREADS + local_id;
          if (edge_local_index < edges_in_bucket) {
            const sycl::uint2 current_edge = (*source_buffer)[duck_edges_a * read_group + edge_local_index];
            if ((current_edge.x() || current_edge.y()) && read_2bit_counter((current_edge.x() & EDGE_MASK) >> 12, edge_counters)) {
              const uint32_t bucket_id = current_edge.y() & BUCKET_MASK_4K;
              const uint32_t bucket_index = sycl::min(c29_global_atomic(acc_i1, bucket_id).fetch_add(1), duck_edges_b - 1);
              acc_b[bucket_id * duck_edges_b + bucket_index] = sycl::uint2(current_edge.y(), current_edge.x()); // Swap edge direction
            }
          }
        }

        if (local_id == 0) acc_i2[work_group_id] = 0;
      });
    });

    profile.add_event("round1", round1_event);

    // FluffyRoundNO1: First trimming round with memory offset optimization
    const sycl::event round_no1_event = compute_queue.submit([&](sycl::handler& handler) {
      sycl::accessor acc_b{buffer_b_u2, handler, sycl::read_only};
      sycl::accessor acc_a1{buffer_a1_u2, handler, sycl::write_only, sycl::no_init};
      sycl::accessor acc_i1{buffer_i1, handler, sycl::read_write};
      sycl::accessor acc_i2{buffer_i2, handler, sycl::read_write};
      sycl::local_accessor<uint32_t, 1> edge_counters{sycl::range<1>(EDGE_COUNTER_WORDS), handler};
      mom_use_bundle(handler, kernel_bundle);
      handler.parallel_for(sycl::nd_range<1>(sycl::range<1>(4096 * COMPUTE_THREADS), sycl::range<1>(COMPUTE_THREADS)),
                          [=](sycl::nd_item<1> item) {
        const uint32_t local_id      = item.get_local_id(0);
        const uint32_t work_group_id = item.get_group(0);

        for (uint32_t counter_index = local_id; counter_index < EDGE_COUNTER_WORDS; counter_index += COMPUTE_THREADS)
          edge_counters[counter_index] = 0;

        item.barrier(sycl::access::fence_space::local_space);

        const uint32_t edges_in_bucket = sycl::min(acc_i1[work_group_id], duck_edges_b);
        const uint32_t processing_loops = (edges_in_bucket + COMPUTE_THREADS - 1) / COMPUTE_THREADS;

        // First pass: count edge occurrences
        for (uint32_t loop_index = 0; loop_index < processing_loops; loop_index++) {
          const uint32_t edge_local_index = loop_index * COMPUTE_THREADS + local_id;
          if (edge_local_index < edges_in_bucket) {
            const sycl::uint2 current_edge = acc_b[(duck_edges_b * work_group_id) + edge_local_index];
            if (current_edge.x() || current_edge.y())
              increment_2bit_counter((current_edge.x() & EDGE_MASK) >> 12, edge_counters);
          }
        }

        item.barrier(sycl::access::fence_space::local_space);

        // Second pass: write filtered edges with memory offset for better cache locality
        for (uint32_t loop_index = 0; loop_index < processing_loops; loop_index++) {
          const uint32_t edge_local_index = loop_index * COMPUTE_THREADS + local_id;
          if (edge_local_index < edges_in_bucket) {
            const sycl::uint2 current_edge = acc_b[(duck_edges_b * work_group_id) + edge_local_index];
            if ((current_edge.x() || current_edge.y()) && read_2bit_counter((current_edge.x() & EDGE_MASK) >> 12, edge_counters)) {
              const uint32_t bucket_id = current_edge.y() & BUCKET_MASK_4K;
              const uint32_t bucket_index = sycl::min(c29_global_atomic(acc_i2, bucket_id).fetch_add(1),
                                                      duck_edges_b - 1 - ((bucket_id & BUCKET_OFFSET) * BUCKET_STEP));
              acc_a1[((bucket_id & BUCKET_OFFSET) * BUCKET_STEP) + (bucket_id * duck_edges_b) + bucket_index] =
                sycl::uint2(current_edge.y(), current_edge.x());
            }
          }
        }

        if (local_id == 0) acc_i1[work_group_id] = 0;
      });
    });

    profile.add_event("round_no1", round_no1_event);

    // FluffyRoundNON: Subsequent trimming rounds with memory offset optimization
    auto trim_round_with_offset = [&](sycl::queue& queue, sycl::buffer<sycl::uint2, 1>& source_buffer,
                                      sycl::buffer<sycl::uint2, 1>& dest_buffer, sycl::buffer<uint32_t, 1>& source_indexes,
                                      sycl::buffer<uint32_t, 1>& dest_indexes) {
      return queue.submit([&](sycl::handler& handler) {
        sycl::accessor acc_source{source_buffer, handler, sycl::read_only};
        sycl::accessor acc_dest{dest_buffer, handler, sycl::write_only, sycl::no_init};
        sycl::accessor acc_source_indexes{source_indexes, handler, sycl::read_write};
        sycl::accessor acc_dest_indexes{dest_indexes, handler, sycl::read_write};
        sycl::local_accessor<uint32_t, 1> edge_counters{sycl::range<1>(EDGE_COUNTER_WORDS), handler};
        mom_use_bundle(handler, kernel_bundle);
        handler.parallel_for(sycl::nd_range<1>(sycl::range<1>(4096 * COMPUTE_THREADS), sycl::range<1>(COMPUTE_THREADS)),
                            [=](sycl::nd_item<1> item) {
          const uint32_t local_id      = item.get_local_id(0);
          const uint32_t work_group_id = item.get_group(0);

          for (uint32_t counter_index = local_id; counter_index < EDGE_COUNTER_WORDS; counter_index += COMPUTE_THREADS)
            edge_counters[counter_index] = 0;

          item.barrier(sycl::access::fence_space::local_space);

          // Clamp the read count to the per-bucket written region: the writer (skewed kernel above)
          // caps each bucket index at duck_edges_b - 1 - ((bucket & BUCKET_OFFSET) * BUCKET_STEP), and
          // the reads here add the same skew, so without this the top work-groups read past the highest
          // written index (OOB device read / stale cross-bucket data). Mirrors the write clamp exactly.
          const uint32_t edges_in_bucket = sycl::min(acc_source_indexes[work_group_id],
                                                     duck_edges_b - ((work_group_id & BUCKET_OFFSET) * BUCKET_STEP));
          const uint32_t processing_loops = (edges_in_bucket + COMPUTE_THREADS - 1) / COMPUTE_THREADS;

          // First pass: count edge occurrences
          for (uint32_t loop_index = 0; loop_index < processing_loops; loop_index++) {
            const uint32_t edge_local_index = loop_index * COMPUTE_THREADS + local_id;
            if (edge_local_index < edges_in_bucket) {
              const sycl::uint2 current_edge = acc_source[((work_group_id & BUCKET_OFFSET) * BUCKET_STEP) +
                                                          (duck_edges_b * work_group_id) + edge_local_index];
              if (current_edge.x() || current_edge.y())
                increment_2bit_counter((current_edge.x() & EDGE_MASK) >> 12, edge_counters);
            }
          }

          item.barrier(sycl::access::fence_space::local_space);

          // Second pass: write filtered edges
          for (uint32_t loop_index = 0; loop_index < processing_loops; loop_index++) {
            const uint32_t edge_local_index = loop_index * COMPUTE_THREADS + local_id;
            if (edge_local_index < edges_in_bucket) {
              const sycl::uint2 current_edge = acc_source[((work_group_id & BUCKET_OFFSET) * BUCKET_STEP) +
                                                          (duck_edges_b * work_group_id) + edge_local_index];
              if ((current_edge.x() || current_edge.y()) && read_2bit_counter((current_edge.x() & EDGE_MASK) >> 12, edge_counters)) {
                const uint32_t bucket_id = current_edge.y() & BUCKET_MASK_4K;
                const uint32_t bucket_index = sycl::min(c29_global_atomic(acc_dest_indexes, bucket_id).fetch_add(1),
                                                        duck_edges_b - 1 - ((bucket_id & BUCKET_OFFSET) * BUCKET_STEP));
                acc_dest[((bucket_id & BUCKET_OFFSET) * BUCKET_STEP) + (bucket_id * duck_edges_b) + bucket_index] =
                  sycl::uint2(current_edge.y(), current_edge.x());
              }
            }
          }

          if (local_id == 0) acc_source_indexes[work_group_id] = 0;
        });
      });
    };

    // Main trimming loop: iteratively reduce edge count by removing low-degree edges
    profile.add_event("trim_a_to_b_initial",
      trim_round_with_offset(compute_queue, buffer_a1_u2, buffer_b_u2, buffer_i2, buffer_i1));

    for (uint32_t round_index = 0; round_index < TRIMMING_ROUNDS; round_index++) {
      profile.add_event("trim_b_to_a",
        trim_round_with_offset(compute_queue, buffer_b_u2, buffer_a1_u2, buffer_i1, buffer_i2));
      profile.add_event("trim_a_to_b",
        trim_round_with_offset(compute_queue, buffer_a1_u2, buffer_b_u2, buffer_i2, buffer_i1));
    }

    // FluffyTailO: Collect final edges into contiguous output buffer
    uint32_t trimmed_edge_count = 0;
    zero_buffer("clear_trimmed_count", buffer_trimmed_edge_count);

    const sycl::event tail_event = compute_queue.submit([&](sycl::handler& handler) {
      sycl::accessor acc_b{buffer_b_u2, handler, sycl::read_only};
      sycl::accessor acc_edges{buffer_trimmed_edges_u2, handler, sycl::write_only, sycl::no_init};
      sycl::accessor acc_i1{buffer_i1, handler, sycl::read_write};
      sycl::accessor acc_trimmed_edge_count{buffer_trimmed_edge_count, handler, sycl::read_write};
      sycl::local_accessor<uint32_t, 1> output_index{sycl::range<1>(1), handler};
      mom_use_bundle(handler, kernel_bundle);
      handler.parallel_for(sycl::nd_range<1>(sycl::range<1>(4096 * COMPUTE_THREADS), sycl::range<1>(COMPUTE_THREADS)),
                          [=](sycl::nd_item<1> item) {
        const uint32_t local_id      = item.get_local_id(0);
        const uint32_t work_group_id = item.get_group(0);
        const uint32_t edges_to_copy = acc_i1[work_group_id];

        // Thread 0 reserves space in output buffer
        if (local_id == 0) output_index[0] = c29_global_atomic(acc_trimmed_edge_count, 0).fetch_add(edges_to_copy);

        item.barrier(sycl::access::fence_space::local_space);

        // Copy edges to contiguous output buffer
        for (uint32_t edge_index = local_id; edge_index < edges_to_copy; edge_index += COMPUTE_THREADS) {
          const uint32_t index = output_index[0] + edge_index;
          if (index < MAX_TRIMMED_EDGE_COUNT)
            acc_edges[index] = acc_b[((work_group_id & BUCKET_OFFSET) * BUCKET_STEP) + work_group_id * duck_edges_b + edge_index];
        }

        if (local_id == 0) acc_i1[work_group_id] = 0;
      });
    });

    profile.add_event("tail", tail_event);

    // Intel OpenCL may busy-spin a host core inside the blocking D2H read if the copy is enqueued
    // before the preceding trim kernels finish. Wait for the final trim event with our poll/sleep
    // helper first; then the small readbacks below only move already-ready data.
    sycl_wait_and_throw(tail_event, compute_queue.get_device());

    // Read final trimmed edges from GPU memory
    c29_read_buffer(compute_queue, buffer_trimmed_edge_count, &trimmed_edge_count, 1);
    trimmed_edge_count = sycl::min(trimmed_edge_count, MAX_TRIMMED_EDGE_COUNT);

    profile.mark("read trimmed count");
    profile.print_events();

    std::vector<sycl::uint2> trimmed_edges(trimmed_edge_count);

    c29_read_buffer(compute_queue, buffer_trimmed_edges_u2, trimmed_edges.data(), trimmed_edge_count);

    profile.mark("read trimmed edges");
    profile.finish(trimmed_edge_count);
