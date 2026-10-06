// Helper functions for 2-bit edge counters used in trimming phases
auto increment_2bit_counter = [](const uint32_t bucket_id,
                                 sycl::local_accessor<uint32_t, 1> counters,
                                 const uint32_t bitset_words) -> void {
  const uint32_t word_index = bucket_id >> 5;
  const uint8_t bit_index = bucket_id & 0x1F;
  const uint32_t bit_mask = 1u << bit_index;
  const uint32_t old_value = local_atomic_ref(counters[word_index]).fetch_or(bit_mask) & bit_mask;
  if (old_value > 0)
    local_atomic_ref(counters[word_index + bitset_words]).fetch_or(bit_mask);
};

auto read_2bit_counter = [](const uint32_t bucket_id,
                            const sycl::local_accessor<uint32_t, 1>& counters,
                            const uint32_t bitset_words) -> bool {
  const uint32_t word_index = bucket_id >> 5;
  const uint8_t bit_index = bucket_id & 0x1F;
  const uint32_t bit_mask = 1u << bit_index;
  return (counters[word_index + bitset_words] & bit_mask) > 0;
};

// FluffyRound1: First trimming round using 2-bit counters to remove low-degree edges
const uint32_t initial_trim_threads = intel_gpu ? 1024u : COMPUTE_THREADS;
const sycl::event round1_event = compute_queue.submit([&](sycl::handler& handler) {
  sycl::accessor acc_a1{buffer_a1_u2, handler, sycl::read_only};
  sycl::accessor acc_a2{buffer_a2_u2, handler, sycl::read_only};
  sycl::accessor acc_b{buffer_b_u2, handler, sycl::write_only, sycl::no_init};
  sycl::accessor acc_i2{buffer_i2, handler, sycl::read_write};
  sycl::accessor acc_i1{buffer_i1, handler, sycl::read_write};
  const uint32_t initial_counter_words = 1u << (EDGE_BITS - initial_bucket_bits - 4);
  sycl::local_accessor<uint32_t, 1> edge_counters{sycl::range<1>(initial_counter_words), handler};
  handler.parallel_for(
      sycl::nd_range<1>(sycl::range<1>(initial_bucket_count * initial_trim_threads),
                        sycl::range<1>(initial_trim_threads)),
      [=](sycl::nd_item<1> item) MOM_SYCL_KERNEL_ARGS_RESTRICT {
        const uint32_t local_id = item.get_local_id(0);
        const uint32_t work_group_id = item.get_group(0);
        const bool is_source_1 = work_group_id < initial_a1_buckets;
        auto* const source_buffer = is_source_1 ? &acc_a1 : &acc_a2;
        const uint32_t read_group =
            is_source_1 ? work_group_id : work_group_id - initial_a1_buckets;

        for (uint32_t counter_index = local_id; counter_index < initial_counter_words;
             counter_index += initial_trim_threads)
          edge_counters[counter_index] = 0;

        item.barrier(sycl::access::fence_space::local_space);

        const uint32_t edges_in_bucket = sycl::min(acc_i2[work_group_id], initial_bucket_edges);
        const uint32_t processing_loops =
            (edges_in_bucket + initial_trim_threads - 1) / initial_trim_threads;

        // First pass: count edge occurrences to identify degree-2+ edges
        for (uint32_t loop_index = 0; loop_index < processing_loops; loop_index++) {
          const uint32_t edge_local_index = loop_index * initial_trim_threads + local_id;
          if (edge_local_index < edges_in_bucket) {
            const sycl::uint2 current_edge =
                (*source_buffer)[initial_bucket_edges * read_group + edge_local_index];
            if (current_edge.x() || current_edge.y())
              increment_2bit_counter((current_edge.x() & EDGE_MASK) >> initial_bucket_bits,
                                     edge_counters, initial_counter_words / 2);
          }
        }

        item.barrier(sycl::access::fence_space::local_space);

        // Second pass: write edges that appear multiple times (degree >= 2)
        for (uint32_t loop_index = 0; loop_index < processing_loops; loop_index++) {
          const uint32_t edge_local_index = loop_index * initial_trim_threads + local_id;
          if (edge_local_index < edges_in_bucket) {
            const sycl::uint2 current_edge =
                (*source_buffer)[initial_bucket_edges * read_group + edge_local_index];
            if ((current_edge.x() || current_edge.y()) &&
                read_2bit_counter((current_edge.x() & EDGE_MASK) >> initial_bucket_bits,
                                  edge_counters, initial_counter_words / 2)) {
              const uint32_t bucket_id = current_edge.y() & trim_bucket_mask;
              const uint32_t bucket_index = sycl::min(
                  c29_global_atomic(acc_i1, bucket_id).fetch_add(1), trim_bucket_edges - 1);
              acc_b[bucket_id * trim_bucket_edges + bucket_index] =
                  sycl::uint2(current_edge.y(), current_edge.x()); // Swap edge direction
            }
          }
        }

        if (local_id == 0)
          acc_i2[work_group_id] = 0;
      });
});

profile.add_event("round1", round1_event);

// FluffyRoundNO1: First subsequent trimming round
const sycl::event round_no1_event = compute_queue.submit([&](sycl::handler& handler) {
  sycl::accessor acc_b{buffer_b_u2, handler, sycl::read_only};
  sycl::accessor acc_a1{buffer_a1_u2, handler, sycl::write_only, sycl::no_init};
  sycl::accessor acc_i1{buffer_i1, handler, sycl::read_write};
  sycl::accessor acc_i2{buffer_i2, handler, sycl::read_write};
  sycl::local_accessor<uint32_t, 1> edge_counters{sycl::range<1>(trim_counter_words), handler};
  handler.parallel_for(
      sycl::nd_range<1>(sycl::range<1>(trim_bucket_count * round_no1_threads),
                        sycl::range<1>(round_no1_threads)),
      [=](sycl::nd_item<1> item) MOM_SYCL_KERNEL_ARGS_RESTRICT {
        const uint32_t local_id = item.get_local_id(0);
        const uint32_t work_group_id = item.get_group(0);

        for (uint32_t counter_index = local_id; counter_index < trim_counter_words;
             counter_index += round_no1_threads)
          edge_counters[counter_index] = 0;

        item.barrier(sycl::access::fence_space::local_space);

        const uint32_t edges_in_bucket = sycl::min(acc_i1[work_group_id], trim_bucket_edges);
        const uint32_t processing_loops =
            (edges_in_bucket + round_no1_threads - 1) / round_no1_threads;

        // First pass: count edge occurrences
        for (uint32_t loop_index = 0; loop_index < processing_loops; loop_index++) {
          const uint32_t edge_local_index = loop_index * round_no1_threads + local_id;
          if (edge_local_index < edges_in_bucket) {
            const sycl::uint2 current_edge =
                acc_b[(trim_bucket_edges * work_group_id) + edge_local_index];
            if (current_edge.x() || current_edge.y())
              increment_2bit_counter((current_edge.x() & EDGE_MASK) >> trim_bucket_bits,
                                     edge_counters, trim_counter_words / 2);
          }
        }

        item.barrier(sycl::access::fence_space::local_space);

        // Second pass: write filtered edges with memory offset for better cache locality
        for (uint32_t loop_index = 0; loop_index < processing_loops; loop_index++) {
          const uint32_t edge_local_index = loop_index * round_no1_threads + local_id;
          if (edge_local_index < edges_in_bucket) {
            const sycl::uint2 current_edge =
                acc_b[(trim_bucket_edges * work_group_id) + edge_local_index];
            if ((current_edge.x() || current_edge.y()) &&
                read_2bit_counter((current_edge.x() & EDGE_MASK) >> trim_bucket_bits, edge_counters,
                                  trim_counter_words / 2)) {
              const uint32_t bucket_id = current_edge.y() & trim_bucket_mask;
              const uint32_t bucket_offset =
                  trim_bucket_bits == 12 ? (bucket_id & BUCKET_OFFSET) * BUCKET_STEP : 0;
              const uint32_t bucket_index =
                  sycl::min(c29_global_atomic(acc_i2, bucket_id).fetch_add(1),
                            trim_bucket_edges - 1 - bucket_offset);
              acc_a1[bucket_id * trim_bucket_edges + bucket_offset + bucket_index] =
                  sycl::uint2(current_edge.y(), current_edge.x());
            }
          }
        }

        if (local_id == 0)
          acc_i1[work_group_id] = 0;
      });
});

profile.add_event("round_no1", round_no1_event);

// FluffyRoundNON: Subsequent trimming rounds
auto trim_round = [&](sycl::queue& queue, sycl::buffer<sycl::uint2, 1>& source_buffer,
                      sycl::buffer<sycl::uint2, 1>& dest_buffer,
                      sycl::buffer<uint32_t, 1>& source_indexes,
                      sycl::buffer<uint32_t, 1>& dest_indexes, const uint32_t threads) {
  return queue.submit([&](sycl::handler& handler) {
    sycl::accessor acc_source{source_buffer, handler, sycl::read_only};
    sycl::accessor acc_dest{dest_buffer, handler, sycl::write_only, sycl::no_init};
    sycl::accessor acc_source_indexes{source_indexes, handler, sycl::read_write};
    sycl::accessor acc_dest_indexes{dest_indexes, handler, sycl::read_write};
    sycl::local_accessor<uint32_t, 1> edge_counters{sycl::range<1>(trim_counter_words), handler};
    handler.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(trim_bucket_count * threads), sycl::range<1>(threads)),
        [=](sycl::nd_item<1> item) MOM_SYCL_KERNEL_ARGS_RESTRICT {
          const uint32_t local_id = item.get_local_id(0);
          const uint32_t work_group_id = item.get_group(0);

          for (uint32_t counter_index = local_id; counter_index < trim_counter_words;
               counter_index += threads)
            edge_counters[counter_index] = 0;

          item.barrier(sycl::access::fence_space::local_space);

          const uint32_t bucket_offset =
              trim_bucket_bits == 12 ? (work_group_id & BUCKET_OFFSET) * BUCKET_STEP : 0;
          const uint32_t edges_in_bucket =
              sycl::min(acc_source_indexes[work_group_id], trim_bucket_edges - bucket_offset);
          const uint32_t processing_loops = (edges_in_bucket + threads - 1) / threads;

          // First pass: count edge occurrences
          for (uint32_t loop_index = 0; loop_index < processing_loops; loop_index++) {
            const uint32_t edge_local_index = loop_index * threads + local_id;
            if (edge_local_index < edges_in_bucket) {
              const sycl::uint2 current_edge =
                  acc_source[trim_bucket_edges * work_group_id + bucket_offset + edge_local_index];
              if (current_edge.x() || current_edge.y())
                increment_2bit_counter((current_edge.x() & EDGE_MASK) >> trim_bucket_bits,
                                       edge_counters, trim_counter_words / 2);
            }
          }

          item.barrier(sycl::access::fence_space::local_space);

          // Second pass: write filtered edges
          for (uint32_t loop_index = 0; loop_index < processing_loops; loop_index++) {
            const uint32_t edge_local_index = loop_index * threads + local_id;
            if (edge_local_index < edges_in_bucket) {
              const sycl::uint2 current_edge =
                  acc_source[trim_bucket_edges * work_group_id + bucket_offset + edge_local_index];
              if ((current_edge.x() || current_edge.y()) &&
                  read_2bit_counter((current_edge.x() & EDGE_MASK) >> trim_bucket_bits,
                                    edge_counters, trim_counter_words / 2)) {
                const uint32_t bucket_id = current_edge.y() & trim_bucket_mask;
                const uint32_t dest_offset =
                    trim_bucket_bits == 12 ? (bucket_id & BUCKET_OFFSET) * BUCKET_STEP : 0;
                const uint32_t bucket_index =
                    sycl::min(c29_global_atomic(acc_dest_indexes, bucket_id).fetch_add(1),
                              trim_bucket_edges - 1 - dest_offset);
                acc_dest[bucket_id * trim_bucket_edges + dest_offset + bucket_index] =
                    sycl::uint2(current_edge.y(), current_edge.x());
              }
            }
          }

          if (local_id == 0)
            acc_source_indexes[work_group_id] = 0;
        });
  });
};

// Main trimming loop: iteratively reduce edge count by removing low-degree edges
profile.add_event("trim_a_to_b_initial", trim_round(compute_queue, buffer_a1_u2, buffer_b_u2,
                                                    buffer_i2, buffer_i1, trim_threads));

const uint32_t trim_rounds = c29_trim_rounds(device);
// MOM_C29_INTEL may also be defined in a combined CUDA build: gate this tuning on the device.
const bool intel_late_trim = device.is_gpu() &&
    device.get_info<sycl::info::device::vendor_id>() == 0x8086;
for (uint32_t round_index = 0; round_index < trim_rounds; round_index++) {
  // Later Intel buckets are sparse: fewer lanes reduce idle work without changing
  // bucket coverage. Keep the dense initial rounds and other vendors unchanged.
  const uint32_t round_threads = intel_late_trim && round_index >= 4u ? 128u : trim_threads;
  profile.add_event("trim_b_to_a", trim_round(compute_queue, buffer_b_u2, buffer_a1_u2, buffer_i1,
                                              buffer_i2, round_threads));
  profile.add_event("trim_a_to_b", trim_round(compute_queue, buffer_a1_u2, buffer_b_u2, buffer_i2,
                                              buffer_i1, round_threads));
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
  handler.parallel_for(
      sycl::nd_range<1>(sycl::range<1>(trim_bucket_count * COMPUTE_THREADS),
                        sycl::range<1>(COMPUTE_THREADS)),
      [=](sycl::nd_item<1> item) MOM_SYCL_KERNEL_ARGS_RESTRICT {
        const uint32_t local_id = item.get_local_id(0);
        const uint32_t work_group_id = item.get_group(0);
        const uint32_t bucket_offset =
            trim_bucket_bits == 12 ? (work_group_id & BUCKET_OFFSET) * BUCKET_STEP : 0;
        const uint32_t edges_to_copy =
            sycl::min(acc_i1[work_group_id], trim_bucket_edges - bucket_offset);

        // Thread 0 reserves space in output buffer
        if (local_id == 0)
          output_index[0] = c29_global_atomic(acc_trimmed_edge_count, 0).fetch_add(edges_to_copy);

        item.barrier(sycl::access::fence_space::local_space);

        // Copy edges to contiguous output buffer
        for (uint32_t edge_index = local_id; edge_index < edges_to_copy;
             edge_index += COMPUTE_THREADS) {
          const uint32_t index = output_index[0] + edge_index;
          if (index < MAX_TRIMMED_EDGE_COUNT)
            acc_edges[index] =
                acc_b[work_group_id * trim_bucket_edges + bucket_offset + edge_index];
        }

        if (local_id == 0)
          acc_i1[work_group_id] = 0;
      });
});

profile.add_event("tail", tail_event);

// Wait for the producer before submitting each raw-host-pointer copy, then wait for that copy before
// its destination can leave scope. An asynchronous variant had no measured benefit and repeatedly
// faulted Intel's dedicated copy engine during process teardown.
profile.mark("before tail wait");
sycl_wait_and_throw(tail_event, device);
profile.mark("tail ready");

c29_read_buffer(compute_queue, buffer_trimmed_edge_count, &trimmed_edge_count, 1);
trimmed_edge_count = sycl::min(trimmed_edge_count, MAX_TRIMMED_EDGE_COUNT);

profile.mark("trimmed count ready");
profile.print_events();

std::vector<sycl::uint2> trimmed_edges(trimmed_edge_count);
c29_read_buffer(compute_queue, buffer_trimmed_edges_u2, trimmed_edges.data(), trimmed_edge_count);

profile.mark("read trimmed edges");
profile.finish(trimmed_edge_count);
