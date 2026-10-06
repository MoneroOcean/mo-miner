int c29(const unsigned job_ref, const unsigned c29_proof_size, const uint8_t* const input,
        const unsigned input_size, uint8_t* const output, uint32_t* const output_edges,
        uint64_t* const pnonce, const std::string& dev_str) {
  try {
    if (!output || !output_edges || !pnonce || (input_size && !input))
      throw std::string("c29 requires output, edge, nonce, and nonempty-input pointers");
    if (c29_proof_size != 32 && c29_proof_size != 42)
      throw std::string("c29 requires proof size 32 or 42");

    // SYCL_PROGRAM_COMPILE_OPTIONS is process-global; pearlhash's ESIMD (VC-backend) image rejects "-O3"
    // ("invalid api option"), which is a no-op for c29 anyway, so leave it empty to stay compatible.
    [[maybe_unused]] static const bool sycl_compile_env_set =
        (set_sycl_env("SYCL_PROGRAM_COMPILE_OPTIONS", ""), true);

    // The bounded guard starts no asynchronous search, so its empty-input poll drains immediately.
    if (std::getenv("MOM_C29_TEST_EDGE"))
      return input_size ? c29_portable_test(input, input_size, output, dev_str) : -1;

    C29State& state = c29_state(dev_str);
    sycl::queue& compute_queue = state.queue;
    const std::shared_ptr<C29ResultState> result_state = state.result_state;
    c29_activate_job(result_state, job_ref);

    // Pop only this job's solution; another device/job may still own the other queue entries.
    bool has_solution = false;
    std::vector<sycl::uint2> solution;
    uint64_t solution_seed[4]{};
    uint64_t solution_nonce = 0;

    {
      std::lock_guard<std::mutex> lock(result_state->mutex);
      for (auto it = result_state->solutions.begin(); it != result_state->solutions.end(); ++it) {
        if (it->job_ref == job_ref) {
          solution = std::move(it->edges);
          solution_seed[0] = it->seed[0];
          solution_seed[1] = it->seed[1];
          solution_seed[2] = it->seed[2];
          solution_seed[3] = it->seed[3];
          solution_nonce = it->nonce;
          result_state->solutions.erase(it);
          has_solution = true;
          break;
        }
      }
    }

    if (has_solution) {
      // The final trim buffer stores (V,U), and cycle reconstruction preserves that order. Packing
      // V into the high half recreates SipHash's U|(V<<32) edge before exact nonce recovery.
      std::vector<uint64_t> recovery_edges;
      recovery_edges.reserve(c29_proof_size);
      for (const auto& edge : solution)
        recovery_edges.push_back(static_cast<uint64_t>(edge.x()) << 32 | edge.y());

      const uint64_t k0 = solution_seed[0], k1 = solution_seed[1], k2 = solution_seed[2],
                     k3 = solution_seed[3];

      sycl::buffer<uint64_t, 1> buffer_edges{recovery_edges.data(), sycl::range<1>{c29_proof_size}};
      sycl::buffer<uint32_t, 1> buffer_nonces{sycl::range<1>{c29_proof_size}};

      // Valid nonces and the unresolved marker are below INT_MAX, so signed/unsigned min agree.
      compute_queue.submit([&](sycl::handler& handler) {
        sycl::accessor acc_nonces{buffer_nonces, handler, sycl::write_only, sycl::no_init};
        handler.fill(acc_nonces, NUM_EDGES);
      });

      // FluffyRecovery kernel - find nonces that generate solution edges
      compute_queue.submit([&](sycl::handler& handler) {
        sycl::accessor acc_edges{buffer_edges, handler, sycl::read_only};
        sycl::accessor acc_nonces{buffer_nonces, handler, sycl::read_write};
        sycl::local_accessor<uint32_t, 1> local_nonces{sycl::range<1>{c29_proof_size}, handler};
        handler.parallel_for(sycl::nd_range<1>(sycl::range<1>(2048 * 256), sycl::range<1>(256)),
                             [=](sycl::nd_item<1> item) MOM_SYCL_KERNEL_ARGS_RESTRICT {
                               const uint32_t gid = item.get_global_id(0);
                               const uint32_t lid = item.get_local_id(0);

                               if (lid < c29_proof_size)
                                 local_nonces[lid] = NUM_EDGES;

                               item.barrier(sycl::access::fence_space::local_space);

                               for (uint32_t block = 0; block < 1024; block += EDGE_BLOCK_SIZE) {
                                 const uint64_t base_nonce = gid * 1024 + block;
                                 uint64_t v0 = k0, v1 = k1, v2 = k2, v3 = k3;
                                 uint64_t sip_block[EDGE_BLOCK_SIZE];

                                 siphash_fill_block(v0, v1, v2, v3, base_nonce, sip_block);
                                 const uint64_t last = sip_block[EDGE_BLOCK_MASK];

                                 // Check each generated edge against target edges
                                 for (int32_t s = EDGE_BLOCK_MASK; s >= 0; s--) {
                                   const uint64_t lookup =
                                       s == EDGE_BLOCK_MASK ? last : sip_block[s] ^ last;
                                   const uint64_t u = lookup & EDGE_MASK;
                                   const uint64_t v = (lookup >> 32) & EDGE_MASK;
                                   const uint64_t edge_a = u | (v << 32);

                                   // U and V share a numeric range but are distinct graph sides.
                                   for (uint32_t idx = 0; idx < c29_proof_size; idx++) {
                                     if (acc_edges[idx] == edge_a)
                                       sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                                        sycl::memory_scope::work_group,
                                                        sycl::access::address_space::local_space>(
                                           local_nonces[idx])
                                           .fetch_min(base_nonce + s);
                                   }
                                 }
                               }

                               item.barrier(sycl::access::fence_space::local_space);

                               if (lid < c29_proof_size)
                                 c29_global_atomic(acc_nonces, lid).fetch_min(local_nonces[lid]);
                             });
      });

      std::vector<uint32_t> nonces(c29_proof_size, NUM_EDGES);

      {
        sycl::host_accessor acc{buffer_nonces, sycl::read_only};
        std::memcpy(nonces.data(), acc.get_pointer(), c29_proof_size * sizeof(uint32_t));
      }

      // Reject unresolved or out-of-range nonces before sorting and packing an incomplete proof.
      for (const uint32_t nonce : nonces) {
        if (!mom_c29::is_valid_nonce(nonce))
          throw std::string("c29 proof recovery left an edge nonce unresolved");
      }

      // Cuckaroo29 requires the recovered edge nonces in ascending order.
      std::sort(nonces.begin(), nonces.end());

      // Pack the sorted nonces little-endian, EDGE_BITS bits each, into the proof bitstream.
      const uint32_t packed_len = (c29_proof_size * EDGE_BITS + 7) / 8;
      std::vector<uint8_t> packed(packed_len, 0);
      uint32_t bit_pos = 0;

      for (const uint32_t nonce : nonces) {
        for (uint32_t bit = 0; bit < EDGE_BITS; bit++, bit_pos++)
          if (nonce & (1u << bit))
            packed[bit_pos / 8] |= (1u << (bit_pos % 8));
      }

      rx_blake2b(output, 32, packed.data(), packed_len);

      // Reverse the 32-byte hash in place to match the pool's expected byte order.
      for (int i = 0; i < 16; i++)
        std::swap(output[i], output[31 - i]);

      *pnonce = solution_nonce;
      std::memcpy(output_edges, nonces.data(), c29_proof_size * sizeof(uint32_t));

      return 1;              // one solution produced
    } else if (input_size) { // start a new asynchronous solution search
      union {
        uint8_t blake_output[32];
        uint64_t k[4];
      };
      rx_blake2b(blake_output, 32, input, input_size);
      start_new_c29_solution_search(k[0], k[1], k[2], k[3], job_ref, *pnonce, c29_proof_size,
                                    compute_queue, state.buffers, result_state);

      return 0; // no immediate results
    } else {    // poll: -1 once every worker has drained, otherwise 0 (still searching)
      std::lock_guard<std::mutex> lock(result_state->mutex);
      const auto it = result_state->running_searches.find(job_ref);
      return it == result_state->running_searches.end() || it->second == 0 ? -1 : 0;
    }
  } catch (const sycl::exception& e) {
    printf("Caught synchronous SYCL exception:\n%s\n", e.what());
    throw;
  }
}
