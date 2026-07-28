// SipHash helpers for edge generation. These used to be statement macros, although every operand
// is strongly typed and visible to the device optimizer through ordinary inline functions.
inline __attribute__((always_inline)) void siphash_round(
  uint64_t& v0, uint64_t& v1, uint64_t& v2, uint64_t& v3
) {
  v0 += v1; v2 += v3; v1 = mo_rotate(v1, static_cast<uint64_t>(13));
  v3 = mo_rotate(v3, static_cast<uint64_t>(16)); v1 ^= v0; v3 ^= v2;
  v0 = mo_rotate(v0, static_cast<uint64_t>(32)); v2 += v1; v0 += v3;
  v1 = mo_rotate(v1, static_cast<uint64_t>(17)); v3 = mo_rotate(v3, static_cast<uint64_t>(21));
  v1 ^= v2; v3 ^= v0; v2 = mo_rotate(v2, static_cast<uint64_t>(32));
}

inline __attribute__((always_inline)) void siphash_fill_block(
  uint64_t& v0, uint64_t& v1, uint64_t& v2, uint64_t& v3,
  const uint64_t base, uint64_t* const output
) {
#pragma vector always
  for (uint32_t offset = 0; offset < EDGE_BLOCK_SIZE; ++offset) {
    v3 ^= base + offset; siphash_round(v0, v1, v2, v3); siphash_round(v0, v1, v2, v3);
    v0 ^= base + offset; v2 ^= 0xff;
    siphash_round(v0, v1, v2, v3); siphash_round(v0, v1, v2, v3);
    siphash_round(v0, v1, v2, v3); siphash_round(v0, v1, v2, v3);
    output[offset] = (v0 ^ v1) ^ (v2 ^ v3);
  }
}

template <typename Destination, typename Store>
inline __attribute__((always_inline)) void c29_write8(
  Destination& destination, const uint32_t index, const Store& store,
  const uint32_t bucket, const uint32_t base
) {
  destination[index] = sycl::ulong4(store[bucket][base], store[bucket][base + 1],
    store[bucket][base + 2], store[bucket][base + 3]);
  destination[index + 1] = sycl::ulong4(store[bucket][base + 4], store[bucket][base + 5],
    store[bucket][base + 6], store[bucket][base + 7]);
}

// Worker function that performs GPU trimming and cycle finding in separate thread
static void start_new_c29_solution_search(const uint64_t seed_k0, const uint64_t seed_k1,
                                          const uint64_t seed_k2, const uint64_t seed_k3,
                                          const unsigned job_ref, const uint64_t nonce,
                                          const unsigned c29_proof_size,
                                          sycl::queue& compute_queue,
                                          MomKernelBundle& kernel_bundle,
                                          C29Buffers& c29_buffers) {
  try {
    C29Profile profile(job_ref, nonce, c29_proof_size, compute_queue, compute_queue.get_device().get_info<sycl::info::device::name>());

#include "seed.inc"
#include "trim.inc"
    // Cycle search runs on a detached thread so the GPU queue is freed for the next graph.
    running_search_threads.fetch_add(1);
    std::thread([=]() {
      const std::list<std::vector<sycl::uint2>> solutions = find_cycles(trimmed_edges, c29_proof_size);

      if (!solutions.empty()) {
        C29SolutionLock lock(global_solutions_mutex);
        for (const auto& solution : solutions) {
          global_solutions.push_back(solution);
          global_seeds.push_back(std::vector<uint64_t>{seed_k0, seed_k1, seed_k2, seed_k3});
          global_job_refs.push_back(job_ref);
          global_nonces.push_back(nonce);
        }
      }

      running_search_threads.fetch_sub(1);
    }).detach();
  } catch (const sycl::exception& e) {
    printf("Error in solution search worker: %s\n", e.what());
  }
}

#include "entry.inc"
