// SipHash helpers for edge generation. These used to be statement macros, although every operand
// is strongly typed and visible to the device optimizer through ordinary inline functions.
inline __attribute__((always_inline)) void siphash_round(uint64_t& v0, uint64_t& v1, uint64_t& v2,
                                                         uint64_t& v3) {
  v0 += v1;
  v2 += v3;
  v1 = mo_rotate(v1, static_cast<uint64_t>(13));
  v3 = mo_rotate(v3, static_cast<uint64_t>(16));
  v1 ^= v0;
  v3 ^= v2;
  v0 = mo_rotate(v0, static_cast<uint64_t>(32));
  v2 += v1;
  v0 += v3;
  v1 = mo_rotate(v1, static_cast<uint64_t>(17));
  v3 = mo_rotate(v3, static_cast<uint64_t>(21));
  v1 ^= v2;
  v3 ^= v0;
  v2 = mo_rotate(v2, static_cast<uint64_t>(32));
}

inline __attribute__((always_inline)) void siphash_fill_block(uint64_t& v0, uint64_t& v1,
                                                              uint64_t& v2, uint64_t& v3,
                                                              const uint64_t base,
                                                              uint64_t* const output) {
#pragma vector always
  for (uint32_t offset = 0; offset < EDGE_BLOCK_SIZE; ++offset) {
    v3 ^= base + offset;
    siphash_round(v0, v1, v2, v3);
    siphash_round(v0, v1, v2, v3);
    v0 ^= base + offset;
    v2 ^= 0xff;
    siphash_round(v0, v1, v2, v3);
    siphash_round(v0, v1, v2, v3);
    siphash_round(v0, v1, v2, v3);
    siphash_round(v0, v1, v2, v3);
    output[offset] = (v0 ^ v1) ^ (v2 ^ v3);
  }
}

template <typename Destination, typename Store>
inline __attribute__((always_inline)) void c29_write8(Destination& destination,
                                                      const uint32_t index, const Store& store,
                                                      const uint32_t bucket, const uint32_t base) {
  destination[index] = sycl::ulong4(store[bucket][base], store[bucket][base + 1],
                                    store[bucket][base + 2], store[bucket][base + 3]);
  destination[index + 1] = sycl::ulong4(store[bucket][base + 4], store[bucket][base + 5],
                                        store[bucket][base + 6], store[bucket][base + 7]);
}

struct C29SearchCompletionGuard {
  std::shared_ptr<C29ResultState> result_state;
  unsigned job_ref;

  C29SearchCompletionGuard(std::shared_ptr<C29ResultState> state, const unsigned ref)
      : result_state(std::move(state)), job_ref(ref) {
    std::lock_guard<std::mutex> lock(result_state->mutex);
    ++result_state->running_searches[job_ref];
  }

  C29SearchCompletionGuard(const C29SearchCompletionGuard&) = delete;
  C29SearchCompletionGuard& operator=(const C29SearchCompletionGuard&) = delete;
  C29SearchCompletionGuard(C29SearchCompletionGuard&& other) noexcept
      : result_state(std::move(other.result_state)), job_ref(other.job_ref) {
  }

  ~C29SearchCompletionGuard() noexcept {
    if (!result_state)
      return;

    std::lock_guard<std::mutex> lock(result_state->mutex);
    const auto it = result_state->running_searches.find(job_ref);
    if (it == result_state->running_searches.end())
      return;
    if (it->second > 1)
      --it->second;
    else
      result_state->running_searches.erase(it);
  }
};

// Worker function that performs GPU trimming and cycle finding in separate thread
static void start_new_c29_solution_search(const uint64_t seed_k0, const uint64_t seed_k1,
                                          const uint64_t seed_k2, const uint64_t seed_k3,
                                          const unsigned job_ref, const uint64_t nonce,
                                          const unsigned c29_proof_size, sycl::queue& compute_queue,
                                          C29Buffers& c29_buffers,
                                          std::shared_ptr<C29ResultState> result_state) {
  try {
    const sycl::device device = compute_queue.get_device();
    C29Profile profile(job_ref, nonce, c29_proof_size, device);

#include "seed.inc"
#include "trim.inc"
    // Cycle search runs on a detached thread so the GPU queue is freed for the next graph.
    try {
      C29SearchCompletionGuard completion_guard(result_state, job_ref);
      std::thread([=, completion_guard = std::move(completion_guard)]() mutable {
        try {
          const auto solutions = mom_c29::find_cycles(trimmed_edges, c29_proof_size);

          if (!solutions.empty()) {
            std::lock_guard<std::mutex> lock(result_state->mutex);
            if (result_state->active_job_ref == job_ref) {
              for (const auto& solution : solutions) {
                C29QueuedSolution queued_solution;
                queued_solution.edges = solution;
                queued_solution.seed[0] = seed_k0;
                queued_solution.seed[1] = seed_k1;
                queued_solution.seed[2] = seed_k2;
                queued_solution.seed[3] = seed_k3;
                queued_solution.job_ref = job_ref;
                queued_solution.nonce = nonce;
                result_state->solutions.push_back(std::move(queued_solution));
              }
            }
          }
        } catch (const std::exception& e) {
          printf("Error in solution search worker: %s\n", e.what());
        } catch (...) {
          printf("Error in solution search worker: unknown exception\n");
        }
      }).detach();
    } catch (const std::exception& e) {
      printf("Error starting solution search worker: %s\n", e.what());
    } catch (...) {
      printf("Error starting solution search worker: unknown exception\n");
    }
  } catch (const sycl::exception& e) {
    printf("Error in solution search worker: %s\n", e.what());
    throw;
  }
}

#include "entry.inc"
