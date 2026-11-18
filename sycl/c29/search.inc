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

// Walk the stateful 64-nonce SipHash block; callers apply the final-word XOR to its edges.
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

struct C29SearchCompletionGuard {
  std::shared_ptr<C29ResultState> result_state;
  unsigned job_ref;

  // Count an async search before launch; transferred ownership decrements it on every exit.
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

  // Release this job's running count without disturbing searches belonging to other jobs.
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

// Worker function that performs GPU trimming and cycle finding in an owned async task.
static void start_new_c29_solution_search(const uint64_t seed_k0, const uint64_t seed_k1,
                                          const uint64_t seed_k2, const uint64_t seed_k3,
                                          const unsigned job_ref, const uint64_t nonce,
                                          const unsigned c29_proof_size, sycl::queue& compute_queue,
                                          C29Buffers& c29_buffers,
                                          std::shared_ptr<C29ResultState> result_state,
                                          std::vector<std::future<void>>& search_futures) {
  try {
    const sycl::device device = compute_queue.get_device();
    C29Profile profile(job_ref, nonce, c29_proof_size, device);

#include "seed.inc"
    profile.mark("seed submitted");
#include "trim.inc"
    // Move the retired graph to its owned async cycle search, freeing the GPU for the next graph
    // without copying the potentially multi-megabyte host handoff.
    try {
      // c29() holds C29State::mutex here; active futures remain owned until a later ready reap.
      for (auto it = search_futures.begin(); it != search_futures.end();) {
        if (it->wait_for(std::chrono::seconds(0)) == std::future_status::ready)
          it = search_futures.erase(it);
        else
          ++it;
      }
      C29SearchCompletionGuard completion_guard(result_state, job_ref);
      auto search = std::async(std::launch::async,
          [=, trimmed_edges = std::move(trimmed_edges),
           completion_guard = std::move(completion_guard)]() mutable {
        // Move out of the future's callable before entering the body: implementations may retain
        // the callable after invocation, but the running-search count must end with this body.
        C29SearchCompletionGuard active_completion_guard = std::move(completion_guard);
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
      });
      search_futures.push_back(std::move(search));
      profile.mark("cycle search launched");
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
