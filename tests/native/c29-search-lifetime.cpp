#include <condition_variable>
#include <cstdio>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

static std::mutex control_mutex;
static std::condition_variable control_cv;
static bool worker_started = false;
static bool allow_body_finish = false;
static bool after_guard = false;
static bool allow_tail_finish = false;
static bool cleanup_started = false;
static bool cleanup_finished = false;

static void require(const bool condition, const char* const message) {
  if (!condition)
    throw std::logic_error(message);
}

static void wait_for(bool& flag) {
  std::unique_lock<std::mutex> lock(control_mutex);
  control_cv.wait(lock, [&] { return flag; });
}

namespace sycl {
struct uint2 {};
}

#include "result-state.inc"
#include "guard.inc"

struct C29State {
  std::shared_ptr<C29ResultState> result_state = std::make_shared<C29ResultState>();
  std::vector<std::future<void>> search_futures;

  ~C29State() {
    {
      std::lock_guard<std::mutex> lock(control_mutex);
      cleanup_started = true;
    }
    control_cv.notify_all();
#include "drain.inc"
  }
};

static void start_blocked_search(C29State& state) {
  const std::shared_ptr<C29ResultState> result_state = state.result_state;
  C29SearchCompletionGuard completion_guard(result_state, 17);
  auto search = std::async(std::launch::async,
      [result_state, completion_guard = std::move(completion_guard)]() mutable {
    {
#include "local-guard.inc"
      {
        std::lock_guard<std::mutex> lock(control_mutex);
        worker_started = true;
      }
      control_cv.notify_all();

      std::unique_lock<std::mutex> lock(control_mutex);
      control_cv.wait(lock, [] { return allow_body_finish; });
    }

    // Retiring the guard precedes this callable's return. State cleanup must still join the
    // future instead of treating an empty running-search map as an exited OS thread.
    {
      std::lock_guard<std::mutex> lock(control_mutex);
      after_guard = true;
    }
    control_cv.notify_all();
    std::unique_lock<std::mutex> lock(control_mutex);
    control_cv.wait(lock, [] { return allow_tail_finish; });
  });
  state.search_futures.push_back(std::move(search));
}

int main() {
  try {
    auto state = std::make_unique<C29State>();
    const std::shared_ptr<C29ResultState> result_state = state->result_state;
    start_blocked_search(*state);
    wait_for(worker_started);
    {
      std::lock_guard<std::mutex> lock(result_state->mutex);
      require(result_state->running_searches.at(17) == 1, "search was not registered");
    }

    std::thread cleanup([&state] {
      state.reset();
      {
        std::lock_guard<std::mutex> lock(control_mutex);
        cleanup_finished = true;
      }
      control_cv.notify_all();
    });
    wait_for(cleanup_started);

    {
      std::lock_guard<std::mutex> lock(control_mutex);
      require(!cleanup_finished, "cleanup passed the completion wait too early");
      allow_body_finish = true;
    }
    control_cv.notify_all();

    wait_for(after_guard);
    {
      std::lock_guard<std::mutex> result_lock(result_state->mutex);
      require(result_state->running_searches.empty(),
              "callable-local completion guard did not retire the search");
    }
    {
      std::lock_guard<std::mutex> lock(control_mutex);
      require(!cleanup_finished, "cleanup treated the running count as thread exit");
      allow_tail_finish = true;
    }
    control_cv.notify_all();
    cleanup.join();

    {
      std::lock_guard<std::mutex> lock(result_state->mutex);
      require(result_state->running_searches.empty(), "cleanup left a running search");
    }
    std::puts("PASS actual-source C29 search lifetime");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL %s\n", error.what());
    return 1;
  }
}
