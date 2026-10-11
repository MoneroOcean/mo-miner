#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

// Model backend handle reuse, not proof of a GPU stall: destroying the
// final node owner returns its handle to a LIFO pool, even while that handle is pending.
// The in-order queue completes all previous submissions only at an existing production wait.
// This is not the complete ACPP DAG: other runtime owners may extend node lifetime.
struct Pool {
  std::vector<bool> pending;
  std::vector<unsigned> available;
  std::vector<std::string> order;
  unsigned early_release = 0, rerecord = 0, waits = 0;
  unsigned acquire(const char* stage) {
    order.emplace_back(stage);
    unsigned id;
    if (available.empty()) {
      id = pending.size();
      pending.push_back(false);
    } else {
      id = available.back();
      available.pop_back();
      rerecord += pending[id];
    }
    pending[id] = true;
    return id;
  }
  void release(unsigned id) {
    early_release += pending[id];
    available.push_back(id);
  }
  void complete() {
    ++waits;
    for (std::size_t i = 0; i < pending.size(); ++i) pending[i] = false;
  }
};
static Pool pool;
static unsigned hazardous_cases = 0;
static bool emulate_in_order = false;
namespace sycl {
struct device {};
struct event {
  struct Owner {
    unsigned id;
    explicit Owner(unsigned value) : id(value) {}
    ~Owner() { pool.release(id); }
  };
  std::shared_ptr<Owner> owner;
  explicit event(const char* stage) : owner(std::make_shared<Owner>(pool.acquire(stage))) {}
};
struct queue {
  std::shared_ptr<event::Owner> previous_submission;
  event submit(const char* stage) {
    // Pinned ACPP queue.hpp execute_submission(): create/record the new node
    // before replacing previous_submission. This owner exists only for in-order
    // emulation; a dedicated bound executor disables that emulation in init().
    event next(stage);
    if (emulate_in_order) previous_submission = next.owner;
    return next;
  }
  event submit() {
    static const char* stages[] = {"stage 1", "stage 3", "stage 4"};
    return submit(stages[pool.order.size()]);
  }
};
}
static void sycl_wait_and_throw(const sycl::event&, const sycl::device&) { pool.complete(); }
struct Result {};
struct State {
  sycl::queue queue;
  sycl::device device;
  const uint8_t* input = nullptr;
  uint64_t* scratch = nullptr;
  const uint8_t* target = nullptr;
  Result* result = nullptr;
  const double* lut = nullptr;
  bool profile = false;
  unsigned profile_reports = 8; // Do not emit ordinary profiling chatter from the fixture.
};
#include "xelis-search.inc"

constexpr unsigned NBUCKETS = 1;
struct ZelHashState {
  sycl::queue queue;
  sycl::device device;
  uint32_t storage[5]{};
  uint32_t* level[4]{storage, storage, storage, storage};
  uint32_t* nslots = storage;
  uint32_t* ref = storage;
  uint32_t* ref_count = storage;
  unsigned slot_capacity = 1;
  bool gpu_compact = true;
  static constexpr unsigned CAND_CAP = 1;
  static uint64_t now_ms() { return 0; }
};
template <unsigned Round, bool Compact, class... Arguments>
static sycl::event submit_round(sycl::queue& queue, Arguments...) {
  static const char* names[] = {"r0", "r1", "r2", "r3"};
  return queue.submit(names[Round]);
}
#include "zel-rounds.inc"

static bool check(const std::vector<std::string>& expected, unsigned waits, bool hazardous) {
  if (pool.order != expected || pool.waits != waits ||
      (pool.early_release != 0) != hazardous || (pool.rerecord != 0) != hazardous) return false;
  for (bool pending : pool.pending) if (pending) return false;
  hazardous_cases += hazardous;
  return true;
}
static bool xelis(bool profile, bool esimd, bool fp64) {
  pool = {};
  {
    State state;
    double lut = 0;
    state.profile = profile;
    state.lut = fp64 ? &lut : nullptr;
    search<true>(state, 0, 1, true, esimd);
  }
#ifdef MOM_SYCL_PORTABLE_OPENCL
  const bool per_stage = true;
#else
  const bool per_stage = profile || esimd;
#endif
  return check({"stage 1", "stage 3", "stage 4"}, per_stage ? 3 : 1,
               !CANDIDATE && !per_stage);
}
static bool zel(bool wait_stages, bool compact) {
  pool = {};
  {
    ZelHashState state;
    state.gpu_compact = compact;
    unsigned wg = 1;
    bool prof = wait_stages;
    uint64_t t_round[4]{};
    // These are the actual production compiler branches, including the caller's owner scope.
#include "zel-call.inc"
    sycl_wait_and_throw(state.queue.submit("recovery"), state.device);
  }
  return check({"r0", "r1", "r2", "r3", "recovery"}, wait_stages ? 5 : 1,
               !CANDIDATE && !wait_stages);
}
int main(int argc, char** argv) {
  if (argc != 3 || std::strlen(argv[1]) != 1 || std::strlen(argv[2]) != 1 ||
      (argv[1][0] != '0' && argv[1][0] != '1') ||
      (argv[2][0] != '0' && argv[2][0] != '1')) return 2;
  const bool compact = argv[1][0] == '1';
  emulate_in_order = argv[2][0] == '1';
  // Check the model's ordering directly: a queue owner protects the first
  // handle through submission two, but releases it before submission three.
  {
    sycl::queue queue;
    queue.submit("first");
    queue.submit("second");
    if (pool.rerecord != (emulate_in_order ? 0u : 1u)) return 13;
    queue.submit("third");
    if (pool.rerecord != (emulate_in_order ? 1u : 2u)) return 14;
    pool.complete();
  }
  for (bool fp64 : {false, true}) {
    if (!xelis(false, false, fp64) || !xelis(true, false, fp64) ||
        !xelis(false, true, fp64)) return 11;
  }
  if (!zel(false, compact) || !zel(true, compact)) return 12;
  std::printf("ASYNC_EVENT_LIFETIME candidate=%d portable=%d emulated=%d cases=8 hazardous=%u\n", CANDIDATE,
#ifdef MOM_SYCL_PORTABLE_OPENCL
              1,
#else
              0,
#endif
              emulate_in_order, hazardous_cases);
  return hazardous_cases ? 17 : 0;
}
