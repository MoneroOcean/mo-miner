#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

constexpr uint32_t HASH_LEN = 32, MAX_ETCHASH_OUTPUTS = 15, ETHASH_MIX_WORDS = 32;
#include "result.inc"
struct FastModData {};
static FastModData make_fast_mod_data(uint32_t) { return {}; }

struct Counters {
  int allocations = 0, frees = 0, copies = 0, copy_waits = 0, drains = 0;
  int inline_probes = 0, reference_probes = 0, inline_searches = 0, reference_searches = 0;
} counts;
static int scenario, stage, kind;
static void* resources[5]{};
static const void* exception_address;
struct PendingCopy {
  void* destination;
  const void* source;
  size_t bytes;
};
static std::vector<PendingCopy> pending_copies;

static void require(bool condition, const char* message) {
  if (!condition)
    throw std::logic_error(message);
}

static void inject() {
  if (kind == 1) {
    try {
      throw std::string("probe fault");
    } catch (const std::string& error) {
      if (!exception_address)
        exception_address = &error;
      throw;
    }
  }
  if (kind == 2) {
    try {
      throw 61;
    } catch (const int& error) {
      if (!exception_address)
        exception_address = &error;
      throw;
    }
  }
  try {
    throw std::runtime_error("probe fault");
  } catch (const std::exception& error) {
    if (!exception_address)
      exception_address = &error;
    throw;
  }
}

namespace sycl {
struct device {};
struct event { int probe = 0; };
struct queue {
  event memcpy(void* destination, const void* source, size_t bytes) {
    const int copy = ++counts.copies;
    for (void* pointer : resources)
      if (!pointer)
        throw std::runtime_error("null allocation");
    if (scenario == 6 && copy == stage)
      inject();
    pending_copies.push_back({destination, source, bytes});
    return {};
  }
  void wait() { ++counts.copy_waits; }
  void wait_and_throw() {
    ++counts.copy_waits;
    for (const PendingCopy& copy : pending_copies)
      std::memcpy(copy.destination, copy.source, copy.bytes);
    pending_copies.clear();
    if (scenario == 7)
      inject();
    if (scenario >= 6)
      ++counts.drains;
  }
};
template <typename T> static T* malloc_device(size_t count, queue&) {
  const int allocation = ++counts.allocations;
  require(allocation <= 5, "probe allocations repeated unexpectedly");
  if (scenario == 5 && allocation == stage)
    inject();
  void* pointer = scenario == 4 && allocation == stage ? nullptr : std::calloc(count, sizeof(T));
  if (!(scenario == 4 && allocation == stage))
    require(pointer != nullptr, "host fixture allocation failed");
  resources[allocation - 1] = pointer;
  return static_cast<T*>(pointer);
}
template <typename T> static T* malloc_shared(size_t count, queue& queue) {
  return malloc_device<T>(count, queue);
}
static void free(void* pointer, queue&) {
  require(pending_copies.empty(), "probe resource released before accepted copies completed");
  bool found = false;
  for (void*& owned : resources) {
    if (owned == pointer) {
      found = true;
      owned = nullptr;
      break;
    }
  }
  require(found && pointer, "probe resource released twice or was not owned");
  std::free(pointer);
  ++counts.frees;
  if (scenario == 12 && counts.frees == 1)
    throw std::string("cleanup failure");
  if (scenario == 15 && counts.frees == 1)
    inject();
}
}

#include "cleanup.inc"
static void sycl_wait_and_throw(sycl::event event, sycl::device) {
  if ((scenario == 7 && event.probe) || (scenario == 9 && event.probe == 1) ||
      ((scenario == 11 || scenario == 12) && event.probe == 2))
    inject();
}
static bool mom_parse_env_ulong(const char* name, unsigned long& value) {
  const bool forced = (scenario == 13 && std::strcmp(name, "MOM_ETCHASH_FORCE_NOINLINE") == 0) ||
      (scenario == 14 && std::strcmp(name, "MOM_ETCHASH_FORCE_INLINE") == 0);
  value = forced ? 1 : 0;
  return forced;
}
template <bool Inline>
static sycl::event submit_etchash_search_gpu(sycl::queue&, const uint8_t*, uint64_t,
    const uint32_t*, FastModData, uint32_t intensity, const uint8_t*, EtchashResult* result,
    bool is_test) {
  if (is_test) {
    require(intensity == 256, "probe nonce count changed");
    if (Inline)
      ++counts.inline_probes;
    else
      ++counts.reference_probes;
    if ((scenario == 8 && Inline) || (scenario == 10 && !Inline))
      inject();
    result->count = scenario == 3 && Inline ? 0 : 1;
    std::memset(result->output[0], scenario == 1 && Inline ? 29 : 17, HASH_LEN);
    std::memset(result->mix_hash[0], scenario == 2 && Inline ? 29 : 17, HASH_LEN);
    return {Inline ? 1 : 2};
  }
  require(intensity == 17, "production caller intensity changed");
  if (Inline)
    ++counts.inline_searches;
  else
    ++counts.reference_searches;
  result->count = 1;
  result->nonce[0] = 99;
  std::memset(result->output[0], 17, HASH_LEN);
  std::memset(result->mix_hash[0], 19, HASH_LEN);
  return {};
}

class EtchashState {
public:
  int inline_pair_ok = -1;
  sycl::device device;
  sycl::queue queue;
  EtchashResult* result = nullptr;
#include "probe.inc"
};

static int caller(EtchashState& state, uint8_t* output, uint8_t* mix_hash, uint64_t* pnonce) {
  sycl::queue& q = state.queue;
  uint8_t input[40]{}, target[HASH_LEN]{};
  uint32_t dag[1]{};
  EtchashResult result{};
  state.result = &result;
  EtchashResult* d_result = &result;
  const uint8_t* d_input = input;
  const uint8_t* d_target = target;
  const uint32_t* d_dag = dag;
  const FastModData dag_mod{};
  const uint64_t start_nonce = 0;
  const uint32_t intensity = 17;
  const bool is_test = false;
#include "selection.inc"
#include "commit.inc"
}

static void run() {
  EtchashState state;
  uint8_t output[HASH_LEN], mix[HASH_LEN];
  std::memset(output, 23, sizeof(output));
  std::memset(mix, 23, sizeof(mix));
  uint64_t nonce = 71;
  bool caught = false, same_exception = false;
  int returned = -1;
  try {
    returned = caller(state, output, mix, &nonce);
  } catch (const std::string& error) {
    caught = true;
    same_exception = kind == 1 && &error == exception_address;
  } catch (const std::exception& error) {
    caught = true;
    same_exception = kind == 0 && &error == exception_address;
  } catch (const int& error) {
    caught = true;
    same_exception = kind == 2 && &error == exception_address;
  }
  const bool fault = (scenario >= 6 && scenario <= 12) || scenario == 15;
  require(caught == fault, "submitted probe fault was swallowed before the real search caller");
  if (fault) {
    require(same_exception, "probe cleanup changed the original fault type/identity");
    require(returned == -1 && nonce == 71 && output[0] == 23 && mix[0] == 23 &&
            counts.inline_searches == 0 && counts.reference_searches == 0 && state.inline_pair_ok == -1,
            "probe fault selected/cached another kernel or committed a result");
    require(counts.frees == 5, "probe fault did not attempt all resource releases once");
    if (scenario <= 7)
      require(counts.inline_probes == 0 && counts.reference_probes == 0,
              "failed input copy continued into a probe search");
    const int failed_scenario = scenario;
    scenario = 0;
    counts = {};
    exception_address = nullptr;
    require(caller(state, output, mix, &nonce) == 1 && state.inline_pair_ok == 1 &&
            counts.inline_probes == 1 && counts.reference_probes == 1 && counts.inline_searches == 1,
            "later explicit job skipped a new probe after failure");
    scenario = failed_scenario;
  } else {
    const bool inline_expected = scenario == 0 || scenario == 14;
    require(returned == 1 && nonce == 99 && output[0] == 17 && mix[0] == 19 &&
            state.inline_pair_ok == (inline_expected ? 1 : 0), "numeric/absence/forced route changed");
    require(counts.inline_searches == (inline_expected ? 1 : 0) &&
            counts.reference_searches == (inline_expected ? 0 : 1), "caller selected wrong search");
    if (scenario == 4 || scenario == 5) {
      require(counts.copies == 0 && counts.inline_probes == 0 && counts.reference_probes == 0,
              "allocation absence submitted probe work");
      require(counts.frees == (scenario == 4 ? 4 : stage - 1), "partial allocation cleanup changed");
    } else if (scenario <= 3) {
      require(counts.inline_probes == 1 && counts.reference_probes == 1 && counts.frees == 5,
              "successful numeric probe did not run/retire both variants");
    }
    const int allocations = counts.allocations;
    require(caller(state, output, mix, &nonce) == 1 && counts.allocations == allocations,
            "successful/unavailable probe stopped being cached");
  }
  for (void* pointer : resources)
    require(!pointer, "probe leaked a resource");
  require(pending_copies.empty(), "probe left an accepted copy pending");
}

int main(int argc, char** argv) {
  if (argc != 4)
    return 2;
  scenario = std::atoi(argv[1]);
  stage = std::atoi(argv[2]);
  kind = std::atoi(argv[3]);
  unsetenv("MOM_ETCHASH_SELFTEST_LOG");
  unsetenv("MOM_SYCL_CLEANUP_DEBUG");
  try {
    run();
    std::puts("PASS actual-source Etchash probe fault boundary");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
