#include "../../sycl/device-state.h"

#include <atomic>
#include <cstdio>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

class TestContext {
  unsigned failures_ = 0;

public:
  void check(const bool condition, const char* message) {
    if (condition)
      return;
    std::fprintf(stderr, "FAIL: %s\n", message);
    ++failures_;
  }

  unsigned failures() const {
    return failures_;
  }
};

struct State {
  int value;
  std::atomic<unsigned>* destructors;

  State(const int value, std::atomic<unsigned>& destructors)
    : value(value), destructors(&destructors) {}

  ~State() {
    destructors->fetch_add(1, std::memory_order_relaxed);
  }
};

using Registry = DeviceStateRegistry<State>;

std::unique_ptr<State> make_state(const int value, std::atomic<unsigned>& destructors) {
  return std::make_unique<State>(value, destructors);
}

void test_throwing_factory(TestContext& context) {
  std::atomic<unsigned> destructors{0};
  Registry registry;
  bool threw = false;
  try {
    registry.get("retry", []() -> std::unique_ptr<State> {
      throw std::runtime_error("factory failed");
    });
  } catch (const std::runtime_error&) {
    threw = true;
  }
  context.check(threw, "throwing factory propagates its exception");

  bool saw_empty_entry = false;
  registry.clear([&](const auto& entry) {
    if (!entry.second)
      saw_empty_entry = true;
    return !entry.second;
  });
  context.check(!saw_empty_entry, "throwing factory leaves no empty entry for clear");

  State& retry = registry.get("retry", [&] {return make_state(7, destructors);});
  context.check(retry.value == 7, "retry after a throwing factory succeeds");
  registry.clear();
  context.check(destructors.load(std::memory_order_relaxed) == 1,
               "retried state is destroyed exactly once");
}

void test_null_factory(TestContext& context) {
  std::atomic<unsigned> destructors{0};
  Registry registry;
  bool threw = false;
  try {
    registry.get("null", []() -> std::unique_ptr<State> {return nullptr;});
  } catch (const std::logic_error&) {
    threw = true;
  } catch (...) {
    context.check(false, "null factory throws the expected exception type");
  }
  context.check(threw, "null factory throws");

  bool saw_entry = false;
  registry.clear([&](const auto&) {
    saw_entry = true;
    return true;
  });
  context.check(!saw_entry, "null factory leaves the registry empty");

  State& retry = registry.get("null", [&] {return make_state(8, destructors);});
  context.check(retry.value == 8, "retry after a null factory succeeds");
  registry.clear();
  context.check(destructors.load(std::memory_order_relaxed) == 1,
               "retried null-factory state is destroyed exactly once");
}

void test_cached_factory(TestContext& context) {
  std::atomic<unsigned> destructors{0};
  Registry registry;
  unsigned factory_calls = 0;
  State& first = registry.get("cached", [&] {
    ++factory_calls;
    return make_state(11, destructors);
  });
  State& second = registry.get("cached", [&] {
    ++factory_calls;
    return make_state(12, destructors);
  });
  context.check(&first == &second, "cached get returns the existing state");
  context.check(factory_calls == 1, "cached get does not invoke the factory");
  context.check(first.value == 11, "cached state keeps its original value");
  registry.clear();
  context.check(destructors.load(std::memory_order_relaxed) == 1,
               "cached state is destroyed exactly once");
}

void test_selective_cleanup(TestContext& context) {
  std::atomic<unsigned> destructors{0};
  Registry registry;
  unsigned keep_factory_calls = 0;
  State& keep = registry.get("keep", [&] {
    ++keep_factory_calls;
    return make_state(21, destructors);
  });
  State& erase = registry.get("erase", [&] {return make_state(22, destructors);});
  context.check(keep.value == 21 && erase.value == 22, "different keys create distinct states");

  registry.clear([](const auto& entry) {return entry.first == "erase";});
  context.check(destructors.load(std::memory_order_relaxed) == 1,
               "selective cleanup destroys only the erased state");
  State& retained = registry.get("keep", [&] {
    ++keep_factory_calls;
    return make_state(23, destructors);
  });
  context.check(&retained == &keep, "selective cleanup retains the other state");
  context.check(keep_factory_calls == 1, "retained state does not invoke its factory");
  State& recreated = registry.get("erase", [&] {return make_state(24, destructors);});
  context.check(recreated.value == 24, "selective cleanup permits erased state recreation");

  registry.clear([](const auto& entry) {return entry.first == "keep";});
  context.check(destructors.load(std::memory_order_relaxed) == 2,
               "second selective cleanup destroys the retained state");
  registry.clear();
  context.check(destructors.load(std::memory_order_relaxed) == 3,
               "final cleanup destroys the recreated state exactly once");
}

void test_concurrent_factory(TestContext& context) {
  std::atomic<unsigned> destructors{0};
  std::atomic<unsigned> factory_calls{0};
  Registry registry;
  std::mutex addresses_mutex;
  std::vector<State*> addresses;
  constexpr unsigned thread_count = 8;
  std::vector<std::thread> threads;
  threads.reserve(thread_count);
  for (unsigned i = 0; i < thread_count; ++i) {
    threads.emplace_back([&] {
      State& state = registry.get("concurrent", [&] {
        factory_calls.fetch_add(1, std::memory_order_relaxed);
        return make_state(31, destructors);
      });
      std::lock_guard<std::mutex> lock(addresses_mutex);
      addresses.push_back(&state);
    });
  }
  for (auto& thread : threads)
    thread.join();

  context.check(factory_calls.load(std::memory_order_relaxed) == 1,
               "concurrent get constructs one state");
  context.check(addresses.size() == thread_count, "concurrent get returns every result");
  for (State* state : addresses)
    context.check(state == addresses.front(), "concurrent get returns one shared state");
  registry.clear();
  context.check(destructors.load(std::memory_order_relaxed) == 1,
               "concurrent state is destroyed exactly once");
}

} // namespace

int main() {
  TestContext context;
  test_throwing_factory(context);
  test_null_factory(context);
  test_cached_factory(context);
  test_selective_cleanup(context);
  test_concurrent_factory(context);
  if (context.failures() != 0) {
    std::fprintf(stderr, "device-state tests failed: %u\n", context.failures());
    return 1;
  }
  std::puts("device-state tests passed");
  return 0;
}
