// GPU-free execution of the real process-wide compiler gate, including nested failure unwind.
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <future>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

extern "C" void mom_native_compiler_lock();
extern "C" void mom_native_compiler_unlock();
struct Guard {
  Guard() { mom_native_compiler_lock(); }
  ~Guard() { mom_native_compiler_unlock(); }
};

static void nested(unsigned depth, bool fail) {
  Guard gate;
  if (depth > 1) nested(depth - 1, fail);
  else if (fail) throw std::runtime_error("expected");
}

int main(int argc, char** argv) {
  assert(argc == 3);
  const std::string mode = argv[1];
  const unsigned count = std::stoul(argv[2]);
  assert(count > 0 && count <= 16);
  if (mode == "nest" || mode == "throw") {
    bool threw = false;
    try { nested(count, mode == "throw"); } catch (const std::runtime_error&) { threw = true; }
    assert(threw == (mode == "throw"));
    // A different thread must acquire it after every recursive level has unwound.
    auto next = std::async(std::launch::async, [] { Guard gate; return true; });
    assert(next.wait_for(std::chrono::seconds(2)) == std::future_status::ready && next.get());
  } else {
    assert(mode == "threads");
    std::atomic<unsigned> active{0}, completed{0};
    std::vector<std::thread> threads;
    for (unsigned i = 0; i < count; ++i) threads.emplace_back([&] {
      for (unsigned round = 0; round < 100; ++round) {
        Guard gate;
        assert(active.fetch_add(1) == 0);
        nested(3, false);
        std::this_thread::yield();
        assert(active.fetch_sub(1) == 1);
        ++completed;
      }
    });
    for (auto& thread : threads) thread.join();
    assert(active == 0 && completed == count * 100);
  }
}
