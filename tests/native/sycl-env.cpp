// SYCL program build options are published on the worker's loop thread before its compute thread
// starts; the compute thread never writes the environment while other threads read it.
#include "async-worker.h"

#include <cassert>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

static std::mutex setenv_mutex;
static std::vector<std::thread::id> setenv_threads;
extern "C" int __real_setenv(const char* name, const char* value, int overwrite);
extern "C" int __wrap_setenv(const char* name, const char* value, int overwrite) {
  {
    const std::lock_guard lock(setenv_mutex);
    setenv_threads.push_back(std::this_thread::get_id());
  }
  return __real_setenv(name, value, overwrite);
}

static size_t setenv_calls() {
  const std::lock_guard lock(setenv_mutex);
  return setenv_threads.size();
}

#include "sycl_env.inc"

constexpr const char* name = "SYCL_PROGRAM_COMPILE_OPTIONS";

// An empty request must leave per-image options alone; unchanged values are not rewritten.
static void test_publication() {
  unsetenv(name);
  mom_set_compile_options_env(nullptr);
  mom_set_compile_options_env("");
  assert(setenv_calls() == 0 && !std::getenv(name));
  mom_set_compile_options_env("-O2");
  assert(setenv_calls() == 1 && std::strcmp(std::getenv(name), "-O2") == 0);
  mom_set_compile_options_env("-O2");
  assert(setenv_calls() == 1);
  mom_set_compile_options_env("-O1");
  assert(setenv_calls() == 2 && std::strcmp(std::getenv(name), "-O1") == 0);
  unsetenv(name);
}

class Events final : public WorkerEvents {
public:
  std::mutex mutex;
  std::condition_variable changed;
  bool finished = false;
  void start() override {}
  void publish(WorkerEvent) noexcept override {}
  void finish() noexcept override {
    const std::lock_guard lock(mutex);
    finished = true;
    changed.notify_all();
  }
};

// Mirrors Core: prepare() publishes the first job's options; Execute only reads the environment.
class Worker final : public AsyncWorker {
public:
  std::thread::id prepare_thread;
  std::string seen;
  explicit Worker(std::unique_ptr<WorkerEvents> events) : AsyncWorker(std::move(events)) {}
  ~Worker() override { stop(); }

protected:
  void prepare(const Message& first) override {
    prepare_thread = std::this_thread::get_id();
    mom_set_compile_options_env(first.values.at("options").c_str());
  }
  void Execute() override {
    for (;;) {
      for (const Message& message : fromNode.drain()) {
        if (message.name == "close")
          return;
        const char* const value = std::getenv(name);
        seen = value ? value : "";
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }
};

static void test_loop_thread_publication() {
  auto events = std::make_unique<Events>();
  Events& channel = *events;
  const size_t before = setenv_calls();
  {
    Worker worker(std::move(events));
    worker.post(Message("job", {{"options", "-ze-opt-large-register-file"}}));
    worker.post(Message("job", {{"options", "-O3"}}));
    worker.post(Message("close", {}));
    std::unique_lock lock(channel.mutex);
    channel.changed.wait(lock, [&] { return channel.finished; });
    lock.unlock();
    assert(worker.prepare_thread == std::this_thread::get_id());
    assert(worker.seen == "-ze-opt-large-register-file");
  }
  const std::lock_guard lock(setenv_mutex);
  assert(setenv_threads.size() == before + 1);
  for (const std::thread::id thread : setenv_threads)
    assert(thread == std::this_thread::get_id());
}

int main() {
  test_publication();
  test_loop_thread_publication();
  return 0;
}
