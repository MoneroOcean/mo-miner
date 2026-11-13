// Runtime-independent contract test; including the real worker must not require Node headers.
#ifndef MOM_WORKER_HEADER
#define MOM_WORKER_HEADER "../../native/async-worker.h"
#endif
#include MOM_WORKER_HEADER
#include <condition_variable>
#include <functional>
#include <stdexcept>
#include <vector>

static void require(bool valid, const char *message) {
  if (!valid) {throw std::runtime_error(message);}
}
struct State {
  std::mutex mutex;
  std::condition_variable changed;
  std::vector<WorkerEvent> events;
  std::thread::id startThread, eventThread, finishThread;
  unsigned starts = 0, finishes = 0, destroyed = 0;
  bool failStart = false;
  std::function<void()> terminalHook;
};
class Channel final : public WorkerEvents {
  std::shared_ptr<State> state;
public:
  explicit Channel(std::shared_ptr<State> value) : state(std::move(value)) {}
  void start() override {
    const std::lock_guard lock(state->mutex);
    ++state->starts;
    state->startThread = std::this_thread::get_id();
    if (state->failStart) {state->failStart = false; throw std::runtime_error("channel start failed");}
  }
  void publish(WorkerEvent event) noexcept override {
    const std::lock_guard lock(state->mutex);
    state->eventThread = std::this_thread::get_id();
    if (event.type == WorkerEvent::Type::complete && state->terminalHook) {state->terminalHook();}
    state->events.push_back(std::move(event));
  }
  void finish() noexcept override {
    const std::lock_guard lock(state->mutex);
    state->finishThread = std::this_thread::get_id();
    ++state->finishes;
    state->changed.notify_all();
  }
};
class Worker final : public AsyncWorker {
  std::shared_ptr<State> state;
  void Execute() override {
    for (;;) {
      for (auto &message : fromNode.drain()) {
        if (message.name == "close" || message.name == "return") {return;}
        if (message.name == "string") {throw std::string("string failure");}
        if (message.name == "standard") {throw std::runtime_error("standard failure");}
        if (message.name == "unknown") {throw 42;}
        sendToNode(std::move(message));
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }
public:
  explicit Worker(std::shared_ptr<State> value)
      : AsyncWorker(std::make_unique<Channel>(value)), state(std::move(value)) {}
  ~Worker() override {stop(); ++state->destroyed;}
};
static void wait(const std::shared_ptr<State> &state) {
  std::unique_lock lock(state->mutex);
  require(state->changed.wait_for(lock, std::chrono::seconds(2), [&] {return state->finishes != 0;}), "worker did not finish");
}
static void stopped(AsyncWorker &worker) {
  try {worker.post({"late", {}});} catch (const std::string &error) {
    require(error == "AsyncWorker is stopped", "wrong stopped error"); return;
  }
  throw std::runtime_error("post after close was accepted");
}

static void run(const std::string &name) {
  auto state = std::make_shared<State>();
  if (name == "idle") {
    {Worker worker(state);}
    require(state->starts == 0 && state->finishes == 0 && state->destroyed == 1, "idle worker started or leaked");
  } else if (name == "echo" || name == "payload" || name == "order" || name == "threads") {
    {Worker worker(state);
      worker.post({"first", {{"value", std::string("é-\0-payload", 13)}}});
      worker.post({"second", {}}); worker.post({"close", {}}); wait(state);
    }
    require(state->events.size() == 3, "wrong event count");
    require(state->events[0].name == "first" && state->events[1].name == "second" &&
        state->events[2].type == WorkerEvent::Type::complete, "progress/completion order changed");
    require(state->events[0].values.at("value") == std::string("é-\0-payload", 13), "event bytes changed");
    require(state->startThread == std::this_thread::get_id() && state->eventThread != state->startThread &&
        state->finishThread == state->eventThread, "channel thread boundary changed");
  } else if (name == "terminalPost") {
    Worker worker(state);
    bool rejected = false;
    state->terminalHook = [&] {
      try {worker.post({"late", {}});} catch (const std::string &) {rejected = true;}
    };
    worker.post({"return", {}}); wait(state);
    require(rejected, "terminal callback could post into a finished worker");
  } else if (name == "postClose") {
    Worker worker(state); worker.post({"close", {}}); stopped(worker); wait(state);
  } else if (name == "destructor" || name == "retainedPayload") {
    {Worker worker(state); worker.post({"saved", {{"value", "retained"}}});}
    require(state->finishes == 1 && state->destroyed == 1 && state->events.size() == 2 &&
        state->events[0].values.at("value") == "retained", "destruction cancelled owned events");
  } else if (name == "retryStart") {
    state->failStart = true;
    {Worker worker(state);
      bool failed = false;
      try {worker.post({"first", {}});} catch (const std::runtime_error &) {failed = true;}
      require(failed, "failed channel startup was ignored");
      worker.post({"close", {}}); wait(state);
    }
    require(state->starts == 2 && state->finishes == 1 && state->events.size() == 1, "failed startup was not retryable");
  } else if (name == "string" || name == "standard" || name == "unknown") {
    Worker worker(state); worker.post({name, {}}); wait(state); stopped(worker);
    require(state->events.size() == 1 && state->events[0].type == WorkerEvent::Type::error,
        "exception did not produce one terminal error");
    const auto expected = name == "unknown" ? "Compute worker exception" : name + " failure";
    require(state->events[0].error == expected, "exception detail lost");
  } else if (name == "repeatedClose") {
    {Worker worker(state); worker.post({"close", {}}); wait(state); stopped(worker);}
    require(state->finishes == 1 && state->destroyed == 1, "channel finish repeated");
  } else if (name == "independent") {
    auto other = std::make_shared<State>();
    {Worker first(state), second(other);
      first.post({"a", {}}); second.post({"b", {}});
      first.post({"close", {}}); second.post({"close", {}}); wait(state); wait(other);
    }
    require(state->events[0].name == "a" && other->events[0].name == "b", "worker channels crossed");
  } else if (name == "nullChannel") {
    // Construction must fail before the abstract execution path can become active.
    class Empty final : public AsyncWorker {void Execute() override {};
      public: Empty() : AsyncWorker(nullptr) {}};
    try {Empty worker;} catch (const std::invalid_argument &) {return;}
    throw std::runtime_error("missing channel was accepted");
  } else if (name == "queue") {
    MessageQueue<Message> queue;
    std::vector<std::thread> producers;
    for (unsigned producer = 0; producer < 4; ++producer) {
      producers.emplace_back([&, producer] {
        for (unsigned index = 0; index < 100; ++index) {queue.write({std::to_string(producer), {{"index", std::to_string(index)}}});}
      });
    }
    for (auto &producer : producers) {producer.join();}
    unsigned next[4]{};
    const auto messages = queue.drain();
    require(messages.size() == 400 && queue.drain().empty(), "queue dropped or duplicated messages");
    for (const auto &message : messages) {
      const auto producer = static_cast<unsigned>(std::stoul(message.name));
      require(producer < 4 && std::stoul(message.values.at("index")) == next[producer]++, "producer FIFO changed");
    }
  } else {throw std::runtime_error("unknown fixture case");}
}
int main(int argc, char **argv) {
  if (argc != 2) {return 2;}
  try {run(argv[1]);} catch (const std::exception &error) {std::fprintf(stderr, "%s\n", error.what()); return 1;}
  return 0;
}
