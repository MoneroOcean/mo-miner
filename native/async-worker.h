// Copyright GNU GPLv3 (c) 2023-2026 MoneroOcean <support@moneroocean.stream>

#pragma once

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <stdexcept>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

using MessageValues = std::map<std::string, std::string>;

// On/off environment switch with the rule of sycl/lib-internal.h mom_env_flag, which native code
// cannot include (the CPU Core builds without SYCL): unset, empty, "0", "false" and "off" are off.
static bool mom_native_env_enabled(const char* const name) {
  const char* const value = std::getenv(name);
  return value && *value && std::strcmp(value, "0") != 0 && std::strcmp(value, "false") != 0 &&
         std::strcmp(value, "off") != 0;
}

// MOM_DEBUG_STARTUP trace line "<message><detail>" for diagnosing worker startup and message
// hangs. The variable is read once, and callers pass dynamic text as `detail` so that no string
// is built when tracing is off (send_msg runs on every hashrate and share message).
static void debug_async_worker(const std::string_view message,
                               const std::string_view detail = {}) {
  static const bool enabled = mom_native_env_enabled("MOM_DEBUG_STARTUP");
  if (!enabled) {
    return;
  }
  std::fprintf(stderr, "MOM_DEBUG_STARTUP %.*s%.*s\n",
               static_cast<int>(message.size()), message.data(),
               static_cast<int>(detail.size()), detail.data());
  std::fflush(stderr);
}

struct Message {
  std::string name;
  MessageValues values;

  Message(std::string name, MessageValues values)
    : name(std::move(name)), values(std::move(values)) {}
};

template <typename T> class MessageQueue {
  std::mutex m_mutex;
  std::deque<T> m_buff;

public:

  // The two trace lines bracket the only blocking step, the queue lock.
  void write(T data) {
    debug_async_worker("MessageQueue write locking");
    {
      const std::lock_guard lock(m_mutex);
      m_buff.emplace_back(std::move(data));
    }
    debug_async_worker("MessageQueue write done");
  }

  std::deque<T> drain() {
    const std::lock_guard lock(m_mutex);
    std::deque<T> result;
    result.swap(m_buff);
    return result;
  }
};

// Runtime-neutral event contract. start() runs on the owner thread before execution; publish()
// and finish() run on the compute thread (or owner on failed launch), never touching JS handles. Each runtime
// adapter owns queued payloads until delivery and releases its keepalive only after the final event.
struct WorkerEvent {
  enum class Type {progress, complete, error};
  Type type;
  std::string name;
  MessageValues values;
  std::string error;
  explicit WorkerEvent(Message message)
      : type(Type::progress), name(std::move(message.name)), values(std::move(message.values)) {}
  WorkerEvent(Type type, std::string error = {}) : type(type), error(std::move(error)) {}
};

class WorkerEvents {
public:
  virtual ~WorkerEvents() = default;
  virtual void start() = 0;
  virtual void publish(WorkerEvent event) noexcept = 0;
  virtual void finish() noexcept = 0;
};

// One message queue, compute thread and shutdown policy serve both Node development and native
// releases. Only event delivery differs; algorithms and command processing stay in Core.
class AsyncWorker {
  std::unique_ptr<WorkerEvents> m_events;
  std::thread m_thread;
  std::atomic<bool> m_started{false};
  std::atomic<bool> m_stopped{false};
  std::atomic<bool> m_accepting{true};

  void terminal(WorkerEvent::Type type, std::string message = {}) noexcept {
    try {m_events->publish(WorkerEvent(type, std::move(message)));} catch (...) {}
  }
  // Compute-thread body: run Execute() and publish exactly one terminal event (complete or
  // error), after which the worker accepts no more messages and its event channel finishes.
  void run() {
    try {
      debug_async_worker("AsyncWorker run entered");
      Execute();
      debug_async_worker("AsyncWorker Execute returned");
      m_accepting = false;
      terminal(WorkerEvent::Type::complete);
    } catch (const std::string& error) {
      m_accepting = false;
      terminal(WorkerEvent::Type::error, error);
    } catch (const std::exception& error) {
      m_accepting = false;
      terminal(WorkerEvent::Type::error, error.what());
    } catch (...) {
      m_accepting = false;
      terminal(WorkerEvent::Type::error, "Compute worker exception");
    }
    m_stopped = true;
    m_events->finish();
  }
  // Open the event channel and start the compute thread on the first post(); a failure before the
  // channel opened leaves the worker startable again.
  void start() {
    bool expected = false;
    if (!m_started.compare_exchange_strong(expected, true)) {return;}
    bool opened = false;
    try {
      m_events->start();
      opened = true;
      m_thread = std::thread([this] {run();});
    } catch (...) {
      if (opened) {
        m_accepting = false;
        m_stopped = true;
        m_events->finish();
      } else {m_started = false;}
      throw;
    }
  }

protected:
  MessageQueue<Message> fromNode;

  // Derived destructors must stop while their resources are still alive; the base destructor
  // repeats this idempotently. Never release an event channel underneath its compute thread.
  void stop() {
    if (m_started && !m_stopped && m_accepting.exchange(false)) {
      fromNode.write(Message("close", {}));
    }
    if (m_thread.joinable()) {m_thread.join();}
  }
  // Runs on the posting (loop) thread with the first message, before the compute thread exists:
  // the only point where process-wide setup such as an environment write cannot race it.
  virtual void prepare(const Message&) {}
  void sendToNode(Message message) {
    try {m_events->publish(WorkerEvent(std::move(message)));} catch (...) {}
  }
  virtual void Execute() = 0;

public:
  explicit AsyncWorker(std::unique_ptr<WorkerEvents> events) : m_events(std::move(events)) {
    if (!m_events) {throw std::invalid_argument("Worker event channel is required");}
  }
  virtual ~AsyncWorker() {stop();}
  // Queue a controller message for the compute thread, starting it on first use. "close" is the
  // last message accepted; posting after the worker stopped throws.
  void post(Message message) {
    if (!m_accepting.load()) {throw std::string("AsyncWorker is stopped");}
    if (!m_started.load()) {prepare(message);}
    start();
    if (!m_accepting.load()) {throw std::string("AsyncWorker is stopped");}
    const bool closes = message.name == "close";
    fromNode.write(std::move(message));
    if (closes) {m_accepting = false;}
  }
};

std::unique_ptr<AsyncWorker> create_worker(std::unique_ptr<WorkerEvents> events);
