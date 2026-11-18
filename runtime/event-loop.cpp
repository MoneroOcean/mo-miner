#include "event-loop.h"
#include "tls.h"
#include "bindings.h"
#include "process.h"
#include "environment.h"
#include "jsi/jsi.h"
#include <uv.h>

#include <cmath>
#include <condition_variable>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <array>
#include <csignal>

using namespace facebook::jsi;
namespace mom::runtime::detail { struct Resource; }

struct SHConsoleContext {
  Runtime &runtime;
  uv_loop_t loop{};
  std::map<uint32_t, std::shared_ptr<mom::runtime::detail::Resource>> resources;
  std::deque<std::function<void()>> deferred;
  std::deque<std::function<void()>> nextTicks;
  uint32_t sequence = 0;
  std::array<uint32_t, 2> outputs{};
  bool closing = false;
  std::string failure;
  std::unique_ptr<Function> rejectionCheckpoint;
  mom::runtime::WorkerFactory workerFactory;
  SHConsoleContext(Runtime &runtime, mom::runtime::WorkerFactory factory) : runtime(runtime), workerFactory(factory) {}
};

namespace mom::runtime::detail {
enum class Kind { timer, immediate, tcp, worker, pipe, process, signal };
struct PlainWrite {std::vector<char> bytes;};

// Keep DNS fan-out bounded. A resolver can return duplicate or attacker-controlled records, but
// one native socket never needs more than a small, deterministic candidate set.
constexpr size_t maximumConnectAddresses = 64;
constexpr uint32_t defaultConnectAttemptTimeout = 250;
struct ConnectAddress {
  sockaddr_storage storage{};
  int family = AF_UNSPEC;
};

struct QueuedEvent {WorkerEvent event; size_t bytes;};

struct WorkerState {
  Function complete, error;
  std::mutex mutex;
  std::condition_variable space;
  std::deque<QueuedEvent> events;
  size_t queuedBytes = 0;
  bool done = false, aborting = false, failed = false;
  std::unique_ptr<AsyncWorker> core;
  WorkerState(Function complete, Function error) : complete(std::move(complete)), error(std::move(error)) {}
};

// Stable heap storage is mandatory: libuv retains handle/request addresses until close callbacks.
struct Resource {
  SHConsoleContext &owner;
  uint32_t id;
  Kind kind;
  uv_timer_t timer{};
  uv_check_t check{};
  uv_idle_t idle{};
  uv_tcp_t tcp{};
  uv_connect_t connect{};
  uv_shutdown_t shutdown{};
  uv_async_t async{};
  uv_pipe_t pipe{};
  uv_process_t process{};
  uv_signal_t signal{};
  uv_getaddrinfo_t *resolver = nullptr;
  Function callback;
  std::unique_ptr<mom::runtime::TlsSession> tls;
  std::unique_ptr<WorkerState> worker;
  std::deque<PlainWrite> plainWrites;
  size_t queued = 0;
  size_t wireQueued = 0;
  bool closing = false;
  bool hadError = false;
  bool ending = false;
  bool readEnded = false;
  bool shutdownStarted = false;
  bool shutdownComplete = false;
  bool secure = false;
  bool pumping = false;
  bool pumpScheduled = false;
  bool processRunning = false;
  bool terminating = false;
  bool idleInitialized = false;
  bool tcpInitialized = false;
  bool connectTimerInitialized = false;
  bool connectTimerActive = false;
  bool connectPending = false;
  bool connectResetPending = false;
  bool tcpRef = true;
  uint32_t connectAttemptTimeout = defaultConnectAttemptTimeout;
  uint64_t connectGeneration = 0;
  size_t nextConnectAddress = 0;
  std::vector<ConnectAddress> connectAddresses;
  unsigned closingHandles = 0;
  std::string channel;
  std::array<uint32_t, 3> pipes{};

  Resource(SHConsoleContext &owner, Kind kind, Function callback)
      : owner(owner), id(++owner.sequence), kind(kind), callback(std::move(callback)) {}
  uv_handle_t *handle() {
    if (kind == Kind::timer) {return reinterpret_cast<uv_handle_t *>(&timer);}
    if (kind == Kind::immediate) {return reinterpret_cast<uv_handle_t *>(&check);}
    if (kind == Kind::tcp) {return reinterpret_cast<uv_handle_t *>(&tcp);}
    if (kind == Kind::pipe) {return reinterpret_cast<uv_handle_t *>(&pipe);}
    if (kind == Kind::process) {return reinterpret_cast<uv_handle_t *>(&process);}
    if (kind == Kind::signal) {return reinterpret_cast<uv_handle_t *>(&signal);}
    return reinterpret_cast<uv_handle_t *>(&async);
  }
  uv_stream_t *stream() {
    return kind == Kind::pipe ? reinterpret_cast<uv_stream_t *>(&pipe) : reinterpret_cast<uv_stream_t *>(&tcp);
  }
  // Stop the per-address connect timer; harmless when it is idle.
  void stopConnectTimer() {
    connectTimerActive = false;
    if (connectTimerInitialized) {uv_timer_stop(&timer);}
  }
};

// libuv read allocator: one 64-KiB heap chunk per read, owned by the read callback.
void allocateRead(uv_handle_t *, size_t, uv_buf_t *buffer) {
  buffer->base = new (std::nothrow) char[65536];
  buffer->len = buffer->base ? 65536 : 0;
}

Object ioError(Runtime &rt, int status) {
  auto error = rt.global().getPropertyAsFunction(rt, "Error")
      .callAsConstructor(rt, String::createFromUtf8(rt, uv_strerror(status))).asObject(rt);
  const char *code = status == UV_EAI_NONAME || status == UV_EAI_NODATA
      ? "ENOTFOUND" : uv_err_name(status);
  error.setProperty(rt, "code", String::createFromAscii(rt, code));
  error.setProperty(rt, "errno", status);
  return error;
}

// No C++/JS exception may cross a libuv C callback. Stop normal dispatch and let shutdown drain
// pending I/O, preserving both libuv storage and JSI references until their respective owners end.
template <typename Action> void guarded(SHConsoleContext &owner, Action action) {
  if (owner.closing || !owner.failure.empty()) {return;}
  try {
    action();
    // Node drains nextTick callbacks before language microtasks; ticks created by microtasks run
    // after that microtask batch, before returning to native I/O. Never recurse through the queue.
    do {
      while (!owner.nextTicks.empty()) {
        auto tick = std::move(owner.nextTicks.front());
        owner.nextTicks.pop_front();
        tick();
      }
      owner.runtime.drainMicrotasks();
      // A rejection handled by this turn's ticks/microtasks is not unhandled. Notification
      // listeners may enqueue more work; drain it before considering the next rejection batch.
    } while (!owner.nextTicks.empty() || (owner.rejectionCheckpoint && owner.rejectionCheckpoint->call(owner.runtime).asBool()));
  } catch (const std::exception &error) {
    owner.failure = error.what();
    uv_stop(&owner.loop);
  } catch (...) {
    owner.failure = "Unknown native I/O callback failure";
    uv_stop(&owner.loop);
  }
}

void emit(Resource &resource, const char *event, Value value = Value::undefined()) {
  guarded(resource.owner, [&] {
    auto &rt = resource.owner.runtime;
    resource.callback.call(rt, String::createFromUtf8(rt,
        resource.channel.empty() ? event : resource.channel + ":" + event), std::move(value));
  });
}

void resumeWorkers(SHConsoleContext &owner);

void finishClose(Resource &resource) {
  // An output failure removes backpressure too. A completed producer may have no further events
  // to wake its paused consumer, so cancellation must resume delivery just like a successful write.
  if (resource.id == resource.owner.outputs[0] || resource.id == resource.owner.outputs[1]) {
    resumeWorkers(resource.owner);
  }
  if (resource.worker) {
    {
      const std::lock_guard lock(resource.worker->mutex);
      resource.worker->aborting = true;
    }
    resource.worker->space.notify_all();
    // Join before uv_close: the compute thread may still enqueue its final event/async wakeup.
    // Cancelling queue backpressure first prevents a main-thread destructor/producer deadlock.
    resource.worker->core.reset();
  }
  if (resource.resolver) {uv_cancel(reinterpret_cast<uv_req_t *>(resource.resolver));}
  const auto closed = [](uv_handle_t *handle) {
    auto &resource = *static_cast<Resource *>(handle->data);
    if (--resource.closingHandles) {return;}
    if (resource.kind == Kind::tcp || resource.kind == Kind::pipe) {emit(resource, "close", Value(resource.hadError));}
    if (resource.kind == Kind::process) {emit(resource, "processClose");}
    resource.owner.resources.erase(resource.id);
  };
  resource.closingHandles = 0;
  const auto queueClose = [&](uv_handle_t *handle) {
    ++resource.closingHandles;
    uv_close(handle, closed);
  };
  if (resource.idleInitialized) {queueClose(reinterpret_cast<uv_handle_t *>(&resource.idle));}
  if (resource.kind == Kind::tcp) {
    if (resource.connectTimerInitialized && !uv_is_closing(reinterpret_cast<uv_handle_t *>(&resource.timer))) {
      resource.stopConnectTimer();
      queueClose(reinterpret_cast<uv_handle_t *>(&resource.timer));
    }
    if (resource.tcpInitialized && !uv_is_closing(reinterpret_cast<uv_handle_t *>(&resource.tcp))) {
      queueClose(reinterpret_cast<uv_handle_t *>(&resource.tcp));
    }
  } else {
    queueClose(resource.handle());
  }
  if (!resource.closingHandles) {resource.owner.resources.erase(resource.id);}
}

void close(Resource &resource) {
  if (resource.closing) {return;}
  if (resource.kind == Kind::process && resource.processRunning) {
    // Only this registry's children are eligible. Reap via exit_cb before closing the watcher;
    // closing a live process handle would orphan it. Fatal teardown must not wait on its stdin.
    if (!resource.terminating) {
      resource.terminating = true;
      const int status = uv_process_kill(&resource.process, SIGKILL);
      if (status && status != UV_ESRCH) {std::fprintf(stderr, "Native child termination failed: %s\n", uv_strerror(status));}
    }
    return;
  }
  // A failed or timed-out connect closes the current family-specific socket before reinitializing
  // the same Resource handle. If JS destroys during that close callback, finish teardown there.
  resource.closing = true;
  if (resource.kind == Kind::tcp && resource.connectResetPending) {
    if (resource.resolver) {uv_cancel(reinterpret_cast<uv_req_t *>(resource.resolver));}
    resource.stopConnectTimer();
    return;
  }
  finishClose(resource);
}

// Leave enough room below the 32-MiB stream cap for one maximum Pearl proof frame. Pausing
// callback delivery (not the loop) lets timers/stdin/shutdown run while the bounded cross-thread
// queue applies backpressure to the producer. No event/proof is dropped or kept off-budget.
bool outputBlocked(SHConsoleContext &owner) {
  for (const auto id : owner.outputs) {
    const auto output = owner.resources.find(id);
    if (output != owner.resources.end() && !output->second->closing && output->second->queued >= 16 * 1024 * 1024) {
      return true;
    }
  }
  return false;
}

void resumeWorkers(SHConsoleContext &owner) {
  if (outputBlocked(owner)) {return;}
  for (const auto &[id, resource] : owner.resources) {
    if (resource->worker && !resource->closing) {uv_async_send(&resource->async);}
  }
}

// Deliver up to 64 queued compute-worker events to their JS callbacks per loop turn, stopping
// early when the worker closes, the loop failed or stdout is backpressured; freed queue space
// wakes the blocked producer.
void drainWorker(Resource &resource) {
  guarded(resource.owner, [&] {
    auto &rt = resource.owner.runtime;
    unsigned delivered = 0;
    while (delivered++ < 64 && !resource.closing && resource.owner.failure.empty() && !outputBlocked(resource.owner)) {
      std::unique_lock lock(resource.worker->mutex);
      if (resource.worker->events.empty()) {break;}
      auto queued = std::move(resource.worker->events.front());
      resource.worker->events.pop_front();
      resource.worker->queuedBytes -= queued.bytes;
      lock.unlock();
      resource.worker->space.notify_all();
      const auto &event = queued.event;
      guarded(resource.owner, [&] {
        if (event.type == WorkerEvent::Type::complete) {resource.worker->complete.call(rt);}
        else if (event.type == WorkerEvent::Type::error) {
          resource.worker->error.call(rt, rt.global().getPropertyAsFunction(rt, "Error")
              .callAsConstructor(rt, String::createFromUtf8(rt, event.error)));
        } else {
          Object values(rt);
          for (const auto &[key, value] : event.values) {values.setProperty(rt, key.c_str(), String::createFromUtf8(rt, value));}
          resource.callback.call(rt, String::createFromUtf8(rt, event.name), std::move(values));
        }
      });
    }
    bool done, failed, pending;
    {
      const std::lock_guard lock(resource.worker->mutex);
      pending = !resource.worker->events.empty();
      done = resource.worker->done && !pending;
      failed = resource.worker->failed;
    }
    if (failed && !resource.closing) {
      resource.worker->error.call(rt, rt.global().getPropertyAsFunction(rt, "Error")
          .callAsConstructor(rt, String::createFromAscii(rt, "Native worker event queue failed")));
    }
    if ((done || failed) && !resource.closing) {close(resource);}
    else if (pending && !resource.closing && !outputBlocked(resource.owner)) {
      // Bound each JS dispatch turn too: a fast producer must not starve timers/other devices.
      uv_async_send(&resource.async);
    }
  });
}

// Only owned bytes cross threads. JS references are read exclusively by drainWorker on the loop.
// A bounded blocking queue preserves proofs rather than dropping results when the consumer is slow.
class NativeWorkerEvents final : public WorkerEvents {
  std::weak_ptr<Resource> resource_;
public:
  explicit NativeWorkerEvents(const std::shared_ptr<Resource> &resource) : resource_(resource) {}
  void start() override {
    const auto resource = resource_.lock();
    if (!resource || resource->closing) {throw std::runtime_error("AsyncWorker is stopped");}
    uv_ref(resource->handle());
  }
  void publish(WorkerEvent event) noexcept override {
    const auto resource = resource_.lock();
    if (!resource) {return;}
    auto &state = *resource->worker;
    constexpr size_t limit = 64 * 1024 * 1024;
    try {
      size_t bytes = sizeof(QueuedEvent);
      const auto add = [&](size_t extra) {
        if (extra > limit || bytes > limit - extra) {throw std::length_error("Native worker event exceeds 64 MiB");}
        bytes += extra;
      };
      add(event.name.size()); add(event.error.size());
      for (const auto &[key, value] : event.values) {
        add(sizeof(std::pair<std::string, std::string>)); add(key.size()); add(value.size());
      }
      std::unique_lock lock(state.mutex);
      state.space.wait(lock, [&] {
        return state.aborting || (state.events.size() < 1024 && bytes <= limit - state.queuedBytes);
      });
      if (state.aborting) {return;}
      state.events.push_back({std::move(event), bytes});
      state.queuedBytes += bytes;
    } catch (...) {
      const std::lock_guard lock(state.mutex);
      state.failed = true;
    }
    uv_async_send(&resource->async);
  }
  void finish() noexcept override {
    if (const auto resource = resource_.lock()) {
      {const std::lock_guard lock(resource->worker->mutex); resource->worker->done = true;}
      uv_async_send(&resource->async);
    }
  }
};

// A weak capability avoids anchoring the JS wrapper through its native callback registry. Its GC
// finalizer joins the worker; loop teardown removes all JS handles before destroying the runtime.
class WorkerHandle final : public HostObject {
  std::weak_ptr<Resource> resource_;
public:
  explicit WorkerHandle(const std::shared_ptr<Resource> &resource) : resource_(resource) {}
  ~WorkerHandle() override {if (const auto resource = resource_.lock()) {close(*resource);}}
  Value get(Runtime &rt, const PropNameID &name) override {
    if (name.utf8(rt) != "post") {return Value::undefined();}
    return Function::createFromHostFunction(rt, name, 2,
        [weak = resource_](Runtime &rt, const Value &, const Value *args, size_t count) -> Value {
      const auto resource = weak.lock();
      if (!resource || resource->closing) {throw JSError(rt, "AsyncWorker is stopped");}
      if (count < 2 || !args[0].isString()) {throw JSError(rt, "Invalid native worker message");}
      auto entries = args[1].asObject(rt).asArray(rt);
      const size_t length = entries.size(rt);
      if (length > 16384) {throw JSError(rt, "Native worker message has too many fields");}
      MessageValues values;
      for (size_t index = 0; index < length; ++index) {
        auto entry = entries.getValueAtIndex(rt, index).asObject(rt).asArray(rt);
        if (entry.size(rt) != 2) {throw JSError(rt, "Invalid native worker field");}
        values[entry.getValueAtIndex(rt, 0).asString(rt).utf8(rt)] = entry.getValueAtIndex(rt, 1).asString(rt).utf8(rt);
      }
      try {resource->worker->core->post({args[0].asString(rt).utf8(rt), std::move(values)});}
      catch (const std::string &error) {throw JSError(rt, error);}
      catch (const std::exception &error) {throw JSError(rt, error.what());}
      return Value::undefined();
    });
  }
};

void fail(Resource &resource, int status) {
  resource.hadError = true;
  guarded(resource.owner, [&] {emit(resource, "error", ioError(resource.owner.runtime, status));});
  close(resource);
}

void pumpTls(Resource &resource);
void scheduleTls(Resource &resource);
void wireWrite(Resource &resource, std::vector<char> bytes, size_t plainCount, bool acknowledge);

void tlsFailure(Resource &resource, const mom::runtime::TlsError &failure) {
  resource.hadError = true;
  guarded(resource.owner, [&] {
    auto &rt = resource.owner.runtime;
    auto error = rt.global().getPropertyAsFunction(rt, "Error")
        .callAsConstructor(rt, String::createFromUtf8(rt, failure.what())).asObject(rt);
    error.setProperty(rt, "code", String::createFromUtf8(rt, failure.code));
    if (resource.tls) {
      error.setProperty(rt, "authorizationError", String::createFromUtf8(rt, resource.tls->authorizationError()));
    }
    emit(resource, "error", std::move(error));
  });
  close(resource);
}

// Connect can fail before libuv schedules a callback. Defer that failure too, preserving Node's
// guarantee that net.connect() returns before emitting network errors to subsequently-added listeners.
void deferFailure(Resource &resource, int status) {
  auto keepAlive = resource.owner.resources.at(resource.id);
  resource.owner.deferred.emplace_back([keepAlive, status] {
    if (!keepAlive->closing) {fail(*keepAlive, status);}
  });
}

// Numeric handles are process-local capabilities, never raw pointers received from JavaScript.
Resource *lookup(SHConsoleContext &owner, const Value &value) {
  if (!value.isNumber()) {throw JSError(owner.runtime, "Native handle must be an integer");}
  const auto id = value.getNumber();
  if (!std::isfinite(id) || id < 1 || id > UINT32_MAX || std::floor(id) != id) {
    throw JSError(owner.runtime, "Invalid native handle");
  }
  const auto found = owner.resources.find(static_cast<uint32_t>(id));
  return found == owner.resources.end() ? nullptr : found->second.get();
}

std::shared_ptr<Resource> reserve(SHConsoleContext &owner, Kind kind, const Value &callback) {
  if (owner.sequence == UINT32_MAX) {throw JSError(owner.runtime, "Native handle space exhausted");}
  auto resource = std::make_shared<Resource>(owner, kind, callback.asObject(owner.runtime).asFunction(owner.runtime));
  // Establish ownership before registering the handle; a failing map allocation after uv_*_init
  // would otherwise free storage that libuv had already linked into its handle list.
  owner.resources.emplace(resource->id, resource);
  return resource;
}

std::shared_ptr<Resource> allocate(SHConsoleContext &owner, Kind kind, const Value &callback) {
  auto resource = reserve(owner, kind, callback);
  int status;
  if (kind == Kind::timer) {status = uv_timer_init(&owner.loop, &resource->timer);}
  else if (kind == Kind::immediate) {status = uv_check_init(&owner.loop, &resource->check);}
  else if (kind == Kind::tcp) {
    status = uv_tcp_init(&owner.loop, &resource->tcp);
    if (!status) {
      resource->tcpInitialized = true;
      resource->tcp.data = resource.get();
      status = uv_timer_init(&owner.loop, &resource->timer);
      if (!status) {
        resource->connectTimerInitialized = true;
        resource->timer.data = resource.get();
      }
    }
  }
  else if (kind == Kind::pipe) {status = uv_pipe_init(&owner.loop, &resource->pipe, 0);}
  else if (kind == Kind::signal) {status = uv_signal_init(&owner.loop, &resource->signal);}
  else {
    status = uv_async_init(&owner.loop, &resource->async, [](uv_async_t *handle) {
      drainWorker(*static_cast<Resource *>(handle->data));
    });
  }
  if (status) {
    if (kind == Kind::tcp && resource->tcpInitialized) {close(*resource);}
    else {owner.resources.erase(resource->id);}
    throw JSError(owner.runtime, ioError(owner.runtime, status));
  }
  resource->handle()->data = resource.get();
  if (kind == Kind::immediate) {
    status = uv_idle_init(&owner.loop, &resource->idle);
    if (status) {close(*resource); throw JSError(owner.runtime, ioError(owner.runtime, status));}
    resource->idle.data = resource.get();
    resource->idleInitialized = true;
  }
  return resource;
}

// Worker pipes share the same ownership and write queues as TCP. Reads only flow after the JS
// consumer subscribes/resumes, so a late listener cannot silently lose the child's output.
void readPipe(Resource &resource, bool reading) {
  if (resource.closing) {return;}
  if (!reading) {checkSystem(resource.owner.runtime, uv_read_stop(resource.stream()), "read_stop"); return;}
  const int status = uv_read_start(resource.stream(), allocateRead, [](uv_stream_t *stream, ssize_t size, const uv_buf_t *buffer) {
        std::unique_ptr<char[]> storage(buffer->base);
        auto &resource = *static_cast<Resource *>(stream->data);
        if (resource.closing) {return;}
        guarded(resource.owner, [&] {
          if (size == UV_EOF) {emit(resource, "end"); close(resource);}
          else if (size < 0) {fail(resource, static_cast<int>(size));}
          else if (size > 0) {
            emit(resource, "data", ArrayBuffer(resource.owner.runtime,
                std::make_shared<OwnedBytes>(buffer->base, static_cast<size_t>(size))));
          }
        });
      });
  if (status && status != UV_EALREADY) {fail(resource, status);}
}

// Use platform constants rather than hard-coded JS signal numbers. In particular a crashing
// worker must report SIGSEGV/SIGABRT to the supervisor, not throw while decoding its exit event.
struct SignalInfo {int number; const char *name;};
static const SignalInfo signals[] = {
  {SIGINT, "SIGINT"}, {SIGTERM, "SIGTERM"}, {SIGKILL, "SIGKILL"},
  {SIGABRT, "SIGABRT"}, {SIGSEGV, "SIGSEGV"}, {SIGILL, "SIGILL"}, {SIGFPE, "SIGFPE"},
#ifdef SIGHUP
  {SIGHUP, "SIGHUP"},
#endif
#ifdef SIGBREAK
  {SIGBREAK, "SIGBREAK"},
#endif
#ifdef SIGQUIT
  {SIGQUIT, "SIGQUIT"},
#endif
#ifdef SIGBUS
  {SIGBUS, "SIGBUS"},
#endif
#ifdef SIGTRAP
  {SIGTRAP, "SIGTRAP"},
#endif
#ifdef SIGSYS
  {SIGSYS, "SIGSYS"},
#endif
#ifdef SIGXCPU
  {SIGXCPU, "SIGXCPU"},
#endif
#ifdef SIGXFSZ
  {SIGXFSZ, "SIGXFSZ"},
#endif
#ifdef SIGPIPE
  {SIGPIPE, "SIGPIPE"},
#endif
#ifdef SIGUSR1
  {SIGUSR1, "SIGUSR1"},
#endif
#ifdef SIGUSR2
  {SIGUSR2, "SIGUSR2"},
#endif
};

int signalNumber(const std::string &name) {
  for (const auto &signal : signals) {if (name == signal.name) {return signal.number;}}
  return 0;
}

Value signalValue(Runtime &rt, int number) {
  if (!number) {return Value::null();}
  for (const auto &signal : signals) {if (number == signal.number) {return String::createFromAscii(rt, signal.name);}}
  return String::createFromUtf8(rt, "SIG" + std::to_string(number));
}

// Validate complete argument/environment vectors before allocating handles or launching anything.
// C strings are owned for the entire uv_spawn call; no shell, PATH search or external tool route.
std::vector<std::string> processStrings(Runtime &rt, const Value &value) {
  const auto array = value.asObject(rt).asArray(rt);
  const size_t count = array.size(rt);
  if (count > 65536) {throw JSError(rt, "Too many process arguments/environment entries");}
  std::vector<std::string> result;
  size_t bytes = 0;
  for (size_t i = 0; i < count; i++) {
    auto text = array.getValueAtIndex(rt, i).asString(rt).utf8(rt);
    if (text.find('\0') != std::string::npos || text.size() > 16 * 1024 * 1024 - bytes) {
      throw JSError(rt, "Invalid or oversized process arguments/environment");
    }
    bytes += text.size();
    result.push_back(std::move(text));
  }
  return result;
}

// Spawn a compute worker: only this executable, with an explicit environment and validated
// working directory, as a resource that reports the child's exit back to JS.
Value spawnProcess(SHConsoleContext &owner, const Value *values) {
  auto &rt = owner.runtime;
  auto args = processStrings(rt, values[0]);
  auto env = processStrings(rt, values[1]);
  char executable[65536];
  size_t executableSize = sizeof(executable);
  checkSystem(rt, uv_exepath(executable, &executableSize), "exepath");
  if (args.empty() || args[0] != std::string(executable, executableSize)) {
    throw JSError(rt, "Native workers may launch only this executable");
  }
  for (const auto &entry : env) {
    if (entry.find('=') == std::string::npos) {throw JSError(rt, "Invalid process environment entry");}
  }
  const char *directory = prepareWorkerDirectory(env); // Before taking pointers into env storage.
  std::vector<char *> argv, environmentPointers;
  for (auto &arg : args) {argv.push_back(arg.data());}
  argv.push_back(nullptr);
  for (auto &entry : env) {environmentPointers.push_back(entry.data());}
  environmentPointers.push_back(nullptr);
  values[2].asObject(rt).asFunction(rt); // Validate before any handle allocation.
  std::array<std::shared_ptr<Resource>, 3> pipes;
  std::shared_ptr<Resource> process;
  bool initialized = false;
  try {
    for (size_t i = 0; i < pipes.size(); i++) {
      pipes[i] = allocate(owner, Kind::pipe, values[2]);
      pipes[i]->channel = i == 0 ? "stdin" : i == 1 ? "stdout" : "stderr";
    }
    process = reserve(owner, Kind::process, values[2]);
    uv_stdio_container_t stdio[3]{};
    for (size_t i = 0; i < pipes.size(); i++) {
      stdio[i].flags = static_cast<uv_stdio_flags>(UV_CREATE_PIPE | (i == 0 ? UV_READABLE_PIPE : UV_WRITABLE_PIPE));
      stdio[i].data.stream = pipes[i]->stream();
      process->pipes[i] = pipes[i]->id;
    }
    uv_process_options_t options{};
    options.file = argv[0]; options.args = argv.data(); options.env = environmentPointers.data();
    options.cwd = directory;
    options.stdio_count = 3; options.stdio = stdio;
    options.exit_cb = [](uv_process_t *handle, int64_t code, int signal) {
      auto &process = *static_cast<Resource *>(handle->data);
      process.processRunning = false;
      const auto input = process.owner.resources.find(process.pipes[0]);
      if (input != process.owner.resources.end()) {close(*input->second);}
      guarded(process.owner, [&] {
        Object result(process.owner.runtime);
        result.setProperty(process.owner.runtime, "code", signal ? Value::null() : Value(static_cast<double>(code)));
        result.setProperty(process.owner.runtime, "signal", signalValue(process.owner.runtime, signal));
        emit(process, "exit", std::move(result));
      });
      close(process); // Never close/reuse a process watcher before its exit callback.
    };
    const int status = uv_spawn(&owner.loop, &process->process, &options);
    initialized = true; // uv_spawn requires uv_close even on failure, unlike other uv_*_init APIs.
    process->process.data = process.get();
    process->processRunning = status == 0;
    if (status) {
      if (status != UV_ENOENT && status != UV_EACCES) {checkSystem(rt, status, "spawn");}
      for (const auto &pipe : pipes) {close(*pipe);}
      owner.deferred.emplace_back([process, status] {
        emit(*process, "error", ioError(process->owner.runtime, status));
        close(*process);
      });
    } else {
      owner.deferred.emplace_back([process] {emit(*process, "spawn");});
    }
    Object result(rt);
    result.setProperty(rt, "id", static_cast<double>(process->id));
    result.setProperty(rt, "pid", status ? Value::undefined() : Value(static_cast<double>(process->process.pid)));
    for (const auto &pipe : pipes) {result.setProperty(rt, pipe->channel.c_str(), static_cast<double>(pipe->id));}
    return result;
  } catch (...) {
    for (const auto &pipe : pipes) {if (pipe) {close(*pipe);}}
    if (process) {
      if (initialized) {close(*process);} else {owner.resources.erase(process->id);}
    }
    throw;
  }
}

// Shutdown is queued after pending writes; closing the handle immediately would cancel their data.
void shutdown(Resource &resource) {
  if (resource.closing || resource.shutdownStarted) {return;}
  resource.shutdownStarted = true;
  resource.shutdown.data = &resource;
  const int status = uv_shutdown(&resource.shutdown, resource.stream(),
      [](uv_shutdown_t *request, int status) {
        auto &resource = *static_cast<Resource *>(request->data);
        if (status && status != UV_ECANCELED && !resource.closing) {fail(resource, status);}
        else if (!resource.closing) {
          resource.shutdownComplete = true;
          emit(resource, "finish");
          if (resource.readEnded || resource.kind == Kind::pipe) {close(resource);}
        }
      });
  if (status) {close(resource);}
}

void end(Resource &resource) {
  if (resource.closing || resource.ending) {return;}
  resource.ending = true;
  if (resource.tls) {scheduleTls(resource);} else {shutdown(resource);}
}

void receivedEnd(Resource &resource) {
  if (resource.readEnded) {return;}
  resource.readEnded = true;
  emit(resource, "end");
  if (resource.kind == Kind::pipe) {close(resource); return;}
  end(resource);
  if (resource.shutdownComplete) {close(resource);}
}

size_t connectAddressSize(int family) {
  if (family == AF_INET) {return sizeof(sockaddr_in);}
  if (family == AF_INET6) {return sizeof(sockaddr_in6);}
  return 0;
}

bool duplicateConnectAddress(const std::vector<ConnectAddress> &addresses,
    int family, const sockaddr_storage &storage) {
  const auto size = connectAddressSize(family);
  for (const auto &address : addresses) {
    if (address.family == family && std::memcmp(&address.storage, &storage, size) == 0) {return true;}
  }
  return false;
}

bool appendConnectAddress(std::vector<ConnectAddress> &addresses, int family,
    const sockaddr *address, size_t length) {
  const auto size = connectAddressSize(family);
  if (!size || !address || address->sa_family != family || length < size || length > sizeof(sockaddr_storage)) {
    return false;
  }
  if (addresses.size() >= maximumConnectAddresses) {return false;}
  ConnectAddress candidate;
  candidate.family = family;
  std::memcpy(&candidate.storage, address, length);
  if (duplicateConnectAddress(addresses, family, candidate.storage)) {return false;}
  addresses.push_back(std::move(candidate));
  return true;
}

void startConnect(Resource &resource);

// Move on from a failed or timed-out address: report the error after the last candidate,
// otherwise close the socket and retry the next address on a freshly initialised handle.
void connectAttemptFailed(Resource &resource, int status) {
  if (resource.closing || resource.connectResetPending) {return;}
  resource.connectPending = false;
  resource.stopConnectTimer();
  if (resource.nextConnectAddress >= resource.connectAddresses.size()) {
    fail(resource, status);
    return;
  }
  resource.connectResetPending = true;
  ++resource.connectGeneration; // Invalidate any deferred immediate-error callback.
  uv_close(reinterpret_cast<uv_handle_t *>(&resource.tcp), [](uv_handle_t *handle) {
    auto &resource = *static_cast<Resource *>(handle->data);
    resource.tcpInitialized = false;
    resource.connectResetPending = false;
    if (resource.closing) {
      finishClose(resource);
      return;
    }
    const int status = uv_tcp_init(&resource.owner.loop, &resource.tcp);
    if (status) {fail(resource, status); return;}
    resource.tcpInitialized = true;
    resource.tcp.data = &resource;
    if (!resource.tcpRef) {uv_unref(reinterpret_cast<uv_handle_t *>(&resource.tcp));}
    startConnect(resource);
  });
}

void deferConnectFailure(Resource &resource, int status, uint64_t generation) {
  auto keepAlive = resource.owner.resources.at(resource.id);
  resource.owner.deferred.emplace_back([keepAlive, status, generation] {
    if (keepAlive->closing || !keepAlive->connectPending || keepAlive->connectGeneration != generation) {return;}
    connectAttemptFailed(*keepAlive, status);
  });
}

// Reading begins only after a successful connection. Deliver byte ownership to the JS adapter;
// it supplies Buffer/StringDecoder semantics without exposing transient libuv storage.
void startConnect(Resource &resource) {
  if (resource.closing || resource.connectResetPending || resource.connectPending) {return;}
  if (resource.nextConnectAddress >= resource.connectAddresses.size()) {
    fail(resource, UV_EAI_NONAME);
    return;
  }
  const auto &address = resource.connectAddresses[resource.nextConnectAddress++];
  resource.connectPending = true;
  const auto generation = ++resource.connectGeneration;
  resource.connect.data = &resource;
  const int status = uv_tcp_connect(&resource.connect, &resource.tcp,
      reinterpret_cast<const sockaddr *>(&address.storage),
      [](uv_connect_t *request, int status) {
        auto &resource = *static_cast<Resource *>(request->data);
        if (resource.closing || resource.connectResetPending || !resource.connectPending) {return;}
        resource.connectPending = false;
        resource.stopConnectTimer();
        if (status) {connectAttemptFailed(resource, status); return;}
        const int reading = uv_read_start(reinterpret_cast<uv_stream_t *>(&resource.tcp), allocateRead,
            [](uv_stream_t *stream, ssize_t size, const uv_buf_t *buffer) {
              std::unique_ptr<char[]> storage(buffer->base);
              auto &resource = *static_cast<Resource *>(stream->data);
              if (resource.closing) {return;}
              guarded(resource.owner, [&] {
                if (size == UV_EOF) {
                  if (resource.tls && !resource.readEnded) {
                    tlsFailure(resource, {"ECONNRESET", "TLS connection ended without close_notify"});
                  } else {receivedEnd(resource);}
                }
                else if (size < 0) {fail(resource, static_cast<int>(size));}
                else if (size > 0) {
                  if (resource.tls) {
                    try {
                      resource.tls->feed(buffer->base, static_cast<size_t>(size));
                      pumpTls(resource);
                    } catch (const mom::runtime::TlsError &error) {tlsFailure(resource, error);}
                  } else {
                    emit(resource, "data", ArrayBuffer(resource.owner.runtime,
                        std::make_shared<OwnedBytes>(buffer->base, static_cast<size_t>(size))));
                  }
                }
              });
            });
        if (reading) {fail(resource, reading); return;}
        emit(resource, "connect");
        if (resource.tls && !resource.closing) {pumpTls(resource);}
      });
  if (status) {
    deferConnectFailure(resource, status, generation);
    return;
  }
  // Node's family-attempt timeout only bounds an address when another candidate remains. The
  // final OS connect keeps its normal platform deadline; an idle auxiliary timer must not hold
  // an unref'd socket open after there is nothing left to race.
  if (resource.nextConnectAddress >= resource.connectAddresses.size()) {
    if (resource.connectTimerActive) {resource.stopConnectTimer();}
    if (uv_has_ref(reinterpret_cast<uv_handle_t *>(&resource.timer))) {
      uv_unref(reinterpret_cast<uv_handle_t *>(&resource.timer));
    }
    return;
  }
  resource.connectTimerActive = true;
  const int timer = uv_timer_start(&resource.timer, [](uv_timer_t *timer) {
    auto &resource = *static_cast<Resource *>(timer->data);
    if (resource.closing || !resource.connectPending) {return;}
    resource.connectPending = false;
    ++resource.connectGeneration;
    resource.stopConnectTimer();
    connectAttemptFailed(resource, UV_ETIMEDOUT);
  }, resource.connectAttemptTimeout, 0);
  if (timer) {
    resource.connectPending = false;
    resource.connectTimerActive = false;
    fail(resource, timer);
    return;
  }
  if (resource.tcpRef) {uv_ref(reinterpret_cast<uv_handle_t *>(&resource.timer));}
  else {uv_unref(reinterpret_cast<uv_handle_t *>(&resource.timer));}
}

void connect(Resource &resource, const sockaddr *address) {
  const auto family = address ? address->sa_family : AF_UNSPEC;
  const auto size = connectAddressSize(family);
  resource.connectAddresses.clear();
  resource.nextConnectAddress = 0;
  if (!appendConnectAddress(resource.connectAddresses, family, address, size)) {
    deferFailure(resource, UV_EAI_NONAME);
    return;
  }
  startConnect(resource);
}

struct ResolveRequest {
  uv_getaddrinfo_t request{};
  std::shared_ptr<Resource> resource;
  std::string host, service;
};

// DNS may complete after socket close. The request independently owns the resource until that
// callback has run, without letting it revive a cancelled connection or outlive the JS runtime.
void resolve(const std::shared_ptr<Resource> &resource, const std::string &host, int port) {
  sockaddr_in ipv4{};
  sockaddr_in6 ipv6{};
  if (uv_ip4_addr(host.c_str(), port, &ipv4) == 0) {
    connect(*resource, reinterpret_cast<sockaddr *>(&ipv4)); return;
  }
  if (uv_ip6_addr(host.c_str(), port, &ipv6) == 0) {
    connect(*resource, reinterpret_cast<sockaddr *>(&ipv6)); return;
  }
  auto request = std::make_unique<ResolveRequest>();
  request->resource = resource;
  request->host = host;
  request->service = std::to_string(port);
  request->request.data = request.get();
  addrinfo hints{};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  const int status = uv_getaddrinfo(&resource->owner.loop, &request->request,
      [](uv_getaddrinfo_t *raw, int status, addrinfo *addresses) {
        std::unique_ptr<ResolveRequest> request(static_cast<ResolveRequest *>(raw->data));
        auto &resource = *request->resource;
        resource.resolver = nullptr;
        guarded(resource.owner, [&] {
          if (resource.closing) {return;}
          if (status) {fail(resource, status);}
          else {
            std::vector<ConnectAddress> preferred, alternate;
            int firstFamily = AF_UNSPEC;
            resource.connectAddresses.clear();
            size_t examined = 0;
            for (auto *entry = addresses; entry && examined < maximumConnectAddresses * 4; entry = entry->ai_next, ++examined) {
              if (entry->ai_family != AF_INET && entry->ai_family != AF_INET6) {continue;}
              const auto size = connectAddressSize(entry->ai_family);
              if (!size || !entry->ai_addr || entry->ai_addrlen < size ||
                  entry->ai_addrlen > sizeof(sockaddr_storage)) {continue;}
              if (firstFamily == AF_UNSPEC) {firstFamily = entry->ai_family;}
              auto &bucket = entry->ai_family == firstFamily ? preferred : alternate;
              if (bucket.size() >= maximumConnectAddresses / 2) {continue;}
              appendConnectAddress(bucket, entry->ai_family, entry->ai_addr, entry->ai_addrlen);
            }
            for (size_t index = 0; index < preferred.size() || index < alternate.size(); ++index) {
              if (index < preferred.size()) {resource.connectAddresses.push_back(std::move(preferred[index]));}
              if (index < alternate.size()) {resource.connectAddresses.push_back(std::move(alternate[index]));}
            }
            resource.nextConnectAddress = 0;
            if (resource.connectAddresses.empty()) {fail(resource, UV_EAI_NONAME);}
            else {startConnect(resource);}
          }
        });
        if (addresses) {uv_freeaddrinfo(addresses);}
      }, request->host.c_str(), request->service.c_str(), &hints);
  if (status) {deferFailure(*resource, status);}
  else {resource->resolver = &request.release()->request;}
}

struct WriteRequest {
  uv_write_t request{};
  std::shared_ptr<Resource> resource;
  std::vector<char> bytes;
  size_t plainCount = 0;
  bool acknowledge = false;
};

// Only OS write failures become socket errors. A JS listener exception must still fail the loop,
// not be mistaken for a reconnectable network error by TLS's asynchronous write pump.
struct WriteFailure {int status;};

// All TLS records, alerts, handshake messages and ordinary TCP writes share this single queue.
// A final-record acknowledgement also covers earlier fragments, because stream writes are ordered.
void wireWrite(Resource &resource, std::vector<char> bytes, size_t plainCount, bool acknowledge) {
  constexpr size_t maximumWire = 64 * 1024 * 1024;
  if (bytes.size() > maximumWire - resource.wireQueued) {
    throw mom::runtime::TlsError("ERR_TLS_BUFFER_LIMIT", "Native wire queue exceeds 64 MiB");
  }
  auto request = std::make_unique<WriteRequest>();
  request->resource = resource.owner.resources.at(resource.id);
  request->bytes = std::move(bytes);
  request->plainCount = plainCount;
  request->acknowledge = acknowledge;
  request->request.data = request.get();
  auto buffer = uv_buf_init(request->bytes.data(), static_cast<unsigned>(request->bytes.size()));
  const int status = uv_write(&request->request, resource.stream(), &buffer, 1,
      [](uv_write_t *raw, int status) {
        std::unique_ptr<WriteRequest> request(static_cast<WriteRequest *>(raw->data));
        auto &resource = *request->resource;
        resource.wireQueued -= request->bytes.size();
        resource.queued -= request->plainCount;
        guarded(resource.owner, [&] {
          if (resource.closing) {return;}
          if (status) {fail(resource, status);}
          else {
            if (request->acknowledge) {emit(resource, "write");}
            if (resource.id == resource.owner.outputs[0] || resource.id == resource.owner.outputs[1]) {
              resumeWorkers(resource.owner);
            }
            if (resource.tls && !resource.closing) {scheduleTls(resource);}
          }
        });
      });
  if (status) {throw WriteFailure{status};}
  resource.wireQueued += request->bytes.size();
  request.release();
}

void flushTls(Resource &resource, size_t acknowledged = 0, bool completed = false) {
  auto bytes = resource.tls->output();
  if (!bytes.empty() || completed) {wireWrite(resource, std::move(bytes), acknowledged, completed);}
}

// Serialize TLS state across callbacks. JS invoked by secureConnect/data can enqueue writes or
// close the socket; it must not re-enter OpenSSL or invalidate a partially processed plaintext write.
void pumpTls(Resource &resource) {
  if (resource.pumping || resource.closing) {return;}
  struct Reset {bool &flag; ~Reset() {flag = false;}} reset{resource.pumping};
  resource.pumping = true;
  guarded(resource.owner, [&] {
    try {
      if (!resource.secure) {
        const bool ready = resource.tls->handshake();
        flushTls(resource);
        if (!ready) {return;}
        resource.secure = true;
        auto &rt = resource.owner.runtime;
        const auto error = resource.tls->authorizationError();
        Object authorization(rt);
        authorization.setProperty(rt, "authorized", error.empty());
        authorization.setProperty(rt, "authorizationError", String::createFromUtf8(rt, error));
        authorization.setProperty(rt, "fingerprint256",
            String::createFromUtf8(rt, resource.tls->fingerprint256()));
        emit(resource, "secureConnect", std::move(authorization));
      }
      while (!resource.closing && !resource.readEnded) {
        char bytes[16384];
        const int size = resource.tls->read(bytes, sizeof(bytes));
        if (!size) {break;}
        if (size < 0) {receivedEnd(resource); break;}
        emit(resource, "data", ArrayBuffer(resource.owner.runtime, std::make_shared<OwnedBytes>(bytes, size)));
      }
      if (resource.closing) {return;}
      while (!resource.plainWrites.empty()) {
        const auto &bytes = resource.plainWrites.front().bytes;
        if (!resource.tls->write(bytes.data(), bytes.size())) {flushTls(resource); return;}
        flushTls(resource, bytes.size(), true);
        resource.plainWrites.pop_front();
      }
      if (resource.ending && !resource.shutdownStarted) {
        resource.tls->shutdown();
        flushTls(resource);
        shutdown(resource);
      } else {flushTls(resource);}
    } catch (const mom::runtime::TlsError &error) {tlsFailure(resource, error);}
    catch (const WriteFailure &error) {fail(resource, error.status);}
  });
}

void scheduleTls(Resource &resource) {
  if (resource.pumping || resource.pumpScheduled || resource.closing) {return;}
  auto keepAlive = resource.owner.resources.at(resource.id);
  resource.pumpScheduled = true;
  resource.owner.deferred.emplace_back([keepAlive] {
    keepAlive->pumpScheduled = false;
    pumpTls(*keepAlive);
  });
}

// Copy before returning to JS: libuv retains the buffer until its completion callback, even when
// the caller mutates/detaches its ArrayBuffer or destroys the socket while a write is pending.
bool write(SHConsoleContext &owner, Resource &resource, const Value &view) {
  auto &rt = owner.runtime;
  const auto input = byteView(rt, view);
  if (resource.closing || resource.ending) {throw JSError(rt, "Native socket is not writable");}
  const size_t length = input.size;
  constexpr size_t maximumQueued = 32 * 1024 * 1024;
  if (length > maximumQueued - resource.queued) {throw JSError(rt, "Native socket write queue exceeded 32 MiB");}
  std::vector<char> bytes;
  if (length) {
    const auto *data = reinterpret_cast<const char *>(input.data(rt));
    bytes.assign(data, data + length);
  }
  if (resource.tls) {
    if (!resource.secure) {throw JSError(rt, "TLS handshake is not complete");}
    resource.plainWrites.push_back({std::move(bytes)});
    scheduleTls(resource);
  } else {
    try {wireWrite(resource, std::move(bytes), static_cast<size_t>(length), true);}
    catch (const WriteFailure &error) {throw JSError(rt, ioError(rt, error.status));}
  }
  resource.queued += static_cast<size_t>(length);
  return resource.queued < 65536;
}

// Centralized registration keeps arity checks out of every OS operation. No callbacks are invoked
// synchronously during connect/start, so JS can install its listeners before the first loop turn.
template <typename Action> void binding(SHConsoleContext &owner, const char *name, size_t arity, Action action) {
  mom::runtime::binding(owner.runtime, name, static_cast<unsigned>(arity),
      [&owner, action](Runtime &, const Value *args) {return action(owner, args);});
}
} // namespace mom::runtime::detail

namespace mom::runtime {
using namespace detail;
// Create the application's libuv loop and install the __mom* event-loop bindings (timers, sockets,
// pipes, processes, signals, workers) on `runtime`; the caller owns the returned context.
SHConsoleContext *createLoop(Runtime &runtime, WorkerFactory workerFactory) {
  auto owner = std::make_unique<SHConsoleContext>(runtime, workerFactory);
  const int status = uv_loop_init(&owner->loop);
  if (status) {throw JSError(runtime, ioError(runtime, status));}
  binding(*owner, "__momNow", 0, [](SHConsoleContext &, const Value *) -> Value {
    return Value(static_cast<double>(uv_hrtime()) / 1000000.0);
  });
  binding(*owner, "__momRejectionCheckpoint", 1, [](SHConsoleContext &owner, const Value *args) -> Value {
    if (owner.rejectionCheckpoint) {throw JSError(owner.runtime, "Promise checkpoint already registered");}
    owner.rejectionCheckpoint = std::make_unique<Function>(args[0].asObject(owner.runtime).asFunction(owner.runtime));
    return Value::undefined();
  });
  binding(*owner, "__momImmediateStart", 1, [](SHConsoleContext &owner, const Value *args) -> Value {
    auto resource = allocate(owner, Kind::immediate, args[0]);
    // A check watcher alone may sleep in poll forever. The paired idle watcher makes this one
    // iteration nonblocking; both are closed/ref-counted together before invoking the callback.
    const int idle = uv_idle_start(&resource->idle, [](uv_idle_t *) {});
    const int check = idle ? idle : uv_check_start(&resource->check, [](uv_check_t *handle) {
      auto &resource = *static_cast<Resource *>(handle->data);
      close(resource);
      guarded(resource.owner, [&] {resource.callback.call(resource.owner.runtime);});
    });
    if (check) {close(*resource); throw JSError(owner.runtime, ioError(owner.runtime, check));}
    return Value(static_cast<double>(resource->id));
  });
  binding(*owner, "__momSpawn", 3, [](SHConsoleContext &owner, const Value *args) {return spawnProcess(owner, args);});
  binding(*owner, "__momProcessKill", 2, [](SHConsoleContext &owner, const Value *args) -> Value {
    auto *resource = lookup(owner, args[0]);
    const int signal = signalNumber(args[1].asString(owner.runtime).utf8(owner.runtime));
    if (!signal) {throw JSError(owner.runtime, "Unsupported process signal");}
    if (!resource || resource->kind != Kind::process || !resource->processRunning) {return Value(false);}
    const int status = uv_process_kill(&resource->process, signal);
    if (status == UV_ESRCH) {return Value(false);}
    checkSystem(owner.runtime, status, "kill");
    return Value(true);
  });
  binding(*owner, "__momPipeRead", 2, [](SHConsoleContext &owner, const Value *args) -> Value {
    auto *resource = lookup(owner, args[0]);
    if (resource && resource->kind == Kind::pipe) {readPipe(*resource, args[1].asBool());}
    return Value::undefined();
  });
  binding(*owner, "__momStdinOpen", 1, [](SHConsoleContext &owner, const Value *args) -> Value {
    auto resource = allocate(owner, Kind::pipe, args[0]);
    const int status = uv_pipe_open(&resource->pipe, 0);
    if (status) {close(*resource); throw JSError(owner.runtime, ioError(owner.runtime, status));}
    return Value(static_cast<double>(resource->id));
  });
  binding(*owner, "__momOutputOpen", 2, [](SHConsoleContext &owner, const Value *args) -> Value {
    const double fd = args[0].asNumber();
    if (fd != 1 && fd != 2) {throw JSError(owner.runtime, "Only stdout/stderr may open as native output");}
    if (uv_guess_handle(static_cast<int>(fd)) != UV_NAMED_PIPE) {return Value::null();}
    auto resource = allocate(owner, Kind::pipe, args[1]);
    const int status = uv_pipe_open(&resource->pipe, static_cast<int>(fd));
    if (status) {close(*resource); throw JSError(owner.runtime, ioError(owner.runtime, status));}
    owner.outputs[static_cast<size_t>(fd) - 1] = resource->id;
    return Value(static_cast<double>(resource->id));
  });
  binding(*owner, "__momSignalStart", 2, [](SHConsoleContext &owner, const Value *args) -> Value {
    const auto name = args[0].asString(owner.runtime).utf8(owner.runtime);
    const int signal = signalNumber(name);
    if (!signal || signal == SIGKILL) {throw JSError(owner.runtime, "Unsupported signal subscription");}
    auto resource = allocate(owner, Kind::signal, args[1]);
    const int status = uv_signal_start(&resource->signal, [](uv_signal_t *handle, int) {
      auto &resource = *static_cast<Resource *>(handle->data);
      guarded(resource.owner, [&] {resource.callback.call(resource.owner.runtime);});
    }, signal);
    if (status) {close(*resource); throw JSError(owner.runtime, ioError(owner.runtime, status));}
    uv_unref(resource->handle()); // Node's signal listeners alone do not keep a process alive.
    return Value(static_cast<double>(resource->id));
  });
  binding(*owner, "__momWorkerCreate", 3, [](SHConsoleContext &owner, const Value *args) -> Value {
    if (!owner.workerFactory) {throw JSError(owner.runtime, "Native compute core is not configured");}
    auto resource = allocate(owner, Kind::worker, args[0]);
    uv_unref(resource->handle()); // Constructing an idle wrapper must not keep the process alive.
    try {
      resource->worker = std::make_unique<WorkerState>(args[1].asObject(owner.runtime).asFunction(owner.runtime),
          args[2].asObject(owner.runtime).asFunction(owner.runtime));
      resource->worker->core = owner.workerFactory(std::make_unique<NativeWorkerEvents>(resource));
      if (!resource->worker->core) {throw std::runtime_error("Native compute factory returned no worker");}
      return Object::createFromHostObject(owner.runtime, std::make_shared<WorkerHandle>(resource));
    } catch (...) {close(*resource); throw;}
  });
  binding(*owner, "__momTimerStart", 3, [](SHConsoleContext &owner, const Value *args) -> Value {
    const double requested = args[1].asNumber();
    const uint64_t delay = std::isfinite(requested) && requested >= 1 && requested <= INT32_MAX
        ? static_cast<uint64_t>(requested) : 1;
    const bool repeat = args[2].asBool();
    auto resource = allocate(owner, Kind::timer, args[0]);
    uv_update_time(&owner.loop);
    const int status = uv_timer_start(&resource->timer, [](uv_timer_t *timer) {
      auto &resource = *static_cast<Resource *>(timer->data);
      if (!uv_timer_get_repeat(timer)) {close(resource);}
      guarded(resource.owner, [&] {resource.callback.call(resource.owner.runtime);});
    }, delay, repeat ? delay : 0);
    if (status) {close(*resource); throw JSError(owner.runtime, ioError(owner.runtime, status));}
    return Value(static_cast<double>(resource->id));
  });
  binding(*owner, "__momHandleClose", 1, [](SHConsoleContext &owner, const Value *args) -> Value {
    if (auto *resource = lookup(owner, args[0])) {close(*resource);}
    return Value::undefined();
  });
  binding(*owner, "__momHandleRef", 2, [](SHConsoleContext &owner, const Value *args) -> Value {
    auto *resource = lookup(owner, args[0]);
    if (resource && !resource->closing) {
      const bool ref = args[1].asBool();
      if (resource->kind == Kind::tcp) {
        resource->tcpRef = ref;
        if (!resource->connectResetPending && resource->tcpInitialized) {
          if (ref) {uv_ref(resource->handle());} else {uv_unref(resource->handle());}
        }
        if (resource->connectTimerInitialized && resource->connectTimerActive &&
            !uv_is_closing(reinterpret_cast<uv_handle_t *>(&resource->timer))) {
          if (ref) {uv_ref(reinterpret_cast<uv_handle_t *>(&resource->timer));}
          else {uv_unref(reinterpret_cast<uv_handle_t *>(&resource->timer));}
        } else if (resource->connectTimerInitialized &&
            !uv_is_closing(reinterpret_cast<uv_handle_t *>(&resource->timer)) &&
            uv_has_ref(reinterpret_cast<uv_handle_t *>(&resource->timer))) {
          // Until an attempt timer is started this is only an auxiliary handle; it must follow
          // no socket reference, even while DNS is still pending.
          uv_unref(reinterpret_cast<uv_handle_t *>(&resource->timer));
        }
      } else if (ref) {uv_ref(resource->handle());}
      else {uv_unref(resource->handle());}
      if (resource->idleInitialized) {
        if (ref) {uv_ref(reinterpret_cast<uv_handle_t *>(&resource->idle));}
        else {uv_unref(reinterpret_cast<uv_handle_t *>(&resource->idle));}
      }
    }
    return Value::undefined();
  });
  binding(*owner, "__momHandleHasRef", 1, [](SHConsoleContext &owner, const Value *args) -> Value {
    auto *resource = lookup(owner, args[0]);
    return Value(resource && !resource->closing &&
      (resource->kind == Kind::tcp && resource->connectResetPending ? resource->tcpRef : uv_has_ref(resource->handle())));
  });
  binding(*owner, "__momTCPConnect", 4, [](SHConsoleContext &owner, const Value *args) -> Value {
    const auto host = args[0].asString(owner.runtime).utf8(owner.runtime);
    const auto port = args[1].asNumber();
    if (host.empty() || host.find('\0') != std::string::npos ||
        !std::isfinite(port) || port < 1 || port > 65535 || port != std::floor(port)) {
      throw JSError(owner.runtime, "Invalid native TCP address");
    }
    uint32_t attemptTimeout = defaultConnectAttemptTimeout;
    if (!args[3].isUndefined() && !args[3].isNull()) {
      if (!args[3].isNumber()) {throw JSError(owner.runtime, "Invalid native TCP attempt timeout");}
      const double requested = args[3].asNumber();
      if (!std::isfinite(requested) || requested < 1 || requested > INT32_MAX || std::floor(requested) != requested) {
        throw JSError(owner.runtime, "Invalid native TCP attempt timeout");
      }
      attemptTimeout = static_cast<uint32_t>(requested < 10 ? 10 : requested);
    }
    auto resource = allocate(owner, Kind::tcp, args[2]);
    resource->connectAttemptTimeout = attemptTimeout;
    try {
      resolve(resource, host, static_cast<int>(port));
    } catch (...) {
      close(*resource);
      throw;
    }
    return Value(static_cast<double>(resource->id));
  });
  binding(*owner, "__momTCPWrite", 2, [](SHConsoleContext &owner, const Value *args) -> Value {
    auto *resource = lookup(owner, args[0]);
    if (!resource || (resource->kind != Kind::tcp && resource->kind != Kind::pipe) || resource->closing || resource->ending) {
      throw JSError(owner.runtime, "Native socket is not writable");
    }
    return Value(write(owner, *resource, args[1]));
  });
  binding(*owner, "__momTLSConfigure", 5, [](SHConsoleContext &owner, const Value *args) -> Value {
    auto *resource = lookup(owner, args[0]);
    if (!resource || resource->kind != Kind::tcp || resource->closing || resource->tls) {
      throw JSError(owner.runtime, "Invalid native TLS socket");
    }
    const auto hostname = args[1].asString(owner.runtime).utf8(owner.runtime);
    const auto servername = args[2].asString(owner.runtime).utf8(owner.runtime);
    if (hostname.empty() || hostname.find('\0') != std::string::npos || servername.find('\0') != std::string::npos) {
      throw JSError(owner.runtime, "Invalid native TLS peer name");
    }
    std::string ca;
    if (!args[4].isNull()) {ca = args[4].asString(owner.runtime).utf8(owner.runtime);}
    resource->tls = std::make_unique<TlsSession>(hostname, servername, args[3].asBool(), args[4].isNull() ? nullptr : &ca);
    return Value::undefined();
  });
  binding(*owner, "__momTCPEnd", 1, [](SHConsoleContext &owner, const Value *args) -> Value {
    auto *resource = lookup(owner, args[0]);
    if (resource && (resource->kind == Kind::tcp || resource->kind == Kind::pipe)) {end(*resource);}
    return Value::undefined();
  });
  binding(*owner, "__momGetEnv", 1, [](SHConsoleContext &owner, const Value *args) -> Value {
    const auto name = args[0].asString(owner.runtime).utf8(owner.runtime);
    // libuv's C-string truncation matches Node's environment access, including embedded NUL.
    std::vector<char> buffer(256);
    size_t size = buffer.size();
    int status = uv_os_getenv(name.c_str(), buffer.data(), &size);
    if (status == UV_ENOBUFS) {
      buffer.resize(size);
      status = uv_os_getenv(name.c_str(), buffer.data(), &size);
    }
    if (status == UV_ENOENT || status == UV_EINVAL) {return Value::undefined();}
    if (status) {throw JSError(owner.runtime, ioError(owner.runtime, status));}
    return String::createFromUtf8(owner.runtime, reinterpret_cast<const uint8_t *>(buffer.data()), size);
  });
  binding(*owner, "__momSetEnv", 2, [](SHConsoleContext &owner, const Value *args) -> Value {
    const auto name = args[0].asString(owner.runtime).utf8(owner.runtime);
    const auto value = args[1].isNull() ? std::string() : args[1].asString(owner.runtime).utf8(owner.runtime);
    const int status = setEnvironment(name.c_str(), args[1].isNull() ? nullptr : value.c_str());
    // Like Node, invalid names (empty or containing '=') have no effect rather than creating JS-only
    // properties. Other OS failures are surfaced; continuing with an unapplied worker env is unsafe.
    if (status && status != UV_EINVAL) {throw JSError(owner.runtime, ioError(owner.runtime, status));}
    return Value::undefined();
  });
  binding(*owner, "__momEnvKeys", 0, [](SHConsoleContext &owner, const Value *) -> Value {
    uv_env_item_t *items = nullptr;
    int count = 0;
    const int status = uv_os_environ(&items, &count);
    if (status) {throw JSError(owner.runtime, ioError(owner.runtime, status));}
    struct Release {uv_env_item_t *items; int count; ~Release() {uv_os_free_environ(items, count);}} release{items, count};
    // Windows keeps drive-current-directory and process-status entries in its environment block,
    // but process.env enumeration omits names beginning with '='. Count first so the compact
    // result never writes past the JSI array's initial length.
    size_t visibleCount = static_cast<size_t>(count);
#ifdef _WIN32
    visibleCount = 0;
    for (int i = 0; i < count; i++) {
      if (items[i].name[0] != '=') {visibleCount++;}
    }
#endif
    Array keys(owner.runtime, visibleCount);
    size_t index = 0;
    for (int i = 0; i < count; i++) {
#ifdef _WIN32
      if (items[i].name[0] == '=') {continue;}
#endif
      keys.setValueAtIndex(owner.runtime, index++, String::createFromUtf8(owner.runtime, items[i].name));
    }
    return keys;
  });
  binding(*owner, "__momNextTick", 1, [](SHConsoleContext &owner, const Value *args) -> Value {
    auto callback = std::make_shared<Function>(args[0].asObject(owner.runtime).asFunction(owner.runtime));
    owner.nextTicks.emplace_back([&owner, callback] {callback->call(owner.runtime);});
    return Value::undefined();
  });
  return owner.release();
}

// Alternate deferred JS jobs with libuv turns until both are exhausted or a callback failed;
// returns false (after printing the failure) when the application must exit unsuccessfully.
bool runLoop(SHConsoleContext *context) {
  guarded(*context, [] {});
  while (context->failure.empty() && (!context->deferred.empty() || uv_loop_alive(&context->loop))) {
    auto batch = std::move(context->deferred);
    for (auto &job : batch) {guarded(*context, job);}
    if (context->failure.empty() && uv_loop_alive(&context->loop)) {
      uv_run(&context->loop, context->deferred.empty() ? UV_RUN_ONCE : UV_RUN_NOWAIT);
    }
  }
  if (!context->failure.empty()) {std::fprintf(stderr, "%s\n", context->failure.c_str());}
  return context->failure.empty();
}

void closeLoop(SHConsoleContext *context) {
  context->closing = true;
  context->deferred.clear();
  context->nextTicks.clear();
  context->rejectionCheckpoint.reset();
  for (auto &[id, resource] : context->resources) {close(*resource);}
  // Close callbacks erase resources; pending DNS/write requests retain independent ownership.
  // An exception before the first uv_run leaves uv_stop pending. The first cleanup iteration can
  // only clear that flag; keep draining until closing handles and active requests are actually gone.
  while (uv_loop_alive(&context->loop)) {uv_run(&context->loop, UV_RUN_DEFAULT);}
  if (!context->resources.empty()) {
    std::fputs("Native cleanup retained handles\n", stderr);
    std::exit(1);
  }
}

void freeLoop(SHConsoleContext *context) {
  if (uv_loop_close(&context->loop)) {
    std::fputs("Native loop did not drain\n", stderr);
    std::exit(1);
  }
  delete context;
}
} // namespace mom::runtime
