// Copyright GNU GPLv3 (c) 2023-2026 MoneroOcean <support@moneroocean.stream>

#pragma once

#include <node_api.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <exception>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>

using MessageValues = std::map<std::string, std::string>;

static void debug_async_worker(const char* message) {
  if (!std::getenv("MOM_DEBUG_STARTUP")) {
    return;
  }
  std::fprintf(stderr, "MOM_DEBUG_STARTUP %s\n", message);
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

  void write(T data) {
    debug_async_worker("MessageQueue write locking");
    {
      const std::lock_guard lock(m_mutex);
      debug_async_worker("MessageQueue write locked");
      m_buff.emplace_back(std::move(data));
      debug_async_worker("MessageQueue write stored");
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

class AsyncWorker {
  class NapiFailure {};

  enum class EventType {
    progress,
    complete,
    error,
  };

  struct CallbackEvent {
    EventType type;
    std::string name;
    MessageValues values;
    std::string error;

    explicit CallbackEvent(Message message)
      : type(EventType::progress), name(std::move(message.name)),
        values(std::move(message.values)) {}

    CallbackEvent(const EventType type, std::string error = {})
      : type(type), error(std::move(error)) {}
  };

  struct CallbackState {
    napi_ref progress{};
    napi_ref complete{};
    napi_ref error{};
  };

  napi_env m_env;
  std::unique_ptr<CallbackState> m_callbacks;
  napi_threadsafe_function m_tsfn{};
  std::thread m_thread;
  std::atomic<bool> m_started{false};
  std::atomic<bool> m_stopped{false};
  std::atomic<bool> m_accepting{true};

  public:

  static void check(napi_env env, const napi_status status) {
    if (status == napi_ok) {
      return;
    }
    if (env && status != napi_pending_exception) {
      const napi_extended_error_info* info = nullptr;
      napi_get_last_error_info(env, &info);
      napi_throw_error(
        env, nullptr, info && info->error_message ? info->error_message : "Node-API call failed");
    }
    throw NapiFailure{};
  }

  static void report_current_exception(napi_env env) noexcept {
    if (!env) {
      return;
    }
    try {
      throw;
    } catch (const NapiFailure&) {
    } catch (const std::string& error) {
      napi_throw_error(env, nullptr, error.c_str());
    } catch (const std::exception& error) {
      napi_throw_error(env, nullptr, error.what());
    } catch (...) {
      napi_throw_error(env, nullptr, "Native addon exception");
    }
  }

  private:

  static napi_value make_string(napi_env env, const std::string& value) {
    napi_value result = nullptr;
    check(env, napi_create_string_utf8(env, value.c_str(), value.size(), &result));
    return result;
  }

  static napi_value callback_value(napi_env env, napi_ref ref) {
    napi_value callback = nullptr;
    check(env, napi_get_reference_value(env, ref, &callback));
    return callback;
  }

  static void call_event(napi_env env, napi_value, void* context, void* data) {
    std::unique_ptr<CallbackEvent> event(static_cast<CallbackEvent*>(data));
    if (!env || !context || !event) {
      return;
    }
    try {
      const CallbackState& callbacks = *static_cast<CallbackState*>(context);
      const napi_ref callback_ref = event->type == EventType::progress ? callbacks.progress
        : event->type == EventType::complete ? callbacks.complete
                                             : callbacks.error;
      const napi_value callback = callback_value(env, callback_ref);
      napi_value global = nullptr;
      check(env, napi_get_global(env, &global));

      if (event->type == EventType::complete) {
        check(env, napi_call_function(env, global, callback, 0, nullptr, nullptr));
        return;
      }

      if (event->type == EventType::error) {
        napi_value error = nullptr;
        const napi_value text = make_string(env, event->error);
        check(env, napi_create_error(env, nullptr, text, &error));
        napi_value argv[] = {error};
        check(env, napi_call_function(env, global, callback, 1, argv, nullptr));
        return;
      }

      napi_value values = nullptr;
      check(env, napi_create_object(env, &values));
      for (const auto& [key, value_text] : event->values) {
        const napi_value value = make_string(env, value_text);
        check(env, napi_set_named_property(env, values, key.c_str(), value));
      }
      const napi_value name = make_string(env, event->name);
      napi_value argv[] = {name, values};
      check(env, napi_call_function(env, global, callback, 2, argv, nullptr));
    } catch (...) {
      report_current_exception(env);
    }
  }

  static void delete_ref(napi_env env, napi_ref& ref) noexcept {
    if (env && ref) {
      napi_delete_reference(env, ref);
    }
    ref = nullptr;
  }

  static void finalize_callbacks(napi_env env, void* data, void*) noexcept {
    std::unique_ptr<CallbackState> callbacks(static_cast<CallbackState*>(data));
    if (!callbacks) {
      return;
    }
    delete_ref(env, callbacks->progress);
    delete_ref(env, callbacks->complete);
    delete_ref(env, callbacks->error);
  }

  static napi_ref create_ref(napi_env env, napi_value value) {
    napi_ref ref = nullptr;
    check(env, napi_create_reference(env, value, 1, &ref));
    return ref;
  }

  void create_tsfn() {
    napi_value resource_name = nullptr;
    check(m_env, napi_create_string_utf8(
      m_env, "core::events", NAPI_AUTO_LENGTH, &resource_name));
    check(m_env, napi_create_threadsafe_function(
      m_env, callback_value(m_env, m_callbacks->progress), nullptr, resource_name, 0, 1,
      m_callbacks.get(), finalize_callbacks, m_callbacks.get(), call_event, &m_tsfn));
    m_callbacks.release();
  }

  void release_tsfn() noexcept {
    if (m_tsfn) {
      napi_release_threadsafe_function(m_tsfn, napi_tsfn_release);
      m_tsfn = nullptr;
    }
  }

  void queue_event(std::unique_ptr<CallbackEvent> event) noexcept {
    try {
      if (m_tsfn && event && napi_call_threadsafe_function(
          m_tsfn, event.get(), napi_tsfn_blocking) == napi_ok) {
        event.release();
      }
    } catch (...) {
      // Event delivery is best-effort if allocating its payload fails.
    }
  }

  void queue_event(const EventType type, std::string message = {}) noexcept {
    try {
      queue_event(std::make_unique<CallbackEvent>(type, std::move(message)));
    } catch (...) {
      // Terminal delivery is best-effort if allocating its payload fails.
    }
  }

  void run() {
    try {
      debug_async_worker("AsyncWorker run entered");
      Execute();
      debug_async_worker("AsyncWorker Execute returned");
      queue_event(EventType::complete);
    } catch (const std::string& err) {
      queue_event(EventType::error, err);
    } catch (const std::exception& err) {
      queue_event(EventType::error, err.what());
    } catch (...) {
      queue_event(EventType::error, "Compute worker exception");
    }
    m_accepting = false;
    m_stopped = true;
    release_tsfn();
  }

  void start() {
    bool expected = false;
    if (!m_started.compare_exchange_strong(expected, true)) {
      return;
    }
    debug_async_worker("AsyncWorker starting thread");
    try {
      create_tsfn();
      m_thread = std::thread([this]() { run(); });
    } catch (...) {
      if (m_tsfn) {
        m_accepting = false;
        m_stopped = true;
        release_tsfn();
      } else {
        m_started = false;
      }
      throw;
    }
    debug_async_worker("AsyncWorker started thread");
  }

  protected:

  MessageQueue<Message> fromNode;

  void stop() {
    if (m_started && !m_stopped) {
      if (m_accepting.exchange(false)) {
        fromNode.write(Message("close", {}));
      }
    }
    if (m_thread.joinable()) {
      m_thread.join();
    }
  }

  void sendToNode(Message message) {
    debug_async_worker("AsyncWorker queueing event for Node");
    try {
      queue_event(std::make_unique<CallbackEvent>(std::move(message)));
    } catch (...) {
      // Progress delivery is best-effort if allocating its payload fails.
    }
    debug_async_worker("AsyncWorker queued event for Node");
  }

  virtual void Execute() = 0;

  public:

  AsyncWorker(napi_env env, napi_value progress, napi_value complete, napi_value error_callback)
    : m_env(env), m_callbacks(std::make_unique<CallbackState>()) {
    try {
      m_callbacks->progress = create_ref(env, progress);
      m_callbacks->complete = create_ref(env, complete);
      m_callbacks->error = create_ref(env, error_callback);
    } catch (...) {
      finalize_callbacks(env, m_callbacks.release(), nullptr);
      throw;
    }
  }

  virtual ~AsyncWorker() {
    stop();
    release_tsfn();
    if (m_callbacks) {
      finalize_callbacks(m_env, m_callbacks.release(), nullptr);
    }
  }

  void post(Message message) {
    if (!m_accepting.load()) {
      throw std::string("AsyncWorker is stopped");
    }
    start();
    if (!m_accepting.load()) {
      throw std::string("AsyncWorker is stopped");
    }
    const bool closes = message.name == "close";
    fromNode.write(std::move(message));
    if (closes) {
      m_accepting = false;
    }
  }
};

std::unique_ptr<AsyncWorker> create_worker(napi_env, napi_value, napi_value, napi_value);

class AsyncWorkerWrapper {
  static std::string to_string(napi_env env, napi_value value) {
    napi_value str = value;
    napi_valuetype type = napi_undefined;
    check(env, napi_typeof(env, value, &type));
    if (type != napi_string) {
      check(env, napi_coerce_to_string(env, value, &str));
    }
    size_t length = 0;
    check(env, napi_get_value_string_utf8(env, str, nullptr, 0, &length));
    std::string buffer(length + 1, '\0');
    check(env, napi_get_value_string_utf8(env, str, buffer.data(), buffer.size(), &length));
    buffer.resize(length);
    return buffer;
  }

  static void check(napi_env env, const napi_status status) {
    AsyncWorker::check(env, status);
  }

  static void finalize(napi_env, void* data, void*) {
    delete static_cast<AsyncWorker*>(data);
  }

  static napi_value New(napi_env env, napi_callback_info info) {
    try {
      size_t argc = 3;
      napi_value args[3]{};
      napi_value self = nullptr;
      check(env, napi_get_cb_info(env, info, &argc, args, &self, nullptr));
      if (argc < 3) {
        napi_throw_type_error(
          env, nullptr, "AsyncWorker requires progress, complete, and error callbacks");
        return nullptr;
      }
      for (const napi_value callback : args) {
        napi_valuetype type = napi_undefined;
        check(env, napi_typeof(env, callback, &type));
        if (type != napi_function) {
          napi_throw_type_error(env, nullptr, "AsyncWorker callbacks must be functions");
          return nullptr;
        }
      }

      std::unique_ptr<AsyncWorker> worker = create_worker(env, args[0], args[1], args[2]);
      check(env, napi_wrap(env, self, worker.get(), finalize, nullptr, nullptr));
      worker.release();
      return self;
    } catch (...) {
      AsyncWorker::report_current_exception(env);
      return nullptr;
    }
  }

  static napi_value sendToCpp(napi_env env, napi_callback_info info) {
    try {
      debug_async_worker("sendToCpp entered");
      size_t argc = 2;
      napi_value args[2]{};
      napi_value self = nullptr;
      check(env, napi_get_cb_info(env, info, &argc, args, &self, nullptr));
      if (argc < 1) {
        napi_throw_type_error(env, nullptr, "sendToCpp requires a message name");
        return nullptr;
      }
      napi_valuetype name_type = napi_undefined;
      check(env, napi_typeof(env, args[0], &name_type));
      if (name_type != napi_string) {
        napi_throw_type_error(env, nullptr, "sendToCpp message name must be a string");
        return nullptr;
      }

      AsyncWorker* worker = nullptr;
      check(env, napi_unwrap(env, self, reinterpret_cast<void**>(&worker)));

      debug_async_worker("sendToCpp reading message name");
      std::string message_name = to_string(env, args[0]);
      debug_async_worker("sendToCpp read message name");

      MessageValues values;
      if (argc > 1) {
        napi_valuetype values_type = napi_undefined;
        check(env, napi_typeof(env, args[1], &values_type));
        if (values_type != napi_undefined && values_type != napi_null) {
          if (values_type != napi_object) {
            napi_throw_type_error(env, nullptr, "sendToCpp values must be an object");
            return nullptr;
          }
          debug_async_worker("sendToCpp reading values");
          napi_value names = nullptr;
          uint32_t length = 0;
          check(env, napi_get_property_names(env, args[1], &names));
          check(env, napi_get_array_length(env, names, &length));
          for (uint32_t i = 0; i < length; ++i) {
            napi_value key = nullptr;
            napi_value value = nullptr;
            check(env, napi_get_element(env, names, i, &key));
            check(env, napi_get_property(env, args[1], key, &value));
            values[to_string(env, key)] = to_string(env, value);
          }
        }
      }

      debug_async_worker("sendToCpp constructing message");
      Message message(std::move(message_name), std::move(values));
      debug_async_worker("sendToCpp queueing message");
      worker->post(std::move(message));
      debug_async_worker("sendToCpp queued message");
      debug_async_worker("sendToCpp done");
    } catch (...) {
      AsyncWorker::report_current_exception(env);
    }
    return nullptr;
  }

  static napi_value exitNow(napi_env env, napi_callback_info info) {
    try {
      size_t argc = 1;
      napi_value args[1]{};
      int32_t code = 0;
      check(env, napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));
      if (argc > 0) {
        napi_valuetype type = napi_undefined;
        check(env, napi_typeof(env, args[0], &type));
        if (type != napi_number) {
          napi_throw_type_error(env, nullptr, "exitNow code must be a number");
          return nullptr;
        }
        check(env, napi_get_value_int32(env, args[0], &code));
      }
      // Flush all streams then terminate without running destructors/atexit; the
      // worker thread may still be busy and a clean shutdown could deadlock.
      std::fflush(nullptr);
      std::_Exit(code);
    } catch (...) {
      AsyncWorker::report_current_exception(env);
      return nullptr;
    }
  }

  public:

  static napi_value Init(napi_env env, napi_value exports) {
    napi_property_descriptor properties[] = {
      { "sendToCpp", nullptr, sendToCpp, nullptr, nullptr, nullptr, napi_default, nullptr }
    };
    napi_value cons = nullptr;
    check(env, napi_define_class(
      env, "AsyncWorker", NAPI_AUTO_LENGTH, New, nullptr,
      std::size(properties), properties, &cons));
    check(env, napi_set_named_property(env, exports, "AsyncWorker", cons));
    napi_property_descriptor module_properties[] = {
      { "exitNow", nullptr, exitNow, nullptr, nullptr, nullptr, napi_default, nullptr }
    };
    check(env, napi_define_properties(
      env, exports, std::size(module_properties), module_properties));
    return exports;
  }
};
