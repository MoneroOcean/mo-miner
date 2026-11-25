// Copyright GNU GPLv3 (c) 2023-2026 MoneroOcean <support@moneroocean.stream>

#pragma once

#include "async-worker.h"
#include <node_api.h>
#include <iterator>

// Development-only adapter. The mining worker owns no Node/V8/N-API state; this channel
// translates its existing events and keeps TSFN/reference lifetimes on the original Node path.
class NodeWorkerEvents final : public WorkerEvents {
  class NapiFailure {};
  using EventType = WorkerEvent::Type;
  using CallbackEvent = WorkerEvent;
  struct CallbackState {
    napi_ref progress{};
    napi_ref complete{};
    napi_ref error{};
  };
  napi_env m_env;
  std::unique_ptr<CallbackState> m_callbacks;
  napi_threadsafe_function m_tsfn{};

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

  // Thread-safe-function callback on the JS thread: take ownership of one queued worker event and
  // invoke the matching JS callback (progress, error or complete). Teardown calls carry no env.
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
      // Node's legacy N-API callback policy can warn and swallow a pending exception, leaving
      // the worker alive forever. Route it through the normal uncaughtException boundary in
      // every supported Node version; an installed handler may still close/recover the worker.
      napi_value exception = nullptr;
      if (napi_get_and_clear_last_exception(env, &exception) == napi_ok && exception) {
        napi_fatal_exception(env, exception);
      }
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

public:
  NodeWorkerEvents(napi_env env, napi_value progress, napi_value complete, napi_value error_callback)
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

  ~NodeWorkerEvents() override {
    release_tsfn();
    if (m_callbacks) {finalize_callbacks(m_env, m_callbacks.release(), nullptr);}
  }
  void start() override {create_tsfn();}
  void publish(WorkerEvent event) noexcept override {
    try {queue_event(std::make_unique<CallbackEvent>(std::move(event)));} catch (...) {}
  }
  void finish() noexcept override {release_tsfn();}
};

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
    NodeWorkerEvents::check(env, status);
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

      std::unique_ptr<AsyncWorker> worker = create_worker(std::make_unique<NodeWorkerEvents>(env, args[0], args[1], args[2]));
      check(env, napi_wrap(env, self, worker.get(), finalize, nullptr, nullptr));
      worker.release();
      return self;
    } catch (...) {
      NodeWorkerEvents::report_current_exception(env);
      return nullptr;
    }
  }

  // JS sendToCpp(name, values?): validate and convert the message, then post it to the worker.
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

      std::string message_name = to_string(env, args[0]);

      MessageValues values;
      if (argc > 1) {
        napi_valuetype values_type = napi_undefined;
        check(env, napi_typeof(env, args[1], &values_type));
        if (values_type != napi_undefined && values_type != napi_null) {
          if (values_type != napi_object) {
            napi_throw_type_error(env, nullptr, "sendToCpp values must be an object");
            return nullptr;
          }
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

      // post() can block on the first message (it starts the compute thread) and on the queue lock.
      debug_async_worker("sendToCpp queueing ", message_name);
      worker->post(Message(std::move(message_name), std::move(values)));
      debug_async_worker("sendToCpp done");
    } catch (...) {
      NodeWorkerEvents::report_current_exception(env);
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
      NodeWorkerEvents::report_current_exception(env);
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
