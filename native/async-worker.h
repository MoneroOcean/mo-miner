// Copyright GNU GPLv3 (c) 2023-2025 MoneroOcean <support@moneroocean.stream>

#pragma once

#include <node_api.h>

#include <atomic>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <iostream>
#include <iterator>
#include <map>
#include <string>
#include <thread>
#include <utility>

typedef std::map<std::string, std::string> MessageValues;

static void debug_async_worker(const char* message) {
  if (!std::getenv("MOMINER_DEBUG_STARTUP")) return;
  std::fprintf(stderr, "MOMINER_DEBUG_STARTUP %s\n", message);
  std::fflush(stderr);
}

struct Message {
  std::string name;
  MessageValues values;
  Message(std::string name, MessageValues values) : name(std::move(name)), values(std::move(values)) {}
};

class SimpleMutex {
  std::atomic_flag m_flag = ATOMIC_FLAG_INIT;

  public:

  void lock() {
    while (m_flag.test_and_set(std::memory_order_acquire)) std::this_thread::yield();
  }

  void unlock() {
    m_flag.clear(std::memory_order_release);
  }
};

class SimpleLock {
  SimpleMutex& m_mutex;

  public:

  explicit SimpleLock(SimpleMutex& mutex) : m_mutex(mutex) { m_mutex.lock(); }
  ~SimpleLock() { m_mutex.unlock(); }

  SimpleLock(const SimpleLock&) = delete;
  SimpleLock& operator=(const SimpleLock&) = delete;
};

template<typename T> class MessageQueue {
  SimpleMutex             m_mutex;
  std::deque<T>           m_buff;

  public:

  void write(T data) {
    debug_async_worker("MessageQueue write locking");
    {
      SimpleLock lock(m_mutex);
      debug_async_worker("MessageQueue write locked");
      m_buff.emplace_back(std::move(data));
      debug_async_worker("MessageQueue write stored");
    }
    debug_async_worker("MessageQueue write done");
  }

  T read() {
    while (true) {
      {
        SimpleLock lock(m_mutex);
        if (!m_buff.empty()) {
          T back = std::move(m_buff.front());
          m_buff.pop_front();
          return back;
        }
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }

  void readAll(std::deque<T>& target) {
    SimpleLock lock(m_mutex);
    std::move(m_buff.begin(), m_buff.end(), std::back_inserter(target));
    m_buff.clear();
  }
};

class AsyncWorker {
  napi_threadsafe_function m_progress_tsfn;
  napi_threadsafe_function m_complete_tsfn;
  napi_threadsafe_function m_error_tsfn;
  MessageQueue<Message> m_toNode;
  std::thread m_thread;
  std::atomic<bool> m_started;
  std::atomic<bool> m_stopped;

  public:

  static void check(napi_env env, napi_status status) {
    if (status != napi_ok) {
      const napi_extended_error_info* info = nullptr;
      napi_get_last_error_info(env, &info);
      napi_throw_error(env, nullptr, info && info->error_message ? info->error_message : "Node-API call failed");
    }
  }

  private:

  static napi_value make_string(napi_env env, const std::string& value) {
    napi_value result;
    check(env, napi_create_string_utf8(env, value.c_str(), value.size(), &result));
    return result;
  }

  static void call_progress(napi_env env, napi_value callback, void* context, void*) {
    if (!env || !callback) return;
    static_cast<AsyncWorker*>(context)->drainQueue(env, callback);
  }

  static void call_complete(napi_env env, napi_value callback, void*, void*) {
    if (!env || !callback) return;
    napi_value global;
    check(env, napi_get_global(env, &global));
    check(env, napi_call_function(env, global, callback, 0, nullptr, nullptr));
  }

  static void call_error(napi_env env, napi_value callback, void*, void* data) {
    if (!env || !callback) {
      delete static_cast<std::string*>(data);
      return;
    }
    std::string* const message = static_cast<std::string*>(data);
    napi_value global, error, text;
    check(env, napi_get_global(env, &global));
    text = make_string(env, *message);
    check(env, napi_create_error(env, nullptr, text, &error));
    napi_value argv[] = { error };
    check(env, napi_call_function(env, global, callback, 1, argv, nullptr));
    delete message;
  }

  static napi_threadsafe_function create_tsfn(napi_env env, napi_value callback, const char* name, napi_threadsafe_function_call_js call_js, void* context) {
    napi_value resource_name;
    napi_threadsafe_function tsfn;
    check(env, napi_create_string_utf8(env, name, NAPI_AUTO_LENGTH, &resource_name));
    check(env, napi_create_threadsafe_function(
      env, callback, nullptr, resource_name, 0, 1, nullptr, nullptr, context, call_js, &tsfn
    ));
    return tsfn;
  }

  void drainQueue(napi_env env, napi_value callback) {
    std::deque<Message> contents;
    m_toNode.readAll(contents);
    napi_value global;
    check(env, napi_get_global(env, &global));

    for (const Message& msg : contents) {
      napi_value values;
      check(env, napi_create_object(env, &values));
      for (const auto& i : msg.values) {
        napi_value value = make_string(env, i.second);
        check(env, napi_set_named_property(env, values, i.first.c_str(), value));
      }
      napi_value argv[] = { make_string(env, msg.name), values };
      check(env, napi_call_function(env, global, callback, 2, argv, nullptr));
    }
  }

  void run() {
    try {
      debug_async_worker("AsyncWorker run entered");
      Execute();
      debug_async_worker("AsyncWorker Execute returned");
      napi_call_threadsafe_function(m_complete_tsfn, nullptr, napi_tsfn_blocking);
    } catch (const std::string& err) {
      napi_call_threadsafe_function(m_error_tsfn, new std::string(err), napi_tsfn_blocking);
    } catch (const std::exception& err) {
      napi_call_threadsafe_function(m_error_tsfn, new std::string(err.what()), napi_tsfn_blocking);
    } catch (...) {
      napi_call_threadsafe_function(m_error_tsfn, new std::string("Compute worker exception"), napi_tsfn_blocking);
    }
    m_stopped = true;
    napi_release_threadsafe_function(m_progress_tsfn, napi_tsfn_release);
    napi_release_threadsafe_function(m_complete_tsfn, napi_tsfn_release);
    napi_release_threadsafe_function(m_error_tsfn, napi_tsfn_release);
  }

  public:

  MessageQueue<Message> fromNode;

  AsyncWorker(napi_env env, napi_value progress, napi_value complete, napi_value error_callback)
    : m_progress_tsfn(create_tsfn(env, progress, "core::progress", call_progress, this)),
      m_complete_tsfn(create_tsfn(env, complete, "core::complete", call_complete, this)),
      m_error_tsfn(create_tsfn(env, error_callback, "core::error", call_error, this)),
      m_started(false), m_stopped(false) {}

  virtual ~AsyncWorker() {
    if (m_started && !m_stopped) fromNode.write(Message("close", {}));
    if (m_thread.joinable()) m_thread.join();
  }

  void start() {
    bool expected = false;
    if (!m_started.compare_exchange_strong(expected, true)) return;
    debug_async_worker("AsyncWorker starting thread");
    m_thread = std::thread([this]() { run(); });
    debug_async_worker("AsyncWorker started thread");
  }

  void sendToNode(const Message& msg) {
    debug_async_worker("AsyncWorker queueing message to Node");
    m_toNode.write(msg);
    debug_async_worker("AsyncWorker calling Node threadsafe function");
    napi_call_threadsafe_function(m_progress_tsfn, nullptr, napi_tsfn_blocking);
    debug_async_worker("AsyncWorker called Node threadsafe function");
  }

  virtual void Execute() = 0;
};

AsyncWorker* create_worker(napi_env, napi_value, napi_value, napi_value, napi_value);

class AsyncWorkerWrapper {
  AsyncWorker* m_worker;

  explicit AsyncWorkerWrapper(AsyncWorker* const worker) : m_worker(worker) {}
  ~AsyncWorkerWrapper() { delete m_worker; }

  static std::string to_string(napi_env env, napi_value value) {
    napi_value str = value;
    napi_valuetype type;
    check(env, napi_typeof(env, value, &type));
    if (type != napi_string) check(env, napi_coerce_to_string(env, value, &str));
    size_t len;
    check(env, napi_get_value_string_utf8(env, str, nullptr, 0, &len));
    std::string result(len + 1, '\0');
    check(env, napi_get_value_string_utf8(env, str, result.data(), result.size(), &len));
    result.resize(len);
    return result;
  }

  static void check(napi_env env, napi_status status) {
    AsyncWorker::check(env, status);
  }

  static void finalize(napi_env, void* data, void*) {
    delete static_cast<AsyncWorkerWrapper*>(data);
  }

  static napi_value New(napi_env env, napi_callback_info info) {
    size_t argc = 4;
    napi_value args[4], self;
    check(env, napi_get_cb_info(env, info, &argc, args, &self, nullptr));
    if (argc < 3) {
      napi_throw_type_error(env, nullptr, "AsyncWorker requires progress, complete, and error callbacks");
      return nullptr;
    }

    AsyncWorkerWrapper* const obj = new AsyncWorkerWrapper(
      create_worker(env, args[0], args[1], args[2], argc > 3 ? args[3] : nullptr)
    );
    check(env, napi_wrap(env, self, obj, finalize, nullptr, nullptr));
    return self;
  }

  static napi_value sendToCpp(napi_env env, napi_callback_info info) {
    debug_async_worker("sendToCpp entered");
    size_t argc = 2;
    napi_value args[2], self;
    check(env, napi_get_cb_info(env, info, &argc, args, &self, nullptr));
    if (argc < 1) {
      napi_throw_type_error(env, nullptr, "sendToCpp requires a message name");
      return nullptr;
    }

    AsyncWorkerWrapper* obj;
    check(env, napi_unwrap(env, self, reinterpret_cast<void**>(&obj)));

    debug_async_worker("sendToCpp reading message name");
    std::string message_name = to_string(env, args[0]);
    debug_async_worker("sendToCpp read message name");

    MessageValues values;
    if (argc > 1) {
      debug_async_worker("sendToCpp reading values");
      napi_value names;
      uint32_t len;
      check(env, napi_get_property_names(env, args[1], &names));
      check(env, napi_get_array_length(env, names, &len));
      for (uint32_t i = 0; i < len; ++i) {
        napi_value key, value;
        check(env, napi_get_element(env, names, i, &key));
        check(env, napi_get_property(env, args[1], key, &value));
        values[to_string(env, key)] = to_string(env, value);
      }
    }

    debug_async_worker("sendToCpp constructing message");
    Message message(std::move(message_name), std::move(values));
    debug_async_worker("sendToCpp queueing message");
    obj->m_worker->fromNode.write(std::move(message));
    debug_async_worker("sendToCpp queued message");
    debug_async_worker("sendToCpp starting worker");
    obj->m_worker->start();
    debug_async_worker("sendToCpp done");
    return nullptr;
  }

  static napi_value exitNow(napi_env env, napi_callback_info info) {
    size_t argc = 1;
    napi_value args[1];
    int32_t code = 0;
    check(env, napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));
    if (argc > 0) check(env, napi_get_value_int32(env, args[0], &code));
    std::fflush(nullptr);
    std::_Exit(code);
  }

  public:

  static napi_value Init(napi_env env, napi_value exports) {
    napi_property_descriptor properties[] = {
      { "sendToCpp", nullptr, sendToCpp, nullptr, nullptr, nullptr, napi_default, nullptr }
    };
    napi_value cons;
    check(env, napi_define_class(
      env, "AsyncWorker", NAPI_AUTO_LENGTH, New, nullptr,
      sizeof(properties) / sizeof(properties[0]), properties, &cons
    ));
    check(env, napi_set_named_property(env, exports, "AsyncWorker", cons));
    napi_property_descriptor module_properties[] = {
      { "exitNow", nullptr, exitNow, nullptr, nullptr, nullptr, napi_default, nullptr }
    };
    check(env, napi_define_properties(
      env, exports, sizeof(module_properties) / sizeof(module_properties[0]), module_properties
    ));
    return exports;
  }
};
