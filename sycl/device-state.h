// Copyright GNU GPLv3 (c) 2023-2026 MoneroOcean <support@moneroocean.stream>

#pragma once

#include <cstdio>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

template <typename State>
class DeviceStateRegistry {
  using StateMap = std::map<std::string, std::unique_ptr<State>>;
  StateMap states_;
  std::mutex mutex_;

public:
  // Construct each device's state once under the registry lock. The returned reference remains
  // valid until clear(); callers must finish all work before registry teardown.
  template <typename Factory>
  State& get(const std::string& device_key, Factory&& factory) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto existing = states_.find(device_key);
    if (existing != states_.end())
      return *existing->second;
    // Publish only fully constructed states; cleanup predicates require a valid owner.
    auto state = std::forward<Factory>(factory)();
    if (!state)
      throw std::logic_error("Device state factory returned null");
    const auto inserted = states_.emplace(device_key, std::move(state)).first;
    return *inserted->second;
  }

  void clear() {
    clear([](const auto&) { return true; });
  }

  // Retain states whose runtime cannot be safely torn down; callers must exclude active users.
  template <typename Predicate>
  void clear(Predicate&& should_erase) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto it = states_.begin(); it != states_.end();) {
      if (should_erase(*it))
        it = states_.erase(it);
      else
        ++it;
    }
  }
};

// Each miner's ordered cleanup releases the device states before its SYCL runtime unloads.
// Keep the registry itself alive across static teardown, with one instance per state type and TU.
template <typename State>
static DeviceStateRegistry<State>& persistentDeviceStates() {
  static auto* const registry = new DeviceStateRegistry<State>;
  return *registry;
}

// Standard miners use the same device name as the cache key and constructor argument.
template <typename State>
static State& persistentDeviceState(const std::string& device_key) {
  return persistentDeviceStates<State>().get(device_key, [&] {
    return std::make_unique<State>(device_key);
  });
}

// Called after workers stop. A failure in one miner must allow the next miner to release its state.
template <typename State>
static void clearPersistentDeviceStates(const char* algorithm) noexcept {
  try {
    persistentDeviceStates<State>().clear();
  } catch (...) {
    std::fprintf(stderr, "%s: ordered SYCL cleanup failed\n", algorithm);
  }
}
