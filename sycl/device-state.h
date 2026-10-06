// Copyright GNU GPLv3 (c) 2023-2026 MoneroOcean <support@moneroocean.stream>

#pragma once

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
