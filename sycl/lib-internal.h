// Copyright GNU GPLv3 (c) 2023-2025 MoneroOcean <support@moneroocean.stream>

#pragma once

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <initializer_list>
#include <limits>
#include <string>
#include <thread>

#include "lib.h"

inline void set_sycl_env(const char* name, const char* value) {
#ifdef _WIN32
  _putenv_s(name, value);
#else
  setenv(name, value, 1);
#endif
}

inline bool sycl_is_level_zero_gpu(const sycl::device& device) {
  return
    device.is_gpu() &&
    device.get_platform().get_info<sycl::info::platform::name>().find("Level-Zero") != std::string::npos;
}

inline unsigned sycl_default_workgroup(
  const sycl::device& device, const std::initializer_list<unsigned> allowed, const unsigned preferred
) {
  const size_t reported_max = device.get_info<sycl::info::device::max_work_group_size>();
  const unsigned max_workgroup = reported_max > std::numeric_limits<unsigned>::max()
    ? std::numeric_limits<unsigned>::max()
    : std::max(1u, static_cast<unsigned>(reported_max));
  unsigned selected = 0;
  for (const unsigned candidate : allowed) {
    if (candidate <= preferred && candidate <= max_workgroup) selected = std::max(selected, candidate);
  }
  return selected ? selected : *std::min_element(allowed.begin(), allowed.end());
}

inline void sycl_wait_and_throw(sycl::event event, const sycl::device& device) {
  if (!sycl_is_level_zero_gpu(device)) {
    event.wait_and_throw();
    return;
  }

  while (event.get_info<sycl::info::event::command_execution_status>() !=
         sycl::info::event_command_status::complete) {
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }
  event.wait_and_throw();
}

sycl::device get_dev(const std::string& dev_str);
