// Copyright GNU GPLv3 (c) 2023-2025 MoneroOcean <support@moneroocean.stream>

#pragma once

#include <sycl/sycl.hpp>

#include <chrono>
#include <cstdlib>
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

inline void sycl_wait_and_throw(sycl::event event, const sycl::device& device) {
  const bool is_level_zero_gpu =
    device.is_gpu() &&
    device.get_platform().get_info<sycl::info::platform::name>().find("Level-Zero") != std::string::npos;
  if (!is_level_zero_gpu) {
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
