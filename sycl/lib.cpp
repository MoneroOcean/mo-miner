// Copyright GNU GPLv3 (c) 2023-2025 MoneroOcean <support@moneroocean.stream>

#include "lib-internal.h"
#include <algorithm>
#include <cstdlib>
#include <cerrno>
#include <limits>
#include <list>
#include <sstream>

static std::map<std::string, sycl::device> str2dev;

static void update_str2dev(const bool verbose = false) {
  unsigned cpu_num = 0, gpu_num = 0;
  for (auto platform : sycl::platform::get_platforms()) {
    const std::string& platform_name = platform.get_info<sycl::info::platform::name>();
    for (auto device : platform.get_devices()) {
      if (device.is_cpu()) {
        str2dev[std::string("cpu") + std::to_string(++cpu_num)] = device;
      } else if (device.is_gpu()) {
        // OpenCL GPU platforms will be available but not used by default if something else is present
        const std::string gpuN = std::string("gpu") + std::to_string(++gpu_num);
        const bool is_opencl = platform_name.find("OpenCL") != std::string::npos;
        const bool is_level_zero = platform_name.find("Level-Zero") != std::string::npos;
        if (is_opencl || is_level_zero) {
          str2dev[gpuN + (is_opencl ? "o" : "z")] = device;
          if (is_level_zero || !str2dev.contains(gpuN)) str2dev[gpuN] = device;
        } else if (verbose) {
          std::cout << "Found unsupported " << platform_name << " GPU platform device" << std::endl;
        }
      }
    }
    gpu_num = 0; // reset gpu counter for every platform
  }
  if (verbose) for (const auto& pair : str2dev) {
    std::cout << pair.first << ": " << pair.second.get_platform().get_info<sycl::info::platform::name>() << std::endl;
  }
}

static std::string available_dev_str() {
  std::ostringstream devices;
  bool first = true;
  for (const auto& pair : str2dev) {
    if (!first) devices << ", ";
    first = false;
    devices << pair.first << " ("
            << pair.second.get_info<sycl::info::device::name>() << " via "
            << pair.second.get_platform().get_info<sycl::info::platform::name>() << ")";
  }
  return first ? "none" : devices.str();
}

static unsigned parse_kawpow_workgroup_override(const bool is_cpu) {
  const unsigned fallback = is_cpu ? 128 : 256;
  const char* const value = std::getenv("MOMINER_KAWPOW_WORKGROUP");
  if (!value || !*value) return fallback;

  char* end = nullptr;
  errno = 0;
  const unsigned long parsed = std::strtoul(value, &end, 10);
  if (errno || end == value || *end) return fallback;

  switch (parsed) {
    case 64:
    case 128:
    case 256:
    case 512:
      return static_cast<unsigned>(parsed);
  }
  return fallback;
}

static unsigned parse_kawpow_intensity_override(const unsigned local) {
  const char* const value = std::getenv("MOMINER_KAWPOW_INTENSITY");
  if (!value || !*value) return 0;

  char* end = nullptr;
  errno = 0;
  const unsigned long parsed = std::strtoul(value, &end, 10);
  if (errno || end == value || *end || parsed < local || parsed > std::numeric_limits<unsigned>::max()) return 0;
  return static_cast<unsigned>(parsed - (parsed % local));
}

static unsigned kawpow_intensity(const sycl::device& dev) {
  const unsigned local = parse_kawpow_workgroup_override(dev.is_cpu());
  const unsigned override = parse_kawpow_intensity_override(local);
  if (override) return override;

  const unsigned max_compute_units = std::max(1u, dev.get_info<sycl::info::device::max_compute_units>());
  unsigned intensity = local * 32768u;
  intensity = static_cast<unsigned>((static_cast<uint64_t>(intensity) * max_compute_units) / 36u);
  intensity -= intensity % local;
  return std::max(intensity, local * 4096u);
}

static void add_result_dev(std::string& result_dev, const std::string& add_str) {
  if (!result_dev.empty()) result_dev += ",";
  result_dev += add_str;
}

static bool is_default_gpu_dev(const std::string& dev_str) {
  return !dev_str.starts_with("cpu") && !dev_str.ends_with("o") && !dev_str.ends_with("z");
}

template <typename Fn>
static void for_each_default_gpu(Fn&& fn) {
  for (const auto& dev_pair : str2dev) {
    if (is_default_gpu_dev(dev_pair.first)) fn(dev_pair.first, dev_pair.second);
  }
}

static std::list<unsigned> cpu_thread_batches(
  const std::string& algo, const unsigned max_cpu_batch, const unsigned socket_count,
  const unsigned thread_count, const unsigned l3cache, const unsigned batch_mem
) {
  unsigned used_l3cache = 0, used_threads = 0;
  std::list<unsigned> threads;
  if (algo.starts_with("rx/")) {
    // for rx algos we emulate parallelism via inprocess batch threads
    // for each CPU socket we start separate process (named "threads" here)
    // normally we only want one separate process per socket
    // to reduce memory usage per process (2GB) and amount of huge pages too
    const unsigned batch = std::max(1u, std::min(thread_count, l3cache / batch_mem) / socket_count);
    for (unsigned i = 0; i != socket_count; ++i) threads.push_back(batch);
    return threads;
  }

  // fill threads list with single batch
  while (++used_threads <= thread_count && (used_l3cache += batch_mem) <= l3cache)
    threads.push_back(algo == "ghostrider" ? 8 : 1);
  if (!algo.starts_with("argon2/")) {
    // increase batch size until we hit L3 cache limit
    while (used_l3cache < l3cache) {
      bool updated = false;
      for (auto& i : threads) {
        if (i < max_cpu_batch && (used_l3cache += batch_mem) <= l3cache) {
          ++i;
          updated = true;
        }
      }
      if (!updated) break; // in case we hit all max_cpu_batch and not L3 cache
    }
  }
  if (threads.empty()) threads.push_back(1);
  return threads;
}

static void append_grouped_cpu_devs(std::string& result_dev, const std::list<unsigned>& threads) {
  unsigned prev_batch = 0, same_batch_threads = 0;
  auto add_last_dev = [&]() {
    if (!same_batch_threads || !prev_batch) return;
    add_result_dev(result_dev, "cpu" + (prev_batch != 1 ? "*" + std::to_string(prev_batch) : ""));
    if (same_batch_threads != 1) result_dev += "^" + std::to_string(same_batch_threads);
    same_batch_threads = 0;
  };
  for (const unsigned batch : threads) {
    if (same_batch_threads && prev_batch != batch) add_last_dev();
    prev_batch = batch;
    ++same_batch_threads;
  }
  add_last_dev();
}

static void add_cpu_algo_dev(
  std::string& result_dev, const std::string& algo, const unsigned max_cpu_batch,
  const unsigned socket_count, const unsigned thread_count, const unsigned l3cache,
  const std::map<std::string, unsigned>& algo2mem
) {
  const auto mem = algo2mem.find(algo);
  if (mem == algo2mem.end()) return add_result_dev(result_dev, "cpu^" + std::to_string(thread_count));
  append_grouped_cpu_devs(result_dev, cpu_thread_batches(
    algo, max_cpu_batch, socket_count, thread_count, l3cache, mem->second
  ));
}

static void add_gpu_cn_algo_dev(
  std::string& result_dev, const std::string& algo, const std::map<std::string, unsigned>& algo2mem
) {
  for_each_default_gpu([&](const std::string& dev_str, const sycl::device& dev) {
    std::string cn_dev_str = dev_str;
    sycl::device cn_dev = dev;
    unsigned batch_multiplier = 6;
    const auto opencl_dev = str2dev.find(dev_str + "o");
    const std::string device_name = dev.get_info<sycl::info::device::name>();
    if (algo == "cn/gpu" && opencl_dev != str2dev.end() &&
        device_name.find("Intel") != std::string::npos) {
      cn_dev_str = dev_str + "o";
      cn_dev = opencl_dev->second;
      batch_multiplier = 8;
    }
    const unsigned max_compute_units = cn_dev.get_info<sycl::info::device::max_compute_units>();
    const auto mem = algo2mem.find(algo);
    if (mem == algo2mem.end()) return add_result_dev(result_dev, cn_dev_str + "*" + std::to_string(max_compute_units));

    const unsigned batch_mem        = mem->second,
                   max_alloc_batch  = (cn_dev.get_info<sycl::info::device::max_mem_alloc_size>() / batch_mem) & 0xFFFFFFF8,
                   max_batch        = cn_dev.get_info<sycl::info::device::global_mem_size>() / batch_mem,
                   max_thread_batch = std::min(max_alloc_batch, max_batch),
                   best_batch       = std::min(max_compute_units * batch_multiplier, max_batch);
    unsigned used_batch = 0;
    while (used_batch < best_batch) {
      const unsigned current_batch = std::min(best_batch - used_batch, max_thread_batch);
      add_result_dev(result_dev, cn_dev_str + "*" + std::to_string(current_batch));
      used_batch += current_batch;
    }
  });
}

static void add_gpu_c29_algo_dev(std::string& result_dev) {
  for_each_default_gpu([&](const std::string& dev_str, const sycl::device&) {
    add_result_dev(result_dev, dev_str + "*1"); // batch is not really used by this algo
  });
}

static void add_gpu_kawpow_algo_dev(std::string& result_dev) {
  for_each_default_gpu([&](const std::string& dev_str, const sycl::device& dev) {
    constexpr uint64_t min_global_mem = 6ULL * 1024ULL * 1024ULL * 1024ULL;
    if (dev.get_info<sycl::info::device::global_mem_size>() >= min_global_mem)
      add_result_dev(result_dev, dev_str + "*" + std::to_string(kawpow_intensity(dev)));
  });
}

sycl::device get_dev(const std::string& dev_str) {
  if (str2dev.empty()) update_str2dev();
  if (!str2dev.contains(dev_str)) {
    throw std::string("Unknown compute platform " + dev_str + ". Available compute platforms: " + available_dev_str());
  }
  return str2dev.at(dev_str);
}

// return list of supported algos with the best device config
std::map<std::string, std::string> algo_params(
  const unsigned max_cpu_batch,
  const unsigned cpu_sockets, const unsigned cpu_threads, const unsigned cpu_l3cache,
  const std::map<std::string, unsigned>& algo2mem,
  const std::set<std::string>& cpu_algos,
  const std::set<std::string>& gpu_cn_algos,
  const std::set<std::string>& gpu_c29_algos,
  const std::set<std::string>& gpu_kawpow_algos
) {
  const bool need_sycl_devices = !gpu_cn_algos.empty() || !gpu_c29_algos.empty() || !gpu_kawpow_algos.empty();
  if (need_sycl_devices && str2dev.empty()) update_str2dev(true);
  const unsigned socket_count = std::max(1u, cpu_sockets);
  const unsigned thread_count = std::max(1u, cpu_threads);
  // Some platforms do not expose L3 topology. Estimate enough cache for at
  // least one CPU worker per logical thread instead of emitting cpu*0.
  const unsigned l3cache = cpu_l3cache ? cpu_l3cache : thread_count * 2u * 1024u * 1024u;
  std::map<std::string, std::string> result;
  std::set<std::string> algos = cpu_algos;
  algos.insert(gpu_cn_algos.begin(), gpu_cn_algos.end());
  algos.insert(gpu_c29_algos.begin(), gpu_c29_algos.end());
  algos.insert(gpu_kawpow_algos.begin(), gpu_kawpow_algos.end());
  for (const auto& algo : algos) {
    std::string result_dev;
    if (cpu_algos.contains(algo))
      add_cpu_algo_dev(result_dev, algo, max_cpu_batch, socket_count, thread_count, l3cache, algo2mem);
    if (gpu_cn_algos.contains(algo)) add_gpu_cn_algo_dev(result_dev, algo, algo2mem);
    else if (gpu_c29_algos.contains(algo)) add_gpu_c29_algo_dev(result_dev);
    else if (gpu_kawpow_algos.contains(algo)) add_gpu_kawpow_algo_dev(result_dev);
    if (!result_dev.empty()) result[algo] = result_dev;
  }
  return result;
}
