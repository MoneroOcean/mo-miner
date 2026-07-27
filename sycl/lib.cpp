// Copyright GNU GPLv3 (c) 2023-2025 MoneroOcean <support@moneroocean.stream>

#include "lib-internal.h"
#include <algorithm>
#include <cctype>
#include <limits>
#include <list>
#include <sstream>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

void mom_sycl_poll_pause() {
  static thread_local HANDLE timer = CreateWaitableTimerExW(
    nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
    TIMER_MODIFY_STATE | SYNCHRONIZE);
  LARGE_INTEGER due{};
  due.QuadPart = -1000; // relative 100 microseconds, expressed in 100-ns units
  if (timer && SetWaitableTimer(timer, &due, 0, nullptr, nullptr, FALSE)) {
    WaitForSingleObject(timer, INFINITE);
  } else {
    Sleep(1);
  }
}
#endif

namespace mom_autolykos2 { void autolykos2_cleanup_states() noexcept; }
namespace mom_beamhash3 { void beamhash3_cleanup_states() noexcept; }
namespace mom_zelhash { void zelhash_cleanup_states() noexcept; }
namespace mom_etchash { void etchash_cleanup_states() noexcept; }
namespace mom_fishhash { void fishhash_cleanup_states() noexcept; }
namespace mom_kawpow { void kawpow_cleanup_states() noexcept; }
void cn_gpu_cleanup_states() noexcept;
void c29_cleanup_states() noexcept;
void pearlhash_cleanup_states() noexcept;

// The registry values are cleared explicitly on runtimes that support ordered destruction. The
// heap-owned shell itself has no process-exit destructor, matching the per-algorithm registries.
static std::map<std::string, sycl::device>& str2dev =
  *new std::map<std::string, sycl::device>;
static std::string available_dev_str();

static bool is_integrated_gpu(const sycl::device& device) {
  if (!device.is_gpu()) return false;
  // SYCL 2020 deprecated this query in favor of USM aspects, but those describe allocation
  // support rather than whether GPU memory is physically shared with the host. Device discovery
  // needs the latter distinction and every supported runtime still implements this SYCL 1.2.1
  // property. Keep the unavoidable warning local instead of hiding unrelated deprecations.
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
  const bool integrated = device.get_info<sycl::info::device::host_unified_memory>();
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
  return integrated;
}

void sycl_cleanup() noexcept {
  // Release every queue, bundle, USM allocation, runtime-compiled module, and device handle before
  // Node unloads the compiler runtime DLL. This specifically prevents AdaptiveCpp/CUDA error 4
  // (cudaErrorCudartUnloading) from late C++ static destruction on Windows and Linux.
  mom_autolykos2::autolykos2_cleanup_states();
  mom_beamhash3::beamhash3_cleanup_states();
  c29_cleanup_states();
  cn_gpu_cleanup_states();
  mom_zelhash::zelhash_cleanup_states();
  mom_etchash::etchash_cleanup_states();
  mom_fishhash::fishhash_cleanup_states();
  mom_kawpow::kawpow_cleanup_states();
  pearlhash_cleanup_states();
  str2dev.clear();
}

constexpr uint64_t MiB = 1024ULL * 1024ULL;
constexpr uint64_t GiB = 1024ULL * MiB;

static std::string gpu_identity_key(const sycl::device& device, const unsigned platform_gpu_num) {
  return device.get_info<sycl::info::device::vendor>() + "|" +
         device.get_info<sycl::info::device::name>() + "|" +
         std::to_string(platform_gpu_num);
}

static unsigned gpu_number(const std::string& name) {
  if (!name.starts_with("gpu") || name.size() < 4) return 0;
  size_t pos = 3;
  unsigned number = 0;
  while (pos < name.size() && name[pos] >= '0' && name[pos] <= '9') {
    const unsigned digit = static_cast<unsigned>(name[pos++] - '0');
    if (number > (std::numeric_limits<unsigned>::max() - digit) / 10u) return 0;
    number = number * 10u + digit;
  }
  if (number == 0 || pos != name.size()) return 0;
  return number;
}

static unsigned selected_stable_gpu_number() {
  const char* const backend = std::getenv("MOM_GPU_BACKEND");
  if (!backend || (std::strncmp(backend, "intel", 5) != 0 &&
      std::strncmp(backend, "opencl", 6) != 0)) return 0;
  unsigned long index = 0;
  if (!mom_parse_env_ulong("MOM_GPU_INDEX", index)) return 0;
  if (index >= std::numeric_limits<unsigned>::max()) return std::numeric_limits<unsigned>::max();
  return static_cast<unsigned>(index) + 1u; // MOM_GPU_INDEX is zero-based; mom names start at gpu1.
}

static void sort_gpus_by_name() {
  std::vector<std::pair<std::string, unsigned>> physical_gpus;
  for (const auto& pair : str2dev) {
    const unsigned number = gpu_number(pair.first);
    if (number != 0 && pair.first == "gpu" + std::to_string(number)) {
      physical_gpus.emplace_back(pair.second.get_info<sycl::info::device::name>(), number);
    }
  }
  std::sort(physical_gpus.begin(), physical_gpus.end());

  std::map<unsigned, unsigned> old_to_new;
  for (size_t pos = 0; pos < physical_gpus.size(); ++pos) {
    old_to_new[physical_gpus[pos].second] = static_cast<unsigned>(pos) + 1u;
  }

  std::map<std::string, sycl::device> sorted;
  for (const auto& pair : str2dev) {
    const unsigned old_number = gpu_number(pair.first);
    if (old_number == 0) {
      sorted.emplace(pair.first, pair.second);
      continue;
    }
    sorted.emplace("gpu" + std::to_string(old_to_new.at(old_number)), pair.second);
  }
  str2dev.swap(sorted);
}

static void update_str2dev(const bool verbose = false) {
  str2dev.clear();
  unsigned cpu_num = 0, next_gpu_num = 0;
  std::map<std::string, std::string> gpu2base;
  for (auto platform : sycl::platform::get_platforms()) {
    unsigned platform_gpu_num = 0;
    const std::string& platform_name = platform.get_info<sycl::info::platform::name>();
    for (auto device : platform.get_devices()) {
      if (device.is_cpu()) {
        str2dev[std::string("cpu") + std::to_string(++cpu_num)] = device;
      } else if (device.is_gpu()) {
        ++platform_gpu_num;
        const std::string identity_key = gpu_identity_key(device, platform_gpu_num);
        if (!gpu2base.contains(identity_key)) {
          gpu2base[identity_key] = std::string("gpu") + std::to_string(++next_gpu_num);
        }
        const std::string& gpuN = gpu2base.at(identity_key);
        const bool supported = mom_is_cuda(device) || mom_is_hip(device) ||
                               mom_is_opencl(device) || sycl_is_level_zero_gpu(device);
        if (!supported) {
          if (verbose) {
            std::cout << "Found unsupported " << platform_name << " GPU platform device" << std::endl;
          }
          continue;
        }
        // The process selects one execution API before loading this addon. Keep the public device
        // identity API-neutral; if a caller exposes duplicate APIs anyway, prefer Level Zero over
        // OpenCL for the canonical entry.
        if (!str2dev.contains(gpuN) || sycl_is_level_zero_gpu(device)) str2dev[gpuN] = device;
      }
    }
  }
  // Backend enumeration order can change with runtime/driver versions. Assign mom's public gpuN
  // names by the full hardware-name string so a mixed iGPU/dGPU system remains deterministic.
  sort_gpus_by_name();
  // A multi-vendor runtime does not promise stable device order. Sort first, then apply
  // MOM_GPU_INDEX to mom's stable physical gpuN names.
  const unsigned selected_gpu = selected_stable_gpu_number();
  if (selected_gpu) {
    const std::string selected_name = "gpu" + std::to_string(selected_gpu);
    if (!str2dev.contains(selected_name)) {
      throw std::string("MOM_GPU_INDEX selects " + selected_name +
                        ", but available GPUs are: " + available_dev_str());
    }
    std::erase_if(str2dev, [selected_gpu](const auto& pair) {
      const unsigned number = gpu_number(pair.first);
      return number != 0 && number != selected_gpu;
    });
  }
  if (verbose) for (const auto& pair : str2dev) {
    std::cout << pair.first << ": " << pair.second.get_info<sycl::info::device::name>()
              << " via " << pair.second.get_platform().get_info<sycl::info::platform::name>();
    // host_unified_memory is the backend-independent SYCL distinction between an integrated GPU
    // sharing system memory and a discrete GPU with its own memory. Test discovery consumes this
    // marker instead of maintaining a product-name or PCI-ID allow/deny list.
    if (is_integrated_gpu(pair.second)) {
      std::cout << " [integrated]";
    }
    std::cout << std::endl;
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

// Round down to a whole multiple of step (step > 0) so intensities stay workgroup-aligned.
static unsigned round_down_to_multiple(const unsigned value, const unsigned step) {
  return value - value % step;
}

static unsigned parse_pow_workgroup_override(
  const sycl::device& dev, const char* const env_name, const unsigned fallback_override = 0
) {
  const unsigned preferred = fallback_override ? fallback_override : (dev.is_cpu() ? 128 : 256);
  const unsigned fallback = sycl_default_workgroup(dev, {32, 64, 128, 256, 512}, preferred);
  unsigned long parsed = 0;
  if (!mom_parse_env_ulong(env_name, parsed)) return fallback;
  switch (parsed) {
    case 32:
    case 64:
    case 128:
    case 256:
    case 512:
      return static_cast<unsigned>(parsed);
  }
  return fallback;
}

static unsigned parse_pow_intensity_override(const unsigned local, const char* const env_name) {
  unsigned long parsed = 0;
  if (!mom_parse_env_ulong(env_name, parsed) ||
      parsed < local || parsed > std::numeric_limits<unsigned>::max()) return 0;
  return round_down_to_multiple(static_cast<unsigned>(parsed), local);
}

struct PowDeviceProfile {
  unsigned compute_units;
  uint64_t global_mem;
  uint64_t max_alloc;
};

struct PowIntensityScale {
  unsigned fallback_workgroup;
  unsigned base_work_items;
  unsigned compute_unit_divisor;
};

struct PowIntensityHeuristic {
  PowIntensityScale compact;
  PowIntensityScale balanced;
  PowIntensityScale wide;
};

static PowDeviceProfile pow_device_profile(const sycl::device& dev) {
  return {
    std::max(1u, dev.get_info<sycl::info::device::max_compute_units>()),
    dev.get_info<sycl::info::device::global_mem_size>(),
    dev.get_info<sycl::info::device::max_mem_alloc_size>()
  };
}

static bool pow_has_dataset_memory(
  const sycl::device& dev, const uint64_t dataset_bytes, const uint64_t min_global_mem
) {
  const PowDeviceProfile profile = pow_device_profile(dev);
  const uint64_t reserve = 512ULL * MiB; // headroom for runtime/scratch allocations
  // DPC++'s HIP adapter reports a 4 GiB max allocation on this Windows/Linux ROCm stack even though
  // hipMalloc and SYCL USM successfully allocate the 4.3--5 GiB datasets on a 16 GiB card. Trust the
  // physical-memory check for HIP; explicit etchash/KawPow vector and benchmark runs exercise it.
  return (mom_is_hip(dev) || profile.max_alloc >= dataset_bytes) &&
         profile.global_mem >= std::max(min_global_mem, dataset_bytes + reserve);
}

static unsigned pow_device_score(const PowDeviceProfile& profile) {
  // Favor deeper in-flight batches only on GPUs with enough parallelism and memory headroom.
  const uint64_t mem_per_cu = profile.global_mem / profile.compute_units;
  unsigned score = 0;
  if (profile.compute_units >= 128) score += 2;
  else if (profile.compute_units >= 64) score += 1;
  if (profile.global_mem >= 8ULL * GiB) score += 2;
  else if (profile.global_mem >= 6ULL * GiB) score += 1;
  if (profile.max_alloc >= 2ULL * GiB) score += 1;
  if (mem_per_cu >= 48ULL * MiB) score += 1;
  return score;
}

static PowIntensityScale select_pow_intensity_scale(
  const PowDeviceProfile& profile, const PowIntensityHeuristic& heuristic
) {
  const unsigned score = pow_device_score(profile);
  if (score >= 5) return heuristic.wide;
  if (score >= 3) return heuristic.balanced;
  return heuristic.compact;
}

static unsigned pow_intensity(
  const sycl::device& dev,
  const char* const workgroup_env,
  const char* const intensity_env,
  const PowIntensityHeuristic& heuristic
) {
  const PowDeviceProfile profile = pow_device_profile(dev);
  const PowIntensityScale scale = select_pow_intensity_scale(profile, heuristic);
  const unsigned local = parse_pow_workgroup_override(dev, workgroup_env, scale.fallback_workgroup);
  const unsigned override = parse_pow_intensity_override(local, intensity_env);
  if (override) return override;

  uint64_t intensity64 = static_cast<uint64_t>(local) * scale.base_work_items * profile.compute_units;
  intensity64 /= scale.compute_unit_divisor;
  intensity64 = std::min<uint64_t>(intensity64, std::numeric_limits<unsigned>::max());
  const unsigned intensity = round_down_to_multiple(static_cast<unsigned>(intensity64), local);
  return std::max(intensity, local * 4096u);
}

static unsigned kawpow_intensity(const sycl::device& dev) {
  return pow_intensity(dev, "MOM_KAWPOW_WORKGROUP", "MOM_KAWPOW_INTENSITY", {
    {256, 16384, 48},
    {256, 32768, 48},
    {256, 32768, 36}
  });
}

static unsigned firopow_intensity(const sycl::device& dev) {
#if defined(_WIN32)
  // WDDM retained 21.13 MH/s at the generic depth; reducing it to 4,194,304 settled at
  // 21.01--21.09 MH/s, so keep the cross-ProgPoW Windows heuristic.
  return kawpow_intensity(dev);
#else
  if (!mom_is_cuda(dev)) return kawpow_intensity(dev);
  // Firo's one-period source-JIT kernel needs less queued work than the other ProgPoW variants.
  // On Linux RTX 5060 Ti, the generic 6,291,456-item depth settled at 21.02--21.11 MH/s, while the
  // adjacent 4,194,304-item depth held 21.79--21.84 MH/s after warm-up. Retain the shared kernel
  // and scale the portable device heuristic instead of introducing an NVIDIA algorithm fork.
  return pow_intensity(dev, "MOM_KAWPOW_WORKGROUP", "MOM_KAWPOW_INTENSITY", {
    {256, 16384, 48},
    {256, 21846, 48},
    {256, 21846, 36}
  });
#endif
}

static unsigned etchash_intensity(const sycl::device& dev) {
  return pow_intensity(dev, "MOM_ETCHASH_WORKGROUP", "MOM_ETCHASH_INTENSITY", {
    {64, 32768, 36},
    {64, 65536, 36},
    {64, 131072, 40}
  });
}

static unsigned autolykos2_intensity(const sycl::device& dev) {
  if (mom_is_cuda(dev)) {
    // NVIDIA: larger batches amortize per-iteration host/sync overhead; throughput
    // climbs to a ~70 MH/s plateau on an L4 around these intensities.
    return pow_intensity(dev, "MOM_AUTOLYKOS2_WORKGROUP", "MOM_AUTOLYKOS2_INTENSITY", {
      {64, 16384, 12},
      {64, 32768, 16},
      {64, 32768, 10}
    });
  }
  if (mom_is_hip(dev)) {
#if defined(_WIN32)
    // Windows AdaptiveCpp/HIP has materially higher per-batch submission overhead. On RX 9060 XT,
    // 2.1M -> 8.4M raised the cooperative kernel from 26.49 to 30.97 MH/s; 16.8M added only 0.02.
    // Scale every device-profile tier by four so other cards keep the same memory/CU heuristic.
    return pow_intensity(dev, "MOM_AUTOLYKOS2_WORKGROUP", "MOM_AUTOLYKOS2_INTENSITY", {
      {128, 32768, 12},
      {128, 65536, 16},
      {128, 65536, 10}
    });
#else
    // DPC++ HIP is fastest with wave32-friendly local 128 and enough work to amortize UR submits.
    return pow_intensity(dev, "MOM_AUTOLYKOS2_WORKGROUP", "MOM_AUTOLYKOS2_INTENSITY", {
      {128, 8192, 12},
      {128, 16384, 16},
      {128, 16384, 10}
    });
#endif
  }
  return pow_intensity(dev, "MOM_AUTOLYKOS2_WORKGROUP", "MOM_AUTOLYKOS2_INTENSITY", {
    {64, 4096, 12},
    {64, 8192, 16},
    {64, 8192, 10}
  });
}

// PearlHash's device batch carries M (not a nonce count). gfx12 performs best with the compact
// rectangular policy profile; Intel/NVIDIA retain the large square profile that fills their matrix
// engines efficiently. N/K/rank travel in the job, so this only selects the matching M.
static unsigned pearlhash_intensity(const sycl::device& dev) {
  unsigned long parsed = 0;
  if (mom_parse_env_ulong("MOM_PEARLHASH_INTENSITY", parsed) && parsed >= 256)
    return round_down_to_multiple(static_cast<unsigned>(parsed), 64);
  std::string vendor = dev.get_info<sycl::info::device::vendor>();
  std::transform(vendor.begin(), vendor.end(), vendor.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return vendor.find("amd") != std::string::npos ||
         vendor.find("advanced micro") != std::string::npos ? 8192 : 131072;
}

// FishHash: memory-gather over the 4.6 GiB DAG (bandwidth-bound like etchash). etchash-class intensities.
static unsigned fishhash_intensity(const sycl::device& dev) {
#if defined(_WIN32)
  if (mom_is_hip(dev)) {
    // The Windows HIP cooperative path reaches its throughput knee at 7.46M (11.43/11.41 MH/s)
    // versus 10.71/10.67 at the old 1.86M default. 14.9M gains under 1%, so retain the lower-latency
    // 4x point. Local 128 matches FishState's AMD workgroup while preserving all profile tiers.
    return pow_intensity(dev, "MOM_FISHHASH_WORKGROUP", "MOM_FISHHASH_INTENSITY", {
      {128, 65536, 36},
      {128, 131072, 36},
      {128, 262144, 40}
    });
  }
#endif
  return pow_intensity(dev, "MOM_FISHHASH_WORKGROUP", "MOM_FISHHASH_INTENSITY", {
    {64, 32768, 36},
    {64, 65536, 36},
    {64, 131072, 40}
  });
}

// Equihash 125,4: the "intensity" is the number of header nonces searched per solve (one full Wagner
// pass over a 3.72--4.25 GiB accelerator-specific arena), so one process searches one nonce per
// dispatch. MOM_ZELHASH_INTENSITY overrides that process batch.
static unsigned zelhash_intensity(const sycl::device&) {
  unsigned long parsed = 0;
  if (mom_parse_env_ulong("MOM_ZELHASH_INTENSITY", parsed) && parsed >= 1)
    return static_cast<unsigned>(std::min<unsigned long>(parsed, std::numeric_limits<unsigned>::max()));
  return 1;
}

// BeamHash III: like zelhash, "intensity" is the nonces searched per dispatch (one Wagner pass).
static unsigned beamhash3_intensity(const sycl::device&) {
  unsigned long parsed = 0;
  if (mom_parse_env_ulong("MOM_BEAMHASH3_INTENSITY", parsed) && parsed >= 1)
    return static_cast<unsigned>(std::min<unsigned long>(parsed, std::numeric_limits<unsigned>::max()));
  return 1;
}

static void add_result_dev(std::string& result_dev, const std::string& add_str) {
  if (!result_dev.empty()) result_dev += ",";
  result_dev += add_str;
}

static bool is_default_gpu_dev(const std::string& dev_str) {
  return dev_str.starts_with("gpu");
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

static unsigned cn_gpu_env_intensity(const unsigned default_batch, const unsigned max_batch) {
  unsigned long requested = 0;
  if (!mom_parse_env_ulong("MOM_CN_GPU_INTENSITY", requested) || requested == 0) return default_batch;
  if (requested > max_batch) return max_batch;
  return static_cast<unsigned>(requested);
}

static void add_gpu_cn_algo_dev(
  std::string& result_dev, const std::string& algo,
  const std::map<std::string, unsigned>& algo2mem, std::string* const backend_hint = nullptr
) {
  bool saw_intel_level_zero = false;
  bool needs_intel_opencl = false;
  for_each_default_gpu([&](const std::string& dev_str, const sycl::device& dev) {
    bool legacy_level_zero = false;
    unsigned batch_multiplier = 6;
    if (algo == "cn/gpu") {
      // Backend choice is policy/configuration, not part of the device name. NEO 32224 still
      // needs the measured shallow queue when the configured worker uses Level Zero.
      const uint32_t build = sycl_is_level_zero_gpu(dev) ? mom_driver_build(dev) : 0;
      legacy_level_zero = build >= 30000u && build < 35000u;
      if (sycl_is_level_zero_gpu(dev) &&
          dev.get_info<sycl::info::device::vendor>().find("Intel") != std::string::npos) {
        saw_intel_level_zero = true;
        needs_intel_opencl = needs_intel_opencl || !legacy_level_zero;
      }
      const unsigned score = pow_device_score(pow_device_profile(dev));
      batch_multiplier = score >= 5 ? 8 : (score >= 3 ? 6 : 4);
      if (mom_is_cuda(dev)) {
        // NVIDIA (sm_89): the FP recurrence needs far more in-flight hashes than the
        // Intel heuristic to fill the SMs. An L4 intensity sweep plateaus near
        // compute_units*64 (~3.7k hashes); below that throughput scales with batch.
        // MOM_CN_GPU_INTENSITY still overrides this.
        batch_multiplier = 64;
      }
      if (mom_is_hip(dev)) {
        // RX 9060 XT reaches its repeatable Linux knee at three 512-hash workers: 3.31 KH/s at
        // 1536 total versus 2.95 at 1024, while 2048 falls to 3.12. Use that cross-platform HIP
        // heuristic while the Windows gate checks its own depth curve explicitly. Scale by compute
        // units so the allocation cap can split the total safely; MOM_CN_GPU_INTENSITY remains
        // authoritative.
        batch_multiplier = 96;
      }
    }
    const unsigned max_compute_units = dev.get_info<sycl::info::device::max_compute_units>();
    const auto mem = algo2mem.find(algo);
    if (mem == algo2mem.end()) return add_result_dev(result_dev, dev_str + "*" + std::to_string(max_compute_units));

    const unsigned batch_mem        = mem->second,
                   // &~7: keep per-allocation batch a multiple of 8 hashes
                   max_alloc_batch  = (dev.get_info<sycl::info::device::max_mem_alloc_size>() / batch_mem) & 0xFFFFFFF8,
                   max_batch        = dev.get_info<sycl::info::device::global_mem_size>() / batch_mem,
                   max_thread_batch = std::min(max_alloc_batch, max_batch),
                   // NEO 32224's measured A770 knee is three hashes per reported compute unit:
                   // ~369 H/s at 1536, versus 231 at 256 and a severe long-dispatch cliff at 4096.
                   default_batch    = legacy_level_zero
                     ? std::min(std::max(8u, (max_compute_units * 3u) & ~7u), max_batch)
                     : std::min(max_compute_units * batch_multiplier, max_batch),
                   best_batch       = algo == "cn/gpu" ? cn_gpu_env_intensity(default_batch, max_batch) : default_batch;
    unsigned used_batch = 0;
    while (used_batch < best_batch) {
      const unsigned current_batch = std::min(best_batch - used_batch, max_thread_batch);
      add_result_dev(result_dev, dev_str + "*" + std::to_string(current_batch));
      used_batch += current_batch;
    }
  });
  if (backend_hint && saw_intel_level_zero) {
    *backend_hint = needs_intel_opencl ? "sycl-opencl" : "sycl-l0";
  }
}

static void add_gpu_c29_algo_dev(std::string& result_dev) {
  for_each_default_gpu([&](const std::string& dev_str, const sycl::device&) {
    add_result_dev(result_dev, dev_str + "*1"); // batch is not really used by this algo
  });
}

// Emit "<dev>*<intensity>" for each default GPU that has room for the algo's dataset.
static void add_gpu_dataset_algo_dev(
  std::string& result_dev, const uint64_t dataset_bytes, const uint64_t min_global_mem,
  unsigned (*intensity)(const sycl::device&)
) {
  for_each_default_gpu([&](const std::string& dev_str, const sycl::device& dev) {
    if (pow_has_dataset_memory(dev, dataset_bytes, min_global_mem))
      add_result_dev(result_dev, dev_str + "*" + std::to_string(intensity(dev)));
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
  const std::set<std::string>& gpu_kawpow_algos,
  const std::set<std::string>& gpu_etchash_algos,
  const std::set<std::string>& gpu_autolykos2_algos,
  const std::set<std::string>& gpu_pearlhash_algos,
  const std::set<std::string>& gpu_fishhash_algos,
  const std::set<std::string>& gpu_karlsenhashv2_algos,
  const std::set<std::string>& gpu_zelhash_algos,
  const std::set<std::string>& gpu_beamhash3_algos
) {
  const bool need_sycl_devices = !gpu_cn_algos.empty() || !gpu_c29_algos.empty() ||
                                 !gpu_kawpow_algos.empty() || !gpu_etchash_algos.empty() ||
                                 !gpu_autolykos2_algos.empty() || !gpu_pearlhash_algos.empty() ||
                                 !gpu_fishhash_algos.empty() ||
                                 !gpu_karlsenhashv2_algos.empty() ||
                                 !gpu_zelhash_algos.empty() || !gpu_beamhash3_algos.empty();
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
  algos.insert(gpu_etchash_algos.begin(), gpu_etchash_algos.end());
  algos.insert(gpu_autolykos2_algos.begin(), gpu_autolykos2_algos.end());
  algos.insert(gpu_pearlhash_algos.begin(), gpu_pearlhash_algos.end());
  algos.insert(gpu_fishhash_algos.begin(), gpu_fishhash_algos.end());
  algos.insert(gpu_karlsenhashv2_algos.begin(), gpu_karlsenhashv2_algos.end());
  algos.insert(gpu_zelhash_algos.begin(), gpu_zelhash_algos.end());
  algos.insert(gpu_beamhash3_algos.begin(), gpu_beamhash3_algos.end());
  for (const auto& algo : algos) {
    std::string result_dev;
    if (cpu_algos.contains(algo))
      add_cpu_algo_dev(result_dev, algo, max_cpu_batch, socket_count, thread_count, l3cache, algo2mem);
    // Dataset bytes (largest single allocation) and minimum global memory per GPU algo.
    if (gpu_cn_algos.contains(algo)) {
      std::string backend_hint;
      add_gpu_cn_algo_dev(result_dev, algo, algo2mem, &backend_hint);
      if (!backend_hint.empty()) result["@backend:" + algo] = backend_hint;
    }
    else if (gpu_c29_algos.contains(algo)) add_gpu_c29_algo_dev(result_dev);
    else if (gpu_kawpow_algos.contains(algo)) add_gpu_dataset_algo_dev(
      result_dev, 5 * GiB, 6 * GiB, algo == "firopow" ? firopow_intensity : kawpow_intensity
    );
    else if (gpu_etchash_algos.contains(algo)) add_gpu_dataset_algo_dev(result_dev, 4300 * MiB, 5 * GiB, etchash_intensity);
    else if (gpu_autolykos2_algos.contains(algo)) add_gpu_dataset_algo_dev(result_dev, 1 * GiB, 3 * GiB, autolykos2_intensity);
    else if (gpu_pearlhash_algos.contains(algo)) add_gpu_dataset_algo_dev(result_dev, 256 * MiB, 2 * GiB, pearlhash_intensity); // small A'/B'/noise buffers
    else if (gpu_fishhash_algos.contains(algo)) add_gpu_dataset_algo_dev(result_dev, 4608 * MiB, 6 * GiB, fishhash_intensity); // 4.6 GiB DAG + 72 MiB light cache
    else if (gpu_karlsenhashv2_algos.contains(algo)) add_gpu_dataset_algo_dev(result_dev, 4608 * MiB, 6 * GiB, fishhash_intensity); // FishHashPlus: same 4.6 GiB DAG/intensity as fishhash
    else if (gpu_zelhash_algos.contains(algo)) add_gpu_dataset_algo_dev(result_dev, 1856 * MiB, 8 * GiB, zelhash_intensity); // The fused final filter removes level 4: compact records use 3.83 GiB on Intel and 3.72 GiB on HIP ({5,4,3,2}); aligned records use 4.25 GiB in DPC++/CUDA ({5,4,4,3}) and 4.13 GiB in AdaptiveCpp/CUDA's smaller-bucket fallback. Parent trees are reconstructed only for final zero pairs; >=8 GiB leaves runtime headroom.
    else if (gpu_beamhash3_algos.contains(algo)) add_gpu_dataset_algo_dev(result_dev, 3000 * MiB, 8 * GiB, beamhash3_intensity); // CUDA/HIP default to two 64-byte arenas totaling 4.375 GiB. Intel's fused layout is 7.184 GiB; MOM_BEAMHASH3_COMPACT=0 retains CUDA's 6.758 GiB linear or HIP's 7.184 GiB fused fallback. All supported layouts fit >=8 GiB.
    if (!result_dev.empty()) result[algo] = result_dev;
  }
  return result;
}
