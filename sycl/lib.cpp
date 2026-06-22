// Copyright GNU GPLv3 (c) 2023-2025 MoneroOcean <support@moneroocean.stream>

#include "lib-internal.h"
#include <algorithm>
#include <limits>
#include <list>
#include <sstream>

static std::map<std::string, sycl::device> str2dev;

constexpr uint64_t MiB = 1024ULL * 1024ULL;
constexpr uint64_t GiB = 1024ULL * MiB;

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
        if (mom_is_cuda(device)) {
          // The DPC++ CUDA backend exposes the GPU through a platform whose name
          // carries no "OpenCL"/"Level-Zero" marker, so register it as the default GPU.
          str2dev[gpuN] = device;
        } else {
          // Detect OpenCL by BACKEND, not the platform-name substring: AMD ("AMD Accelerated Parallel
          // Processing") and Mesa ("rusticl") OpenCL platforms don't contain "OpenCL" in their name, so a
          // name match would drop them. By backend they register as gpuNo and -- via the default-alias
          // fallback below -- become the automatic gpuN on an OpenCL-only (e.g. AMD) box. Level-Zero stays
          // name-based (matches sycl_is_level_zero_gpu in lib-internal.h).
          const bool is_opencl = device.get_backend() == sycl::backend::opencl;
          const bool is_level_zero = platform_name.find("Level-Zero") != std::string::npos;
          if (is_opencl || is_level_zero) {
            str2dev[gpuN + (is_opencl ? "o" : "z")] = device;
            if (is_level_zero || !str2dev.contains(gpuN)) str2dev[gpuN] = device;
          } else if (verbose) {
            std::cout << "Found unsupported " << platform_name << " GPU platform device" << std::endl;
          }
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
  return profile.max_alloc >= dataset_bytes &&
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
  return pow_intensity(dev, "MOM_AUTOLYKOS2_WORKGROUP", "MOM_AUTOLYKOS2_INTENSITY", {
    {64, 4096, 12},
    {64, 8192, 16},
    {64, 8192, 10}
  });
}

// pearl "intensity" is the square NoisyGEMM edge m=n (not a nonce batch). Default 131072 -- the
// network-standard shape for HeroMiners/LuckyPool (k=4096, rank=256), ~53 TH/s on a B580 (~1.2GB
// VRAM). Low-mem cards / pearlpool.cloud use 16384 via MOM_PEARL_INTENSITY (k=1024, rank=64).
static unsigned pearl_intensity(const sycl::device&) {
  unsigned long parsed = 0;
  if (mom_parse_env_ulong("MOM_PEARL_INTENSITY", parsed) && parsed >= 256)
    return round_down_to_multiple(static_cast<unsigned>(parsed), 64);
  return 131072;
}

// kHeavyHash is compute-bound (no dataset): each nonce is 4096 nibble MACs + two Keccak-f[1600].
// Larger batches amortize launch overhead; workgroup 256 matches the kernel's SLM-matrix tiling.
static unsigned kheavyhash_intensity(const sycl::device& dev) {
  return pow_intensity(dev, "MOM_KHEAVYHASH_WORKGROUP", "MOM_KHEAVYHASH_INTENSITY", {
    {256, 16384, 36},
    {256, 32768, 36},
    {256, 32768, 28}
  });
}

// FishHash: memory-gather over the 4.6 GiB DAG (bandwidth-bound like etchash). etchash-class intensities.
static unsigned fishhash_intensity(const sycl::device& dev) {
  return pow_intensity(dev, "MOM_FISHHASH_WORKGROUP", "MOM_FISHHASH_INTENSITY", {
    {64, 32768, 36},
    {64, 65536, 36},
    {64, 131072, 40}
  });
}

// Equihash 125,4: the "intensity" is the number of header nonces searched per solve (one full Wagner
// pass over the ~10.7 GiB bucket arenas per nonce), so it stays small -- a handful of nonces per dispatch.
// Defaults are deliberately modest until M3/M4 tune throughput against real Sol/s. Env override:
// MOM_EQUIHASH_INTENSITY (workgroup is internal to the kernel pipeline; MOM_EQUIHASH_WORKGROUP reserved).
static unsigned equihash125_4_intensity(const sycl::device&) {
  unsigned long parsed = 0;
  if (mom_parse_env_ulong("MOM_EQUIHASH_INTENSITY", parsed) && parsed >= 1)
    return static_cast<unsigned>(std::min<unsigned long>(parsed, std::numeric_limits<unsigned>::max()));
  return 1;
}

// BeamHash III: like equihash125_4, "intensity" is the nonces searched per dispatch (one Wagner pass).
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
    if (algo == "cn/gpu") {
      // cn/gpu runs long kernels that trip Level-Zero scheduling resets on
      // Intel Xe2; the OpenCL backend runs them cleanly, so prefer it.
      const auto opencl_dev = str2dev.find(dev_str + "o");
      if (opencl_dev != str2dev.end()) {
        cn_dev_str = opencl_dev->first;
        cn_dev = opencl_dev->second;
      }
      const unsigned score = pow_device_score(pow_device_profile(cn_dev));
      batch_multiplier = score >= 5 ? 8 : (score >= 3 ? 6 : 4);
      if (mom_is_cuda(cn_dev)) {
        // NVIDIA (sm_89): the FP recurrence needs far more in-flight hashes than the
        // Intel heuristic to fill the SMs. An L4 intensity sweep plateaus near
        // compute_units*64 (~3.7k hashes); below that throughput scales with batch.
        // MOM_CN_GPU_INTENSITY still overrides this.
        batch_multiplier = 64;
      }
    }
    const unsigned max_compute_units = cn_dev.get_info<sycl::info::device::max_compute_units>();
    const auto mem = algo2mem.find(algo);
    if (mem == algo2mem.end()) return add_result_dev(result_dev, cn_dev_str + "*" + std::to_string(max_compute_units));

    const unsigned batch_mem        = mem->second,
                   // &~7: keep per-allocation batch a multiple of 8 hashes
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
  const std::set<std::string>& gpu_pearl_algos,
  const std::set<std::string>& gpu_kheavyhash_algos,
  const std::set<std::string>& gpu_fishhash_algos,
  const std::set<std::string>& gpu_karlsenhashv2_algos,
  const std::set<std::string>& gpu_pyrinhashv2_algos,
  const std::set<std::string>& gpu_equihash125_4_algos,
  const std::set<std::string>& gpu_beamhash3_algos
) {
  const bool need_sycl_devices = !gpu_cn_algos.empty() || !gpu_c29_algos.empty() ||
                                 !gpu_kawpow_algos.empty() || !gpu_etchash_algos.empty() ||
                                 !gpu_autolykos2_algos.empty() || !gpu_pearl_algos.empty() ||
                                 !gpu_kheavyhash_algos.empty() || !gpu_fishhash_algos.empty() ||
                                 !gpu_karlsenhashv2_algos.empty() || !gpu_pyrinhashv2_algos.empty() ||
                                 !gpu_equihash125_4_algos.empty() || !gpu_beamhash3_algos.empty();
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
  algos.insert(gpu_pearl_algos.begin(), gpu_pearl_algos.end());
  algos.insert(gpu_kheavyhash_algos.begin(), gpu_kheavyhash_algos.end());
  algos.insert(gpu_fishhash_algos.begin(), gpu_fishhash_algos.end());
  algos.insert(gpu_karlsenhashv2_algos.begin(), gpu_karlsenhashv2_algos.end());
  algos.insert(gpu_pyrinhashv2_algos.begin(), gpu_pyrinhashv2_algos.end());
  algos.insert(gpu_equihash125_4_algos.begin(), gpu_equihash125_4_algos.end());
  algos.insert(gpu_beamhash3_algos.begin(), gpu_beamhash3_algos.end());
  for (const auto& algo : algos) {
    std::string result_dev;
    if (cpu_algos.contains(algo))
      add_cpu_algo_dev(result_dev, algo, max_cpu_batch, socket_count, thread_count, l3cache, algo2mem);
    // Dataset bytes (largest single allocation) and minimum global memory per GPU algo.
    if (gpu_cn_algos.contains(algo)) add_gpu_cn_algo_dev(result_dev, algo, algo2mem);
    else if (gpu_c29_algos.contains(algo)) add_gpu_c29_algo_dev(result_dev);
    else if (gpu_kawpow_algos.contains(algo)) add_gpu_dataset_algo_dev(result_dev, 5 * GiB, 6 * GiB, kawpow_intensity);
    else if (gpu_etchash_algos.contains(algo)) add_gpu_dataset_algo_dev(result_dev, 4300 * MiB, 5 * GiB, etchash_intensity);
    else if (gpu_autolykos2_algos.contains(algo)) add_gpu_dataset_algo_dev(result_dev, 1 * GiB, 3 * GiB, autolykos2_intensity);
    else if (gpu_pearl_algos.contains(algo)) add_gpu_dataset_algo_dev(result_dev, 256 * MiB, 2 * GiB, pearl_intensity); // small A'/B'/noise buffers
    else if (gpu_kheavyhash_algos.contains(algo)) add_gpu_dataset_algo_dev(result_dev, 1 * MiB, 512 * MiB, kheavyhash_intensity); // compute-bound: only the 8 KiB matrix + I/O
    else if (gpu_fishhash_algos.contains(algo)) add_gpu_dataset_algo_dev(result_dev, 4608 * MiB, 6 * GiB, fishhash_intensity); // 4.6 GiB DAG + 72 MiB light cache
    else if (gpu_karlsenhashv2_algos.contains(algo)) add_gpu_dataset_algo_dev(result_dev, 4608 * MiB, 6 * GiB, fishhash_intensity); // FishHashPlus: same 4.6 GiB DAG/intensity as fishhash
    else if (gpu_pyrinhashv2_algos.contains(algo)) add_gpu_dataset_algo_dev(result_dev, 1 * MiB, 512 * MiB, kheavyhash_intensity); // kHeavyHash-family: no DAG, just the 8 KiB matrix + I/O
    else if (gpu_equihash125_4_algos.contains(algo)) add_gpu_dataset_algo_dev(result_dev, 1856 * MiB, 8 * GiB, equihash125_4_intensity); // ~7.03 GiB Wagner bucket arenas total: per-level slot widths 6/6/5/5/3 words WITH the tree log co-located inline at each slot's tail (no separate tree arena) -> one coalesced write stream per survivor (rounds 420->95 ms, +280% solve/s on B580). Level 3 is padded to 20 B (not 16) to dodge the 2-slots-per-32B-line false-share. Largest single alloc ~1.69 GiB level-0 arena. Fits a >=8 GiB card (was 7.3 GiB split-tree / 10.7 GiB / >=12 GiB before).
    else if (gpu_beamhash3_algos.contains(algo)) add_gpu_dataset_algo_dev(result_dev, 3200 * MiB, 8 * GiB, beamhash3_intensity); // ~6.9 GiB Wagner arenas: ONE shared transient scratch for levels 0..5 (~2.6 GiB; leaf index == level-0 slot so no persistent mixed[0], and collide never re-reads its input level so each level overwrites the prior in place -- no ping-pong) + 5 tree logs (~0.26 GiB each) + a reused bucket arena (~3.0 GiB, the largest single alloc). Fits a >=8 GiB card (was ~12.8 GiB / >=16 GiB before the shrink).
    if (!result_dev.empty()) result[algo] = result_dev;
  }
  return result;
}
