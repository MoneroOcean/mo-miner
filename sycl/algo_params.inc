static void add_result_dev(std::string& result_dev, const std::string& add_str) {
  if (!result_dev.empty()) result_dev += ",";
  result_dev += add_str;
}

static std::string gpu_tuning_entry(
  const std::string& dev_str, const char* const field, const unsigned value
) {
  return dev_str + "*[" + field + "=" + std::to_string(value) + "]";
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
#if defined(_WIN32)
  constexpr unsigned hip_cn_total_per_cu = 96;
  constexpr unsigned hip_cn_worker_per_cu = 48;
#else
  constexpr unsigned hip_cn_total_per_cu = 120;
  constexpr unsigned hip_cn_worker_per_cu = 40;
#endif
  bool saw_intel_level_zero = false;
  bool needs_intel_opencl = false;
  for_each_default_gpu([&](const std::string& dev_str, const sycl::device& dev) {
    const unsigned max_compute_units = dev.get_info<sycl::info::device::max_compute_units>();
    bool prefer_intel_level_zero = false;
    unsigned batch_multiplier = 6;
    if (algo == "cn/gpu") {
      // Backend choice is policy/configuration, not part of the device name. Older NEO builds and
      // Intel GPUs exposing whole 64-EU slice groups run this long recurrence much faster through
      // Level Zero with a shallow queue. Current Xe2-class geometry does not expose that grouping
      // and needs OpenCL to avoid long-dispatch scheduler resets. Use reported execution geometry,
      // not a product-name/device-ID table, so the rule extends to unlisted cards.
      const uint32_t build = sycl_is_level_zero_gpu(dev) ? mom_driver_build(dev) : 0;
      prefer_intel_level_zero = build >= 30000u && build < 35000u;
      if (sycl_is_level_zero_gpu(dev) &&
          dev.get_info<sycl::info::device::vendor>().find("Intel") != std::string::npos) {
        saw_intel_level_zero = true;
        prefer_intel_level_zero = prefer_intel_level_zero || max_compute_units % 64u == 0u;
        needs_intel_opencl = needs_intel_opencl || !prefer_intel_level_zero;
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
        // HIP reaches its broad knee through independent queues. Linux peaks near three workers
        // of forty hashes per reported CU. AdaptiveCpp/HIP on Windows favors two workers of
        // forty-eight; a third worker can overlap independent JIT/runtime initialization and stall
        // every queue. Scale both total work and worker depth from execution geometry so the rule
        // extends to unlisted devices rather than depending on a product ID.
        batch_multiplier = hip_cn_total_per_cu;
      }
    }
    const auto mem = algo2mem.find(algo);
    if (mem == algo2mem.end()) return add_result_dev(result_dev, dev_str + "*" + std::to_string(max_compute_units));

    const unsigned batch_mem        = mem->second,
                   // &~7: keep per-allocation batch a multiple of 8 hashes
                   max_alloc_batch  = (dev.get_info<sycl::info::device::max_mem_alloc_size>() / batch_mem) & 0xFFFFFFF8,
                   max_batch        = dev.get_info<sycl::info::device::global_mem_size>() / batch_mem,
                   // HIP's long recurrence scales through independent queues/processes, not a
                   // single deeper queue. Keep the platform's geometry-scaled worker count inside
                   // the total heuristic; an allocation-only split collapses it to one worker.
                   transport_batch = algo == "cn/gpu" && mom_is_hip(dev)
                     ? std::max(8u, max_compute_units * hip_cn_worker_per_cu)
                     : max_alloc_batch,
                   max_thread_batch = std::min(std::min(max_alloc_batch, transport_batch), max_batch),
                   // Intel Level Zero's measured broad knee is 2--3 hashes per reported compute
                   // unit. Three retains peak throughput while avoiding the severe deep-dispatch
                   // cliff; other transports use their independently measured occupancy scales.
                   default_batch    = prefer_intel_level_zero
                     ? std::min(std::max(8u, (max_compute_units * 3u) & ~7u), max_batch)
                     : std::min(max_compute_units * batch_multiplier, max_batch),
                   best_batch       = algo == "cn/gpu" ? cn_gpu_env_intensity(default_batch, max_batch) : default_batch;
    unsigned used_batch = 0;
    while (used_batch < best_batch) {
      const unsigned current_batch = std::min(best_batch - used_batch, max_thread_batch);
      add_result_dev(result_dev, gpu_tuning_entry(dev_str, "intensity", current_batch));
      used_batch += current_batch;
    }
  });
  if (backend_hint && saw_intel_level_zero) {
    *backend_hint = needs_intel_opencl ? "sycl-opencl" : "sycl-l0";
  }
}

static void add_gpu_c29_algo_dev(std::string& result_dev) {
  for_each_default_gpu([&](const std::string& dev_str, const sycl::device&) {
    add_result_dev(result_dev, dev_str + "*[seed_workgroup=" +
      std::to_string(c29_seed_local_size()) + ";seed_blocks=" +
      std::to_string(c29_seed_blocks_per_item()) + "]");
  });
}

// Emit the resolved primary tuning dimension for each GPU that has room for the
// algorithm's dataset. Secondary launch controls stay algorithm-specific and
// can be overridden in the same bracket/object tuning model.
static void add_gpu_dataset_algo_dev(
  std::string& result_dev, const uint64_t dataset_bytes, const uint64_t min_global_mem,
  unsigned (*intensity)(const sycl::device&), const char* const field = "intensity"
) {
  for_each_default_gpu([&](const std::string& dev_str, const sycl::device& dev) {
    if (!pow_has_dataset_memory(dev, dataset_bytes, min_global_mem)) return;
    add_result_dev(result_dev, field
      ? gpu_tuning_entry(dev_str, field, intensity(dev))
      : dev_str);
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
    else if (gpu_pearlhash_algos.contains(algo)) add_gpu_dataset_algo_dev(
      result_dev, 256 * MiB, 2 * GiB, pearlhash_intensity, "m"
    ); // small A'/B'/noise buffers
    else if (gpu_fishhash_algos.contains(algo)) add_gpu_dataset_algo_dev(result_dev, 4608 * MiB, 6 * GiB, fishhash_intensity); // 4.6 GiB DAG + 72 MiB light cache
    else if (gpu_karlsenhashv2_algos.contains(algo)) add_gpu_dataset_algo_dev(result_dev, 4608 * MiB, 6 * GiB, fishhash_intensity); // FishHashPlus: same 4.6 GiB DAG/intensity as fishhash
    else if (gpu_zelhash_algos.contains(algo)) add_gpu_dataset_algo_dev(
      result_dev, 1856 * MiB, 8 * GiB, mom_zelhash::zelhash_slot_capacity, "slots"
    ); // The reported slot count and the solver use the same allocation-derived heuristic.
    else if (gpu_beamhash3_algos.contains(algo)) add_gpu_dataset_algo_dev(
      result_dev, 3000 * MiB, 8 * GiB, mom_beamhash3::beamhash3_workgroup, "workgroup"
    ); // The reported workgroup and the solver share one device/compiler heuristic.
    if (!result_dev.empty()) result[algo] = result_dev;
  }
  return result;
}
