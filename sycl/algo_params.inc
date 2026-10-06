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
  const auto eligible_device = [](const sycl::device& dev) {
    if constexpr (!mom_sycl_portable_opencl) {
      if (!mom_has_usm_shared(dev) || !mom_has_usm_device(dev) ||
          !sycl_reqd_subgroup_16_supported(dev))
        return false;
    }
    return true;
  };
  for_each_default_gpu([&](const std::string& dev_str, const sycl::device& dev) {
    if (!eligible_device(dev))
      return;
    const unsigned max_compute_units = dev.get_info<sycl::info::device::max_compute_units>();
    bool prefer_intel_level_zero = false;
    unsigned batch_multiplier = 6;
    if (algo == "cn/gpu") {
      if (!max_compute_units)
        return;
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

    if (!mem->second)
      return;
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
    if (!max_thread_batch)
      return;
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
  for_each_default_gpu([&](const std::string& dev_str, const sycl::device& dev) {
    // The fixed C29 arenas, 1024-work-item stages, and largest trim counter all have to fit.
    if (!pow_has_dataset_memory(dev, 4 * GiB, 8 * GiB) ||
        !sycl_workgroup_fits(dev, 1024) || !sycl_local_memory_fits(dev, 32 * KiB))
      return;
    add_result_dev(result_dev,
                   dev_str + "*[seed_workgroup=" + std::to_string(c29_seed_local_size()) +
                       ";seed_blocks=" + std::to_string(c29_seed_blocks_per_item()) + "]");
  });
}

static void add_gpu_c30_algo_dev(std::string& result_dev) {
#if defined(MOM_SYCL_HAS_HIP)
  constexpr uint64_t minimum_local_memory = 64 * KiB;
#else
  constexpr uint64_t minimum_local_memory = 32 * KiB;
#endif
  for_each_default_gpu([&](const std::string& dev_str, const sycl::device& dev) {
    if (!pow_has_dataset_memory(dev, 6240 * MiB, 11 * GiB) ||
        !mom_has_usm_shared(dev) || !mom_has_usm_device(dev) ||
        !sycl_workgroup_fits(dev, 512) ||
        !sycl_local_memory_fits(dev, minimum_local_memory)) return;
    add_result_dev(result_dev, gpu_tuning_entry(dev_str, "intensity", 1));
  });
}

// Emit the resolved primary tuning dimension for each GPU that has room for the
// algorithm's dataset. Secondary launch controls stay algorithm-specific and
// can be overridden in the same bracket/object tuning model.
enum class GpuUsm { none, shared, device, both };

static bool gpu_usm_supported(const sycl::device& dev, const GpuUsm required) {
  switch (required) {
  case GpuUsm::none: return true;
  case GpuUsm::shared: return mom_has_usm_shared(dev);
  case GpuUsm::device: return mom_has_usm_device(dev);
  case GpuUsm::both: return mom_has_usm_shared(dev) && mom_has_usm_device(dev);
  }
  return false;
}

static void add_gpu_dataset_algo_dev(std::string& result_dev, const uint64_t dataset_bytes,
                                     const uint64_t min_global_mem,
                                     unsigned (*intensity)(const sycl::device&),
                                     const char* const field,
                                     const GpuUsm required_usm = GpuUsm::both,
                                     const unsigned minimum_workgroup = 0,
                                     const uint64_t minimum_local_memory = 0,
                                     const bool requires_subgroup_16 = false,
                                     const bool emit_tuning = true) {
  for_each_default_gpu([&](const std::string& dev_str, const sycl::device& dev) {
    if (!pow_has_dataset_memory(dev, dataset_bytes, min_global_mem) ||
        !gpu_usm_supported(dev, required_usm) ||
        (minimum_workgroup && !sycl_workgroup_fits(dev, minimum_workgroup)) ||
        !sycl_local_memory_fits(dev, minimum_local_memory) ||
        (requires_subgroup_16 && !sycl_reqd_subgroup_16_supported(dev)))
      return;
    const unsigned value = intensity(dev);
    if (!value)
      return;
    add_result_dev(result_dev, emit_tuning ? gpu_tuning_entry(dev_str, field, value) : dev_str);
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
  const std::set<std::string>& gpu_misc_algos,
  const std::set<std::string>& gpu_verthash_algos,
  const std::set<std::string>& gpu_zelhash_algos,
  const std::set<std::string>& gpu_beamhash3_algos
) {
  const bool need_sycl_devices = !gpu_cn_algos.empty() || !gpu_c29_algos.empty() ||
                                 !gpu_kawpow_algos.empty() || !gpu_etchash_algos.empty() ||
                                 !gpu_autolykos2_algos.empty() || !gpu_pearlhash_algos.empty() ||
                                 !gpu_fishhash_algos.empty() ||
                                 !gpu_karlsenhashv2_algos.empty() || !gpu_misc_algos.empty() ||
                                 !gpu_verthash_algos.empty() || !gpu_zelhash_algos.empty() || !gpu_beamhash3_algos.empty();
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
  algos.insert(gpu_misc_algos.begin(), gpu_misc_algos.end());
  algos.insert(gpu_verthash_algos.begin(), gpu_verthash_algos.end());
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
    else if (gpu_c29_algos.contains(algo)) {
      if (algo == "c30") add_gpu_c30_algo_dev(result_dev);
      else add_gpu_c29_algo_dev(result_dev);
    }
    else if (gpu_kawpow_algos.contains(algo)) add_gpu_dataset_algo_dev(
      result_dev, 5 * GiB, 6 * GiB, algo == "firopow" ? firopow_intensity : kawpow_intensity,
      "intensity", GpuUsm::shared, 64, 20 * KiB, true
    );
    else if (gpu_etchash_algos.contains(algo)) {
      if (algo == "octopus")
        add_gpu_dataset_algo_dev(result_dev, 4300 * MiB, 6 * GiB, octopus_intensity,
                                 "intensity", GpuUsm::both, 128);
      else
        add_gpu_dataset_algo_dev(result_dev, 4300 * MiB, 5 * GiB, etchash_intensity,
                                 "intensity", GpuUsm::both, 128, 9 * KiB, true);
    }
    else if (gpu_autolykos2_algos.contains(algo)) add_gpu_dataset_algo_dev(result_dev, 1 * GiB, 3 * GiB, autolykos2_intensity,
                               "intensity", mom_sycl_portable_opencl ? GpuUsm::none : GpuUsm::both, 32);
    else if (gpu_pearlhash_algos.contains(algo)) add_gpu_dataset_algo_dev(
      result_dev, 512 * MiB, 2 * GiB, pearlhash_intensity, "m", GpuUsm::both, 256
    );
    else if (gpu_fishhash_algos.contains(algo)) add_gpu_dataset_algo_dev(result_dev, 4608 * MiB, 6 * GiB, fishhash_intensity,
                               "intensity", GpuUsm::both, 128, 6 * KiB, true); // 4.6 GiB DAG + 72 MiB light cache
    else if (gpu_karlsenhashv2_algos.contains(algo)) add_gpu_dataset_algo_dev(result_dev, 4608 * MiB, 6 * GiB, fishhash_intensity,
                               "intensity", GpuUsm::both, 128); // FishHashPlus: same 4.6 GiB DAG/intensity as fishhash
    else if (gpu_misc_algos.contains(algo)) {
      if (algo == "hoohash")
      for_each_default_gpu([&](const std::string& dev_str, const sycl::device& dev) {
        if (hoohash_supported(dev))
          add_result_dev(result_dev, gpu_tuning_entry(dev_str, "intensity", hoohash_intensity(dev)));
      });
      else if (algo == "nexapow")
        for_each_default_gpu([&](const std::string& dev_str, const sycl::device& dev) {
          if (mom_has_usm_device(dev))
            add_result_dev(result_dev,
                           gpu_tuning_entry(dev_str, "intensity", nexapow_intensity(dev)));
        });
    }
    else if (gpu_verthash_algos.contains(algo))
      add_gpu_dataset_algo_dev(result_dev, 1224 * MiB, 2 * GiB, verthash_intensity,
                               "intensity", GpuUsm::both, 64, 8 * KiB);
    else if (gpu_zelhash_algos.contains(algo)) {
      if (algo == "equihash192_7")
        add_gpu_dataset_algo_dev(
            result_dev, 1152 * MiB, 8 * GiB, [](const sycl::device&) { return 1u; },
            "intensity", GpuUsm::device, 1024, 34 * KiB);
      else if (algo == "zhash")
        add_gpu_dataset_algo_dev(
          result_dev, 906 * MiB, 4 * GiB, [](const sycl::device&) { return 1u; },
          "intensity", GpuUsm::device, 1024, 34 * KiB);
      else add_gpu_dataset_algo_dev(
        result_dev, 1856 * MiB, 8 * GiB, mom_zelhash::zelhash_slot_capacity, "slots",
        GpuUsm::both, 1024, 0, true
      ); // The reported slot count and the solver use the same allocation-derived heuristic.
    }
    else if (gpu_beamhash3_algos.contains(algo)) add_gpu_dataset_algo_dev(
      result_dev, 3000 * MiB, 8 * GiB, mom_beamhash3::beamhash3_workgroup, "workgroup",
      GpuUsm::both, 128, 24580
    ); // The reported workgroup and the solver share one device/compiler heuristic.
    if (!result_dev.empty()) result[algo] = result_dev;
  }
  return result;
}
