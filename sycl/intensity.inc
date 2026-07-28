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

// PearlHash's device batch carries M (not a nonce count). Choose the same vendor/OS profile used
// for N/K/rank by the compiler policy: this keeps automatic discovery and an explicitly saved
// profile identical without recognizing individual GPU products.
static unsigned pearlhash_intensity(const sycl::device& dev) {
  unsigned long parsed = 0;
  if (mom_parse_env_ulong("MOM_PEARLHASH_INTENSITY", parsed) && parsed >= 256)
    return round_down_to_multiple(static_cast<unsigned>(parsed), 64);
  std::string vendor = dev.get_info<sycl::info::device::vendor>();
  std::transform(vendor.begin(), vendor.end(), vendor.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (vendor.find("nvidia") != std::string::npos) return 65536;
  if (vendor.find("amd") != std::string::npos ||
      vendor.find("advanced micro") != std::string::npos) {
#if defined(_WIN32)
    return 131072;
#else
    return 32768;
#endif
  }
  return 131072;
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
