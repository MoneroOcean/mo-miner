#include "pow-intensity.h"

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
      if (sycl_workgroup_fits(dev, static_cast<unsigned>(parsed)))
        return static_cast<unsigned>(parsed);
      break;
  }
  return fallback;
}

static unsigned parse_pow_intensity_override(const unsigned local, const char* const env_name) {
  unsigned long parsed = 0;
  if (!mom_parse_env_ulong(env_name, parsed) ||
      parsed < local || parsed > std::numeric_limits<unsigned>::max()) return 0;
  return round_down_to_multiple(static_cast<unsigned>(parsed), local);
}

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
  return (mom_is_hip(dev) || sycl_is_level_zero_gpu(dev) || profile.max_alloc >= dataset_bytes) &&
         profile.global_mem >= std::max(min_global_mem, dataset_bytes + reserve);
}

static unsigned pow_intensity(
  const sycl::device& dev,
  const char* const workgroup_env,
  const char* const intensity_env,
  const PowIntensityHeuristic& heuristic,
  const unsigned max_work_items_per_gib = 0
) {
  const PowDeviceProfile profile = pow_device_profile(dev);
  const PowIntensityScale scale = select_pow_intensity_scale(profile, heuristic);
  const unsigned local = parse_pow_workgroup_override(dev, workgroup_env, scale.fallback_workgroup);
  const unsigned override = parse_pow_intensity_override(local, intensity_env);
  if (override) return override;

  return pow_intensity(profile, heuristic, local, max_work_items_per_gib);
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
  // Four Mi work-items/GiB places high-EU devices on the measured broad Etchash plateau; current
  // Xe2, NVIDIA, and AMD heuristics are already shallower and therefore remain unchanged.
  return pow_intensity(dev, "MOM_ETCHASH_WORKGROUP", "MOM_ETCHASH_INTENSITY", {
    {64, 32768, 36},
    {64, 65536, 36},
    {64, 131072, 40}
  }, 4);
}

static unsigned octopus_intensity(const sycl::device& dev) {
  if (!mom_is_hip(dev))
    return etchash_intensity(dev);
  // AMD's queue-native WMMA pipeline reaches its throughput plateau with a deeper dispatch. On the
  // RX 9060 XT, 4,194,304 work-items sustained 29.18 MH/s while Etchash's 1,864,128 held 26.34 MH/s.
  // Scale the same device profiles instead of introducing a card-specific setting.
  return pow_intensity(dev, "MOM_ETCHASH_WORKGROUP", "MOM_ETCHASH_INTENSITY",
                       {{64, 73728, 36}, {64, 147456, 36}, {64, 294912, 40}}, 9);
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
  if (mom_parse_env_ulong("MOM_PEARLHASH_INTENSITY", parsed) && parsed >= 256 &&
      parsed <= std::numeric_limits<unsigned>::max())
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

static bool walahash_supported(const sycl::device& dev) {
  return dev.is_gpu() && mom_has_usm_device(dev) && sycl_workgroup_fits(dev, 32);
}

static unsigned walahash_intensity(const sycl::device& dev) {
  const unsigned override = parse_pow_intensity_override(1, "MOM_WALAHASH_INTENSITY");
  if (override)
    return override;
  const uint64_t global_mem = dev.get_info<sycl::info::device::global_mem_size>();
  const uint64_t max_alloc = dev.get_info<sycl::info::device::max_mem_alloc_size>();
  if (mom_is_cuda(dev))
    return 67108864;
  // The 33,554,432-nonce accelerated layout allocates two 1-GiB buffers. High-memory HIP
  // devices sustain this batch on both operating systems; select by backend/capacity rather than
  // a GPU model so future compatible devices inherit the measured default.
  if (mom_is_hip(dev) && global_mem >= 8ULL * GiB)
    return 33554432;
  if (sycl_is_level_zero_gpu(dev) && global_mem >= 8ULL * GiB && max_alloc >= 2ULL * GiB)
    return 33554432; // Memory-guarded Intel matrix batch also fits under WDDM.
#ifdef _WIN32
  return 16777216; // smaller fallback for constrained Windows devices.
#else
  return sycl_is_level_zero_gpu(dev) ? 2097152 : 524288;
#endif
}

static unsigned verthash_intensity(const sycl::device& dev) {
  const unsigned override = parse_pow_intensity_override(16, "MOM_VERTHASH_INTENSITY");
  if (override)
    return override;
  if (sycl_is_level_zero_gpu(dev) && !is_integrated_gpu(dev)) {
#ifdef _WIN32
    return 1u << 18; // 0.8% below the 524K peak with about twice the WDDM dispatch headroom.
#else
    return 1u << 19; // Measured B580 throughput knee.
#endif
  }
  return 1u << 16;
}

// Equihash 125,4: the "intensity" is the number of header nonces searched per solve (one full Wagner
// pass over a 3.72--4.25 GiB accelerator-specific arena), so one process searches one nonce per
// dispatch. MOM_ZELHASH_INTENSITY overrides that process batch.

static bool xelishashv3_supported(const sycl::device& dev) {
  return mom_has_usm_device(dev);
}

static uint64_t xelishashv3_scratch_capacity(const sycl::device& dev) {
  constexpr uint64_t scratch_bytes = 531ULL * 1024;
  const uint64_t global_mem = dev.get_info<sycl::info::device::global_mem_size>();
  const uint64_t global_limit = (global_mem / 5) * 2 + (global_mem % 5) * 2 / 5;
  return std::min<uint64_t>(dev.get_info<sycl::info::device::max_mem_alloc_size>(), global_limit) /
         scratch_bytes;
}

static unsigned xelishashv3_intensity(const sycl::device& dev) {
  if (!xelishashv3_supported(dev))
    return 0;
  const uint64_t memory_limit = xelishashv3_scratch_capacity(dev);
  if (!memory_limit)
    return 0;
  const unsigned override = parse_pow_intensity_override(1, "MOM_XELISHASHV3_INTENSITY");
  if (override)
    return static_cast<unsigned>(std::min<uint64_t>(override, memory_limit));
  const uint64_t compute_limit =
      static_cast<uint64_t>(dev.get_info<sycl::info::device::max_compute_units>()) * 384ULL;
  const uint64_t automatic_limit = std::min(
      {memory_limit, compute_limit, static_cast<uint64_t>(std::numeric_limits<unsigned>::max())});
  return static_cast<unsigned>(automatic_limit & ~255ULL);
}

static unsigned nexapow_intensity(const sycl::device& dev) {
  const unsigned override = parse_pow_intensity_override(1, "MOM_NEXAPOW_INTENSITY");
  if (override)
    return override;
  const uint64_t memory = dev.get_info<sycl::info::device::global_mem_size>();
  if (memory >= 15 * GiB)
    return 1u << 23;
  if (memory >= 10 * GiB)
    return 1u << 22;
  return std::max(256u, dev.get_info<sycl::info::device::max_compute_units>() * 64u);
}

static bool hoohash_supported(const sycl::device& dev) {
  return dev.has(sycl::aspect::fp64) && mom_has_usm_device(dev);
}

static unsigned hoohash_intensity(const sycl::device& dev) {
  const unsigned override = parse_pow_intensity_override(1, "MOM_HOOHASH_INTENSITY");
  if (override)
    return override;
#if defined(_WIN32)
  // A longer WDDM/CUDA batch amortizes dispatch cost: 114688 measured 1.4164 MH/s versus
  // 1.3392 MH/s at 65536. Raising it to 262144 added only 0.45% while more than doubling
  // job-switch latency, so keep the smaller point on the throughput plateau.
  if (mom_is_cuda(dev))
    return 114688;
#else
  (void)dev;
#endif
  return 65536;
}
