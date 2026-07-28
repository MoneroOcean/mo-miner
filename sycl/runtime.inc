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
namespace mom_beamhash3 {
void beamhash3_cleanup_states() noexcept;
unsigned beamhash3_workgroup(const sycl::device&);
}
namespace mom_zelhash {
void zelhash_cleanup_states() noexcept;
unsigned zelhash_slot_capacity(const sycl::device&);
}
namespace mom_etchash { void etchash_cleanup_states() noexcept; }
namespace mom_fishhash { void fishhash_cleanup_states() noexcept; }
namespace mom_kawpow { void kawpow_cleanup_states() noexcept; }
void cn_gpu_cleanup_states() noexcept;
void c29_cleanup_states() noexcept;
uint32_t c29_seed_local_size();
uint32_t c29_seed_blocks_per_item();
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
