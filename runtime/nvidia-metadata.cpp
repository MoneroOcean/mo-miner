#include "nvidia-metadata.h"
#ifdef MOM_NATIVE_NVML_METADATA
#include "system-driver.h"
#include <nvml.h>
#include <cstdint>
#include <limits>
#endif

namespace mom::runtime {
std::vector<unsigned> nvidiaComputeCapabilities() {
#ifdef MOM_NATIVE_NVML_METADATA
  // Derive every calling convention/type from the pinned SDK; link only the system driver.
  static const auto init = systemDriverSymbol<SystemDriver::nvml, decltype(&nvmlInit_v2)>("nvmlInit_v2");
  static const auto shutdown = systemDriverSymbol<SystemDriver::nvml, decltype(&nvmlShutdown)>("nvmlShutdown");
  static const auto countDevices = systemDriverSymbol<SystemDriver::nvml, decltype(&nvmlDeviceGetCount_v2)>("nvmlDeviceGetCount_v2");
  static const auto deviceAt = systemDriverSymbol<SystemDriver::nvml, decltype(&nvmlDeviceGetHandleByIndex_v2)>("nvmlDeviceGetHandleByIndex_v2");
  static const auto capability = systemDriverSymbol<SystemDriver::nvml, decltype(&nvmlDeviceGetCudaComputeCapability)>("nvmlDeviceGetCudaComputeCapability");
  if (!init || !shutdown || !countDevices || !deviceAt || !capability || init() != NVML_SUCCESS) return {};
  // NVML init/shutdown are reference-counted. Own exactly our successful initialization,
  // including error/allocation unwinding; do not retire another adapter's NVML session.
  struct Session {
    decltype(shutdown) release;
    bool active = true;
    ~Session() {if (active) (void)release();}
    bool close() {active = false; return release() == NVML_SUCCESS;}
  } session{shutdown};
  unsigned count = 0;
  if (countDevices(&count) != NVML_SUCCESS || count > 4096) return {};
  std::vector<unsigned> result;
  result.reserve(count);
  for (unsigned index = 0; index < count; ++index) {
    nvmlDevice_t device = nullptr;
    int major = -1, minor = -1;
    if (deviceAt(index, &device) != NVML_SUCCESS || !device ||
        capability(device, &major, &minor) != NVML_SUCCESS || major <= 0 || minor < 0 || minor > 9) return {};
    const auto sm = static_cast<uint64_t>(major) * 10 + static_cast<unsigned>(minor);
    if (sm > std::numeric_limits<unsigned>::max()) return {};
    result.push_back(static_cast<unsigned>(sm));
  }
  if (!session.close()) return {};
  return result;
#else
  return {}; // CPU-only proof builds must not look up any GPU driver.
#endif
}
}
