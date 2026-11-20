#include "nvidia-metadata.h"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <setupapi.h>
#include <cwchar>
#endif
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

std::vector<DisplayAdapter> displayAdapters() {
#ifdef _WIN32
  // GUID_DEVCLASS_DISPLAY, spelled out so no translation unit needs INITGUID.
  static const GUID displayClass = {0x4d36e968, 0xe325, 0x11ce, {0xbf, 0xc1, 0x08, 0x00, 0x2b, 0xe1, 0x03, 0x18}};
  const HDEVINFO set = SetupDiGetClassDevsW(&displayClass, L"PCI", nullptr, DIGCF_PRESENT);
  if (set == INVALID_HANDLE_VALUE) return {};
  struct List {
    HDEVINFO set;
    ~List() {SetupDiDestroyDeviceInfoList(set);}
  } list{set};
  std::vector<DisplayAdapter> result;
  SP_DEVINFO_DATA device{};
  device.cbSize = sizeof(device);
  for (DWORD index = 0; SetupDiEnumDeviceInfo(set, index, &device); ++index) {
    wchar_t id[256]; // MAX_DEVICE_ID_LEN is 200.
    DWORD bus = 0;
    unsigned vendor = 0, product = 0;
    // Four hex digits; a terminating NUL fails the digit test before any read past it.
    const auto hex = [](const wchar_t *text, unsigned &value) {
      value = 0;
      for (int i = 0; i < 4; ++i) {
        const wchar_t c = text[i];
        const unsigned digit = c >= L'0' && c <= L'9' ? c - L'0' : c >= L'A' && c <= L'F' ? c - L'A' + 10 :
                               c >= L'a' && c <= L'f' ? c - L'a' + 10 : 16;
        if (digit > 15) return false;
        value = value * 16 + digit;
      }
      return true;
    };
    // Instance IDs read PCI\VEN_xxxx&DEV_xxxx&... A partial list could look single-vendor, so
    // any unreadable adapter makes the whole answer unknown.
    if (index >= 64 || !SetupDiGetDeviceInstanceIdW(set, &device, id, 256, nullptr) ||
        !SetupDiGetDeviceRegistryPropertyW(set, &device, SPDRP_BUSNUMBER, nullptr,
                                           reinterpret_cast<PBYTE>(&bus), sizeof(bus), nullptr) ||
        std::wcsncmp(id, L"PCI\\VEN_", 8) != 0 || !hex(id + 8, vendor) ||
        std::wcsncmp(id + 12, L"&DEV_", 5) != 0 || !hex(id + 17, product)) return {};
    result.push_back({vendor, product, bus});
  }
  return GetLastError() == ERROR_NO_MORE_ITEMS ? result : std::vector<DisplayAdapter>{};
#else
  return {};
#endif
}
}
