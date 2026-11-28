// Copyright GNU GPLv3 (c) 2026 MoneroOcean <support@moneroocean.stream>

#pragma once

// Where Windows keeps this process's GPU allocations. WDDM can place a device allocation in shared
// GPU memory (system RAM) instead of dedicated VRAM, for example when the commit limit (RAM plus
// page file) is nearly reached; kernels then reach it through the host at a small fraction of
// their speed, without any error.
#if defined(_WIN32)

#include <cstdint>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <dxgi1_4.h>

struct MomGpuMemoryUse {
  uint64_t dedicated = 0, shared = 0;
  bool valid = false;
};

// This process's bytes in the dedicated (local) and shared (non-local) memory segments, summed
// over the hardware adapters of one PCI vendor. `valid` stays false when DXGI reports nothing.
inline MomGpuMemoryUse mom_windows_gpu_memory_use(const unsigned vendor_id) {
  // Resolved at run time from System32 only, like runtime/system-driver.h (a pinned SDK input this
  // header leaves alone): the release application admits no DXGI import, and the check is optional,
  // so a missing DXGI just reports nothing. The module stays loaded.
  using CreateFactory = HRESULT(WINAPI*)(REFIID, void**);
  static const CreateFactory create_factory = [] {
    const HMODULE dxgi = LoadLibraryExW(L"dxgi.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    return dxgi ? reinterpret_cast<CreateFactory>(GetProcAddress(dxgi, "CreateDXGIFactory1")) : nullptr;
  }();
  MomGpuMemoryUse use;
  IDXGIFactory1* factory = nullptr;
  if (!create_factory ||
      FAILED(create_factory(__uuidof(IDXGIFactory1), reinterpret_cast<void**>(&factory))))
    return use;
  for (UINT index = 0;; ++index) {
    IDXGIAdapter1* adapter = nullptr;
    if (factory->EnumAdapters1(index, &adapter) != S_OK)
      break;
    DXGI_ADAPTER_DESC1 description{};
    IDXGIAdapter3* adapter3 = nullptr;
    if (SUCCEEDED(adapter->GetDesc1(&description)) && description.VendorId == vendor_id &&
        !(description.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) &&
        SUCCEEDED(adapter->QueryInterface(__uuidof(IDXGIAdapter3),
                                          reinterpret_cast<void**>(&adapter3)))) {
      DXGI_QUERY_VIDEO_MEMORY_INFO local{}, non_local{};
      if (SUCCEEDED(adapter3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &local)) &&
          SUCCEEDED(
              adapter3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &non_local))) {
        use.dedicated += local.CurrentUsage;
        use.shared += non_local.CurrentUsage;
        use.valid = true;
      }
      adapter3->Release();
    }
    adapter->Release();
  }
  factory->Release();
  return use;
}

#endif
