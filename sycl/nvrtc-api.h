// Copyright GNU GPLv3 (c) 2026 MoneroOcean <support@moneroocean.stream>

#pragma once

// Compiler-only NVRTC contract; CUDA driver declarations remain in cuda-api.h.

#include <nvrtc.h>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <string>

namespace mom {

// Shared compiler contract: optional lazy loading in Engineering Mode, archive binding in native
// releases. MOM_STATIC_NVRTC is a whole-build policy and must be identical in every host TU.
class NvrtcApi {
  bool loaded_ = false;
#if !defined(MOM_STATIC_NVRTC)
#if defined(_WIN32)
  HMODULE library_ = nullptr;
#else
  void* library_ = nullptr;
  void* builtins_library_ = nullptr;
#endif

  template <typename T>
  bool load_symbol(T& output, const char* const name) {
#if defined(_WIN32)
    output = reinterpret_cast<T>(GetProcAddress(library_, name));
#else
    output = reinterpret_cast<T>(dlsym(library_, name));
#endif
    if (output)
      return true;
    error = std::string("missing NVRTC symbol ") + name;
    return false;
  }
#endif

  // All native algorithms share this compiler. Prefer the checksum-pinned, MoM-owned
  // runtime without relinking an application's SDK; older installs remain optional fallbacks.
  NvrtcApi() {
#if defined(MOM_STATIC_NVRTC)
    // Release mode binds the same compiler API to archive symbols. Its matching builtins and
    // PTX compiler must also be linked statically; there is no application-library search or
    // fallback/extraction path. Engineering keeps the optional dynamic loader below.
    version = &nvrtcVersion;
    create_program = &nvrtcCreateProgram;
    compile_program = &nvrtcCompileProgram;
    get_log_size = &nvrtcGetProgramLogSize;
    get_log = &nvrtcGetProgramLog;
    get_ptx_size = &nvrtcGetPTXSize;
    get_ptx = &nvrtcGetPTX;
    get_cubin_size = &nvrtcGetCUBINSize;
    get_cubin = &nvrtcGetCUBIN;
    get_num_supported_archs = &nvrtcGetNumSupportedArchs;
    get_supported_archs = &nvrtcGetSupportedArchs;
    destroy_program = &nvrtcDestroyProgram;
    error_string = &nvrtcGetErrorString;
    loaded_ = true;
#else
#if defined(_WIN32)
    const char* const names[] = {
      "nvrtc64_130_0.dll", "nvrtc64_120_0.dll", "nvrtc64_121_0.dll", "nvrtc64_122_0.dll"
    };
    for (const char* const name : names) {
      if ((library_ = LoadLibraryA(name)))
        break;
    }
#else
    const char* const names[] = {
      "/opt/mom/nvrtc-13.3.33/lib/libnvrtc.so.13.3.33",
      "libnvrtc.so", "libnvrtc.so.13", "libnvrtc.so.12"
    };
    for (const char* const name : names) {
      if ((library_ = dlopen(name, RTLD_NOW | RTLD_LOCAL))) {
        // NVRTC opens its builtins by SONAME, not relative to this absolute
        // compiler path. Keep the matching dependency loaded without changing
        // the application's SDK, global linker configuration, or environment.
        if (name[0] == '/' && !(builtins_library_ = dlopen(
              "/opt/mom/nvrtc-13.3.33/lib/libnvrtc-builtins.so.13.3", RTLD_NOW | RTLD_LOCAL))) {
          error = "the managed NVRTC builtins library is not installed";
          return;
        }
        break;
      }
    }
#endif
    if (!library_) {
      error = "NVRTC is not installed";
      return;
    }
    loaded_ = true;
#if defined(_WIN32)
    version = reinterpret_cast<decltype(version)>(GetProcAddress(library_, "nvrtcVersion"));
#else
    version = reinterpret_cast<decltype(version)>(dlsym(library_, "nvrtcVersion"));
#endif
    const bool ptx =
      load_symbol(create_program, "nvrtcCreateProgram") &&
      load_symbol(compile_program, "nvrtcCompileProgram") &&
      load_symbol(get_log_size, "nvrtcGetProgramLogSize") &&
      load_symbol(get_log, "nvrtcGetProgramLog") &&
      load_symbol(get_ptx_size, "nvrtcGetPTXSize") &&
      load_symbol(get_ptx, "nvrtcGetPTX") &&
      load_symbol(destroy_program, "nvrtcDestroyProgram") &&
      load_symbol(error_string, "nvrtcGetErrorString");
    if (ptx) {
      (void)(load_symbol(get_cubin_size, "nvrtcGetCUBINSize") &&
             load_symbol(get_cubin, "nvrtcGetCUBIN") &&
             load_symbol(get_num_supported_archs, "nvrtcGetNumSupportedArchs") &&
             load_symbol(get_supported_archs, "nvrtcGetSupportedArchs"));
    }
#endif
  }

public:
  NvrtcApi(const NvrtcApi&) = delete;
  NvrtcApi& operator=(const NvrtcApi&) = delete;

  decltype(&nvrtcCreateProgram) create_program = nullptr;
  // Optional: individual kernels may require a compiler version without disabling legacy JIT.
  decltype(&nvrtcVersion) version = nullptr;
  decltype(&nvrtcCompileProgram) compile_program = nullptr;
  decltype(&nvrtcGetProgramLogSize) get_log_size = nullptr;
  decltype(&nvrtcGetProgramLog) get_log = nullptr;
  decltype(&nvrtcGetPTXSize) get_ptx_size = nullptr;
  decltype(&nvrtcGetPTX) get_ptx = nullptr;
  decltype(&nvrtcGetCUBINSize) get_cubin_size = nullptr;
  decltype(&nvrtcGetCUBIN) get_cubin = nullptr;
  decltype(&nvrtcGetNumSupportedArchs) get_num_supported_archs = nullptr;
  decltype(&nvrtcGetSupportedArchs) get_supported_archs = nullptr;
  decltype(&nvrtcDestroyProgram) destroy_program = nullptr;
  decltype(&nvrtcGetErrorString) error_string = nullptr;
  std::string error;

  bool ptx_available() const {
    return loaded_ && create_program && compile_program && get_log_size && get_log &&
           get_ptx_size && get_ptx && destroy_program && error_string;
  }

  bool cubin_available() const {
    return ptx_available() && get_cubin_size && get_cubin && get_num_supported_archs &&
           get_supported_archs;
  }

  static NvrtcApi& instance() {
    // NVRTC stays loaded for the same process lifetime as cached native modules.
    static NvrtcApi api;
    return api;
  }
};

// Own only the temporary compiler program; release it before loading the resulting module.
class NvrtcProgram {
  NvrtcApi& api_;
  nvrtcProgram program_ = nullptr;

public:
  // Construct after ptx_available()/cubin_available() validates the compiler entry points.
  explicit NvrtcProgram(NvrtcApi& api) noexcept : api_(api) {}
  NvrtcProgram(const NvrtcProgram&) = delete;
  NvrtcProgram& operator=(const NvrtcProgram&) = delete;

  nvrtcProgram* address() noexcept { return &program_; }
  nvrtcProgram get() const noexcept { return program_; }

  // Clear ownership before destruction so cleanup faults cannot cause a repeated destroy.
  void reset() noexcept {
    if (!program_)
      return;
    nvrtcProgram program = program_;
    program_ = nullptr;
    (void)api_.destroy_program(&program);
  }

  ~NvrtcProgram() {
    reset();
  }
};

// Attach the compiler's build log while the temporary program still owns it.
inline std::string rtc_error(NvrtcApi& api, const nvrtcResult status, nvrtcProgram program,
                             const char* const operation) {
  size_t size = 0;
  std::string log;
  if (program && api.get_log_size(program, &size) == NVRTC_SUCCESS && size) {
    log.resize(size);
    (void)api.get_log(program, log.data());
  }
  return std::string(operation) + ": " + api.error_string(status) +
         (log.empty() ? "" : "\n" + log);
}

} // namespace mom
