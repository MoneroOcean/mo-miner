#if defined(MOM_SYCL_HAS_HIP)
// HIP itself is already required by AdaptiveCpp's HIP backend, but HIPRTC is optional: loading it
// dynamically lets a release worker retain the fully functional SYCL path when the source compiler
// is absent or broken. Generated code is cached locally and is never part of a release archive.
struct PearlHashHiprtcApi {
#if defined(_WIN32)
  HMODULE library = nullptr;
#else
  void* library = nullptr;
#endif
  decltype(&hiprtcCreateProgram) create_program = nullptr;
  decltype(&hiprtcCompileProgram) compile_program = nullptr;
  decltype(&hiprtcGetProgramLogSize) get_log_size = nullptr;
  decltype(&hiprtcGetProgramLog) get_log = nullptr;
  decltype(&hiprtcGetCodeSize) get_code_size = nullptr;
  decltype(&hiprtcGetCode) get_code = nullptr;
  decltype(&hiprtcDestroyProgram) destroy_program = nullptr;
  decltype(&hiprtcGetErrorString) error_string = nullptr;
  std::string error;

  template <typename T>
  bool symbol(T& output, const char* name) {
#if defined(_WIN32)
    output = reinterpret_cast<T>(GetProcAddress(library, name));
#else
    output = reinterpret_cast<T>(dlsym(library, name));
#endif
    if (output) return true;
    error = std::string("missing HIPRTC symbol ") + name;
    return false;
  }

  PearlHashHiprtcApi() {
#if defined(_WIN32)
    const char* names[] = {"hiprtc.dll", "hiprtc0701.dll", "hiprtc0700.dll",
                           "hiprtc0604.dll", "hiprtc0603.dll", "hiprtc0507.dll"};
    for (const char* name : names)
      if ((library = LoadLibraryA(name))) break;
#else
    const char* names[] = {"libhiprtc.so", "libhiprtc.so.7", "libhiprtc.so.6"};
    for (const char* name : names)
      if ((library = dlopen(name, RTLD_NOW | RTLD_LOCAL))) break;
#endif
    if (!library) {
      error = "HIPRTC library is not installed";
      return;
    }
    (void)(symbol(create_program, "hiprtcCreateProgram") &&
           symbol(compile_program, "hiprtcCompileProgram") &&
           symbol(get_log_size, "hiprtcGetProgramLogSize") &&
           symbol(get_log, "hiprtcGetProgramLog") &&
           symbol(get_code_size, "hiprtcGetCodeSize") &&
           symbol(get_code, "hiprtcGetCode") &&
           symbol(destroy_program, "hiprtcDestroyProgram") &&
           symbol(error_string, "hiprtcGetErrorString"));
  }

  bool available() const {
    return library && create_program && compile_program && get_log_size && get_log &&
           get_code_size && get_code && destroy_program && error_string;
  }
  static PearlHashHiprtcApi& instance() {
    static PearlHashHiprtcApi api;
    return api;
  }
};
#endif

#if defined(MOM_SYCL_HAS_HIP) || defined(MOM_SYCL_HAS_CUDA)
static uint64_t pearlhash_jit_hash(const std::string& value) {
  uint64_t hash = 1469598103934665603ULL;
  for (const unsigned char byte : value) {
    hash ^= byte;
    hash *= 1099511628211ULL;
  }
  return hash;
}

static std::filesystem::path pearlhash_jit_cache_dir() {
  if (const char* configured = std::getenv("MOM_JIT_CACHE_DIR"); configured && *configured)
    return configured;
#if defined(_WIN32)
  if (const char* base = std::getenv("LOCALAPPDATA"); base && *base)
    return std::filesystem::path(base) / "mom" / "jit";
#else
  if (const char* base = std::getenv("XDG_CACHE_HOME"); base && *base)
    return std::filesystem::path(base) / "mom" / "jit";
  if (const char* home = std::getenv("HOME"); home && *home)
    return std::filesystem::path(home) / ".cache" / "mom" / "jit";
#endif
  return {};
}

static std::vector<char> pearlhash_read_cache(const std::filesystem::path& path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) return {};
  const std::streamsize size = stream.tellg();
  if (size <= 0 || size > 64 * 1024 * 1024) return {};
  std::vector<char> code(static_cast<size_t>(size));
  stream.seekg(0);
  if (!stream.read(code.data(), size)) return {};
  return code;
}

static void pearlhash_write_cache(const std::filesystem::path& path,
                              const std::vector<char>& code) noexcept {
  try {
    if (path.empty() || code.empty()) return;
    std::filesystem::create_directories(path.parent_path());
    const auto temporary = path.string() + ".tmp";
    {
      std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
      if (!stream.write(code.data(), static_cast<std::streamsize>(code.size()))) return;
    }
    std::error_code error;
    std::filesystem::rename(temporary, path, error);
    if (error) {
      std::filesystem::remove(path, error);
      error.clear();
      std::filesystem::rename(temporary, path, error);
      if (error) std::filesystem::remove(temporary, error);
    }
  } catch (...) {
    // A read-only home/cache directory is not a mining failure; the in-process module still works.
  }
}
#endif
