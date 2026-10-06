#pragma once

#if !defined(__SYCL_DEVICE_ONLY__)

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace mom::jit_cache {

inline uint64_t hash(const std::string_view value) {
  uint64_t result = 1469598103934665603ULL;
  for (const unsigned char byte : value) {
    result ^= byte;
    result *= 1099511628211ULL;
  }
  return result;
}

inline std::filesystem::path directory() {
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

inline std::vector<char> read(const std::filesystem::path& path,
                              const size_t max_bytes = 64 * 1024 * 1024) {
  try {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream)
      return {};
    const std::streamsize size = stream.tellg();
    if (size <= 0 || static_cast<uintmax_t>(size) > max_bytes ||
        static_cast<uintmax_t>(size) > std::numeric_limits<size_t>::max())
      return {};
    std::vector<char> bytes(static_cast<size_t>(size));
    stream.seekg(0);
    if (!stream.read(bytes.data(), size))
      return {};
    return bytes;
  } catch (...) {
    return {};
  }
}

inline void write(const std::filesystem::path& path, const std::vector<char>& bytes) noexcept {
  std::filesystem::path temporary;
  try {
    if (path.empty() || bytes.empty())
      return;

    static std::atomic<uint64_t> counter{0};
    std::random_device random;
    const uint64_t sequence = counter.fetch_add(1, std::memory_order_relaxed);
    const uint64_t entropy = (static_cast<uint64_t>(random()) << 32) ^ random();
    temporary = path;
    temporary += ".tmp-" + std::to_string(sequence) + "-" + std::to_string(entropy);

    const std::filesystem::path parent = path.parent_path();
    if (!parent.empty())
      std::filesystem::create_directories(parent);
    {
      std::ofstream stream;
      stream.exceptions(std::ios::failbit | std::ios::badbit);
      stream.open(temporary, std::ios::binary | std::ios::trunc);
      stream.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
      stream.close();
    }

    std::filesystem::rename(temporary, path);
  } catch (...) {
    if (!temporary.empty()) {
      std::error_code ignored;
      std::filesystem::remove(temporary, ignored);
    }
    // A cache failure is not a mining failure; the in-process module remains usable.
  }
}

} // namespace mom::jit_cache

#endif // !__SYCL_DEVICE_ONLY__
