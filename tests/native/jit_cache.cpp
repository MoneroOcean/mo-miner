#include "../../sycl/jit-cache.h"

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

namespace {

struct TempDirectory {
  std::filesystem::path path;

  ~TempDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path, error);
  }
};

TempDirectory make_temp_directory() {
  const std::filesystem::path base = std::filesystem::temp_directory_path();
  const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
  for (unsigned attempt = 0; attempt < 1000; ++attempt) {
    const std::filesystem::path path =
        base / ("mom-jit-cache-test-" + std::to_string(stamp) + "-" + std::to_string(attempt));
    std::error_code error;
    if (std::filesystem::create_directory(path, error))
      return {path};
    assert(!error);
  }
  assert(false);
  return {};
}

void test_hash() {
  assert(mom::jit_cache::hash("") == 1469598103934665603ULL);
  assert(mom::jit_cache::hash("abc") == 0xe16801510db89efdULL);
  assert(mom::jit_cache::hash("abc") == mom::jit_cache::hash(std::string("abc")));
}

unsigned temporary_entries(const std::filesystem::path& parent,
                           const std::filesystem::path& target) {
  const std::string prefix = target.filename().string() + ".tmp-";
  unsigned entries = 0;
  for (const auto& entry : std::filesystem::directory_iterator(parent)) {
    if (entry.path().filename().string().compare(0, prefix.size(), prefix) == 0)
      ++entries;
  }
  return entries;
}

void test_read_write(const std::filesystem::path& root) {
  const std::vector<char> bytes{'j', 'i', 't', '\0', 'c', 'a', 'c', 'h', 'e'};
  const std::filesystem::path explicit_path = root / "nested" / "cache.bin";
  mom::jit_cache::write(explicit_path, bytes);
  assert(std::filesystem::exists(explicit_path));
  assert(mom::jit_cache::read(explicit_path) == bytes);

  assert(mom::jit_cache::read(root / "missing.bin").empty());
  const std::filesystem::path empty_path = root / "empty.bin";
  {
    std::ofstream stream(empty_path, std::ios::binary);
    assert(stream);
  }
  assert(mom::jit_cache::read(empty_path).empty());
  mom::jit_cache::write(empty_path, bytes);
  assert(mom::jit_cache::read(empty_path) == bytes);

  const std::vector<char> limited_bytes{'1', '2', '3', '4', '5'};
  const std::filesystem::path limited_path = root / "limited.bin";
  mom::jit_cache::write(limited_path, limited_bytes);
  assert(mom::jit_cache::read(limited_path, 0).empty());
  assert(mom::jit_cache::read(limited_path, 4).empty());
  assert(mom::jit_cache::read(limited_path, limited_bytes.size()) == limited_bytes);

  const std::filesystem::path replacement_path = root / "replacement.bin";
  const std::vector<char> old_bytes{'o', 'l', 'd'};
  {
    std::ofstream stream(replacement_path, std::ios::binary | std::ios::trunc);
    assert(stream);
    stream.write(old_bytes.data(), static_cast<std::streamsize>(old_bytes.size()));
  }
  assert(mom::jit_cache::read(replacement_path) == old_bytes);
  mom::jit_cache::write(replacement_path, bytes);
  assert(mom::jit_cache::read(replacement_path) == bytes);

  const std::filesystem::path empty_write_path = root / "empty-write.bin";
  mom::jit_cache::write(empty_write_path, {});
  assert(!std::filesystem::exists(empty_write_path));
}

void test_failed_writes_preserve_targets(const std::filesystem::path& root) {
  const std::vector<char> bytes{'x'};

  const std::filesystem::path empty_directory = root / "empty-directory";
  assert(std::filesystem::create_directory(empty_directory));
  mom::jit_cache::write(empty_directory, bytes);
  assert(std::filesystem::is_directory(empty_directory));
  assert(std::filesystem::is_empty(empty_directory));
  assert(temporary_entries(root, empty_directory) == 0);

  const std::filesystem::path nonempty_directory = root / "nonempty-directory";
  assert(std::filesystem::create_directory(nonempty_directory));
  const std::filesystem::path sentinel = nonempty_directory / "sentinel";
  const std::vector<char> sentinel_bytes{'s', 'e', 'n', 't', 'i', 'n', 'e', 'l'};
  {
    std::ofstream stream(sentinel, std::ios::binary | std::ios::trunc);
    assert(stream);
    stream.write(sentinel_bytes.data(), static_cast<std::streamsize>(sentinel_bytes.size()));
  }
  mom::jit_cache::write(nonempty_directory, bytes);
  assert(std::filesystem::is_directory(nonempty_directory));
  assert(mom::jit_cache::read(sentinel) == sentinel_bytes);
  assert(temporary_entries(root, nonempty_directory) == 0);

  const std::filesystem::path parent_file = root / "parent-file";
  const std::vector<char> parent_bytes{'p', 'a', 'r', 'e', 'n', 't'};
  {
    std::ofstream stream(parent_file, std::ios::binary | std::ios::trunc);
    assert(stream);
    stream.write(parent_bytes.data(), static_cast<std::streamsize>(parent_bytes.size()));
  }
  const std::filesystem::path invalid_target = parent_file / "cache.bin";
  mom::jit_cache::write(invalid_target, bytes);
  assert(std::filesystem::is_regular_file(parent_file));
  assert(mom::jit_cache::read(parent_file) == parent_bytes);
  assert(!std::filesystem::exists(invalid_target));
  assert(temporary_entries(root, parent_file) == 0);
}

void test_concurrent_writes(const std::filesystem::path& root) {
  const std::filesystem::path directory = root / "concurrent";
  const std::filesystem::path path = directory / "cache.bin";
  const std::vector<char> bytes(4096, static_cast<char>(0xa5));
  constexpr unsigned writers = 16;
  std::atomic<bool> start{false};
  std::vector<std::thread> threads;
  threads.reserve(writers);
  for (unsigned i = 0; i < writers; ++i) {
    threads.emplace_back([&] {
      while (!start.load(std::memory_order_acquire))
        std::this_thread::yield();
      mom::jit_cache::write(path, bytes);
    });
  }
  start.store(true, std::memory_order_release);
  for (auto& thread : threads)
    thread.join();

  assert(mom::jit_cache::read(path) == bytes);
  unsigned entries = 0;
  for (const auto& entry : std::filesystem::directory_iterator(directory)) {
    ++entries;
    assert(entry.path() == path);
  }
  assert(entries == 1);
}

} // namespace

int main() {
  const TempDirectory temporary = make_temp_directory();
  test_hash();
  test_read_write(temporary.path);
  test_failed_writes_preserve_targets(temporary.path);
  test_concurrent_writes(temporary.path);
  return 0;
}
