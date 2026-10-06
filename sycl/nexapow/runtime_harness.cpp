// Standalone physical-runtime correctness and throughput harness.

#include <sycl/sycl.hpp>

#include <charconv>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace mom_nexapow {
#include "device.inc"
} // namespace mom_nexapow

namespace {

constexpr uint32_t kDefaultCount = 8;
constexpr uint32_t kMaximumCount = 1u << 20;
constexpr uint64_t kRecordedNonce = 0x1182dc5800000000ULL;
constexpr uint32_t kCompletedMarker = 0x4e455841u;
static_assert(kMaximumCount <= std::numeric_limits<uint64_t>::max() - kRecordedNonce - 1u);

constexpr uint8_t kRecordedHeader[32] = {
    0x0a, 0x4a, 0xc4, 0x9b, 0x2d, 0x02, 0xe3, 0xc8, 0xd1, 0x2c, 0x70, 0x93, 0x25, 0x5b, 0xa7, 0xc4,
    0x96, 0x24, 0xf9, 0xc3, 0x74, 0xd9, 0xf1, 0xc2, 0xf8, 0xe3, 0x7c, 0x58, 0x70, 0x5e, 0x74, 0xb0,
};
constexpr uint8_t kRecordedExtranonce[8] = {0x10, 0, 0, 0, 0, 0, 0, 0};
constexpr uint8_t kRecordedHash[32] = {
    0x00, 0x00, 0x00, 0x42, 0xcb, 0xc2, 0x40, 0x37, 0x52, 0x42, 0xe1, 0x46, 0x41, 0x48, 0x8a, 0x0e,
    0x2d, 0xca, 0x7b, 0x54, 0x45, 0x8a, 0x2c, 0xea, 0x23, 0xdc, 0x1d, 0x2c, 0x17, 0x8b, 0xb1, 0x88,
};

struct Work {
  uint8_t header[32];
  uint8_t extranonce[8];
};

struct RecordedResult {
  uint64_t nonce;
  uint32_t ok;
  uint32_t marker;
  uint8_t hash[32];
};

struct BenchmarkResult {
  uint64_t nonce;
  uint64_t digest;
  uint32_t ordinal;
  uint32_t ok;
};

class RecordedKernel;
class BenchmarkKernel;

template <typename T> class DeviceBuffer {
public:
  DeviceBuffer(sycl::queue& queue, const size_t count)
      : queue_(queue), data_(sycl::malloc_device<T>(count, queue)) {
    if (!data_)
      throw std::runtime_error("SYCL device allocation failed");
  }
  ~DeviceBuffer() {
    try {
      sycl::free(data_, queue_);
    } catch (...) {
    }
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  T* get() const {
    return data_;
  }

private:
  sycl::queue& queue_;
  T* data_;
};

struct Config {
  uint32_t count = kDefaultCount;
  std::string device = "0";
  bool list_devices = false;
};

uint32_t parse_count(const std::string& text) {
  uint32_t value = 0;
  const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
  if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() || !value ||
      value > kMaximumCount)
    throw std::runtime_error("count must be in [1, " + std::to_string(kMaximumCount) + "]");
  return value;
}

Config parse_args(const int argc, char** argv) {
  Config config;
  if (const char* value = std::getenv("NEXAPOW_COUNT"))
    config.count = parse_count(value);
  if (const char* value = std::getenv("NEXAPOW_DEVICE"))
    config.device = value;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--count" && i + 1 < argc)
      config.count = parse_count(argv[++i]);
    else if (arg.compare(0, 8, "--count=") == 0)
      config.count = parse_count(arg.substr(8));
    else if (arg == "--device" && i + 1 < argc)
      config.device = argv[++i];
    else if (arg.compare(0, 9, "--device=") == 0)
      config.device = arg.substr(9);
    else if (arg == "--list-devices")
      config.list_devices = true;
    else if (arg == "--help") {
      std::cout << "usage: nexapow-runtime [--device INDEX|SUBSTRING] [--count 1.." << kMaximumCount
                << "] [--list-devices]\n"
                   "env: NEXAPOW_DEVICE, NEXAPOW_COUNT\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown or incomplete argument: " + arg);
    }
  }
  if (config.device.empty())
    throw std::runtime_error("device selector must not be empty");
  return config;
}

std::string lowercase(std::string value) {
  for (char& character : value)
    character = static_cast<char>(std::tolower(static_cast<unsigned char>(character)));
  return value;
}

size_t select_device(const std::vector<sycl::device>& devices, const std::string& selector) {
  size_t index = 0;
  const auto parsed = std::from_chars(selector.data(), selector.data() + selector.size(), index);
  if (parsed.ec == std::errc{} && parsed.ptr == selector.data() + selector.size()) {
    if (index >= devices.size())
      throw std::runtime_error("device index is out of range");
    return index;
  }
  const std::string needle = lowercase(selector);
  size_t match = devices.size();
  for (size_t i = 0; i < devices.size(); ++i) {
    const std::string name = lowercase(devices[i].get_info<sycl::info::device::name>());
    if (name.find(needle) == std::string::npos)
      continue;
    if (match != devices.size())
      throw std::runtime_error("device selector matches more than one GPU");
    match = i;
  }
  if (match == devices.size())
    throw std::runtime_error("device selector matches no GPU");
  return match;
}

void print_devices(const std::vector<sycl::device>& devices) {
  for (size_t i = 0; i < devices.size(); ++i) {
    const auto& device = devices[i];
    std::cout << "device_index=" << i
              << " name=" << std::quoted(device.get_info<sycl::info::device::name>())
              << " vendor=" << std::quoted(device.get_info<sycl::info::device::vendor>())
              << " driver=" << std::quoted(device.get_info<sycl::info::device::driver_version>())
              << " platform="
              << std::quoted(device.get_platform().get_info<sycl::info::platform::name>()) << '\n';
  }
}

uint64_t digest_hash(const uint8_t hash[32]) {
  uint64_t digest = 1469598103934665603ULL;
  for (unsigned i = 0; i < 32; ++i) {
    digest ^= hash[i];
    digest *= 1099511628211ULL;
  }
  return digest;
}

double event_milliseconds(const sycl::event& event) {
  const uint64_t start = event.get_profiling_info<sycl::info::event_profiling::command_start>();
  const uint64_t end = event.get_profiling_info<sycl::info::event_profiling::command_end>();
  if (end <= start)
    throw std::runtime_error("invalid SYCL event profiling timestamps");
  return static_cast<double>(end - start) / 1.0e6;
}

template <typename Clock>
double wall_milliseconds(const std::chrono::time_point<Clock>& start,
                         const std::chrono::time_point<Clock>& end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

std::string hash_hex(const uint8_t hash[32]) {
  static constexpr char digits[] = "0123456789abcdef";
  std::string value(64, '0');
  for (unsigned i = 0; i < 32; ++i) {
    value[2 * i] = digits[hash[i] >> 4];
    value[2 * i + 1] = digits[hash[i] & 15u];
  }
  return value;
}

} // namespace

int main(int argc, char** argv) {
  try {
    const Config config = parse_args(argc, argv);
    const std::vector<sycl::device> devices =
        sycl::device::get_devices(sycl::info::device_type::gpu);
    if (devices.empty())
      throw std::runtime_error("no visible SYCL GPU devices");
    if (config.list_devices) {
      print_devices(devices);
      return 0;
    }

    const size_t device_index = select_device(devices, config.device);
    const sycl::device device = devices[device_index];
    if (!device.has(sycl::aspect::usm_device_allocations))
      throw std::runtime_error("selected GPU lacks device USM support");
    sycl::queue queue(device, sycl::property_list{
                                  sycl::property::queue::in_order{},
                                  sycl::property::queue::enable_profiling{},
                              });

    Work work{};
    std::memcpy(work.header, kRecordedHeader, sizeof(work.header));
    std::memcpy(work.extranonce, kRecordedExtranonce, sizeof(work.extranonce));

    std::cout << "device_index=" << device_index
              << " name=" << std::quoted(device.get_info<sycl::info::device::name>())
              << " vendor=" << std::quoted(device.get_info<sycl::info::device::vendor>())
              << " driver=" << std::quoted(device.get_info<sycl::info::device::driver_version>())
              << " platform="
              << std::quoted(device.get_platform().get_info<sycl::info::platform::name>()) << '\n';
    std::cout << "config count=" << config.count << " default_count=" << kDefaultCount
              << " maximum_count=" << kMaximumCount << '\n';

    DeviceBuffer<RecordedResult> recorded_device(queue, 1);
    queue.memset(recorded_device.get(), 0, sizeof(RecordedResult)).wait_and_throw();
    const auto recorded_wall_start = std::chrono::steady_clock::now();
    sycl::event recorded_event = queue.submit([&](sycl::handler& handler) {
      RecordedResult* output = recorded_device.get();
      handler.single_task<RecordedKernel>([=]() {
        uint8_t hash[32]{};
        const bool ok =
            mom_nexapow::np_hash_one(work.header, work.extranonce, kRecordedNonce, hash);
        output->nonce = kRecordedNonce;
        output->ok = ok ? 1u : 0u;
        for (unsigned i = 0; i < 32; ++i)
          output->hash[i] = hash[i];
        output->marker = kCompletedMarker;
      });
    });
    recorded_event.wait_and_throw();
    RecordedResult recorded{};
    queue.memcpy(&recorded, recorded_device.get(), sizeof(recorded)).wait_and_throw();
    const auto recorded_wall_end = std::chrono::steady_clock::now();
    if (recorded.marker != kCompletedMarker || recorded.nonce != kRecordedNonce ||
        recorded.ok != 1 || std::memcmp(recorded.hash, kRecordedHash, sizeof(kRecordedHash)) != 0)
      throw std::runtime_error("recorded device vector mismatch");
    std::cout << std::fixed << std::setprecision(3) << "recorded status=PASS nonce=0x" << std::hex
              << recorded.nonce << std::dec << " hash=" << hash_hex(recorded.hash)
              << " event_ms=" << event_milliseconds(recorded_event)
              << " wall_ms=" << wall_milliseconds(recorded_wall_start, recorded_wall_end) << '\n';

    const uint64_t first_nonce = kRecordedNonce + 1;
    std::vector<BenchmarkResult> results(config.count);
    DeviceBuffer<BenchmarkResult> result_device(queue, config.count);
    queue.memset(result_device.get(), 0xff, results.size() * sizeof(results[0])).wait_and_throw();
    const auto benchmark_wall_start = std::chrono::steady_clock::now();
    sycl::event benchmark_event = queue.submit([&](sycl::handler& handler) {
      BenchmarkResult* output = result_device.get();
      const uint32_t count = config.count;
      handler.parallel_for<BenchmarkKernel>(sycl::range<1>(count), [=](sycl::id<1> id) {
        const uint32_t ordinal = static_cast<uint32_t>(id[0]);
        const uint64_t nonce = first_nonce + ordinal;
        uint8_t hash[32]{};
        const bool ok = mom_nexapow::np_hash_one(work.header, work.extranonce, nonce, hash);
        output[ordinal] = {nonce, ok ? digest_hash(hash) : 0, ordinal, ok ? 1u : 0u};
      });
    });
    benchmark_event.wait_and_throw();
    queue.memcpy(results.data(), result_device.get(), results.size() * sizeof(results[0]))
        .wait_and_throw();
    const auto benchmark_wall_end = std::chrono::steady_clock::now();

    uint32_t processed = 0, success = 0;
    uint64_t checksum = 1469598103934665603ULL;
    for (uint32_t i = 0; i < config.count; ++i) {
      const BenchmarkResult& result = results[i];
      if (result.ordinal != i || result.nonce != first_nonce + i || result.ok > 1)
        throw std::runtime_error("benchmark count accounting mismatch at ordinal " +
                                 std::to_string(i));
      ++processed;
      success += result.ok;
      checksum = (checksum ^ result.nonce) * 1099511628211ULL;
      checksum = (checksum ^ result.digest) * 1099511628211ULL;
      checksum = (checksum ^ result.ok) * 1099511628211ULL;
    }
    if (processed != config.count)
      throw std::runtime_error("benchmark processed count mismatch");
    if (success != config.count)
      throw std::runtime_error("one or more benchmark nonces failed to hash");
    const double benchmark_event_ms = event_milliseconds(benchmark_event);
    const double rate = static_cast<double>(config.count) * 1000.0 / benchmark_event_ms;
    std::cout << "benchmark status=PASS count=" << config.count << " processed=" << processed
              << " success=" << success << " failed=" << (config.count - success)
              << " first_nonce=0x" << std::hex << first_nonce << " last_nonce=0x"
              << (first_nonce + config.count - 1u) << " checksum=0x" << checksum << std::dec
              << " event_ms=" << benchmark_event_ms
              << " wall_ms=" << wall_milliseconds(benchmark_wall_start, benchmark_wall_end)
              << " rate_hps=" << rate << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "status=FAIL error=" << std::quoted(error.what()) << '\n';
    return 1;
  }
}
