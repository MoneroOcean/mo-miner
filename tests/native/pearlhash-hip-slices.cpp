// CPU-only launch/retirement model for the final ten-argument fused HIP ABI.
// The driver supplies the actual final launch, wait, wave grouping, and allocation snippets.
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#define DEV
#ifndef PEARLHASH_ER
#define PEARLHASH_ER 12
#endif
#ifndef PEARLHASH_EC
#define PEARLHASH_EC 4
#endif
#ifndef PEARLHASH_K
#define PEARLHASH_K 2048
#endif
#include "matrix-offset.inc"
#include "hip-wave-groups.inc"

static int selected_tile_er(int k, int rank) {
  int tile_er = 12;
#include "hip-selector.inc"
  return tile_er;
}

static void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

struct Address {
  uintptr_t base;
  void* operator+(size_t offset) const { return reinterpret_cast<void*>(base + offset); }
};

struct Buffers {
  Address Ap{0x10000000};
  void* Bp = reinterpret_cast<void*>(0x20000000);
  void* cA = reinterpret_cast<void*>(0x40000000);
  void* target = reinterpret_cast<void*>(0x50000000);
  void* result = reinterpret_cast<void*>(0x60000000);
};

struct Slice {
  uintptr_t a;
  int rows;
  int n;
  int offset;
  int debug;
};

static std::vector<Slice> slices;
static const Buffers* active_buffers = nullptr;
static int records = 0;
static int pending = 0;
static int fail_launch = 0;
static int fail_record = 0; // 1=start event, 2=done event
static int drains = 0;

static int cache_rows_for(int k) {
  require(k > 0, "non-positive K reached cache sizing");
  return std::max(16, std::min(64, 131072 / k));
}

static int expected_chunk_rows(int m, int k, bool debug) {
  const int row_group = cache_rows_for(k) * PEARLHASH_ER * 16;
  const int debug_rows = std::min(row_group, std::max(128, (m / 2 / 128) * 128));
  return std::min(m, debug ? debug_rows : row_group);
}

using hipError_t = int;
constexpr int hipSuccess = 0;
constexpr int hipErrorNotReady = 2;
static int query_status = hipSuccess;
static int queries = 0;
static int activation_status = hipSuccess;

static int hipEventQuery(int event) {
  require(event == 4, "queried foreign event");
  return ++queries == 1 ? hipErrorNotReady : query_status;
}
static int hipEventElapsedTime(float* elapsed, int, int) {
  *elapsed = 0.0f;
  return hipSuccess;
}
static int hipStreamSynchronize(int stream) {
  require(stream == 7, "drained foreign stream");
  ++drains;
  return 1;
}
static int hipEventRecord(int event, int stream) {
  require(stream == 7 && event == (records == 0 ? 3 : 4),
          "completion event order changed");
  ++records;
  if ((event == 3 && fail_record == 1) || (event == 4 && fail_record == 2)) return 1;
  return hipSuccess;
}

static int hipModuleLaunchKernel(int function, unsigned blocks, unsigned y, unsigned z,
                                unsigned threads, unsigned ty, unsigned tz, unsigned shared,
                                int stream, void** args, void*) {
  require(active_buffers != nullptr, "launch had no owning buffer graph");
  require(y == 1 && z == 1 && threads == 256 && ty == 1 && tz == 1 && !shared && stream == 7,
          "fused launch geometry or owning stream changed");
  require(function == 11, "legacy matrix/hash entry point was launched");
  if (++pending == fail_launch) return 1;

  const int rows = *static_cast<int*>(args[6]);
  const int n = *static_cast<int*>(args[7]);
  const int row_offset = *static_cast<int*>(args[8]);
  const int debug = *static_cast<int*>(args[9]);
  require(*static_cast<void**>(args[1]) == active_buffers->Bp &&
              *static_cast<void**>(args[2]) == active_buffers->cA &&
              *static_cast<void**>(args[3]) == active_buffers->target &&
              *static_cast<void**>(args[4]) == active_buffers->result,
          "fused ABI changed the shared B/key/target/result ownership");
  require(*static_cast<uint32_t*>(args[5]) == 19 && (debug == 0 || debug == 1),
          "fused ABI changed seed or debug representation");
  require(row_offset >= 0 && rows > 0 && row_offset + rows <= 100000000,
          "invalid fused row slice");
  const uint64_t waves =
      uint64_t((rows + PEARLHASH_ER * 16 - 1) / (PEARLHASH_ER * 16)) *
      uint64_t((n + 63) / 64);
  require(blocks == (waves + 3) / 4, "fused wave block coverage changed");
  require(row_offset % 16 == 0, "fused row offsets lost 16-row granularity");
  const uintptr_t expected_a = active_buffers->Ap.base + size_t(row_offset) * PEARLHASH_K;
  require(reinterpret_cast<uintptr_t>(*static_cast<void**>(args[0])) == expected_a,
          "fused A slice offset changed");
  slices.push_back({expected_a, rows, n, row_offset, debug});
  return hipSuccess;
}

struct Search {
  unsigned wmma_threads = 256;
  int tile_er = PEARLHASH_ER;
  int tile_ec = 4;
  int native_stream = 7;
  int kernel = 11;
  int started = 3;
  int done = 4;
  int compiled_k = 0;
  int compiled_rank = 0;
  bool enabled = true;
  bool checked = true;
  int module = 9;
  unsigned stats_printed = 0;
  void activate() {
    if (activation_status != hipSuccess)
      throw std::runtime_error("hipSetDevice(PearlHash)");
  }
  void release() noexcept {}
  static void hip_check(int status, const char* operation) {
    if (status) throw std::runtime_error(operation);
  }
#include "hip-launch.inc"
#include "hip-wait.inc"
};

static int allocated_rows(int m, int n, bool hip_jit, bool cuda_jit) {
#include "transcript-allocation.inc"
  require(transcript_bytes == uint64_t(transcript_rows / 16) * (n / 16) * 64,
          "transcript allocation does not match fused word count");
  if (hip_jit || cuda_jit) require(transcript_bytes <= transcript_limit,
                                   "scratch cap exceeded");
  return transcript_rows;
}

static void check_wave_mapping(int m, int n) {
  const int groups_m = (m + PEARLHASH_ER * 16 - 1) / (PEARLHASH_ER * 16);
  const int groups_n = (n + 63) / 64;
  const uint64_t waves = uint64_t(groups_m) * uint64_t(groups_n);
  std::vector<bool> seen(static_cast<size_t>(waves));
  uint64_t proofs = 0;
  for (uint64_t wave = 0; wave < waves; ++wave) {
    int row = -1, column = -1;
    wave_groups(static_cast<uint32_t>(wave), m, n, &row, &column);
    require(row >= 0 && row < groups_m && column >= 0 && column < groups_n,
            "fused cache grouping left the row slice");
    const size_t tile = static_cast<size_t>(row) * static_cast<size_t>(groups_n) +
                        static_cast<size_t>(column);
    require(!seen[tile], "fused cache grouping duplicated an output tile");
    seen[tile] = true;
    const int remaining_rows = m - row * PEARLHASH_ER * 16;
    proofs += static_cast<uint64_t>(
        std::min(std::max(remaining_rows, 0), PEARLHASH_ER * 16) / 4);
  }
  require(proofs == uint64_t(m / 4) * uint64_t(n / 64),
          "fused cache grouping changed proof coverage");
  int first_row = -1, first_column = -1, last_row = -1, last_column = -1;
  wave_groups(0, m, n, &first_row, &first_column);
  wave_groups(static_cast<uint32_t>(waves - 1), m, n, &last_row, &last_column);
  require(first_row == 0 && first_column == 0 && last_row == groups_m - 1 &&
              last_column == groups_n - 1,
          "fused cache grouping changed first or tail coordinate");
}

static void check_wave_boundary(int m, int n) {
  const int groups_m = (m + PEARLHASH_ER * 16 - 1) / (PEARLHASH_ER * 16);
  const int groups_n = (n + 63) / 64;
  const int cache_rows = cache_rows_for(PEARLHASH_K);
  const uint64_t waves = uint64_t(groups_m) * uint64_t(groups_n);
  auto expect = [&](uint64_t wave, int expected_row, int expected_column) {
    require(wave < waves, "fused cache boundary exceeded wave count");
    int row = -1, column = -1;
    wave_groups(static_cast<uint32_t>(wave), m, n, &row, &column);
    require(row == expected_row && column == expected_column,
            "fused cache transition coordinate changed");
  };
  if (groups_m % cache_rows == 0 && groups_n % 8 == 0) {
    expect(uint64_t(cache_rows) * 8 - 1, cache_rows - 1, 7);
    if (groups_n >= 16)
      expect(uint64_t(cache_rows) * 8, 0, 8);
    if (groups_m >= cache_rows * 2)
      expect(uint64_t(groups_n) * cache_rows, cache_rows, 0);
  } else if (groups_m > 1) {
    expect(static_cast<uint64_t>(groups_n), 1, 0);
  }
  expect(waves - 1, groups_m - 1, groups_n - 1);
}

static void check_wave_groups() {
  require(cache_rows_for(1024) == 64 && cache_rows_for(2048) == 64 &&
              cache_rows_for(3072) == 42 && cache_rows_for(4096) == 32 &&
              cache_rows_for(8192) == 16 && cache_rows_for(16384) == 16,
          "cache row clamp changed");
  require(cache_rows_for(PEARLHASH_K) * PEARLHASH_ER * 16 == expected_chunk_rows(
              cache_rows_for(PEARLHASH_K) * PEARLHASH_ER * 16, PEARLHASH_K, false),
          "compile-time cache row profile changed");
  const int chunk = cache_rows_for(PEARLHASH_K) * PEARLHASH_ER * 16;
  for (const auto shape : {std::vector<int>{256, 128}, {640, 512}, {2048, 2048},
                           {2048, 524288}, {8192, 131072},
                           std::vector<int>{chunk, 512}, {chunk * 2, 512},
                           std::vector<int>{chunk + 64, 512}, {524288, 1024}})
    check_wave_mapping(shape[0], shape[1]);
  check_wave_boundary(chunk * 2, 1024);
  check_wave_boundary(chunk + 64, 1024);
  check_wave_boundary(524288, 524288);
}

static void check_host_selector() {
  require(selected_tile_er(2048, 128) == 13 &&
              selected_tile_er(4096, 256) == 13 &&
              selected_tile_er(4096, 128) == 12 &&
              selected_tile_er(8192, 128) == 12,
          "HIP host q16/q32/q64 tile selector changed");
}

static void check_wait() {
  unsetenv("MOM_PEARLHASH_STATS");
  activation_status = hipSuccess;
  for (const int status : {hipSuccess, 1}) {
    query_status = status;
    queries = drains = 0;
    bool caught = false;
    try {
      Search{}.wait(0);
    } catch (const std::runtime_error& error) {
      caught = true;
      require(std::string(error.what()) == "hipEventQuery(PearlHash)",
              "query error was replaced");
    }
    require(caught == (status != hipSuccess) && queries == 2,
            "event polling changed");
    require(drains == (caught ? 1 : 0),
            "query fault did not retire native work exactly once");
  }
  query_status = hipSuccess;
  queries = drains = 0;
  activation_status = 1;
  bool caught = false;
  try {
    Search{}.wait(0);
  } catch (const std::runtime_error& error) {
    caught = true;
    require(std::string(error.what()) == "hipSetDevice(PearlHash)",
            "activation error was replaced");
  }
  require(caught && queries == 0 && drains == 1,
          "activation fault did not retire native work on its own stream");
  activation_status = hipSuccess;
}

static void check_allocation() {
  require(amd_wmma_offset(0, 0, 2048) == 0,
          "current matrix helper changed its origin layout");
  require(amd_wmma_offset(17, 35, 2048) == 33291,
          "current matrix helper changed its packed byte offset");
  require(allocated_rows(524288, 524288, true, false) == 0,
          "HIP fused path retained an unnecessary global transcript");
  require(allocated_rows(131072, 524288, false, true) == 8192,
          "qualified CUDA scratch cap changed");
  require(allocated_rows(256, 128, true, false) == 0,
          "small fused path retained an unnecessary global transcript");
  require(allocated_rows(131072, 131072, false, false) == 131072,
          "ordinary layout was chunked");
  require(allocated_rows(128, 4194304, true, false) == 0,
          "inadmissible fused slice did not fall back");
}

static void check_launches() {
  for (const auto shape : {std::vector<int>{2048, 2048, 1024},
                           std::vector<int>{256, 128, 256},
                           std::vector<int>{1536, 64, 512},
                           std::vector<int>{640, 512, 512},
                           std::vector<int>{524288, 524288, 3072},
                           std::vector<int>{448, 192}}) {
    for (const bool debug : {false, true}) {
      for (const int launch_failure : {0, 1, 2, 3}) {
        slices.clear();
        records = pending = drains = 0;
        fail_launch = launch_failure;
        fail_record = 0;
        const int m = shape[0], n = shape[1];
        Buffers buffers;
        active_buffers = &buffers;
        bool caught = false;
        try {
          Search{}.launch(buffers, 19, m, n, PEARLHASH_K, 128, debug);
        } catch (const std::runtime_error&) {
          caught = true;
        }
        const int expected_rows = expected_chunk_rows(m, PEARLHASH_K, debug);
        const int launches = (m + expected_rows - 1) / expected_rows;
        require(caught == (launch_failure != 0 && launch_failure <= launches),
                "fused launch failure was swallowed or invented");
        require(records == (caught ? 1 : 2),
                "fused completion recorded after a launch fault");
        require(drains == (caught ? 1 : 0),
                "partial fused submissions escaped without retirement");
        int row = 0;
        for (const auto& slice : slices) {
          require(slice.a == buffers.Ap.base + size_t(row) * PEARLHASH_K &&
                      slice.n == n && slice.debug == (debug ? 1 : 0) &&
                      slice.offset == row,
                  "fused row offset or debug ABI changed");
          require(slice.rows == std::min(expected_rows, m - row),
                  "fused cache group/scratch/tail mismatch");
          row += slice.rows;
        }
        if (!caught)
          require(row == m && pending == launches, "fused slices omitted rows or launches");
      }
    }
  }

  for (const int record_failure : {1, 2}) {
    slices.clear();
    records = pending = drains = 0;
    fail_launch = 0;
    fail_record = record_failure;
    Buffers buffers;
    active_buffers = &buffers;
    bool caught = false;
    try {
      Search{}.launch(buffers, 19, 1024, 192, PEARLHASH_K, 128, false);
    } catch (const std::runtime_error&) {
      caught = true;
    }
    require(caught && records == record_failure && drains == 1,
            "start/done event fault did not retire exactly once");
    if (record_failure == 1)
      require(pending == 0 && slices.empty(), "start fault submitted fused work");
    else
      require(pending == static_cast<int>(slices.size()) && !slices.empty(),
              "done fault lost already-submitted fused work");
  }
  active_buffers = nullptr;
}

int main() try {
  check_host_selector();
  check_wave_groups();
  check_wait();
  check_allocation();
  check_launches();
  std::printf("CPU fused launch oracle passed: K=%d cache_rows=%d chunk_rows=%d, allocation, wait, "
              "ten-argument ABI, tails and faults\n",
              PEARLHASH_K, cache_rows_for(PEARLHASH_K),
              expected_chunk_rows(100000, PEARLHASH_K, false));
  return 0;
} catch (const std::exception& error) {
  std::fprintf(stderr, "%s\n", error.what());
  return 1;
}
