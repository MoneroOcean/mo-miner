// Compare the DG2 search and packed operands against the portable SYCL search.
// Build/run instructions: scripts/test-pearlhash-alchemist.md
#define PEARLHASH_STANDALONE
#define PEARLHASH_ESIMD
#define PEARLHASH_TEST
#include "../sycl/pearlhash/pearlhash.cpp"

static void free_buffers(sycl::queue& q, const mom_pearlhash::Buffers& b) {
  for (void* ptr : {(void*)b.EAL, (void*)b.EBR, (void*)b.EBRt, (void*)b.Ap, (void*)b.Bp,
                   (void*)b.EARp1, (void*)b.EARp2, (void*)b.EBLq1, (void*)b.EBLq2,
                   (void*)b.cA, (void*)b.cB, (void*)b.key, (void*)b.target,
                   (void*)b.CVA, (void*)b.CVB, (void*)b.transcript, (void*)b.result})
    if (ptr) sycl::free(ptr, q);
}

static void benchmark(sycl::queue& q, int samples, int m, int n) {
  using namespace mom_pearlhash;
  constexpr int k=4096, rank=256;
  constexpr uint32_t seed=0x12345678;
  Buffers b=alloc_buffers(q,m,n,k,rank);
  uint8_t header[76],key[32],target[32]{};
  for (int i=0;i<76;++i) header[i]=static_cast<uint8_t>(i*37+11);
  derive_key(header,k,rank,key);
  q.memcpy(b.key,key,32); q.memcpy(b.target,target,32); q.wait_and_throw();
  // Zero target forces the same full search workload, without share/proof work.
  // Prepare once for search-only timings; vary the seed in end-to-end attempts.
  *b.result={};
  attempt(q,b,seed,m,n,k,rank,"auto"); q.wait_and_throw();
  std::printf("BENCH_CONFIG rows=2 columns=1 dpasw=paired block=64x64 m=%d n=%d k=%d rank=%d samples=%d\n",
              m,n,k,rank,samples);
  for (int phase=0;phase<3;++phase) {
    const char* phase_name=phase==0?"search":phase==1?"pipeline":"prepare";
    std::vector<double> wall,device;
    for (int i=-5;i<samples;++i) {
      *b.result={};
      const auto start=std::chrono::steady_clock::now();
      sycl::event event;
      if (phase==0) event=search_esimd_alchemist(q,b,seed,m,n,k,rank,false);
      else if (phase==1) attempt(q,b,seed+static_cast<uint32_t>(i+5),m,n,k,rank,"auto");
      else {
        const uint32_t next_seed=seed+static_cast<uint32_t>(i+5);
        k_roots(q,b,next_seed,m,n,k);
        k_noise(q,b,m,n,k,rank);
        compute_ab(q,b,next_seed,m,n,k,rank,false,true);
      }
      q.wait_and_throw();
      const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
      if (b.result->found) throw std::string("Unexpected winner for zero benchmark target");
      if (i<0) continue;
      wall.push_back(ms);
      double device_ms=0;
      if (phase==0) {
        const auto begin=event.get_profiling_info<sycl::info::event_profiling::command_start>();
        const auto end=event.get_profiling_info<sycl::info::event_profiling::command_end>();
        device_ms=static_cast<double>(end-begin)/1e6;
        device.push_back(device_ms);
      }
      std::printf("SAMPLE,%s,%d,%.6f,%.6f\n",phase_name,i,ms,device_ms);
    }
    std::sort(wall.begin(),wall.end()); std::sort(device.begin(),device.end());
    const double median=(wall[(samples-1)/2]+wall[samples/2])/2;
    const double kernel=device.empty()?0:(device[(samples-1)/2]+device[samples/2])/2;
    std::printf("BENCH_RESULT phase=%s median_ms=%.6f p10_ms=%.6f p90_ms=%.6f kernel_ms=%.6f TMAC_s=%.6f\n",
                phase_name,median,wall[samples/10],wall[(samples*9)/10],kernel,
                phase==2?0:static_cast<double>(m)*n*k/(median*1e9));
  }
  free_buffers(q,b);
}

int main(int argc,char** argv) {
  using namespace mom_pearlhash;
  try {
    const bool bench=argc>=2 && std::string(argv[1])=="--benchmark";
    const int samples=argc>=3?std::atoi(argv[2]):40;
    const int bench_m=argc>=4?std::atoi(argv[3]):16384;
    const int bench_n=argc>=5?std::atoi(argv[4]):bench_m;
    if (bench_m<2048 || bench_m>65536 || bench_n<2048 || bench_n>65536 ||
        (bench_m & (bench_m-1)) || (bench_n & (bench_n-1)))
      throw std::string("Benchmark dimensions must be powers of two from 2048 to 65536");
    if (samples<10 || samples>200) throw std::string("Samples must be between 10 and 200");
    sycl::queue q{sycl::gpu_selector_v, {sycl::property::queue::in_order{},sycl::property::queue::enable_profiling{}}};
    std::string name = q.get_device().get_info<sycl::info::device::name>();
    std::printf("Testing %s\n", name.c_str());
    for (char& c : name) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (name.find("a770") == std::string::npos && name.find("a750") == std::string::npos &&
        name.find("a580") == std::string::npos && name.find("dg2") == std::string::npos)
      throw std::string("Select an Alchemist GPU with ONEAPI_DEVICE_SELECTOR");
    // Intel OpenCL normally selects the portable implementation. Exercise ESIMD
    // explicitly here, including the same dispatch/layout decision used by mining.
#ifdef _WIN32
    _putenv_s("MOM_PEARLHASH_ESIMD", "1");
#else
    setenv("MOM_PEARLHASH_ESIMD", "1", 1);
#endif
    struct Shape { int m, n, k, rank; };
    const Shape shapes[] = {
      {32, 64, 1024, 64}, {64, 32, 4096, 256},
      {16 * pearlhash_alchemist_rows * 64,
       16 * pearlhash_alchemist_columns * 64, 64, 32},
      // Common shape across every tuning variant, with production K/rank.
      {2048,2048,4096,256},
      // Valid partial unroll group on the generic-rank fallback and fewer ranks.
      // Portable reference requires K divisible by rank and K/rank <= 16.
      {32,32,320,64}, {32,32,1280,256},
      // Incomplete two-tile row group must select portable packing and search.
      {16,32,256,64}, {48,48,256,64},
      // Odd column grid must fall back to DPAS; multiple cache blocks test pairing.
      {32,16 * pearlhash_alchemist_columns * 3,256,64}, {4096,4096,64,32}
    };
    for (const auto& shape : shapes) {
      const auto [m, n, k, rank] = shape;
      Buffers b = alloc_buffers(q, m, n, k, rank);
      uint8_t header[76], key[32], target[32];
      for (int i = 0; i < 76; ++i) header[i] = static_cast<uint8_t>(i * 37 + 11);
      derive_key(header, k, rank, key);
      q.memcpy(b.key, key, 32);
      for (uint32_t seed : {0U, 0x12345678U}) {
        // Exercise both no-win and all-win paths, while checksumming EVERY tile.
        std::memset(target, seed == 0 ? 0 : 255, 32);
        q.memcpy(b.target, target, 32).wait_and_throw();
        *b.result = {};
        attempt(q, b, seed, m, n, k, rank, "auto", nullptr, nullptr, nullptr, nullptr, true);
        q.wait_and_throw();
        const Result native = *b.result;
        *b.result = {};
        attempt(q, b, seed, m, n, k, rank, "sycl", nullptr, nullptr, nullptr, nullptr, true);
        q.wait_and_throw();
        if (native.chk != b.result->chk || native.found != b.result->found)
          throw std::string("Alchemist/portable checksum or winner mismatch");
        if (native.found && (native.seed != seed || native.row >= static_cast<uint32_t>(m) ||
            native.col >= static_cast<uint32_t>(n) || native.row % 16 || native.col % 16))
          throw std::string("Invalid winning tile coordinates");
        std::printf("PASS m=%d n=%d k=%d rank=%d seed=%u checksum=%08x\n",
                    m, n, k, rank, seed, native.chk);
      }
      free_buffers(q,b);
    }
    if (bench) benchmark(q,samples,bench_m,bench_n);
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
  } catch (const std::string& error) {
    std::fprintf(stderr, "%s\n", error.c_str());
  }
  return 1;
}
