class FishDagKernel;
static sycl::event submit_fishhash_dag_gen(
  sycl::queue& q, MomKernelBundle& kb, const uint64_t* __restrict d_cache, uint64_t* __restrict d_dag,
  uint32_t start, uint32_t count
) {
  constexpr unsigned WG = 64;
  const size_t global = (static_cast<size_t>(count) + WG - 1) / WG * WG;
  return q.submit([&](sycl::handler& h) {
    mom_use_bundle(h, kb);
    h.parallel_for<FishDagKernel>(sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) {
      const uint32_t i = (uint32_t)it.get_global_id(0);
      if (i >= count) return;
      const uint32_t idx = start + i;
      uint64_t item[16];
      fish_dataset_item(d_cache, idx, item);
      for (unsigned k = 0; k < 16; ++k) d_dag[(size_t)idx * 16 + k] = item[k];
    });
  });
}

// ---- HOST: build the 72 MB light cache (keccak512 chain + 3 rounds), == ethash light cache ----
static void keccak512_host(uint64_t out[8], const uint8_t* in, unsigned in_len) { keccak512_dev(out, in, in_len); }
static void build_light_cache_host(std::vector<uint64_t>& cache /* N*8 */) {
  cache.resize((size_t)LIGHT_CACHE_NUM_ITEMS * 8);
  keccak512_host(&cache[0], FISH_SEED, 32);
  for (int i = 1; i < LIGHT_CACHE_NUM_ITEMS; ++i) {
    uint8_t buf[64]; for (unsigned k = 0; k < 8; ++k) for (unsigned b = 0; b < 8; ++b) buf[k*8+b] = (uint8_t)(cache[(size_t)(i-1)*8+k] >> (8*b));
    keccak512_host(&cache[(size_t)i*8], buf, 64);
  }
  for (int q = 0; q < LIGHT_CACHE_ROUNDS; ++q) {
    for (int i = 0; i < LIGHT_CACHE_NUM_ITEMS; ++i) {
      const uint32_t t = (uint32_t)cache[(size_t)i*8];                 // word32s[0] = low 32 of word64s[0]
      const uint32_t v = t % (uint32_t)LIGHT_CACHE_NUM_ITEMS;
      const uint32_t w = (uint32_t)(LIGHT_CACHE_NUM_ITEMS + (i - 1)) % (uint32_t)LIGHT_CACHE_NUM_ITEMS;
      uint8_t x[64];
      for (unsigned k = 0; k < 8; ++k) { const uint64_t xr = cache[(size_t)v*8+k] ^ cache[(size_t)w*8+k];
        for (unsigned b = 0; b < 8; ++b) x[k*8+b] = (uint8_t)(xr >> (8*b)); }
      keccak512_host(&cache[(size_t)i*8], x, 64);
    }
  }
}
