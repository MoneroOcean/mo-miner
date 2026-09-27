// icpx -fsycl -std=c++20 tests/pearlhash_seed.cpp -o pearlhash_seed
// Run with ONEAPI_DEVICE_SELECTOR=level_zero:gpu for the device checks.
#include <sycl/sycl.hpp>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include "../sycl/pearlhash/blake3.inc"

static bool equals_hex(const uint8_t* data, const char* hex) {
  char actual[65];
  for (int i=0;i<32;++i) std::sprintf(actual+2*i,"%02x",data[i]);
  return std::strcmp(actual,hex)==0;
}
int main() {
  // Official pinned legacy/V3 vectors from Pearl commit 3fe226761a139a9652b8f28a6464a4bbc25986c8.
  const char* expected_a[] = {
    "483b07b6f73105030b9482255f37723f3fed69ae916724ee8291848b8c28794b",
    "301784168005ec833ab0aa60006f7fe7faaa95307d8c1fc6819b2ffdd717eccf"};
  const char* expected_b[] = {
    "add6f7ea5feebf89c8a77e2ebfa0d82442e7dbb0046dbd48971861d12fcb0177",
    "60ed9b73c5a9599b200b6cd563e7f0d5d9a67d2402d85fd4ef966c580080d0e5"};
  sycl::queue q;
  auto* out=sycl::malloc_shared<uint8_t>(64,q);
  for (unsigned version=1;version<=3;++version) {
    const int vector=version==3?1:0;
    uint8_t key[32],ra[32],rb[32],a[32],b[32];
    std::memset(key,0x11,32); std::memset(ra,0xaa,32); std::memset(rb,0xbb,32);
    pearlhash_b3::noise_seeds(key,ra,rb,192,320,version,a,b);
    if (!equals_hex(a,expected_a[vector]) || !equals_hex(b,expected_b[vector])) return 1;
    q.single_task([=]() {
      uint8_t dk[32],da[32],db[32];
      for(int i=0;i<32;++i) {dk[i]=0x11; da[i]=0xaa; db[i]=0xbb;}
      pearlhash_b3::noise_seeds(dk,da,db,192,320,version,out,out+32);
    }).wait_and_throw();
    if (!equals_hex(out,expected_a[vector]) || !equals_hex(out+32,expected_b[vector])) return 2;
    if (!equals_hex(ra,"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")) return 3;
    std::printf("PASS certificate %u host and device consensus seeds\n",version);
  }
  sycl::free(out,q);
}

