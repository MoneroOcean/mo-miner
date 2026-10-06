// Check the synthetic CPU geometry against the independent scalar Ethash implementation.
// This is not a consensus-size vector; physical GPU checks retain their recorded share gold.
#include "../../xmrig/3rdparty/libethash/ethash_internal.h"

#include <cstdio>
#include <cstring>

static void encode_hex(const ethash_h256_t& hash, char* const output) {
  constexpr char digits[] = "0123456789abcdef";
  for (unsigned i = 0; i < 32; ++i) {
    output[i * 2] = digits[hash.b[i] >> 4];
    output[i * 2 + 1] = digits[hash.b[i] & 15];
  }
  output[64] = '\0';
}

int main() {
  const ethash_h256_t seed{};
  ethash_h256_t header{};
  for (unsigned i = 0; i < 32; ++i)
    header.b[i] = static_cast<uint8_t>(i);
  ethash_light_t light = ethash_light_new_internal(16u * 1024u, &seed);
  if (!light)
    return 1;
  const auto result = ethash_light_compute_internal(light, 80u * 1024u, header, 1);
  ethash_light_delete(light);
  char hash[65], mix[65];
  encode_hex(result.result, hash);
  encode_hex(result.mix_hash, mix);
  if (!result.success ||
      std::strcmp(hash, "a57562c7a275ab2bae0b3a6afd5b6b425d9dc3b5f6e3d106adfc5401b687b424") ||
      std::strcmp(mix, "c854bad41b003055e79a241a6e071341021b972d0a0090064210b95bb60db98a")) {
    std::fprintf(stderr, "Etchash compact reference mismatch: %s %s\n", hash, mix);
    return 1;
  }
  std::puts("etchash-small-reference:passed");
  return 0;
}
