#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "../../sycl/hoohash/host_math.h"

static int hex_digit(char digit) {
  if (digit >= '0' && digit <= '9')
    return digit - '0';
  if (digit >= 'a' && digit <= 'f')
    return digit - 'a' + 10;
  return -1;
}

static int decode_hex(const char* text, uint8_t* output, size_t output_size) {
  if (strlen(text) != output_size * 2)
    return 0;
  for (size_t i = 0; i < output_size; ++i) {
    const int high = hex_digit(text[2 * i]);
    const int low = hex_digit(text[2 * i + 1]);
    if (high < 0 || low < 0)
      return 0;
    output[i] = (uint8_t)(high << 4 | low);
  }
  return 1;
}

int main(void) {
  char blob_hex[161], expected_hex[65];
  uint8_t blob[80], expected[32], actual[32];
  if (scanf("%160s%64s", blob_hex, expected_hex) != 2 ||
      !decode_hex(blob_hex, blob, sizeof(blob)) ||
      !decode_hex(expected_hex, expected, sizeof(expected)))
    return 2;

  uint64_t nonce = 0;
  for (unsigned i = 0; i < 8; ++i)
    nonce |= (uint64_t)blob[72 + i] << (8 * i);
  if (mom_hoohash_math_error())
    return 3;
  mom_hoohash_canonical_hash(blob, nonce, actual);
  return memcmp(actual, expected, sizeof(actual)) ? 1 : 0;
}
