// Copyright GNU GPLv3 (c) 2026 MoneroOcean <support@moneroocean.stream>

#define MOM_NEXAPOW_HOST_TEST
#include "../../sycl/nexapow/nexapow.cpp"

// Exercise recorded and width-specific host consensus vectors without a SYCL runtime.
int main() {
  uint8_t input[48]{};
  const uint8_t header[32] = {
      0x0a, 0x4a, 0xc4, 0x9b, 0x2d, 0x02, 0xe3, 0xc8, 0xd1, 0x2c, 0x70,
      0x93, 0x25, 0x5b, 0xa7, 0xc4, 0x96, 0x24, 0xf9, 0xc3, 0x74, 0xd9,
      0xf1, 0xc2, 0xf8, 0xe3, 0x7c, 0x58, 0x70, 0x5e, 0x74, 0xb0,
  };
  for (unsigned i = 0; i < 32; ++i)
    input[i] = header[i];
  input[32] = 0x10;
  const uint64_t nonce = 0x1182dc5800000000ULL;
  for (unsigned i = 0; i < 8; ++i)
    input[40 + i] = static_cast<uint8_t>(nonce >> (8u * (7u - i)));
  const uint8_t header2[32] = {
      0x0a, 0x4a, 0xc4, 0x9b, 0x2d, 0x02, 0xe3, 0xc8, 0xd1, 0x2c, 0x70,
      0x93, 0x25, 0x5b, 0xa7, 0xc4, 0x96, 0x24, 0xf9, 0xc3, 0x74, 0xd9,
      0xf1, 0xc2, 0xf8, 0xe3, 0x7c, 0x58, 0x70, 0x5e, 0x74, 0xb0,
  };
  const uint8_t extranonce2[8] = {0x10, 0, 0, 0, 0, 0, 0, 0};
  const uint64_t nonce2 = 0xb787915d00000000ULL;
  const uint8_t expected2[32] = {
      0x00, 0x00, 0x00, 0x5f, 0x0b, 0x59, 0xe1, 0x10, 0x86, 0x35, 0x66,
      0xe7, 0x7d, 0x85, 0xe1, 0xb4, 0xfc, 0x71, 0x37, 0x54, 0xe5, 0xc7,
      0x5f, 0x8f, 0xc3, 0xdf, 0x44, 0x13, 0x38, 0x66, 0xa6, 0x69,
  };
  uint8_t hash2[32];
  if (!mom_nexapow::np_hash_one(header2, extranonce2, nonce2, hash2) ||
      !mom_nexapow::np_equal(hash2, expected2, sizeof(hash2)))
    return 1;
  uint8_t target[32];
  for (uint8_t& byte : target)
    byte = 0xff;
  uint8_t output[32]{};
  uint64_t found = 0;
  if (mom_nexapow::nexapow(0, 0, input, sizeof(input), output, nullptr, &found, target, nullptr, 1,
                           true, false, "") != 1 ||
      found != nonce)
    return 1;

  const uint8_t short_expected[32] = {
      0xb0, 0x34, 0xdd, 0x02, 0xa3, 0xba, 0x15, 0xd1, 0x6b, 0x1f, 0x03,
      0x25, 0xe4, 0x4d, 0xe3, 0xe3, 0x74, 0x48, 0x91, 0x82, 0xb2, 0x30,
      0x5c, 0xa1, 0x37, 0x40, 0x77, 0x21, 0x84, 0xc3, 0xbc, 0xd9,
  };
  uint8_t short_input[40]{};
  uint8_t short_output[32]{};
  uint64_t short_nonce = 0;
  if (mom_nexapow::nexapow(0, 0, short_input, sizeof(short_input), short_output, nullptr,
                           &short_nonce, target, nullptr, 1, true, false, "") != 1 ||
      short_nonce != 0 ||
      !mom_nexapow::np_equal(short_output, short_expected, sizeof(short_output)))
    return 1;

  uint8_t prefixed_input[44]{};
  const uint8_t prefix[4] = {0x10, 0x20, 0x30, 0x40};
  std::memcpy(prefixed_input, mom_nexapow::kRecordedHeader,
              sizeof(mom_nexapow::kRecordedHeader));
  std::memcpy(prefixed_input + 32, prefix, sizeof(prefix));
  uint64_t prefixed_nonce = 0x0102030405060708ULL;
  uint8_t prefixed_output[32]{};
  const uint8_t prefixed_expected_hash[32] = {
      0xa9, 0xbd, 0x7f, 0xfd, 0xfc, 0xc1, 0x81, 0x9e, 0xa1, 0xe4, 0x06, 0x87,
      0x45, 0xb4, 0x3e, 0xe2, 0x33, 0xa2, 0xcf, 0xe0, 0x4f, 0x42, 0xe3, 0x36,
      0xe3, 0xd9, 0x3c, 0x24, 0xf3, 0xaf, 0x9d, 0xee,
  };
  uint8_t prefixed_serialized[49]{};
  mom_nexapow::np_nonce_bytes(mom_nexapow::kRecordedHeader, prefix, prefixed_nonce,
                              prefixed_serialized, sizeof(prefix));
  bool serialized_ok = prefixed_serialized[32] == 0x0c;
  for (unsigned i = 0; i < 32; ++i)
    serialized_ok = serialized_ok &&
        prefixed_serialized[i] == mom_nexapow::kRecordedHeader[31u - i];
  for (unsigned i = 0; i < sizeof(prefix); ++i)
    serialized_ok = serialized_ok && prefixed_serialized[33u + i] == prefix[i];
  for (unsigned i = 0; i < 8; ++i)
    serialized_ok = serialized_ok && prefixed_serialized[37u + i] ==
        static_cast<uint8_t>(prefixed_nonce >> (8u * (7u - i)));
  if (mom_nexapow::nexapow(0, 0, prefixed_input, sizeof(prefixed_input), prefixed_output,
                           nullptr, &prefixed_nonce, target, nullptr, 1, true, false, "") != 1 ||
      prefixed_nonce != 0x0102030405060708ULL || !serialized_ok ||
      !mom_nexapow::np_equal(prefixed_output, prefixed_expected_hash, sizeof(prefixed_output)))
    return 1;
  return 0;
}
