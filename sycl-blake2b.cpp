// Copyright GNU GPLv3 (c) 2026-2026 MoneroOcean <support@moneroocean.stream>

#include <cstddef>

#include "crypto/randomx/blake2/blake2.h"

// Windows builds sycl.dll separately, so provide the RandomX blake2b hooks used
// by c29.cpp inside that DLL instead of importing them from the Node addon.
void (*rx_blake2b_compress)(blake2b_state* S, const uint8_t* block) = rx_blake2b_compress_integer;
int (*rx_blake2b)(void* out, size_t outlen, const void* in, size_t inlen) = rx_blake2b_default;
