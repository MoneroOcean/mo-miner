// Copyright GNU GPLv3 (c) 2023-2026 MoneroOcean <support@moneroocean.stream>

#pragma once

constexpr unsigned HASH_LEN = 32; // length of a PoW hash / target in bytes

// Out-of-band GPU proof/test buffers. The validation path returns 256 expanded Equihash rows.
constexpr unsigned EQUIHASH_TEST_ROWS  = 256;
constexpr unsigned EQUIHASH_ROW_LEN    = 20;   // (K+1)*COLLISION_BYTE_LENGTH expanded collision fields
constexpr unsigned SMALL_BLOB_SOL_LEN  = EQUIHASH_TEST_ROWS * EQUIHASH_ROW_LEN;
constexpr unsigned EQUIHASH_SOL_BUFFER_LEN = 1 + 64 * 400; // count + worst planned 192,7 proofs
