// Copyright GNU GPLv3 (c) 2023-2025 MoneroOcean <support@moneroocean.stream>

#pragma once

constexpr unsigned HASH_LEN = 32; // length of a PoW hash / target in bytes

// Out-of-band solution buffer for small-blob GPU pow that returns a solution larger than 32 bytes.
// Equihash 125,4 needs the 52-byte compressed solution, and its M1 gen-kernel validation path dumps
// the first EQUIHASH_TEST_ROWS entries' 20-byte expanded rows (256*20 = 5120 B) back for comparison.
constexpr unsigned EQUIHASH_TEST_ROWS  = 256;
constexpr unsigned EQUIHASH_ROW_LEN    = 20;   // (K+1)*COLLISION_BYTE_LENGTH expanded collision fields
constexpr unsigned SMALL_BLOB_SOL_LEN  = EQUIHASH_TEST_ROWS * EQUIHASH_ROW_LEN; // >= 52 and the M1 dump

