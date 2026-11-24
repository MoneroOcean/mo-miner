// Copyright GNU GPLv3 (c) 2026 MoneroOcean <support@moneroocean.stream>

#pragma once

// Proof-level helpers shared by the Equihash-family adapters (ZHash, ZelHash, Equihash 192,7 and
// BeamHash III): leaf canonicalisation and distinctness, the host SHA-256 used by share targets, and
// the count-prefixed solution buffer handed to the native worker.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "../native/job-boundary.h"

namespace mom_equihash::pow {

// Reorder recovered leaves into the canonical IndicesBefore form: at every tree level the sibling
// subtree with the smaller first leaf comes first. Leaves must be distinct; host and device safe.
template <unsigned K>
inline void canonicalize_leaves(std::uint32_t* const leaves) {
  for (unsigned level = 0; level < K; ++level) {
    const unsigned half = 1u << level;
    for (unsigned begin = 0; begin < (1u << K); begin += 2 * half) {
      if (leaves[begin] < leaves[begin + half]) continue;
      for (unsigned i = 0; i < half; ++i) {
        const std::uint32_t swap = leaves[begin + i];
        leaves[begin + i] = leaves[begin + half + i];
        leaves[begin + half + i] = swap;
      }
    }
  }
}

// Equihash proofs require every leaf index to be unique; host and device safe.
inline bool leaves_distinct(const std::uint32_t* const leaves, const unsigned count) {
  for (unsigned i = 0; i < count; ++i)
    for (unsigned j = i + 1; j < count; ++j)
      if (leaves[i] == leaves[j]) return false;
  return true;
}

inline std::uint32_t rotr(const std::uint32_t value, const unsigned bits) {
  return (value >> bits) | (value << (32 - bits));
}

// Hash a bounded host proof preimage; this is target verification, not the Equihash device kernel.
inline void sha256(const std::uint8_t* message, const std::size_t length,
                   std::uint8_t output[32]) {
  static constexpr std::uint32_t k[64] = {
      0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
      0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
      0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
      0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
      0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
      0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
      0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
      0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
  std::uint32_t state[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
                            0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
  const std::size_t padded = ((length + 8) / 64 + 1) * 64;
  std::vector<std::uint8_t> bytes(padded, 0);
  std::memcpy(bytes.data(), message, length);
  bytes[length] = 0x80;
  const std::uint64_t bit_length = static_cast<std::uint64_t>(length) * 8;
  for (unsigned i = 0; i < 8; ++i) {
    bytes[padded - 1 - i] = static_cast<std::uint8_t>(bit_length >> (8 * i));
  }
  for (std::size_t offset = 0; offset < padded; offset += 64) {
    std::uint32_t w[64];
    for (unsigned i = 0; i < 16; ++i)
      w[i] = static_cast<std::uint32_t>(bytes[offset + 4 * i]) << 24 |
             static_cast<std::uint32_t>(bytes[offset + 4 * i + 1]) << 16 |
             static_cast<std::uint32_t>(bytes[offset + 4 * i + 2]) << 8 |
             bytes[offset + 4 * i + 3];
    for (unsigned i = 16; i < 64; ++i) {
      const auto s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      const auto s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    std::uint32_t a = state[0];
    std::uint32_t b = state[1];
    std::uint32_t c = state[2];
    std::uint32_t d = state[3];
    std::uint32_t e = state[4];
    std::uint32_t f = state[5];
    std::uint32_t g = state[6];
    std::uint32_t h = state[7];
    for (unsigned i = 0; i < 64; ++i) {
      const auto s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
      const auto t1 = h + s1 + ((e & f) ^ (~e & g)) + k[i] + w[i];
      const auto s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
      const auto t2 = s0 + ((a & b) ^ (a & c) ^ (b & c));
      h = g;
      g = f;
      f = e;
      e = d + t1;
      d = c;
      c = b;
      b = a;
      a = t1 + t2;
    }
    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
  }
  for (unsigned i = 0; i < 8; ++i) {
    output[4 * i] = static_cast<std::uint8_t>(state[i] >> 24);
    output[4 * i + 1] = static_cast<std::uint8_t>(state[i] >> 16);
    output[4 * i + 2] = static_cast<std::uint8_t>(state[i] >> 8);
    output[4 * i + 3] = static_cast<std::uint8_t>(state[i]);
  }
}

// Hash header plus CompactSize-prefixed proof, then compare the reversed SHA256d digest to target.
// Callers supply canonical proofs shorter than 65536 bytes and a 32-byte big-endian target.
inline bool meets_target(const std::uint8_t* header, const std::size_t header_length,
                         const std::uint8_t* solution, const std::size_t solution_length,
                         const std::uint8_t* target) {
  if (solution_length > UINT16_MAX)
    return false;
  std::uint8_t prefix[3];
  const unsigned prefix_size =
      mom::job_boundary::write_compact_size16(static_cast<std::uint16_t>(solution_length), prefix);
  std::vector<std::uint8_t> preimage(header, header + header_length);
  preimage.insert(preimage.end(), prefix, prefix + prefix_size);
  preimage.insert(preimage.end(), solution, solution + solution_length);
  std::uint8_t first[32];
  std::uint8_t hash[32];
  sha256(preimage.data(), preimage.size(), first);
  sha256(first, sizeof(first), hash);
  for (unsigned i = 0; i < 32; ++i)
    if (hash[31 - i] != target[i]) return hash[31 - i] < target[i];
  return true;
}

// All solvers use [count:u8][count * proof bytes]. A null byte pointer skips a proof.
template <typename Solutions, typename Encode>
inline unsigned pack_solutions(const Solutions& solutions, std::size_t proof_length,
                               std::size_t buffer_length, bool first_only, std::uint8_t* out,
                               Encode encode) {
  if (!buffer_length)
    return 0;
  const std::size_t capacity = proof_length
      ? std::min<std::size_t>(first_only ? 1 : 255, (buffer_length - 1) / proof_length) : 0;
  unsigned count = 0;
  for (const auto& solution : solutions) {
    if (count == capacity)
      break;
    if (const auto* encoded = encode(solution)) {
      std::memcpy(out + 1 + count * proof_length, encoded, proof_length);
      ++count;
    }
  }
  out[0] = static_cast<std::uint8_t>(count);
  return count;
}

// Diagnostics use the bytes accepted by the worker, including the buffer capacity limit.
inline void log_solutions(const std::uint8_t* packed, std::size_t proof_length) {
  const auto* byte = packed + 1;
  for (unsigned proof = 0; proof < packed[0]; ++proof) {
    std::fputs("  ", stderr);
    for (std::size_t i = 0; i < proof_length; ++i)
      std::fprintf(stderr, "%02x", static_cast<unsigned>(*byte++));
    std::fputc('\n', stderr);
  }
  std::fflush(stderr);
}

// A non-null target keeps only qualifying canonical proofs. Tests request the first proof.
template <typename Solutions>
inline unsigned write_solutions(const Solutions& solutions, const std::size_t proof_length,
                                const std::size_t buffer_length, const bool first_only,
                                const std::uint8_t* const header, const std::size_t header_length,
                                const std::uint8_t* const target, std::uint8_t* const out) {
  return pack_solutions(solutions, proof_length, buffer_length, first_only, out,
                        [&](const auto& solution) -> const std::uint8_t* {
    const auto* proof = solution.encoded.data();
    return !target || meets_target(header, header_length, proof, proof_length, target)
        ? proof : nullptr;
  });
}

}  // namespace mom_equihash::pow
