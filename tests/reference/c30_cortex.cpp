// CPU-only Cortex c30 oracle.
// Vector: official block 0xee4093 (hash
// 8a533ecead09281b20c05caa2d575994cfe3ba0f518167601ecbd170af98dea1).
// Consensus sources:
// https://github.com/CortexFoundation/CortexTheseus/blob/master/consensus/cuckoo/consensus.go
// https://github.com/CortexFoundation/solution/blob/master/src/cuckoo.cc

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>

#include "../../xmrig/base/crypto/keccak.h"
#include "../../xmrig/crypto/randomx/blake2/blake2.h"

void (*rx_blake2b_compress)(blake2b_state *state, const std::uint8_t *block) =
    rx_blake2b_compress_integer;
int (*rx_blake2b)(void *out, std::size_t outlen, const void *in, std::size_t inlen) =
    rx_blake2b_default;

namespace {

using u32 = std::uint32_t;
using u64 = std::uint64_t;

constexpr u32 kProofSize = 42;
constexpr u32 kEdgeBits = 30;
constexpr u32 kEdgeMask = (1u << kEdgeBits) - 1;
constexpr u32 kEdgeBlockMask = 63;
constexpr u64 kNonce = 0x1e000000d90820d4ULL;

constexpr std::array<std::uint8_t, 32> kSealHash = {
    0x8b, 0xbb, 0x88, 0x97, 0xa7, 0x96, 0x76, 0x34,
    0xe1, 0x5b, 0xae, 0x66, 0x2e, 0xe2, 0x3e, 0x16,
    0xe8, 0xc8, 0x56, 0x69, 0xf3, 0xca, 0x0a, 0x9e,
    0x65, 0x84, 0xf8, 0xf4, 0xaa, 0x41, 0xf2, 0x20
};

constexpr std::array<u32, kProofSize> kSolution = {
    20556923, 49328084, 66386344, 128738901, 132651276, 149372831,
    181150355, 187948573, 212976704, 230374362, 238671778, 262353442,
    322221503, 373413741, 375602693, 477881229, 482331279, 513034595,
    547999576, 554241182, 559490951, 565679447, 614213668, 760389481,
    821628732, 858636160, 863402811, 866207996, 875109483, 881247921,
    884674238, 896486016, 909490614, 914020222, 915733019, 926780695,
    959049446, 979837634, 989118169, 1000739073, 1028175605, 1045923045
};

constexpr std::array<std::uint8_t, 32> kGraphKey = {
    0x00, 0x14, 0xb9, 0x44, 0x7c, 0x5d, 0x27, 0x3a,
    0x93, 0xd8, 0x03, 0x90, 0x95, 0xe1, 0xbe, 0x26,
    0xee, 0x91, 0x68, 0x85, 0x78, 0x6a, 0xeb, 0x54,
    0x9a, 0x7f, 0x9a, 0x94, 0xaf, 0x12, 0x13, 0x65
};

constexpr std::array<std::uint8_t, 32> kSolutionHash = {
    0x20, 0x28, 0xe2, 0x2a, 0xa1, 0x2a, 0xb3, 0xb0,
    0x7e, 0xc6, 0xf0, 0xc5, 0x74, 0xb2, 0x82, 0x98,
    0x56, 0x5d, 0x47, 0xd8, 0xb0, 0x26, 0x25, 0xfb,
    0xfb, 0x85, 0xb3, 0x7d, 0xa0, 0xe9, 0xb4, 0xe4
};

struct SipKeys {
    u64 k0;
    u64 k1;
    u64 k2;
    u64 k3;
};

u64 load_le64(const std::uint8_t *p)
{
    u64 value = 0;
    for (unsigned i = 0; i < 8; ++i)
        value |= static_cast<u64>(p[i]) << (i * 8);
    return value;
}

u64 rotate_left(u64 value, unsigned bits)
{
    return (value << bits) | (value >> (64 - bits));
}

void sip_round(u64 &v0, u64 &v1, u64 &v2, u64 &v3)
{
    v0 += v1;
    v2 += v3;
    v1 = rotate_left(v1, 13);
    v3 = rotate_left(v3, 16);
    v1 ^= v0;
    v3 ^= v2;
    v0 = rotate_left(v0, 32);
    v2 += v1;
    v0 += v3;
    v1 = rotate_left(v1, 17);
    v3 = rotate_left(v3, 21);
    v1 ^= v2;
    v3 ^= v0;
    v2 = rotate_left(v2, 32);
}

u64 sipblock(const SipKeys &keys, u32 edge)
{
    u64 v0 = keys.k0;
    u64 v1 = keys.k1;
    u64 v2 = keys.k2;
    u64 v3 = keys.k3;
    std::array<u64, 64> values{};
    const u32 first = edge & ~kEdgeBlockMask;

    for (u32 i = 0; i < values.size(); ++i) {
        const u64 nonce = first + i;
        v3 ^= nonce;
        for (unsigned round = 0; round < 4; ++round)
            sip_round(v0, v1, v2, v3);
        v0 ^= nonce;
        v2 ^= 0xff;
        for (unsigned round = 0; round < 8; ++round)
            sip_round(v0, v1, v2, v3);
        values[i] = (v0 ^ v1) ^ (v2 ^ v3);
    }

    const u64 last = values.back();
    for (u32 i = 0; i < kEdgeBlockMask; ++i)
        values[i] ^= last;
    return values[edge & kEdgeBlockMask];
}

bool verify_graph(const SipKeys &keys)
{
    std::array<u32, 2 * kProofSize> endpoints{};
    u32 xor_u = 0;
    u32 xor_v = 0;

    for (u32 i = 0; i < kProofSize; ++i) {
        if (kSolution[i] > kEdgeMask || (i && kSolution[i] <= kSolution[i - 1]))
            return false;
        const u64 edge = sipblock(keys, kSolution[i]);
        endpoints[2 * i] = static_cast<u32>(edge) & kEdgeMask;
        endpoints[2 * i + 1] = static_cast<u32>(edge >> 32) & kEdgeMask;
        xor_u ^= endpoints[2 * i];
        xor_v ^= endpoints[2 * i + 1];
    }
    if (xor_u || xor_v)
        return false;

    u32 length = 0;
    u32 current = 0;
    do {
        u32 match = current;
        for (u32 i = current; (i = (i + 2) % (2 * kProofSize)) != current;) {
            if (endpoints[i] == endpoints[current]) {
                if (match != current)
                    return false;
                match = i;
            }
        }
        if (match == current)
            return false;
        current = match ^ 1;
        ++length;
    } while (current != 0);
    return length == kProofSize;
}

bool equal_bytes(const std::array<std::uint8_t, 32> &a,
                 const std::array<std::uint8_t, 32> &b)
{
    return std::equal(a.begin(), a.end(), b.begin());
}

} // namespace

int main()
{
    std::array<std::uint8_t, 40> header_nonce{};
    std::copy(kSealHash.begin(), kSealHash.end(), header_nonce.begin());
    for (unsigned i = 0; i < sizeof(kNonce); ++i)
        header_nonce[32 + i] = static_cast<std::uint8_t>(kNonce >> (i * 8));

    std::array<std::uint8_t, 32> graph_key{};
    if (rx_blake2b_default(graph_key.data(), graph_key.size(),
                           header_nonce.data(), header_nonce.size()) != 0 ||
        !equal_bytes(graph_key, kGraphKey)) {
        std::fprintf(stderr, "c30 graph key mismatch\n");
        return 1;
    }

    const SipKeys keys = {
        load_le64(graph_key.data()), load_le64(graph_key.data() + 8),
        load_le64(graph_key.data() + 16), load_le64(graph_key.data() + 24)
    };
    if (!verify_graph(keys)) {
        std::fprintf(stderr, "c30 graph proof mismatch\n");
        return 1;
    }

    std::array<std::uint8_t, kProofSize * sizeof(u32)> solution_bytes{};
    for (u32 i = 0; i < kProofSize; ++i) {
        for (unsigned byte = 0; byte < sizeof(u32); ++byte)
            solution_bytes[4 * i + byte] =
                static_cast<std::uint8_t>(kSolution[i] >> (8 * (3 - byte)));
    }
    std::array<std::uint8_t, 32> solution_hash{};
    xmrig::keccak(solution_bytes.data(), solution_bytes.size(), solution_hash.data(),
                  solution_hash.size());
    if (!equal_bytes(solution_hash, kSolutionHash)) {
        std::fprintf(stderr, "c30 solution hash mismatch\n");
        return 1;
    }

    std::puts("Cortex c30 vector: PASS (official block 0xee4093)");
    return 0;
}
