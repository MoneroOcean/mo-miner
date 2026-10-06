// Bitcoin Gold's authoritative validator_144_5_btg_salt regression:
// github.com/BTCGPU/BTCGPU/blob/master/src/test/equihash_tests.cpp
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "../../sycl/equihash_pow.hpp"

namespace {

constexpr unsigned N = 144;
constexpr unsigned K = 5;
constexpr unsigned COLLISION_BITS = N / (K + 1);
constexpr unsigned COLLISION_WORDS = 6;
constexpr unsigned HASH_LENGTH = 54;
constexpr unsigned SEGMENT_LENGTH = HASH_LENGTH / 3;
constexpr unsigned INDEX_COUNT = 1u << K;
constexpr unsigned INDEX_BITS = COLLISION_BITS + 1;
constexpr unsigned SOLUTION_LENGTH = INDEX_COUNT * INDEX_BITS / 8;

static_assert(N % (K + 1) == 0, "invalid Equihash parameters");
static_assert(COLLISION_BITS == 24, "unexpected collision width");
static_assert(SEGMENT_LENGTH == 18, "unexpected hash segment width");
static_assert(SOLUTION_LENGTH == 100, "unexpected solution width");

constexpr std::uint64_t BLAKE2B_IV[8] = {
    0x6a09e667f3bcc908ull, 0xbb67ae8584caa73bull,
    0x3c6ef372fe94f82bull, 0xa54ff53a5f1d36f1ull,
    0x510e527fade682d1ull, 0x9b05688c2b3e6c1full,
    0x1f83d9abfb41bd6bull, 0x5be0cd19137e2179ull,
};

constexpr std::uint8_t BLAKE2B_SIGMA[12][16] = {
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
    { 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
    { 11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
    { 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
    { 9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13 },
    { 2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9 },
    { 12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11 },
    { 13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10 },
    { 6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5 },
    { 10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0 },
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
    { 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
};

constexpr int hex_digit(const char c) {
    return c >= '0' && c <= '9' ? c - '0' :
           c >= 'a' && c <= 'f' ? c - 'a' + 10 :
           c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1;
}

template <std::size_t Bytes, std::size_t Characters>
constexpr std::array<std::uint8_t, Bytes> from_hex(const char (&text)[Characters]) {
    static_assert(Characters == Bytes * 2 + 1, "wrong hexadecimal length");
    std::array<std::uint8_t, Bytes> bytes{};
    for (std::size_t i = 0; i < Bytes; ++i) {
        bytes[i] = static_cast<std::uint8_t>((hex_digit(text[2 * i]) << 4) |
                                             hex_digit(text[2 * i + 1]));
    }
    return bytes;
}

constexpr std::uint64_t load64_le(const std::uint8_t* p) {
    return static_cast<std::uint64_t>(p[0]) |
           (static_cast<std::uint64_t>(p[1]) << 8) |
           (static_cast<std::uint64_t>(p[2]) << 16) |
           (static_cast<std::uint64_t>(p[3]) << 24) |
           (static_cast<std::uint64_t>(p[4]) << 32) |
           (static_cast<std::uint64_t>(p[5]) << 40) |
           (static_cast<std::uint64_t>(p[6]) << 48) |
           (static_cast<std::uint64_t>(p[7]) << 56);
}

void store64_le(std::uint8_t* p, const std::uint64_t x) {
    for (unsigned i = 0; i < 8; ++i) {
        p[i] = static_cast<std::uint8_t>(x >> (8 * i));
    }
}

std::uint64_t rotate_right(const std::uint64_t x, const unsigned amount) {
    return (x >> amount) | (x << (64 - amount));
}

void mix(std::uint64_t& a, std::uint64_t& b, std::uint64_t& c, std::uint64_t& d,
         const std::uint64_t x, const std::uint64_t y) {
    a = a + b + x;
    d = rotate_right(d ^ a, 32);
    c += d;
    b = rotate_right(b ^ c, 24);
    a = a + b + y;
    d = rotate_right(d ^ a, 16);
    c += d;
    b = rotate_right(b ^ c, 63);
}

void compress(std::uint64_t h[8], const std::uint8_t block[128], const std::uint64_t count,
              const bool last) {
    std::uint64_t m[16];
    std::uint64_t v[16];
    for (unsigned i = 0; i < 16; ++i) {
        m[i] = load64_le(block + 8 * i);
    }
    for (unsigned i = 0; i < 8; ++i) {
        v[i] = h[i];
        v[i + 8] = BLAKE2B_IV[i];
    }
    v[12] ^= count;
    if (last) {
        v[14] ^= 0xffffffffffffffffull;
    }
    for (unsigned round = 0; round < 12; ++round) {
        const std::uint8_t* s = BLAKE2B_SIGMA[round];
        mix(v[0], v[4], v[8], v[12], m[s[0]], m[s[1]]);
        mix(v[1], v[5], v[9], v[13], m[s[2]], m[s[3]]);
        mix(v[2], v[6], v[10], v[14], m[s[4]], m[s[5]]);
        mix(v[3], v[7], v[11], v[15], m[s[6]], m[s[7]]);
        mix(v[0], v[5], v[10], v[15], m[s[8]], m[s[9]]);
        mix(v[1], v[6], v[11], v[12], m[s[10]], m[s[11]]);
        mix(v[2], v[7], v[8], v[13], m[s[12]], m[s[13]]);
        mix(v[3], v[4], v[9], v[14], m[s[14]], m[s[15]]);
    }
    for (unsigned i = 0; i < 8; ++i) {
        h[i] ^= v[i] ^ v[i + 8];
    }
}

std::array<std::uint8_t, HASH_LENGTH> hash_index(
    const std::array<std::uint8_t, 140>& header, const std::uint32_t index) {
    std::uint8_t parameter[64] = {};
    parameter[0] = HASH_LENGTH;
    parameter[2] = 1;
    parameter[3] = 1;
    const std::uint8_t personal[16] = {
        'B', 'g', 'o', 'l', 'd', 'P', 'o', 'W',
        static_cast<std::uint8_t>(N), 0, 0, 0,
        static_cast<std::uint8_t>(K), 0, 0, 0,
    };
    for (unsigned i = 0; i < 16; ++i) {
        parameter[48 + i] = personal[i];
    }

    std::uint64_t h[8];
    for (unsigned i = 0; i < 8; ++i) {
        h[i] = BLAKE2B_IV[i] ^ load64_le(parameter + 8 * i);
    }

    std::uint8_t block[128] = {};
    for (unsigned i = 0; i < 128; ++i) {
        block[i] = header[i];
    }
    compress(h, block, 128, false);

    for (unsigned i = 0; i < 12; ++i) {
        block[i] = header[128 + i];
    }
    block[12] = static_cast<std::uint8_t>(index & 0xffu);
    block[13] = static_cast<std::uint8_t>((index >> 8) & 0xffu);
    block[14] = static_cast<std::uint8_t>((index >> 16) & 0xffu);
    block[15] = static_cast<std::uint8_t>((index >> 24) & 0xffu);
    for (unsigned i = 16; i < 128; ++i) {
        block[i] = 0;
    }
    compress(h, block, 144, true);

    std::array<std::uint8_t, HASH_LENGTH> digest{};
    std::uint8_t full_digest[64];
    for (unsigned i = 0; i < 8; ++i) {
        store64_le(full_digest + 8 * i, h[i]);
    }
    for (unsigned i = 0; i < HASH_LENGTH; ++i) {
        digest[i] = full_digest[i];
    }
    return digest;
}

std::array<std::uint32_t, INDEX_COUNT> decode_indices(
    const std::array<std::uint8_t, SOLUTION_LENGTH>& solution) {
    std::array<std::uint32_t, INDEX_COUNT> indices{};
    for (unsigned i = 0; i < INDEX_COUNT; ++i) {
        std::uint32_t value = 0;
        for (unsigned bit = 0; bit < INDEX_BITS; ++bit) {
            const unsigned position = i * INDEX_BITS + bit;
            value = (value << 1) | ((solution[position / 8] >> (7 - position % 8)) & 1u);
        }
        indices[i] = value;
    }
    return indices;
}

struct Node {
    std::array<std::uint32_t, COLLISION_WORDS> words{};
    std::uint32_t first_index = 0;
};

bool check_subtree(const std::array<Node, INDEX_COUNT>& leaves, const unsigned begin,
                  const unsigned level, Node& result) {
    if (level == 0) {
        result = leaves[begin];
        return true;
    }

    const unsigned half = 1u << (level - 1);
    Node left;
    Node right;
    if (!check_subtree(leaves, begin, level - 1, left) ||
        !check_subtree(leaves, begin + half, level - 1, right)) {
        return false;
    }
    if (left.first_index >= right.first_index || left.words[0] != right.words[0]) {
        return false;
    }

    result.first_index = left.first_index;
    const unsigned output_words = COLLISION_WORDS - level;
    for (unsigned i = 0; i < output_words; ++i) {
        result.words[i] = left.words[i + 1] ^ right.words[i + 1];
    }
    return true;
}

bool verify(const std::array<std::uint8_t, 140>& header,
            const std::array<std::uint8_t, SOLUTION_LENGTH>& solution) {
    const auto indices = decode_indices(solution);
    for (unsigned i = 0; i < INDEX_COUNT; ++i) {
        for (unsigned j = 0; j < i; ++j) {
            if (indices[i] == indices[j]) {
                return false;
            }
        }
    }

    std::array<Node, INDEX_COUNT> leaves;
    for (unsigned i = 0; i < INDEX_COUNT; ++i) {
        const std::uint32_t index = indices[i];
        const auto digest = hash_index(header, index / 3);
        const unsigned offset = (index % 3) * SEGMENT_LENGTH;
        leaves[i].first_index = index;
        for (unsigned word = 0; word < COLLISION_WORDS; ++word) {
            const unsigned byte = offset + word * 3;
            leaves[i].words[word] = static_cast<std::uint32_t>(digest[byte]) |
                                    (static_cast<std::uint32_t>(digest[byte + 1]) << 8) |
                                    (static_cast<std::uint32_t>(digest[byte + 2]) << 16);
        }
    }

    Node root;
    return check_subtree(leaves, 0, K, root) && root.words[0] == 0;
}

}  // namespace

int main() {
    constexpr auto header = from_hex<140>(
        "0400000008e9694cc2120ec1b5733cc12687b609058eec4f7046a521ad1d1e3049b400003e7420ed6f40659de0305ef9b7ec037f4380ed9848bc1c015691c90aa16ff3930000000000000000000000000000000000000000000000000000000000000000c9310d5874e0001f000000000000000000000000000000010b000000000000000000000000666666");
    constexpr auto solution = from_hex<100>(
        "01629b3779fd498defb2b0a551f7e111a8a003711acfe129622eb80bc98df66b9d8178b9670bacdc972b250fcb6715f437eb0addf858f9419c03f93a1be742e6377d4dcc4b9196afd811592ee4589cecfa321e7a9d5675338e7834923fe12b49f743a8d4");

    if (!verify(header, solution)) {
        return 1;
    }

    auto negative = solution;
    negative[0] ^= 0x80;
    if (verify(header, negative)) {
        return 2;
    }

    std::vector<std::uint8_t> preimage(header.begin(), header.end());
    preimage.push_back(solution.size());
    preimage.insert(preimage.end(), solution.begin(), solution.end());
    std::uint8_t first[32], digest[32], target[32];
    mom_equihash::pow::sha256(preimage.data(), preimage.size(), first);
    mom_equihash::pow::sha256(first, sizeof(first), digest);
    for (unsigned i = 0; i < 32; ++i) target[i] = digest[31 - i];
    if (!mom_equihash::pow::meets_target(header.data(), header.size(), solution.data(),
                                        solution.size(), target)) return 3;
    for (int i = 31; i >= 0; --i) {
        if (target[i]-- != 0) break;
    }
    if (mom_equihash::pow::meets_target(header.data(), header.size(), solution.data(),
                                       solution.size(), target)) return 4;
    return 0;
}
