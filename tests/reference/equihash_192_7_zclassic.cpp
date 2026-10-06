// CPU-only Equihash(192,7) acceptance oracle for Zclassic block 3209920.
// Compile/run: c++ -std=c++17 -O2 -Wall -Wextra -pedantic tests/reference/equihash_192_7_zclassic.cpp -o /tmp/equihash_192_7_zclassic && /tmp/equihash_192_7_zclassic

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr unsigned N = 192;
constexpr unsigned K = 7;
constexpr unsigned COLLISION_BITS = N / (K + 1);
constexpr unsigned COLLISION_BYTES = COLLISION_BITS / 8;
constexpr unsigned COLLISION_WORDS = K + 1;
constexpr unsigned INDICES_PER_HASH = 2;
constexpr unsigned SEGMENT_LENGTH = COLLISION_WORDS * COLLISION_BYTES;
constexpr unsigned HASH_LENGTH = INDICES_PER_HASH * SEGMENT_LENGTH;
constexpr unsigned INDEX_COUNT = 1u << K;
constexpr unsigned INDEX_BITS = COLLISION_BITS + 1;
constexpr unsigned SOLUTION_LENGTH = INDEX_COUNT * INDEX_BITS / 8;
constexpr unsigned HEADER_LENGTH = 140;
constexpr unsigned COMPACT_SIZE_LENGTH = 3;
constexpr unsigned TEST_BUFFER_LENGTH = 5120;
constexpr unsigned TEST_ROWS = TEST_BUFFER_LENGTH / SEGMENT_LENGTH;
constexpr unsigned MAX_SOLUTIONS = (TEST_BUFFER_LENGTH - 1) / SOLUTION_LENGTH;

static_assert(N % (K + 1) == 0, "invalid Equihash parameters");
static_assert(COLLISION_BITS == 24 && COLLISION_BYTES == 3, "unexpected collision width");
static_assert(HASH_LENGTH == 48 && SEGMENT_LENGTH == 24, "unexpected hash layout");
static_assert(SOLUTION_LENGTH == 400, "unexpected solution width");

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
    const std::array<std::uint8_t, HEADER_LENGTH>& header, const std::uint32_t index) {
    std::uint8_t parameter[64] = {};
    parameter[0] = HASH_LENGTH;
    parameter[2] = 1;
    parameter[3] = 1;
    const std::uint8_t personal[16] = {
        'Z', 'c', 'a', 's', 'h', 'P', 'o', 'W',
        static_cast<std::uint8_t>(N), static_cast<std::uint8_t>(N >> 8),
        static_cast<std::uint8_t>(N >> 16), static_cast<std::uint8_t>(N >> 24),
        static_cast<std::uint8_t>(K), static_cast<std::uint8_t>(K >> 8),
        static_cast<std::uint8_t>(K >> 16), static_cast<std::uint8_t>(K >> 24),
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
    block[12] = static_cast<std::uint8_t>(index);
    block[13] = static_cast<std::uint8_t>(index >> 8);
    block[14] = static_cast<std::uint8_t>(index >> 16);
    block[15] = static_cast<std::uint8_t>(index >> 24);
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

bool verify_subtree(const std::array<Node, INDEX_COUNT>& leaves, const unsigned begin,
                   const unsigned level, Node& result) {
    if (level == 0) {
        result = leaves[begin];
        return true;
    }

    const unsigned half = 1u << (level - 1);
    Node left;
    Node right;
    if (!verify_subtree(leaves, begin, level - 1, left) ||
        !verify_subtree(leaves, begin + half, level - 1, right)) {
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

bool verify_solution(const std::array<std::uint8_t, HEADER_LENGTH>& header,
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
        const auto digest = hash_index(header, index / INDICES_PER_HASH);
        const unsigned offset = (index % INDICES_PER_HASH) * SEGMENT_LENGTH;
        leaves[i].first_index = index;
        for (unsigned word = 0; word < COLLISION_WORDS; ++word) {
            const unsigned byte = offset + word * COLLISION_BYTES;
            leaves[i].words[word] = static_cast<std::uint32_t>(digest[byte]) |
                                    (static_cast<std::uint32_t>(digest[byte + 1]) << 8) |
                                    (static_cast<std::uint32_t>(digest[byte + 2]) << 16);
        }
    }

    Node root;
    return verify_subtree(leaves, 0, K, root) && root.words[0] == 0;
}

constexpr std::uint32_t SHA256_K[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u,
    0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u, 0xe49b69c1u, 0xefbe4786u,
    0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
    0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, 0xa2bfe8a1u, 0xa81a664bu,
    0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au,
    0x5b9cca4fu, 0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffaU, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
};

std::uint32_t rotate_right_32(const std::uint32_t x, const unsigned amount) {
    return (x >> amount) | (x << (32 - amount));
}

std::array<std::uint8_t, 32> sha256(const std::uint8_t* message, const std::size_t length) {
    const std::size_t padded_length = ((length + 9 + 63) / 64) * 64;
    std::vector<std::uint8_t> padded(padded_length, 0);
    std::memcpy(padded.data(), message, length);
    padded[length] = 0x80;
    const std::uint64_t bit_length = static_cast<std::uint64_t>(length) * 8;
    for (unsigned i = 0; i < 8; ++i) {
        padded[padded_length - 1 - i] = static_cast<std::uint8_t>(bit_length >> (8 * i));
    }

    std::uint32_t h[8] = {
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u,
    };
    for (std::size_t offset = 0; offset < padded_length; offset += 64) {
        std::uint32_t w[64];
        for (unsigned i = 0; i < 16; ++i) {
            w[i] = (static_cast<std::uint32_t>(padded[offset + 4 * i]) << 24) |
                   (static_cast<std::uint32_t>(padded[offset + 4 * i + 1]) << 16) |
                   (static_cast<std::uint32_t>(padded[offset + 4 * i + 2]) << 8) |
                   static_cast<std::uint32_t>(padded[offset + 4 * i + 3]);
        }
        for (unsigned i = 16; i < 64; ++i) {
            const std::uint32_t s0 = rotate_right_32(w[i - 15], 7) ^
                                      rotate_right_32(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const std::uint32_t s1 = rotate_right_32(w[i - 2], 17) ^
                                      rotate_right_32(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }

        std::uint32_t a = h[0];
        std::uint32_t b = h[1];
        std::uint32_t c = h[2];
        std::uint32_t d = h[3];
        std::uint32_t e = h[4];
        std::uint32_t f = h[5];
        std::uint32_t g = h[6];
        std::uint32_t hh = h[7];
        for (unsigned i = 0; i < 64; ++i) {
            const std::uint32_t s1 = rotate_right_32(e, 6) ^
                                      rotate_right_32(e, 11) ^ rotate_right_32(e, 25);
            const std::uint32_t ch = (e & f) ^ (~e & g);
            const std::uint32_t t1 = hh + s1 + ch + SHA256_K[i] + w[i];
            const std::uint32_t s0 = rotate_right_32(a, 2) ^
                                      rotate_right_32(a, 13) ^ rotate_right_32(a, 22);
            const std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            const std::uint32_t t2 = s0 + maj;
            hh = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }
        h[0] += a;
        h[1] += b;
        h[2] += c;
        h[3] += d;
        h[4] += e;
        h[5] += f;
        h[6] += g;
        h[7] += hh;
    }

    std::array<std::uint8_t, 32> digest{};
    for (unsigned i = 0; i < 8; ++i) {
        digest[4 * i] = static_cast<std::uint8_t>(h[i] >> 24);
        digest[4 * i + 1] = static_cast<std::uint8_t>(h[i] >> 16);
        digest[4 * i + 2] = static_cast<std::uint8_t>(h[i] >> 8);
        digest[4 * i + 3] = static_cast<std::uint8_t>(h[i]);
    }
    return digest;
}

std::array<std::uint8_t, 32> sha256d(const std::uint8_t* message, const std::size_t length) {
    const auto first = sha256(message, length);
    return sha256(first.data(), first.size());
}

bool matches_block_hash(const std::array<std::uint8_t, HEADER_LENGTH>& header,
                        const std::array<std::uint8_t, COMPACT_SIZE_LENGTH>& compact_size,
                        const std::array<std::uint8_t, SOLUTION_LENGTH>& solution,
                        const std::array<std::uint8_t, 32>& expected_display_hash) {
    std::array<std::uint8_t, HEADER_LENGTH + COMPACT_SIZE_LENGTH + SOLUTION_LENGTH> serialized{};
    std::memcpy(serialized.data(), header.data(), header.size());
    std::memcpy(serialized.data() + header.size(), compact_size.data(), compact_size.size());
    std::memcpy(serialized.data() + header.size() + compact_size.size(), solution.data(), solution.size());
    const auto digest = sha256d(serialized.data(), serialized.size());
    for (unsigned i = 0; i < digest.size(); ++i) {
        if (digest[i] != expected_display_hash[31 - i]) {
            return false;
        }
    }
    return true;
}

void print_hex(const std::uint8_t* bytes, const std::size_t length) {
    constexpr char digits[] = "0123456789abcdef";
    for (std::size_t i = 0; i < length; ++i) {
        std::cout << digits[bytes[i] >> 4] << digits[bytes[i] & 0x0f];
    }
    std::cout << '\n';
}

bool parse_hex(const std::string& text, std::uint8_t* bytes, const std::size_t length) {
    if (text.size() != length * 2) {
        return false;
    }
    for (std::size_t i = 0; i < length; ++i) {
        const int high = hex_digit(text[2 * i]);
        const int low = hex_digit(text[2 * i + 1]);
        if (high < 0 || low < 0) {
            return false;
        }
        bytes[i] = static_cast<std::uint8_t>((high << 4) | low);
    }
    return true;
}

int verify_solver_dump(const std::array<std::uint8_t, HEADER_LENGTH>& header,
                       const std::array<std::uint8_t, SOLUTION_LENGTH>& known_solution,
                       const std::array<std::uint8_t, TEST_BUFFER_LENGTH>& dump) {
    const unsigned count = dump[0];
    if (count == 0 || count > MAX_SOLUTIONS) {
        std::cerr << "solutions=" << count << " (expected 1.." << MAX_SOLUTIONS << ")\n";
        return 10;
    }

    bool known = false;
    for (unsigned i = 0; i < count; ++i) {
        std::array<std::uint8_t, SOLUTION_LENGTH> solution{};
        std::memcpy(solution.data(), dump.data() + 1 + i * SOLUTION_LENGTH,
                    SOLUTION_LENGTH);
        if (!verify_solution(header, solution)) {
            std::cerr << "solutions=" << count << " verified=" << i
                      << " invalid_proof=" << i << "\n";
            return 11;
        }
        known = known || solution == known_solution;
        for (unsigned j = 0; j < i; ++j) {
            if (std::memcmp(solution.data(),
                            dump.data() + 1 + j * SOLUTION_LENGTH,
                            SOLUTION_LENGTH) == 0) {
                std::cerr << "solutions=" << count << " verified=" << i
                          << " duplicate_proof=" << i << "\n";
                return 12;
            }
        }
    }
    for (std::size_t i = 1 + static_cast<std::size_t>(count) * SOLUTION_LENGTH;
         i < dump.size(); ++i) {
        if (dump[i] != 0) {
            std::cerr << "solutions=" << count << " verified=" << count
                      << " nonzero_padding_byte=" << i << "\n";
            return 13;
        }
    }
    std::cerr << "solutions=" << count << " verified=" << count
              << " known=" << (known ? 1 : 0) << '\n';
    return known ? 0 : 14;
}

}  // namespace

int main(const int argc, char** argv) {
    constexpr auto header = from_hex<HEADER_LENGTH>(
        "04000000ecf888bb9e8440dff1eca5ff69c277e85462f306ec785719a76e4dd20f0b0000c450f3fd"
        "2a66b462f4133c48cc636655ac055de95072bd693514c9c7156dfa8de2004d086a6929b60cb4e4ef"
        "bbfcf41d3fda50ad985fc421c990217a1daef400c58d776ad03c141e8001fde00f6dcbadb169c913"
        "1b3a07c44e9b11ca00000000000000005b15db75");
    constexpr auto compact_size = from_hex<COMPACT_SIZE_LENGTH>("fd9001");
    constexpr auto solution = from_hex<SOLUTION_LENGTH>(
        "000870397e46deb5da1b012d60132a670f44b56bcc3d62efab0f5fe274a4d7c74b5ac57ba1f80d89"
        "873459c67a1dbd5c11b10af944f4a507b01d143e4c10fa3f99975a00fb4f2b9dc5c9db0c9ce7fd89"
        "d8aeab27b9ab4b65974673cfd566c56e63c90e4911756fab9a579ed290934b91545505ecb0796eae"
        "d74172eac24232aec77064e87335363a5e67a282a69cfd3e15abe16cabcb28bbb9b268e90a9785c8"
        "0be303e2713b47b43188f7e3361c8045200feefe0368b5a1fdd5527a45c3365f2ccebaf29576a7e8"
        "01eae51de4694c3a678da0f0b04ce00a654cfc20fe0eb132ae0d295a07afb7c51514124bdd8180dd"
        "614363f5bca625f9dd2d02cdf0d40a0066ad121de47c736e894bf5de9b8bd29ff09a72284877514d"
        "695635121f1a4b63c2859da3279a6ad375fba1e7022f9428b76950fea7140ab892e5c64adb33f90e"
        "d025ec12a417c13551ab551408d72d1111a4b14ea4efa729bf64db883f38131bdb8aa6324f1fd4bf"
        "e2f99781bc554e5d6635d4e15b4335301b0cc3272d1f18c13226aef428fe049fe36241d98fafc610");
    constexpr auto expected_hash = from_hex<32>(
        "0000047969f6f267b909cc01a006fb05b91bea8c2ac51cc857f1cdf605b95978");
    // The first two 24-byte generation rows are the complete digest for hash index zero.
    constexpr auto expected_first_digest = from_hex<48>(
        "d22c1ef2a4fcd68dbc24ef7b7ac6df11c264375afae2010eba986ec05717f17e"
        "f9617c716cd961115eff3f21fd401b73");

    if (argc == 2 && std::string(argv[1]) == "--generation") {
        std::array<std::uint8_t, TEST_BUFFER_LENGTH> dump{};
        for (unsigned row = 0; row < TEST_ROWS; ++row) {
            const auto digest = hash_index(header, row / INDICES_PER_HASH);
            const unsigned offset = (row % INDICES_PER_HASH) * SEGMENT_LENGTH;
            std::memcpy(dump.data() + row * SEGMENT_LENGTH,
                        digest.data() + offset, SEGMENT_LENGTH);
        }
        print_hex(dump.data(), dump.size());
        return 0;
    }

    if (argc == 2 && std::string(argv[1]) == "--verify-solutions") {
        std::string text;
        if (!(std::cin >> text)) {
            std::cerr << "missing solver dump\n";
            return 15;
        }
        std::array<std::uint8_t, TEST_BUFFER_LENGTH> dump{};
        if (!parse_hex(text, dump.data(), dump.size())) {
            std::cerr << "solver dump must be exactly " << TEST_BUFFER_LENGTH * 2
                      << " hexadecimal characters\n";
            return 16;
        }
        return verify_solver_dump(header, solution, dump);
    }
    if (argc != 1) {
        std::cerr << "usage: " << argv[0]
                  << " [--generation|--verify-solutions]\n";
        return 17;
    }

    if (compact_size[0] != 0xfd || compact_size[1] != 0x90 || compact_size[2] != 0x01) {
        return 1;
    }
    const auto first_digest = hash_index(header, 0);
    if (std::memcmp(first_digest.data(), expected_first_digest.data(), expected_first_digest.size()) != 0) {
        return 5;
    }
    if (!verify_solution(header, solution)) {
        return 2;
    }
    if (!matches_block_hash(header, compact_size, solution, expected_hash)) {
        return 3;
    }

    auto invalid = solution;
    invalid.fill(0);
    if (verify_solution(header, invalid)) {
        return 4;
    }
    return 0;
}
