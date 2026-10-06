// CPU-only NexaPow Echelon conformance oracle.
// Compile/run: c++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic tests/reference/nexapow_echelon.cpp -o /tmp/nexapow_echelon && /tmp/nexapow_echelon
// Vectors: nexa/specification@bb81346868dd41b1d0c0d6b20360aef96af9a8b0.
// Protocol: echelon-mining-protocol@a525f07f212b85bb12b854f54cc23e6fe124cebf.
// Nexa@0d48a091116f9fc87a641c928fc0fa84364e1f59 changes Tailstorm's h1 to
// SHA256(SHA256d(mining_hash || prevhash)). Echelon v1 has no prevhash field, so
// this oracle covers its published null-prevhash path; Tailstorm input is unresolved.

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <boost/multiprecision/cpp_int.hpp>

namespace {

using boost::multiprecision::cpp_int;
using byte = std::uint8_t;
using Hash = std::array<byte, 32>;
using Nonce = std::array<byte, 16>;
template <std::size_t NonceBytes>
using PowNonce = std::array<byte, NonceBytes>;
template <std::size_t NonceBytes>
using PowSerialized = std::array<byte, 33u + NonceBytes>;
using Serialized = PowSerialized<16>;
using Signature = std::array<byte, 64>;

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

constexpr int hex_digit(const char c) {
    return c >= '0' && c <= '9' ? c - '0' :
           c >= 'a' && c <= 'f' ? c - 'a' + 10 :
           c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1;
}

template <std::size_t Bytes, std::size_t Characters>
constexpr std::array<byte, Bytes> from_hex(const char (&text)[Characters]) {
    static_assert(Characters == Bytes * 2 + 1, "wrong hexadecimal length");
    std::array<byte, Bytes> bytes{};
    for (std::size_t i = 0; i < Bytes; ++i) {
        const int hi = hex_digit(text[2 * i]);
        const int lo = hex_digit(text[2 * i + 1]);
        if (hi < 0 || lo < 0) throw "invalid hex";
        bytes[i] = static_cast<byte>((hi << 4) | lo);
    }
    return bytes;
}

template <std::size_t Bytes>
constexpr std::array<byte, Bytes> reversed(const std::array<byte, Bytes>& input) {
    std::array<byte, Bytes> output{};
    for (std::size_t i = 0; i < Bytes; ++i) {
        output[i] = input[Bytes - 1 - i];
    }
    return output;
}

constexpr Nonce solution_nonce(const std::uint64_t extranonce, const std::uint64_t miner_nonce) {
    Nonce result{};
    for (unsigned i = 0; i < 8; ++i) {
        result[7u - i] = static_cast<byte>(extranonce >> (8u * i));
        result[15u - i] = static_cast<byte>(miner_nonce >> (8u * i));
    }
    return result;
}

template <std::size_t NonceBytes>
constexpr PowSerialized<NonceBytes> serialize_echelon(const Hash& header_display,
                                                      const PowNonce<NonceBytes>& nonce) {
    static_assert(NonceBytes == 8 || NonceBytes == 12 || NonceBytes == 16,
                  "unsupported Echelon nonce width");
    PowSerialized<NonceBytes> result{};
    for (unsigned i = 0; i < 32; ++i) result[i] = header_display[31u - i];
    result[32] = static_cast<byte>(NonceBytes);  // CompactSize(8-, 12-, or 16-byte nonce).
    for (std::size_t i = 0; i < NonceBytes; ++i) result[33u + i] = nonce[i];
    return result;
}

Hash scalar_mod_order(const Hash& value);

std::uint32_t rotate_right(const std::uint32_t value, const unsigned amount) {
    return (value >> amount) | (value << (32u - amount));
}

Hash sha256(const byte* data, const std::size_t length) {
    const std::size_t padded_length = ((length + 9u + 63u) / 64u) * 64u;
    std::vector<byte> padded(padded_length, 0);
    for (std::size_t i = 0; i < length; ++i) {
        padded[i] = data[i];
    }
    padded[length] = 0x80;
    const std::uint64_t bit_length = static_cast<std::uint64_t>(length) * 8u;
    for (unsigned i = 0; i < 8; ++i) {
        padded[padded_length - 1 - i] = static_cast<byte>(bit_length >> (8u * i));
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
            const std::uint32_t s0 = rotate_right(w[i - 15], 7) ^
                                      rotate_right(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const std::uint32_t s1 = rotate_right(w[i - 2], 17) ^
                                      rotate_right(w[i - 2], 19) ^ (w[i - 2] >> 10);
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
            const std::uint32_t s1 = rotate_right(e, 6) ^ rotate_right(e, 11) ^ rotate_right(e, 25);
            const std::uint32_t ch = (e & f) ^ (~e & g);
            const std::uint32_t t1 = hh + s1 + ch + SHA256_K[i] + w[i];
            const std::uint32_t s0 = rotate_right(a, 2) ^ rotate_right(a, 13) ^ rotate_right(a, 22);
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

    Hash digest{};
    for (unsigned i = 0; i < 8; ++i) {
        digest[4 * i] = static_cast<byte>(h[i] >> 24);
        digest[4 * i + 1] = static_cast<byte>(h[i] >> 16);
        digest[4 * i + 2] = static_cast<byte>(h[i] >> 8);
        digest[4 * i + 3] = static_cast<byte>(h[i]);
    }
    return digest;
}

Hash sha256d(const byte* data, const std::size_t length) {
    const Hash first = sha256(data, length);
    return sha256(first.data(), first.size());
}

Hash hmac_sha256(const Hash& key, const byte* data, const std::size_t length) {
    std::array<byte, 64> ipad{};
    std::array<byte, 64> opad{};
    ipad.fill(0x36);
    opad.fill(0x5c);
    for (unsigned i = 0; i < 32; ++i) {
        ipad[i] ^= key[i];
        opad[i] ^= key[i];
    }

    std::vector<byte> inner(64u + length);
    std::vector<byte> outer(64u + 32u);
    for (unsigned i = 0; i < 64; ++i) {
        inner[i] = ipad[i];
        outer[i] = opad[i];
    }
    for (std::size_t i = 0; i < length; ++i) {
        inner[64u + i] = data[i];
    }
    const Hash inner_hash = sha256(inner.data(), inner.size());
    for (unsigned i = 0; i < 32; ++i) {
        outer[64u + i] = inner_hash[i];
    }
    return sha256(outer.data(), outer.size());
}

Hash rfc6979_nonce(const Hash& message, const Hash& key, const unsigned counter) {
    constexpr std::array<byte, 16> algorithm = {
        'S', 'c', 'h', 'n', 'o', 'r', 'r', '+', 'S', 'H', 'A', '2', '5', '6', ' ', ' ',
    };
    const Hash reduced_message = scalar_mod_order(message);
    std::array<byte, 80> seed{};
    for (unsigned i = 0; i < 32; ++i) {
        seed[i] = key[i];
        seed[32u + i] = reduced_message[i];
    }
    for (unsigned i = 0; i < 16; ++i) {
        seed[64u + i] = algorithm[i];
    }

    Hash k{};
    Hash v{};
    v.fill(1);
    const auto update_key = [&](const byte marker, Hash& state) {
        std::vector<byte> input(33u + seed.size());
        for (unsigned i = 0; i < 32; ++i) {
            input[i] = state[i];
        }
        input[32] = marker;
        for (std::size_t i = 0; i < seed.size(); ++i) {
            input[33u + i] = seed[i];
        }
        k = hmac_sha256(k, input.data(), input.size());
    };
    update_key(0, v);
    v = hmac_sha256(k, v.data(), v.size());
    update_key(1, v);
    v = hmac_sha256(k, v.data(), v.size());
    for (unsigned i = 0; i <= counter; ++i) {
        v = hmac_sha256(k, v.data(), v.size());
    }
    return v;
}

template <std::size_t Bytes>
cpp_int from_big_endian(const std::array<byte, Bytes>& bytes) {
    cpp_int value = 0;
    for (const byte item : bytes) {
        value <<= 8;
        value += item;
    }
    return value;
}

Hash to_big_endian_256(cpp_int value) {
    Hash bytes{};
    for (std::size_t i = bytes.size(); i-- > 0;) {
        bytes[i] = static_cast<byte>((value & 0xff).convert_to<unsigned>());
        value >>= 8;
    }
    return bytes;
}

const cpp_int& field_prime() {
    static const cpp_int value = (cpp_int(1) << 256) - (cpp_int(1) << 32) - 977;
    return value;
}

const cpp_int& group_order() {
    static const cpp_int value = from_big_endian(from_hex<32>(
        "fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141"));
    return value;
}

cpp_int modulo(cpp_int value, const cpp_int& modulus) {
    value %= modulus;
    if (value < 0) {
        value += modulus;
    }
    return value;
}

Hash scalar_mod_order(const Hash& value) {
    return to_big_endian_256(modulo(from_big_endian(value), group_order()));
}

bool display_hash_meets_target(const Hash& hash, const Hash& target) {
    return from_big_endian(hash) <= from_big_endian(target);
}

cpp_int modular_power(cpp_int base, cpp_int exponent, const cpp_int& modulus) {
    base = modulo(base, modulus);
    cpp_int result = 1;
    while (exponent > 0) {
        if ((exponent & 1) != 0) {
            result = modulo(result * base, modulus);
        }
        base = modulo(base * base, modulus);
        exponent >>= 1;
    }
    return result;
}

cpp_int modular_inverse(const cpp_int& value, const cpp_int& modulus) {
    return modular_power(value, modulus - 2, modulus);
}

struct Point {
    cpp_int x{};
    cpp_int y{};
    bool infinity = true;
};

const Point& generator() {
    static const Point value{
        from_big_endian(from_hex<32>(
            "79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798")),
        from_big_endian(from_hex<32>(
            "483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8")),
        false,
    };
    return value;
}

Point point_add(const Point& first, const Point& second) {
    if (first.infinity) {
        return second;
    }
    if (second.infinity) {
        return first;
    }

    const cpp_int& p = field_prime();
    cpp_int slope;
    if (first.x == second.x) {
        if (modulo(first.y + second.y, p) == 0 || first.y == 0) {
            return {};
        }
        slope = modulo(3 * first.x * first.x *
                           modular_inverse(modulo(2 * first.y, p), p), p);
    } else {
        slope = modulo((second.y - first.y) *
                           modular_inverse(modulo(second.x - first.x, p), p), p);
    }

    const cpp_int x = modulo(slope * slope - first.x - second.x, p);
    const cpp_int y = modulo(slope * (first.x - x) - first.y, p);
    return Point{x, y, false};
}

Point scalar_multiply(cpp_int scalar, const Point& point) {
    Point result;
    Point addend = point;
    while (scalar > 0) {
        if ((scalar & 1) != 0) {
            result = point_add(result, addend);
        }
        addend = point_add(addend, addend);
        scalar >>= 1;
    }
    return result;
}

bool is_quadratic_residue(const cpp_int& value) {
    return value != 0 && modular_power(value, (field_prime() - 1) / 2, field_prime()) == 1;
}

std::array<byte, 33> compressed_public_key(const Point& point) {
    std::array<byte, 33> result{};
    result[0] = static_cast<byte>(2u + static_cast<unsigned>((point.y & 1) != 0));
    const Hash x = to_big_endian_256(point.x);
    for (unsigned i = 0; i < 32; ++i) {
        result[1u + i] = x[i];
    }
    return result;
}

Hash challenge_digest(const Hash& r, const Point& public_key, const Hash& message) {
    const auto compressed = compressed_public_key(public_key);
    // The legacy Nexa/Cash Schnorr implementation hashes r || pubkey || m.
    std::array<byte, 97> ordered{};
    for (unsigned i = 0; i < 32; ++i) {
        ordered[i] = r[i];
    }
    for (unsigned i = 0; i < 33; ++i) {
        ordered[32u + i] = compressed[i];
    }
    for (unsigned i = 0; i < 32; ++i) {
        ordered[65u + i] = message[i];
    }
    return sha256(ordered.data(), ordered.size());
}

bool compute_challenge(const Hash& r, const Point& public_key, const Hash& message, cpp_int& e) {
    e = from_big_endian(challenge_digest(r, public_key, message));
    return e != 0 && e < group_order();
}

template <typename NonceFunction>
bool sign_schnorr_with_nonce(const Hash& message, const Hash& private_key,
                             const Point& public_key, Signature& signature,
                             const NonceFunction& nonce_function) {
    const cpp_int d = from_big_endian(private_key);
    if (d == 0 || d >= group_order() || public_key.infinity) {
        return false;
    }

    for (unsigned counter = 0; ; ++counter) {
        const cpp_int k = from_big_endian(nonce_function(message, private_key, counter));
        if (k == 0 || k >= group_order()) {
            continue;
        }

        const Point nonce_point = scalar_multiply(k, generator());
        cpp_int adjusted_k = k;
        if (!is_quadratic_residue(nonce_point.y)) {
            adjusted_k = group_order() - adjusted_k;
        }
        const Hash r = to_big_endian_256(nonce_point.x);
        cpp_int e;
        if (!compute_challenge(r, public_key, message, e)) {
            return false;
        }

        const Hash s = to_big_endian_256(modulo(adjusted_k + e * d, group_order()));
        for (unsigned i = 0; i < 32; ++i) {
            signature[i] = r[i];
            signature[32u + i] = s[i];
        }
        return true;
    }
}

bool sign_schnorr(const Hash& message, const Hash& private_key, const Point& public_key,
                  Signature& signature) {
    return sign_schnorr_with_nonce(message, private_key, public_key, signature, rfc6979_nonce);
}

bool verify_schnorr(const Signature& signature, const Hash& message, const Point& public_key) {
    if (public_key.infinity) {
        return false;
    }
    Hash r_bytes{};
    Hash s_bytes{};
    for (unsigned i = 0; i < 32; ++i) {
        r_bytes[i] = signature[i];
        s_bytes[i] = signature[32u + i];
    }
    const cpp_int r = from_big_endian(r_bytes);
    const cpp_int s = from_big_endian(s_bytes);
    if (r >= field_prime() || s >= group_order()) {
        return false;
    }

    // Verification reduces the challenge as scalar_set_b32 does in Nexa's
    // legacy secp256k1 fork; signing rejects overflow and zero challenges.
    const cpp_int e = modulo(from_big_endian(challenge_digest(r_bytes, public_key, message)),
                             group_order());

    const Point s_generator = scalar_multiply(s, generator());
    const Point e_public = scalar_multiply(modulo(group_order() - e, group_order()), public_key);
    const Point recovered = point_add(s_generator, e_public);
    return !recovered.infinity && recovered.x == r && is_quadratic_residue(recovered.y);
}

template <std::size_t NonceBytes>
struct PowVector {
    PowNonce<NonceBytes> nonce;
    Hash mining_hash_display;
    Hash h1_display;
    Signature signature;
    Hash final_hash_display;
};

using Vector = PowVector<16>;

struct Work {
    Hash mining_hash;
    Hash h1;
    Point public_key;
};

template <std::size_t NonceBytes>
bool check_vector(const Hash& header_display, const PowVector<NonceBytes>& expected, Work& work) {
    const PowSerialized<NonceBytes> serialized = serialize_echelon(header_display, expected.nonce);
    work.mining_hash = sha256d(serialized.data(), serialized.size());
    work.h1 = sha256(work.mining_hash.data(), work.mining_hash.size());
    if (reversed(work.mining_hash) != expected.mining_hash_display ||
        reversed(work.h1) != expected.h1_display) {
        return false;
    }

    const cpp_int private_scalar = from_big_endian(work.mining_hash);
    if (private_scalar == 0 || private_scalar >= group_order()) return false;
    work.public_key = scalar_multiply(private_scalar, generator());
    Signature signature{};
    if (!sign_schnorr(work.h1, work.mining_hash, work.public_key, signature) ||
        signature != expected.signature) {
        return false;
    }
    return reversed(sha256(signature.data(), signature.size())) == expected.final_hash_display;
}

}  // namespace

int main() {
    constexpr auto header_display = from_hex<32>(
        "0a4ac49b2d02e3c8d12c7093255ba7c49624f9c374d9f1c2f8e37c58705e74b0");
    constexpr std::array<Vector, 2> vectors{{
        {
            from_hex<16>("10000000000000001182dc5800000000"),
            from_hex<32>("7077c3549e088001ff681c2291bbc171f6ea681b5bc79bca9f2e0f563ad3dfef"),
            from_hex<32>("f97e4711d32f29f4f9f3aa254290eaf486330b3b43507cc03ba93b596c6cce36"),
            from_hex<64>(
                "93bb5bd98bb9d0817c5594782d4648a995e76755a3c94fdd96673115a06b8483"
                "efe485e73c1e4732ce9c34ef6891e7f25ff7c080aacbf71660bf782b067bc4f1"),
            from_hex<32>("00000042cbc240375242e14641488a0e2dca7b54458a2cea23dc1d2c178bb188"),
        },
        {
            from_hex<16>("1000000000000000b787915d00000000"),
            from_hex<32>("5cddd556e7972ee8b3e4ce01f954336cbb223813c1f01c768f299b82f20a0ea9"),
            from_hex<32>("6c55ec4a361007edaf6561e20a2eccfb51572309f66bf7b04aea315fadc039f9"),
            from_hex<64>(
                "fc1894a100e6b33d19abac5289403b4c942d75d061b5453904be740681526cae"
                "30ed3a59f1a9d3ca165151ddb1ca3069de832cd5cc5fc95a7339f058f1fd155b"),
            from_hex<32>("0000005f0b59e110863566e77d85e1b4fc713754e5c75f8fc3df44133866a669"),
        },
    }};
    constexpr std::array<Hash, 2> short_headers{{
        from_hex<32>("0000000000000000000000000000000000000000000000000000000000000000"),
        from_hex<32>("0104070a0d101316191c1f2225282b2e3134373a3d404346494c4f5255585b5e")
    }};
    constexpr std::array<PowVector<8>, 2> short_vectors{{
        {
            from_hex<8>("0000000000000000"),
            from_hex<32>("14eea5a43d2f39ecde694fc685a43257d598bb4f38df836c367251e3dcb681ca"),
            from_hex<32>("ebb48ba1d7e6e0249901cf2f6d30f1fc1290c47d36641aded0ee8807088803b1"),
            from_hex<64>(
                "b1d32f2a232da70e642582d7b86a7dfabf0a5a526b0a60e0dd2d72dfcbe95c94"
                "93ca144885f694349aec38dcdaff9554156a3f57f71bdb45c0ddbf90d430e6dc"),
            from_hex<32>("b034dd02a3ba15d16b1f0325e44de3e374489182b2305ca13740772184c3bcd9"),
        },
        {
            from_hex<8>("0102030405060708"),
            from_hex<32>("15495bb7f9464e233c54d2179b2b05dab0ad4b42a6bb6f284eee35409d242c31"),
            from_hex<32>("1ce3e7e6e7027fc57c77be0955fd2d6af6259abddc024c68e892c13df5995cc0"),
            from_hex<64>(
                "94e0e0eba4c8f0aec3aaa660516feb138ce3e42098e06b41bf5181fadb1d398c"
                "b1ef0dc6b95d0453739a71d819c7816518341b9190eb4dd48716a07897a8cdda"),
            from_hex<32>("a254ce16b11c81a3bebf46569b2598d28e3fd2223a565a9245cfa8bbca03e997"),
        }
    }};
    constexpr std::array<PowVector<12>, 1> four_byte_prefix_vectors{{
        {
            // WildRig-compatible short jobs concatenate the fixed prefix and worker nonce.
            from_hex<12>("102030400102030405060708"),
            from_hex<32>("50feb0496eb956fbb90df8fcf012302dcb22373a5492b90fe54ec206304d17ff"),
            from_hex<32>("30e99bc61cf5096497ec0ddad9b5671b31f151c347192be7d8cafa73e24ed816"),
            from_hex<64>(
                "e5b0c0ed297f905d2771905ea3fdecab8c6606c6d48c0a7b203cb9fafb4abc30"
                "86ef5b94736ba6985d53f5053ba015f9d7481847353386bccb86bbc26b11d714"),
            from_hex<32>("a9bd7ffdfcc1819ea1e4068745b43ee233a2cfe04f42e336e3d93c24f3af9dee"),
        },
    }};
    constexpr auto expected_serialized = from_hex<49>(
        "b0745e70587ce3f8c2f1d974c3f92496c4a75b2593702cd1c8e3022d9bc44a0a"
        "1010000000000000001182dc5800000000");
    constexpr auto expected_four_byte_prefix_serialized = from_hex<45>(
        "b0745e70587ce3f8c2f1d974c3f92496c4a75b2593702cd1c8e3022d9bc44a0a"
        "0c102030400102030405060708");

    if (serialize_echelon(header_display, vectors[0].nonce) != expected_serialized) return 1;
    if (serialize_echelon(header_display, four_byte_prefix_vectors[0].nonce) !=
        expected_four_byte_prefix_serialized) return 11;
    constexpr auto carry_before = solution_nonce(0x0102030405060708ull, 0xffull);
    constexpr auto carry_after = solution_nonce(0x0102030405060708ull, 0x100ull);
    constexpr auto extranonce_max = solution_nonce(0xffffffffffffffffull, 0);
    constexpr auto miner_nonce_max = solution_nonce(0x0102030405060708ull, 0xffffffffffffffffull);
    if (carry_before != from_hex<16>("010203040506070800000000000000ff") ||
        carry_after != from_hex<16>("01020304050607080000000000000100") ||
        extranonce_max != from_hex<16>("ffffffffffffffff0000000000000000") ||
        miner_nonce_max != from_hex<16>("0102030405060708ffffffffffffffff") ||
        serialize_echelon(header_display, extranonce_max) != from_hex<49>(
            "b0745e70587ce3f8c2f1d974c3f92496c4a75b2593702cd1c8e3022d9bc44a0a"
            "10ffffffffffffffff0000000000000000")) {
        return 2;
    }

    std::array<Work, vectors.size()> work{};
    for (std::size_t i = 0; i < vectors.size(); ++i) {
        if (!check_vector(header_display, vectors[i], work[i])) return static_cast<int>(3 + i);
    }

    std::array<Work, short_vectors.size()> short_work{};
    for (std::size_t i = 0; i < short_vectors.size(); ++i) {
        if (!check_vector(short_headers[i], short_vectors[i], short_work[i]))
            return static_cast<int>(11 + i);
        if (!verify_schnorr(short_vectors[i].signature, short_work[i].h1, short_work[i].public_key))
            return static_cast<int>(13 + i);
    }

    Work four_byte_prefix_work{};
    if (!check_vector(header_display, four_byte_prefix_vectors[0], four_byte_prefix_work) ||
        !verify_schnorr(four_byte_prefix_vectors[0].signature, four_byte_prefix_work.h1,
                        four_byte_prefix_work.public_key)) {
        return 15;
    }

    if (!verify_schnorr(vectors[0].signature, work[0].h1, work[0].public_key)) return 5;
    auto tampered = vectors[0].signature;
    tampered[63] ^= 1;
    if (verify_schnorr(tampered, work[0].h1, work[0].public_key)) return 6;

    Signature scratch{};
    const Hash zero{};
    const Hash order = to_big_endian_256(group_order());
    if (sign_schnorr(work[0].h1, zero, generator(), scratch) ||
        sign_schnorr(work[0].h1, order, generator(), scratch)) {
        return 7;
    }

    const Hash just_above = to_big_endian_256(from_big_endian(vectors[0].final_hash_display) + 1);
    if (!display_hash_meets_target(vectors[0].final_hash_display, vectors[0].final_hash_display) ||
        display_hash_meets_target(just_above, vectors[0].final_hash_display)) {
        return 8;
    }

    constexpr auto nonce0 = from_hex<32>(
        "81ce30cee9394f8cb2284de714f93109abf01df36950b99f01c92d05078bdc5b");
    constexpr auto nonce1 = from_hex<32>(
        "e6854fbdfaf77efb67b5e736a2d8ff09dea80067f8c9a32405a7f4113d3501b2");
    Hash one{};
    one.back() = 1;
    const Hash order_plus_one = to_big_endian_256(group_order() + 1);
    if (rfc6979_nonce(work[0].h1, work[0].mining_hash, 0) != nonce0 ||
        rfc6979_nonce(work[0].h1, work[0].mining_hash, 1) != nonce1 ||
        rfc6979_nonce(order, work[0].mining_hash, 0) != rfc6979_nonce(zero, work[0].mining_hash, 0) ||
        rfc6979_nonce(order_plus_one, work[0].mining_hash, 0) !=
            rfc6979_nonce(one, work[0].mining_hash, 0)) {
        return 9;
    }

    unsigned attempts = 0;
    const auto retry_nonce = [&](const Hash& message, const Hash& key, const unsigned counter) {
        ++attempts;
        if (counter == 0) return Hash{};
        if (counter == 1) return order;
        return rfc6979_nonce(message, key, 0);
    };
    Signature retried{};
    if (!sign_schnorr_with_nonce(work[0].h1, work[0].mining_hash, work[0].public_key,
                                 retried, retry_nonce) ||
        attempts != 3 || retried != vectors[0].signature) {
        return 10;
    }
    return 0;
}
