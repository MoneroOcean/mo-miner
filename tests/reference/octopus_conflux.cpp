// Independent Conflux Octopus light-mode correctness oracle (C++17).
// Consensus source pin: Conflux-Chain/conflux-rust@3790f62a7bfc24ffecbfec55d9882aa73bd3c4d3c
// Relevant files: crates/primitives/src/block_header.rs and crates/cfxcore/pow/src/{shared,
// seed_compute,cache,compute,lib}.rs. Vector: mainnet epoch 0x5f5e100 (stage 190).

#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using Bytes = std::vector<std::uint8_t>;
using Hash256 = std::array<std::uint8_t, 32>;
using Node = std::array<std::uint8_t, 64>;

[[noreturn]] static void fail(const char* message) { throw std::runtime_error(message); }
static void require(bool condition, const char* message) { if (!condition) fail(message); }

static std::uint64_t load_le64(const std::uint8_t* p) {
    std::uint64_t x = 0;
    for (unsigned i = 0; i < 8; ++i) x |= std::uint64_t(p[i]) << (8U * i);
    return x;
}

static std::uint32_t load_le32(const std::uint8_t* p) {
    std::uint32_t x = 0;
    for (unsigned i = 0; i < 4; ++i) x |= std::uint32_t(p[i]) << (8U * i);
    return x;
}

static void store_le64(std::uint8_t* p, std::uint64_t x) {
    for (unsigned i = 0; i < 8; ++i) p[i] = static_cast<std::uint8_t>(x >> (8U * i));
}

static void store_le32(std::uint8_t* p, std::uint32_t x) {
    for (unsigned i = 0; i < 4; ++i) p[i] = static_cast<std::uint8_t>(x >> (8U * i));
}

static std::uint64_t rotl(std::uint64_t x, unsigned n) {
    return n == 0 ? x : (x << n) | (x >> (64U - n));
}

static void keccak_f(std::array<std::uint64_t, 25>& a) {
    static constexpr std::array<std::uint64_t, 24> rc{{
        0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL,
        0x8000000080008000ULL, 0x000000000000808bULL, 0x0000000080000001ULL,
        0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008aULL,
        0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
        0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL,
        0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
        0x000000000000800aULL, 0x800000008000000aULL, 0x8000000080008081ULL,
        0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL,
    }};
    static constexpr std::array<unsigned, 25> rho{{
        0, 1, 62, 28, 27, 36, 44, 6, 55, 20, 3, 10, 43, 25, 39, 41, 45,
        15, 21, 8, 18, 2, 61, 56, 14,
    }};
    for (std::uint64_t round_constant : rc) {
        std::array<std::uint64_t, 5> c{}, d{};
        std::array<std::uint64_t, 25> b{};
        for (std::size_t x = 0; x < 5; ++x)
            c[x] = a[x] ^ a[x + 5] ^ a[x + 10] ^ a[x + 15] ^ a[x + 20];
        for (std::size_t x = 0; x < 5; ++x) d[x] = c[(x + 4) % 5] ^ rotl(c[(x + 1) % 5], 1);
        for (std::size_t y = 0; y < 5; ++y)
            for (std::size_t x = 0; x < 5; ++x) a[x + 5 * y] ^= d[x];
        for (std::size_t y = 0; y < 5; ++y)
            for (std::size_t x = 0; x < 5; ++x)
                b[y + 5 * ((2 * x + 3 * y) % 5)] = rotl(a[x + 5 * y], rho[x + 5 * y]);
        for (std::size_t y = 0; y < 5; ++y)
            for (std::size_t x = 0; x < 5; ++x)
                a[x + 5 * y] = b[x + 5 * y] ^ ((~b[(x + 1) % 5 + 5 * y]) & b[(x + 2) % 5 + 5 * y]);
        a[0] ^= round_constant;
    }
}

template <std::size_t Out>
static std::array<std::uint8_t, Out> keccak(const std::uint8_t* data, std::size_t size,
                                            std::size_t rate) {
    std::array<std::uint64_t, 25> state{};
    std::size_t offset = 0;
    while (size - offset >= rate) {
        for (std::size_t i = 0; i < rate / 8; ++i) state[i] ^= load_le64(data + offset + 8 * i);
        keccak_f(state);
        offset += rate;
    }
    std::array<std::uint8_t, 200> block{};
    for (std::size_t i = 0; i < size - offset; ++i) block[i] = data[offset + i];
    block[size - offset] = 0x01; // Keccak padding, not SHA-3's 0x06 domain.
    block[rate - 1] |= 0x80;
    for (std::size_t i = 0; i < rate / 8; ++i) state[i] ^= load_le64(block.data() + 8 * i);
    keccak_f(state);

    std::array<std::uint8_t, Out> out{};
    std::size_t done = 0;
    while (done < Out) {
        const std::size_t take = (Out - done < rate) ? Out - done : rate;
        for (std::size_t i = 0; i < take; ++i)
            out[done + i] = static_cast<std::uint8_t>(state[i / 8] >> (8U * (i % 8)));
        done += take;
        if (done < Out) keccak_f(state);
    }
    return out;
}

static Hash256 keccak256(const std::uint8_t* p, std::size_t n) { return keccak<32>(p, n, 136); }
static Node keccak512(const std::uint8_t* p, std::size_t n) { return keccak<64>(p, n, 72); }
static Hash256 keccak256(const Bytes& v) { return keccak256(v.data(), v.size()); }

static unsigned hex_digit(char c) {
    if (c >= '0' && c <= '9') return static_cast<unsigned>(c - '0');
    if (c >= 'a' && c <= 'f') return static_cast<unsigned>(c - 'a' + 10);
    if (c >= 'A' && c <= 'F') return static_cast<unsigned>(c - 'A' + 10);
    fail("invalid hex digit");
}

static Bytes from_hex(std::string s) {
    if (s.size() >= 2 && s[0] == '0' && s[1] == 'x') s.erase(0, 2);
    require(s.size() % 2 == 0, "odd hex length");
    Bytes out(s.size() / 2);
    for (std::size_t i = 0; i < out.size(); ++i)
        out[i] = static_cast<std::uint8_t>((hex_digit(s[2 * i]) << 4U) | hex_digit(s[2 * i + 1]));
    return out;
}

template <std::size_t N>
static std::string to_hex(const std::array<std::uint8_t, N>& v) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string out;
    out.reserve(2 * N);
    for (std::uint8_t x : v) {
        out.push_back(digits[x >> 4U]);
        out.push_back(digits[x & 15U]);
    }
    return out;
}

static Bytes rlp_length(std::size_t length, std::uint8_t short_base, std::uint8_t long_base) {
    if (length <= 55) return Bytes{static_cast<std::uint8_t>(short_base + length)};
    Bytes encoded;
    for (std::size_t n = length; n != 0; n >>= 8U)
        encoded.insert(encoded.begin(), static_cast<std::uint8_t>(n));
    Bytes prefix{static_cast<std::uint8_t>(long_base + encoded.size())};
    prefix.insert(prefix.end(), encoded.begin(), encoded.end());
    return prefix;
}

static Bytes rlp_bytes(const Bytes& value) {
    if (value.size() == 1 && value[0] < 0x80) return Bytes{value[0]};
    Bytes out = rlp_length(value.size(), 0x80, 0xb7);
    out.insert(out.end(), value.begin(), value.end());
    return out;
}

static Bytes rlp_uint(std::uint64_t value) {
    Bytes bytes;
    while (value != 0) {
        bytes.insert(bytes.begin(), static_cast<std::uint8_t>(value));
        value >>= 8U;
    }
    return rlp_bytes(bytes);
}

static Bytes rlp_list(const std::vector<Bytes>& values) {
    Bytes body;
    for (const Bytes& value : values) body.insert(body.end(), value.begin(), value.end());
    Bytes out = rlp_length(body.size(), 0xc0, 0xf7);
    out.insert(out.end(), body.begin(), body.end());
    return out;
}

struct HeaderVector {
    Bytes parent = from_hex("3b0169382426207060f5aac490c49ae7c33cb770716e8e333b5eeb0bdc628fbc");
    std::uint64_t height = 0x5f5e100;
    std::uint64_t timestamp = 0x668ee92c;
    Bytes author = from_hex("1468d2a77f4f7acc66387e427e58e49b5f347932");
    Bytes transactions_root = from_hex("c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470");
    Bytes state_root = from_hex("1140ea389846f3a8d49e135276a824e1408b46be1416bb951ceb736d1389e368");
    Bytes receipts_root = from_hex("12af19d53c378426ebe08ad33e48caf3efdaaade0994770c161c0637e65a6566");
    Bytes bloom_hash = from_hex("d397b3b043d87fcd6fad1291ff0bfd16401c274896d8c63a923727f077b8e0b5");
    std::uint64_t difficulty = 0x1d30a3352ccULL;
    std::uint64_t gas_limit = 0x1c9c380;
    std::vector<Bytes> referees{
        from_hex("6868eefc305484a91fd2754349295416d3c12e051ceb4f0104adc2de000c7070"),
        from_hex("a5f6754d2431dd0646331f27dd165fe5f0126a638b171f54bb397decf1416bef"),
    };
    std::uint64_t nonce = 0x35847700061b0795ULL;
    Bytes pos_reference = from_hex("ab010f45b50280114f697fdf72b3c338cc25b7947dec7caaaac4a9435500825e");
};

static Bytes encode_header(const HeaderVector& h, bool with_nonce) {
    std::vector<Bytes> fields{
        rlp_bytes(h.parent), rlp_uint(h.height), rlp_uint(h.timestamp), rlp_bytes(h.author),
        rlp_bytes(h.transactions_root), rlp_bytes(h.state_root), rlp_bytes(h.receipts_root),
        rlp_bytes(h.bloom_hash), rlp_uint(0), rlp_uint(h.difficulty), rlp_uint(0),
        rlp_uint(h.gas_limit),
    };
    std::vector<Bytes> refs;
    for (const Bytes& referee : h.referees) refs.push_back(rlp_bytes(referee));
    fields.push_back(rlp_list(refs));
    if (with_nonce) fields.push_back(rlp_uint(h.nonce));
    fields.push_back(rlp_list({rlp_bytes(h.pos_reference)})); // Option<H256>::Some is a list.
    fields.push_back(rlp_bytes(Bytes{0x03}));                 // Current custom field; no base price.
    return rlp_list(fields);
}

static bool is_prime(std::uint64_t n) {
    if (n < 2 || n % 2 == 0) return n == 2;
    for (std::uint64_t d = 3; d <= n / d; d += 2)
        if (n % d == 0) return false;
    return true;
}

static std::uint64_t cache_size(std::uint64_t stage) {
    std::uint64_t size = (1ULL << 24U) + (1ULL << 16U) * stage - 64;
    while (!is_prime(size / 64)) size -= 128;
    return size;
}

static std::uint64_t dataset_size(std::uint64_t stage) {
    std::uint64_t size = (1ULL << 32U) + (1ULL << 24U) * stage - 256;
    while (!is_prime(size / 256)) size -= 512;
    return size;
}

static std::uint32_t fnv32(std::uint32_t a, std::uint32_t b) {
    return (a * 0x01000193U) ^ b;
}

static std::uint64_t fnv64(std::uint64_t a, std::uint64_t b) {
    return (a * 0x01000193ULL) ^ b;
}

static std::vector<Node> make_cache(std::uint64_t stage) {
    Hash256 seed{};
    for (std::uint64_t i = 0; i < stage; ++i) seed = keccak256(seed.data(), seed.size());
    std::vector<Node> cache(static_cast<std::size_t>(cache_size(stage) / 64));
    cache[0] = keccak512(seed.data(), seed.size());
    for (std::size_t i = 1; i < cache.size(); ++i)
        cache[i] = keccak512(cache[i - 1].data(), cache[i - 1].size());
    for (unsigned round = 0; round < 3; ++round) {
        for (std::size_t i = 0; i < cache.size(); ++i) {
            const std::size_t index = load_le32(cache[i].data()) % cache.size();
            const Node previous = cache[(i + cache.size() - 1) % cache.size()];
            const Node indexed = cache[index];
            Node mixed{};
            for (std::size_t j = 0; j < mixed.size(); ++j) mixed[j] = previous[j] ^ indexed[j];
            cache[i] = keccak512(mixed.data(), mixed.size());
        }
    }
    return cache;
}

static Node dataset_item(const std::vector<Node>& cache, std::uint32_t node_index) {
    Node item = cache[node_index % cache.size()];
    store_le32(item.data(), load_le32(item.data()) ^ node_index);
    item = keccak512(item.data(), item.size());
    for (std::uint32_t i = 0; i < 256; ++i) {
        const std::uint32_t parent = fnv32(node_index ^ i, load_le32(item.data() + 4 * (i % 16))) %
                                     static_cast<std::uint32_t>(cache.size());
        for (std::size_t word = 0; word < 16; ++word) {
            const std::uint32_t mixed = fnv32(load_le32(item.data() + 4 * word),
                                              load_le32(cache[parent].data() + 4 * word));
            store_le32(item.data() + 4 * word, mixed);
        }
    }
    return keccak512(item.data(), item.size());
}

static std::uint64_t gcd(std::uint64_t a, std::uint64_t b) {
    while (b != 0) {
        const std::uint64_t r = a % b;
        a = b;
        b = r;
    }
    return a;
}

static constexpr std::uint64_t kMod = 1032193;

static std::uint64_t pow_mod(std::uint64_t base, std::uint64_t exponent) {
    std::uint64_t result = 1;
    while (exponent != 0) {
        if ((exponent & 1U) != 0) result = (result * base) % kMod;
        base = (base * base) % kMod;
        exponent >>= 1U;
    }
    return result;
}

static std::uint64_t remap(std::uint64_t h) {
    std::uint64_t exponent = h % (kMod - 2) + 1;
    for (std::uint64_t divisor = gcd(exponent, kMod - 1); divisor != 1;
         divisor = gcd(exponent, kMod - 1)) exponent /= divisor;
    return pow_mod(11, exponent);
}

struct SipHashState {
    std::uint64_t v0, v1, v2, v3;

    void round() {
        v0 += v1;
        v1 = rotl(v1, 13);
        v1 ^= v0;
        v0 = rotl(v0, 32);
        v2 += v3;
        v3 = rotl(v3, 16);
        v3 ^= v2;
        v0 += v3;
        v3 = rotl(v3, 21);
        v3 ^= v0;
        v2 += v1;
        v1 = rotl(v1, 17);
        v1 ^= v2;
        v2 = rotl(v2, 32);
    }

    void hash24(std::uint64_t nonce) {
        v3 ^= nonce;
        round();
        round();
        v0 ^= nonce;
        v2 ^= 0xff;
        round();
        round();
        round();
        round();
    }

    std::uint64_t value() const {
        return v0 ^ v1 ^ v2 ^ v3;
    }
};

static Hash256 octopus_hash(const Hash256& problem, std::uint64_t nonce,
                            const std::vector<Node>& cache, std::uint64_t full_size) {
    std::array<std::uint64_t, 4> v{};
    for (std::size_t i = 0; i < v.size(); ++i) v[i] = load_le64(problem.data() + 8 * i);
    const std::uint64_t a = remap(v[0]);
    const std::uint64_t b = remap(v[1]);
    std::uint64_t c_input = v[2];
    std::uint64_t c = remap(c_input);
    while ((b * b) % kMod == (4 * a * c) % kMod) c = remap(++c_input);
    const std::uint64_t w = remap(v[3]);

    std::array<std::uint32_t, 1024> coefficients{};
    for (std::uint64_t lane = 0; lane < 32; ++lane) {
        SipHashState sip{v[0], v[1], v[2], v[3]};
        sip.hash24((nonce / 32) * 32 + lane);
        for (std::size_t row = 0; row < 32; ++row) {
            sip.round();
            coefficients[row * 32 + lane] = static_cast<std::uint32_t>(sip.value()) %
                                                 static_cast<std::uint32_t>(kMod);
        }
    }

    std::uint64_t wpow = 1;
    std::uint64_t w2pow = 1;
    const std::uint64_t w2 = (w * w) % kMod;
    for (std::uint64_t i = 0; i < nonce % 32; ++i) {
        wpow = (wpow * w) % kMod;
        w2pow = (w2pow * w2) % kMod;
    }
    std::uint64_t full_wpow = wpow;
    std::uint64_t full_w2pow = w2pow;
    for (std::uint64_t i = nonce % 32; i < 32; ++i) {
        full_wpow = (full_wpow * w) % kMod;
        full_w2pow = (full_w2pow * w2) % kMod;
    }

    std::array<std::uint32_t, 32> polynomial{};
    std::uint64_t result = 0;
    for (std::size_t lane = 0; lane < 32; ++lane) {
        const std::uint64_t x = (a * w2pow + b * wpow + c) % kMod;
        std::uint64_t value = 0;
        for (std::size_t i = coefficients.size(); i != 0; --i)
            value = (value * x + coefficients[i - 1]) % kMod;
        polynomial[lane] = static_cast<std::uint32_t>(value);
        result = fnv64(result, value);
        wpow = (wpow * full_wpow) % kMod;
        w2pow = (w2pow * full_w2pow) % kMod;
    }

    std::array<std::uint8_t, 40> seed_input{};
    for (std::size_t i = 0; i < problem.size(); ++i) seed_input[i] = problem[i];
    store_le64(seed_input.data() + problem.size(), result);
    const Node seed = keccak512(seed_input.data(), seed_input.size());
    std::array<std::uint32_t, 64> mix{};
    for (std::size_t i = 0; i < mix.size(); ++i) mix[i] = load_le32(seed.data() + 4 * (i % 16));
    const std::uint64_t pages = full_size / 256;
    for (std::uint32_t access = 0; access < 32; ++access) {
        const std::uint64_t page = fnv32(load_le32(seed.data()) ^ access ^ polynomial[access],
                                         mix[access]) % pages;
        for (std::uint32_t node = 0; node < 4; ++node) {
            const Node item = dataset_item(cache, static_cast<std::uint32_t>(page * 4 + node));
            for (std::size_t word = 0; word < 16; ++word)
                mix[node * 16 + word] = fnv32(mix[node * 16 + word],
                                              load_le32(item.data() + 4 * word));
        }
    }

    std::array<std::uint8_t, 32> compressed{};
    for (std::size_t i = 0; i < 8; ++i) {
        std::uint32_t left = mix[4 * i];
        std::uint32_t right = mix[4 * (8 + i)];
        for (std::size_t j = 1; j < 4; ++j) {
            left = fnv32(left, mix[4 * i + j]);
            right = fnv32(right, mix[4 * (8 + i) + j]);
        }
        store_le32(compressed.data() + 4 * i, fnv32(left, right));
    }
    std::array<std::uint8_t, 96> final_input{};
    for (std::size_t i = 0; i < seed.size(); ++i) final_input[i] = seed[i];
    for (std::size_t i = 0; i < compressed.size(); ++i) final_input[seed.size() + i] = compressed[i];
    return keccak256(final_input.data(), final_input.size());
}

struct U256 {
    std::array<std::uint64_t, 4> limb{}; // Little-endian limbs.
};

static U256 from_be(const Hash256& bytes) {
    U256 value;
    for (std::size_t i = 0; i < 4; ++i) {
        std::uint64_t limb = 0;
        for (std::size_t j = 0; j < 8; ++j) limb = (limb << 8U) | bytes[8 * (3 - i) + j];
        value.limb[i] = limb;
    }
    return value;
}

static Hash256 to_be(const U256& value) {
    Hash256 bytes{};
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 8; ++j)
            bytes[8 * (3 - i) + 7 - j] = static_cast<std::uint8_t>(value.limb[i] >> (8U * j));
    return bytes;
}

static bool less(const U256& a, const U256& b) {
    for (std::size_t i = 4; i != 0; --i)
        if (a.limb[i - 1] != b.limb[i - 1]) return a.limb[i - 1] < b.limb[i - 1];
    return false;
}

static bool equal(const U256& a, const U256& b) { return a.limb == b.limb; }

static U256 subtract_mod(const U256& a, const U256& b) {
    U256 out;
    std::uint64_t borrow = 0;
    for (std::size_t i = 0; i < 4; ++i) {
        const std::uint64_t first = a.limb[i] - b.limb[i];
        const std::uint64_t borrow1 = a.limb[i] < b.limb[i] ? 1U : 0U;
        out.limb[i] = first - borrow;
        const std::uint64_t borrow2 = first < borrow ? 1U : 0U;
        borrow = borrow1 | borrow2;
    }
    return out;
}

static bool increment(U256& value) {
    for (std::uint64_t& limb : value.limb) {
        ++limb;
        if (limb != 0) return false;
    }
    return true;
}

static U256 lower_bound(U256 nonce) {
    nonce.limb[0] = 0;
    nonce.limb[1] = 0;
    nonce.limb[3] &= 0x7fffffffffffffffULL;
    return nonce;
}

static U256 boundary_from_scalar(std::uint64_t divisor) {
    require(divisor != 0, "zero target divisor");
    U256 quotient;
    if (divisor == 1) {
        quotient.limb.fill(std::numeric_limits<std::uint64_t>::max());
        return quotient;
    }
    require(divisor <= std::numeric_limits<std::uint64_t>::max() / 2, "scalar divisor too large");
    std::uint64_t remainder = 0;
    for (int bit = 256; bit >= 0; --bit) {
        remainder = remainder * 2 + (bit == 256 ? 1U : 0U);
        if (remainder >= divisor) {
            remainder -= divisor;
            if (bit < 256) quotient.limb[static_cast<std::size_t>(bit) / 64] |=
                               1ULL << (static_cast<unsigned>(bit) % 64U);
        }
    }
    return quotient;
}

static U256 nonce_value(std::uint64_t nonce) { return U256{{nonce, 0, 0, 0}}; }

static U256 target_value(const Hash256& hash, const U256& nonce) {
    return subtract_mod(from_be(hash), lower_bound(nonce));
}

static bool valid_pow(const Hash256& hash, const U256& nonce, const U256& boundary) {
    return less(target_value(hash, nonce), boundary);
}

static bool has_quality(const Hash256& hash, const U256& nonce, std::uint64_t quality) {
    U256 denominator = target_value(hash, nonce);
    if (increment(denominator)) return quality == 1;
    const U256 lower = boundary_from_scalar(quality + 1);
    const U256 upper = boundary_from_scalar(quality);
    return less(lower, denominator) && (less(denominator, upper) || equal(denominator, upper));
}

int main() try {
    const Bytes empty;
    require(to_hex(keccak256(empty)) ==
                "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470",
            "Keccak-256 self-test failed");
    require(to_hex(keccak512(empty.data(), empty.size())) ==
                "0eab42de4c3ceb9235fc91acffe746b29c29a8c366b7c60e4e67c466f36a4304"
                "c00fa9caf9d87976ba469bcbe06713b435f091ef2769fb160cdab33d3670680e",
            "Keccak-512 self-test failed");

    HeaderVector header;
    const Bytes problem_rlp = encode_header(header, false);
    const Hash256 problem = keccak256(problem_rlp);
    require(problem_rlp.size() == 316, "unexpected problem-header RLP length");
    require(to_hex(problem) == "8c03f99c72afad07ad0fe89d28c83a6aad5118558d52a2800fcfaec46bd5d97f",
            "problem-header RLP/Keccak mismatch");
    const Bytes full_header = encode_header(header, true);
    require(to_hex(keccak256(full_header)) ==
                "029130a0617e10df8e1bcb032b9a9fe2937b117314535e05eb8c93d46b20b143",
            "accepted block-header RLP/Keccak mismatch");

    constexpr std::uint64_t stage = 190;
    require(cache_size(stage) == 29228608, "stage-190 cache size mismatch");
    require(dataset_size(stage) == 7482636032ULL, "stage-190 dataset size mismatch");
    const U256 boundary = boundary_from_scalar(header.difficulty);
    require(to_hex(to_be(boundary)) ==
                "00000000008c527378f8e567fbad0605ee3c235a3b61d3bf7ccf51a8ce43f25b",
            "difficulty boundary mismatch");
    const U256 synthetic_nonce{{1, 2, 3, 0xffffffffffffffffULL}};
    require(lower_bound(synthetic_nonce).limb ==
                std::array<std::uint64_t, 4>{{0, 0, 3, 0x7fffffffffffffffULL}},
            "nonce lower-bound layout mismatch");

    const std::vector<Node> cache = make_cache(stage);
    const Hash256 accepted = octopus_hash(problem, header.nonce, cache, dataset_size(stage));
    // Derived here, then locked only after the target and recorded-quality checks below passed.
    require(to_hex(accepted) == "000000000013df09cedb71556c0744d461ab9acb3cce92a4b4ec924be585e704",
            "final pow_hash mismatch");
    require(valid_pow(accepted, nonce_value(header.nonce), boundary), "accepted nonce misses target");
    require(has_quality(accepted, nonce_value(header.nonce), 0xce2083c3a2fULL),
            "accepted nonce quality mismatch");

    const std::uint64_t bad_nonce = header.nonce + 1;
    const Hash256 nonce_tamper = octopus_hash(problem, bad_nonce, cache, dataset_size(stage));
    require(!valid_pow(nonce_tamper, nonce_value(bad_nonce), boundary), "tampered nonce passed target");
    ++header.timestamp;
    const Hash256 field_problem = keccak256(encode_header(header, false));
    require(field_problem != problem, "tampered field preserved problem hash");
    const Hash256 field_tamper = octopus_hash(field_problem, header.nonce, cache, dataset_size(stage));
    require(!valid_pow(field_tamper, nonce_value(header.nonce), boundary), "tampered field passed target");

    std::cout << "octopus_conflux: pow_hash=0x" << to_hex(accepted)
              << " stage=190 cache_bytes=" << cache_size(stage) << " PASS\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "octopus_conflux: FAIL: " << error.what() << '\n';
    return 1;
}
