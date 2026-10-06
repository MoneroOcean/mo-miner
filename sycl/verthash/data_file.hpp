// Copyright GNU GPLv3 (c) 2026 MoneroOcean <support@moneroocean.stream>
// Verthash data generation follows the GPLv2+ reference implementation from
// github.com/CryptoGraphics/VerthashMiner, rewritten to use bounded C++ storage and checked I/O.

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace mom_verthash::data_file {

using Node = std::array<std::uint8_t, 32>;

inline constexpr std::uint64_t expected_bytes = 1283457024;
inline constexpr unsigned expected_index = 17;
inline constexpr std::array<std::uint8_t, 32> expected_sha256 = {
    0xa5, 0x55, 0x31, 0xe8, 0x43, 0xcd, 0x56, 0xb0, 0x10, 0x11, 0x4a, 0xaf, 0x63, 0x25, 0xb0, 0xd5,
    0x29, 0xec, 0xf8, 0x8f, 0x8a, 0xd4, 0x76, 0x39, 0xb6, 0xed, 0xed, 0xaf, 0xd7, 0x21, 0xaa, 0x48};

inline void keccak(std::uint64_t state[25]) {
  static constexpr std::uint64_t constants[24] = {
      0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL,
      0x8000000080008000ULL, 0x000000000000808bULL, 0x0000000080000001ULL,
      0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008aULL,
      0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
      0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL,
      0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
      0x000000000000800aULL, 0x800000008000000aULL, 0x8000000080008081ULL,
      0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL};
  static constexpr unsigned rotations[24] = {1,  3,  6,  10, 15, 21, 28, 36, 45, 55, 2,  14,
                                              27, 41, 56, 8,  25, 43, 62, 18, 39, 61, 20, 44};
  static constexpr unsigned positions[24] = {10, 7,  11, 17, 18, 3, 5,  16, 8,  21, 24, 4,
                                              15, 23, 19, 13, 12, 2, 20, 14, 22, 9,  6,  1};
  for (unsigned round = 0; round < 24; ++round) {
    std::uint64_t column[5];
    for (unsigned i = 0; i < 5; ++i)
      column[i] = state[i] ^ state[i + 5] ^ state[i + 10] ^ state[i + 15] ^ state[i + 20];
    for (unsigned i = 0; i < 5; ++i) {
      const std::uint64_t value = column[(i + 4) % 5] ^
                                  ((column[(i + 1) % 5] << 1) | (column[(i + 1) % 5] >> 63));
      for (unsigned j = 0; j < 25; j += 5)
        state[j + i] ^= value;
    }
    std::uint64_t value = state[1];
    for (unsigned i = 0; i < 24; ++i) {
      const unsigned position = positions[i];
      column[0] = state[position];
      state[position] = (value << rotations[i]) | (value >> (64 - rotations[i]));
      value = column[0];
    }
    for (unsigned j = 0; j < 25; j += 5) {
      for (unsigned i = 0; i < 5; ++i)
        column[i] = state[j + i];
      for (unsigned i = 0; i < 5; ++i)
        state[j + i] ^= (~column[(i + 1) % 5]) & column[(i + 2) % 5];
    }
    state[0] ^= constants[round];
  }
}

inline Node sha3_256(const std::uint8_t* input, const std::size_t size) {
  if (size >= 136)
    throw std::runtime_error("Verthash generator SHA3 input is too large");
  std::array<std::uint8_t, 200> bytes{};
  std::memcpy(bytes.data(), input, size);
  bytes[size] = 0x06;
  bytes[135] |= 0x80;
  std::uint64_t state[25]{};
  for (unsigned word = 0; word < 25; ++word)
    for (unsigned byte = 0; byte < 8; ++byte)
      state[word] |= static_cast<std::uint64_t>(bytes[word * 8 + byte]) << (byte * 8);
  keccak(state);
  Node result{};
  for (unsigned byte = 0; byte < result.size(); ++byte)
    result[byte] = static_cast<std::uint8_t>(state[byte / 8] >> ((byte % 8) * 8));
  return result;
}

class Sha256 {
public:
  void update(const std::uint8_t* input, std::size_t size) {
    total_bytes_ += size;
    while (size) {
      const std::size_t count = std::min(size, block_.size() - buffered_);
      std::memcpy(block_.data() + buffered_, input, count);
      buffered_ += count;
      input += count;
      size -= count;
      if (buffered_ == block_.size()) {
        transform(block_.data());
        buffered_ = 0;
      }
    }
  }

  Node finish() {
    const std::uint64_t bits = total_bytes_ * 8;
    block_[buffered_++] = 0x80;
    if (buffered_ > 56) {
      std::fill(block_.begin() + static_cast<std::ptrdiff_t>(buffered_), block_.end(), 0);
      transform(block_.data());
      buffered_ = 0;
    }
    std::fill(block_.begin() + static_cast<std::ptrdiff_t>(buffered_), block_.begin() + 56, 0);
    for (unsigned i = 0; i < 8; ++i)
      block_[63 - i] = static_cast<std::uint8_t>(bits >> (i * 8));
    transform(block_.data());
    Node result{};
    for (unsigned i = 0; i < 8; ++i) {
      result[i * 4] = static_cast<std::uint8_t>(state_[i] >> 24);
      result[i * 4 + 1] = static_cast<std::uint8_t>(state_[i] >> 16);
      result[i * 4 + 2] = static_cast<std::uint8_t>(state_[i] >> 8);
      result[i * 4 + 3] = static_cast<std::uint8_t>(state_[i]);
    }
    return result;
  }

private:
  static std::uint32_t rotate(const std::uint32_t value, const unsigned bits) {
    return (value >> bits) | (value << (32 - bits));
  }

  void transform(const std::uint8_t* input) {
    static constexpr std::uint32_t constants[64] = {
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
    std::uint32_t words[64];
    for (unsigned i = 0; i < 16; ++i)
      words[i] = static_cast<std::uint32_t>(input[i * 4]) << 24 |
                 static_cast<std::uint32_t>(input[i * 4 + 1]) << 16 |
                 static_cast<std::uint32_t>(input[i * 4 + 2]) << 8 | input[i * 4 + 3];
    for (unsigned i = 16; i < 64; ++i) {
      const std::uint32_t s0 = rotate(words[i - 15], 7) ^ rotate(words[i - 15], 18) ^
                               (words[i - 15] >> 3);
      const std::uint32_t s1 = rotate(words[i - 2], 17) ^ rotate(words[i - 2], 19) ^
                               (words[i - 2] >> 10);
      words[i] = words[i - 16] + s0 + words[i - 7] + s1;
    }
    std::uint32_t a = state_[0];
    std::uint32_t b = state_[1];
    std::uint32_t c = state_[2];
    std::uint32_t d = state_[3];
    std::uint32_t e = state_[4];
    std::uint32_t f = state_[5];
    std::uint32_t g = state_[6];
    std::uint32_t h = state_[7];
    for (unsigned i = 0; i < 64; ++i) {
      const std::uint32_t s1 = rotate(e, 6) ^ rotate(e, 11) ^ rotate(e, 25);
      const std::uint32_t choice = (e & f) ^ (~e & g);
      const std::uint32_t first = h + s1 + choice + constants[i] + words[i];
      const std::uint32_t s0 = rotate(a, 2) ^ rotate(a, 13) ^ rotate(a, 22);
      const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
      const std::uint32_t second = s0 + majority;
      h = g;
      g = f;
      f = e;
      e = d + first;
      d = c;
      c = b;
      b = a;
      a = first + second;
    }
    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
  }

  std::array<std::uint32_t, 8> state_ = {
      0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
      0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
  std::array<std::uint8_t, 64> block_{};
  std::size_t buffered_ = 0;
  std::uint64_t total_bytes_ = 0;
};

inline Node sha256(const std::uint8_t* input, const std::size_t size) {
  Sha256 hash;
  hash.update(input, size);
  return hash.finish();
}

inline Node sha256_file(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input)
    throw std::runtime_error("Cannot read Verthash data file: " + path.string());
  Sha256 hash;
  std::array<std::uint8_t, 1024 * 1024> buffer{};
  while (input) {
    input.read(reinterpret_cast<char*>(buffer.data()), buffer.size());
    const std::streamsize count = input.gcount();
    if (count > 0)
      hash.update(buffer.data(), static_cast<std::size_t>(count));
  }
  if (!input.eof())
    throw std::runtime_error("Cannot verify Verthash data file: " + path.string());
  return hash.finish();
}

inline constexpr std::uint64_t graph_nodes(const unsigned index) {
  return index >= 1 && index <= expected_index
             ? (std::uint64_t{1} << index) * (index + 1) * index
             : 0;
}

static_assert(graph_nodes(expected_index) * Node{}.size() == expected_bytes);

inline Node varint(const std::int64_t value) {
  Node result{};
  std::uint64_t encoded = static_cast<std::uint64_t>(value) << 1;
  if (value < 0)
    encoded = ~encoded;
  unsigned index = 0;
  while (encoded >= 0x80) {
    result[index++] = static_cast<std::uint8_t>(encoded) | 0x80;
    encoded >>= 7;
  }
  result[index] = static_cast<std::uint8_t>(encoded);
  return result;
}

class Graph {
public:
  Graph(const std::filesystem::path& path, const unsigned index, const Node& public_key)
      : file_(path, std::ios::binary | std::ios::in | std::ios::out | std::ios::trunc),
        index_(index), public_key_(public_key) {
    if (!file_)
      throw std::runtime_error("Cannot create Verthash data file: " + path.string());
    if (graph_nodes(index_) == 0)
      throw std::runtime_error("Unsupported Verthash data index");
    std::uint64_t power = 1;
    while (power <= graph_nodes(index_))
      power <<= 1;
    power_ = power;
  }

  void generate() {
    std::uint64_t count = power_;
    const std::uint64_t width = std::uint64_t{1} << index_;
    for (std::uint64_t i = 0; i < width; ++i) {
      write_node(count, make_node(count));
      ++count;
    }
    if (index_ == 1) {
      butterfly(1, count);
      file_.flush();
      if (!file_)
        throw std::runtime_error("Failed writing Verthash data file");
      return;
    }

    std::vector<unsigned> indices(5, index_);
    std::vector<unsigned> graphs = {4, 3, 2, 1, 0};
    while (!indices.empty() && !graphs.empty()) {
      const unsigned index = indices.back();
      const unsigned graph = graphs.back();
      indices.pop_back();
      graphs.pop_back();
      const std::uint64_t inner_width = std::uint64_t{1} << index;
      const std::uint64_t half = inner_width / 2;

      if (graph == 0) {
        const std::uint64_t sources = count - inner_width;
        for (std::uint64_t i = 0; i < half; ++i) {
          const Node left = read_node(sources + i);
          const Node right = read_node(sources + i + half);
          write_node(count, make_node(count, &left, &right));
          ++count;
        }
      } else if (graph < 4) {
        const std::uint64_t first = count;
        for (std::uint64_t i = 0; i < half; ++i) {
          const Node parent = read_node(first - half + i);
          write_node(count, make_node(first + i, &parent));
          ++count;
        }
      } else {
        const std::uint64_t sinks = count;
        const std::uint64_t sources = sinks + inner_width - graph_nodes(index);
        for (std::uint64_t i = 0; i < half; ++i) {
          const Node parent = read_node(sinks - half + i);
          const Node left = read_node(sources + i);
          const Node right = read_node(sources + i + half);
          write_node(sinks + i, make_node(sinks + i, &parent, &left));
          write_node(sinks + i + half, make_node(sinks + i + half, &parent, &right));
          count += 2;
        }
      }

      if (graph == 0 || graph == 3 || ((graph == 1 || graph == 2) && index == 2)) {
        butterfly(index - 1, count);
      } else if (graph == 1 || graph == 2) {
        indices.insert(indices.end(), 5, index - 1);
        graphs.insert(graphs.end(), {4, 3, 2, 1, 0});
      }
    }
    file_.flush();
    if (!file_)
      throw std::runtime_error("Failed writing Verthash data file");
  }

private:
  std::uint64_t stored_id(const std::uint64_t id) const {
    return id & ~power_;
  }

  Node read_node(const std::uint64_t id) {
    Node result{};
    file_.clear();
    file_.seekg(static_cast<std::streamoff>(stored_id(id) * result.size()));
    file_.read(reinterpret_cast<char*>(result.data()), result.size());
    if (!file_)
      throw std::runtime_error("Failed reading temporary Verthash graph");
    return result;
  }

  void write_node(const std::uint64_t id, const Node& value) {
    file_.clear();
    file_.seekp(static_cast<std::streamoff>(stored_id(id) * value.size()));
    file_.write(reinterpret_cast<const char*>(value.data()), value.size());
    if (!file_)
      throw std::runtime_error("Failed writing temporary Verthash graph");
  }

  Node make_node(const std::uint64_t id, const Node* first = nullptr,
                 const Node* second = nullptr) const {
    std::array<std::uint8_t, 128> input{};
    std::size_t size = 0;
    auto append = [&](const Node& value) {
      std::memcpy(input.data() + size, value.data(), value.size());
      size += value.size();
    };
    append(public_key_);
    append(varint(static_cast<std::int64_t>(id)));
    if (first)
      append(*first);
    if (second)
      append(*second);
    return sha3_256(input.data(), size);
  }

  void butterfly(unsigned index, std::uint64_t& count) {
    if (index == 0)
      index = 1;
    const std::uint64_t width = std::uint64_t{1} << index;
    const std::uint64_t begin = count - width;
    for (unsigned level = 1; level < 2 * index; ++level) {
      const unsigned shift = level > index ? level - index : index - level;
      for (std::uint64_t i = 0; i < width; ++i) {
        const std::uint64_t previous = ((i >> shift) & 1) == 0
                                           ? i + (std::uint64_t{1} << shift)
                                           : i - (std::uint64_t{1} << shift);
        const Node first = read_node(begin + (level - 1) * width + previous);
        const Node second = read_node(count - width);
        write_node(count, make_node(count, &first, &second));
        ++count;
      }
    }
  }

  std::fstream file_;
  unsigned index_;
  std::uint64_t power_ = 0;
  Node public_key_;
};

inline bool has_expected_size(const std::filesystem::path& path) {
  std::error_code error;
  return std::filesystem::is_regular_file(path, error) && !error &&
         std::filesystem::file_size(path, error) == expected_bytes && !error;
}

inline bool has_expected_hash(const std::filesystem::path& path) {
  return has_expected_size(path) && sha256_file(path) == expected_sha256;
}

inline std::filesystem::path temporary_path(const std::filesystem::path& path) {
  static std::atomic<std::uint64_t> sequence{0};
  std::random_device random;
  for (unsigned attempt = 0; attempt < 32; ++attempt) {
    const std::uint64_t entropy =
        (static_cast<std::uint64_t>(random()) << 32) ^ random() ^
        static_cast<std::uint64_t>(std::chrono::high_resolution_clock::now()
                                       .time_since_epoch().count()) ^
        sequence.fetch_add(1, std::memory_order_relaxed);
    std::filesystem::path temporary = path;
    temporary += ".tmp." + std::to_string(entropy);
    std::error_code error;
    const bool exists = std::filesystem::exists(temporary, error);
    if (error)
      throw std::runtime_error("Cannot inspect Verthash temporary path: " + error.message());
    if (!exists)
      return temporary;
  }
  throw std::runtime_error("Cannot allocate a unique Verthash temporary path");
}

inline void generate(const std::filesystem::path& path, const unsigned index = expected_index) {
  if (graph_nodes(index) == 0)
    throw std::runtime_error("Unsupported Verthash data index");
  const std::filesystem::path parent = path.parent_path();
  if (!parent.empty()) {
    std::error_code error;
    std::filesystem::create_directories(parent, error);
    if (error)
      throw std::runtime_error("Cannot create Verthash data directory: " + error.message());
  }
  const auto complete = [&] {
    if (index == expected_index)
      return has_expected_hash(path);
    std::error_code error;
    return std::filesystem::is_regular_file(path, error) && !error &&
           std::filesystem::file_size(path, error) == graph_nodes(index) * Node{}.size() && !error;
  };
  if (complete())
    return;
  std::error_code target_error;
  const bool target_exists = std::filesystem::exists(path, target_error);
  if (target_error)
    throw std::runtime_error("Cannot access Verthash data file: " + target_error.message());
  if (target_exists && complete())
    return;
  if (target_exists)
    throw std::runtime_error("Invalid Verthash data file; remove it to regenerate: " +
                             path.string());

  // Concurrent first runs may duplicate work, but each writes a private verified temporary and
  // reuses a valid file installed by another process instead of sharing mutable generation state.
  const std::filesystem::path temporary = temporary_path(path);
  try {
    static constexpr std::array<std::uint8_t, 32> seed = {
        'V','e','r','t','h','a','s','h',' ','P','r','o','o','f','-','o','f','-','S','p','a','c','e',' ',
        'D','a','t','a','f','i','l','e'};
    const Node public_key = sha3_256(seed.data(), seed.size());
    {
      Graph graph(temporary, index, public_key);
      graph.generate();
    }
    const std::uint64_t bytes = graph_nodes(index) * Node{}.size();
    if (std::filesystem::file_size(temporary) != bytes ||
        (index == expected_index && sha256_file(temporary) != expected_sha256))
      throw std::runtime_error("Generated Verthash data failed verification");
    if (complete()) {
      std::error_code ignored;
      std::filesystem::remove(temporary, ignored);
      return;
    }
    std::error_code rename_error;
    std::filesystem::rename(temporary, path, rename_error);
    if (rename_error) {
      if (complete()) {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        return;
      }
      throw std::runtime_error("Cannot install Verthash data file: " + rename_error.message());
    }
  } catch (...) {
    std::error_code ignored;
    std::filesystem::remove(temporary, ignored);
    throw;
  }
}

inline std::filesystem::path selection() {
  if (const char* configured = std::getenv("MOM_VERTHASH_DATA"); configured && *configured)
    return configured;
  return "verthash.dat";
}

} // namespace mom_verthash::data_file
