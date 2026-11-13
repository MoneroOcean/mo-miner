// Copyright GNU GPLv3 (c) 2026 MoneroOcean <support@moneroocean.stream>
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string_view>
#include <vector>
#include <zlib.h>

namespace mom {

// Own decompressed include text for the complete lifetime of every NVRTC header pointer. There
// is deliberately no filesystem resolver or extraction fallback. The build payload is immutable,
// but validate framing, bounds and names too so corrupt executables fail before entering a JIT.
class JitHeaders {
  std::vector<char> bytes_;
  std::vector<const char *> names_, sources_;
  static bool valid_name(std::string_view name) {
    if (name.empty() || name.size() > 255 || name.front() == '/' || name.back() == '/') {return false;}
    size_t start = 0;
    for (size_t i = 0; i <= name.size(); ++i) {
      if (i == name.size() || name[i] == '/') {
        const auto part = name.substr(start, i - start);
        if (part.empty() || part == "." || part == "..") {return false;} start = i + 1;
      } else {
        const unsigned char c = static_cast<unsigned char>(name[i]);
        if (!((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') ||
              c == '_' || c == '.' || c == '+' || c == '-')) {return false;}
      }
    }
    return name.front() != '.' && name.front() != '+' && name.front() != '-';
  }

public:
  JitHeaders(const JitHeaders &) = delete;
  JitHeaders &operator=(const JitHeaders &) = delete;
  explicit JitHeaders(std::string_view compressed, size_t size) {
    constexpr size_t limit = 64U << 20;
    if (size < 12 || size > limit || compressed.empty() || compressed.size() > limit) {throw std::runtime_error("JIT header size bounds");}
    bytes_.resize(size);
    z_stream stream{};
    stream.next_in = reinterpret_cast<Bytef *>(const_cast<char *>(compressed.data()));
    stream.avail_in = static_cast<uInt>(compressed.size());
    stream.next_out = reinterpret_cast<Bytef *>(bytes_.data()); stream.avail_out = static_cast<uInt>(size);
    if (inflateInit2(&stream, 15 + 16) != Z_OK) {throw std::runtime_error("JIT header inflate initialization");}
    struct End {z_stream &stream; ~End() {inflateEnd(&stream);}} end{stream};
    if (inflate(&stream, Z_FINISH) != Z_STREAM_END || stream.total_out != size || stream.total_in != compressed.size()) {
      throw std::runtime_error("Corrupt JIT header compression");
    }
    const char *cursor = bytes_.data(), *const stop = cursor + bytes_.size();
    if (std::memcmp(cursor, "MOMHDR01", 8)) {throw std::runtime_error("Invalid JIT header format");} cursor += 8;
    const auto integer = [&]() -> uint32_t {
      if (stop - cursor < 4) {throw std::runtime_error("Truncated JIT header length");}
      const auto *p = reinterpret_cast<const unsigned char *>(cursor); cursor += 4;
      return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
    };
    const auto text = [&](size_t length) -> const char * {
      if (length >= static_cast<size_t>(stop - cursor) || cursor[length] || std::memchr(cursor, 0, length)) {
        throw std::runtime_error("Invalid JIT header byte range");
      }
      const auto *value = cursor; cursor += length + 1; return value;
    };
    const auto count = integer();
    if (!count || count > 16384 || count > static_cast<size_t>(stop - cursor) / 11) {throw std::runtime_error("JIT header count bounds");}
    names_.reserve(count); sources_.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
      const auto name_size = integer(), data_size = integer();
      const char *name = text(name_size), *source = text(data_size);
      if (!valid_name({name, name_size}) || (!names_.empty() && std::strcmp(names_.back(), name) >= 0)) {
        throw std::runtime_error("Invalid or unordered JIT header name");
      }
      names_.push_back(name); sources_.push_back(source);
    }
    if (cursor != stop) {throw std::runtime_error("Trailing JIT header bytes");}
  }
  int count() const {return static_cast<int>(names_.size());}
  const char *const *names() const {return names_.data();}
  const char *const *sources() const {return sources_.data();}
  std::string_view find(std::string_view name) const {
    const auto found = std::lower_bound(names_.begin(), names_.end(), name,
        [](const char *entry, std::string_view key) {return std::string_view(entry) < key;});
    return found != names_.end() && std::string_view(*found) == name ? sources_[found - names_.begin()] : std::string_view{};
  }
};

} // namespace mom
