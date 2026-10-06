#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string_view>
#include <type_traits>

namespace mom::job_boundary {

template <typename UInt>
inline bool parse_unsigned(std::string_view text, UInt& out, int base = 10,
                           bool allow_hex_prefix = false) {
  static_assert(std::is_integral_v<UInt> && std::is_unsigned_v<UInt> &&
                !std::is_same_v<UInt, bool>,
                "UInt must be an unsigned integer type");
  if (base < 2 || base > 36)
    return false;
  std::size_t begin = 0;
  if (allow_hex_prefix && base == 16 && text.size() >= 2 && text[0] == '0' &&
      (text[1] == 'x' || text[1] == 'X'))
    begin = 2;
  if (begin == text.size())
    return false;
  const UInt radix = static_cast<UInt>(base);
  const UInt maximum = std::numeric_limits<UInt>::max();
  UInt value = 0;
  for (std::size_t i = begin; i < text.size(); ++i) {
    const unsigned char character = static_cast<unsigned char>(text[i]);
    unsigned digit = 0;
    if (character >= static_cast<unsigned char>('0') &&
        character <= static_cast<unsigned char>('9'))
      digit = static_cast<unsigned>(character - static_cast<unsigned char>('0'));
    else if (character >= static_cast<unsigned char>('a') &&
             character <= static_cast<unsigned char>('z'))
      digit = static_cast<unsigned>(character - static_cast<unsigned char>('a')) + 10;
    else if (character >= static_cast<unsigned char>('A') &&
             character <= static_cast<unsigned char>('Z'))
      digit = static_cast<unsigned>(character - static_cast<unsigned char>('A')) + 10;
    else
      return false;
    if (digit >= static_cast<unsigned>(base))
      return false;
    const UInt digit_value = static_cast<UInt>(digit);
    if (value > (maximum - digit_value) / radix)
      return false;
    value = value * radix + digit_value;
  }
  out = value;
  return true;
}

inline bool split_once(std::string_view text, char delimiter, std::string_view& left,
                       std::string_view& right) {
  const std::size_t position = text.find(delimiter);
  if (position == std::string_view::npos || position == 0 || position + 1 == text.size())
    return false;
  if (text.find(delimiter, position + 1) != std::string_view::npos)
    return false;
  left = text.substr(0, position);
  right = text.substr(position + 1);
  return true;
}

inline bool nonce_big_endian(std::string_view algo, std::size_t blob_size) {
  return (algo == "fishhash" && blob_size == 180) || algo == "xelishashv3" ||
    algo == "nexapow" || algo == "beamhash3";
}

inline std::size_t fishhash_nonce_offset(std::size_t blob_size) {
  if (blob_size == 40)
    return 32;
  if (blob_size == 180)
    return 172;
  return std::numeric_limits<std::size_t>::max();
}

inline bool valid_fishhash_layout(std::size_t blob_size, std::size_t offset) {
  const std::size_t expected = fishhash_nonce_offset(blob_size);
  return expected != std::numeric_limits<std::size_t>::max() && offset == expected;
}

inline bool valid_karlsenhashv2_layout(std::size_t blob_size, std::size_t offset) {
  return blob_size == 80 && offset == 72;
}

inline bool valid_nexapow_layout(std::size_t blob_size, std::size_t offset) {
  return (blob_size == 40 && offset == 32) || (blob_size == 44 && offset == 36) ||
    (blob_size == 48 && offset == 40);
}

inline bool valid_cn_gpu_input(std::size_t size) {
  return size > 0 && size <= 135;
}

inline bool valid_beamhash3_input(std::size_t size) {
  return size == 44;
}

inline bool padded_u32_range_fits(std::uint32_t count, std::uint32_t alignment) {
  return alignment != 0 &&
    count <= std::numeric_limits<std::uint32_t>::max() - (alignment - 1);
}

inline bool valid_pearlhash_shape(std::uint64_t m, std::uint64_t n, std::uint64_t k,
                                 std::uint64_t rank) {
  const bool rank_power_of_two = rank && (rank & (rank - 1)) == 0;
  constexpr std::uint64_t max_dimension = std::uint64_t{1} << 24;
  constexpr std::uint64_t max_k = std::uint64_t{1} << 16;
  if (m < 128 || m > max_dimension || m % 32 ||
      n < 128 || n > max_dimension || n % 32 ||
      k < 1024 || k > max_k || k % 64 || rank < 128 || rank > 1024 ||
      !rank_power_of_two || k < 16 * rank || k > 4 * rank * rank) {
    return false;
  }
  const std::uint64_t tile_rows = m / 16;
  const std::uint64_t tile_columns = n / 16;
  const std::uint64_t maximum = std::numeric_limits<std::uint64_t>::max();
  const std::uint64_t signed_max = std::numeric_limits<std::int32_t>::max();
  return m <= signed_max / rank && n <= signed_max / rank && m <= signed_max / k &&
      tile_rows <= signed_max / tile_columns && n <= maximum / m && m * n <= maximum / k;
}

// For non-test C29 dispatches, 0 means a new graph, 1 recovers an earlier proof, and -1 is drained.
inline constexpr std::uint64_t c29_completed_graphs(int result) {
  return result == 0 ? 1 : 0;
}

inline bool nonce_overflowed(std::uint64_t previous, std::uint64_t next,
                             std::uint64_t protected_mask) {
  return previous > next || (previous & protected_mask) != (next & protected_mask);
}

template <typename UInt>
inline bool nonce_with_offset(UInt base, std::uint64_t index, std::uint64_t stride,
                              UInt protected_mask, UInt& result) {
  static_assert(std::is_integral_v<UInt> && std::is_unsigned_v<UInt> &&
                !std::is_same_v<UInt, bool>,
                "UInt must be an unsigned integer type");
  if (index && stride > std::numeric_limits<std::uint64_t>::max() / index)
    return false;
  const std::uint64_t offset = index * stride;
  const UInt maximum = std::numeric_limits<UInt>::max();
  if (offset > static_cast<std::uint64_t>(maximum - base))
    return false;
  const UInt candidate = static_cast<UInt>(base + static_cast<UInt>(offset));
  if (protected_mask && (base & protected_mask) != (candidate & protected_mask))
    return false;
  result = candidate;
  return true;
}

// PearlHash kernels consume a uint32_t seed even though the native ABI carries it in a uint64_t.
// Keep the worker/thread partition arithmetic wide, but refuse any starting point or successor
// that would leave the low-32-bit domain and repeat an earlier seed.
inline bool pearl_seed_start(std::uint32_t slot, std::uint32_t thread_id,
                             std::uint32_t thread_num, std::uint32_t stride,
                             std::uint32_t& start) {
  if (!thread_num || thread_id >= thread_num || !stride ||
      !nonce_with_offset<std::uint32_t>(slot, thread_id, stride, 0, start)) {
    return false;
  }
  return true;
}

inline bool pearl_seed_next(std::uint32_t current, std::uint64_t step,
                            std::uint32_t& next) {
  constexpr std::uint64_t maximum = std::numeric_limits<std::uint32_t>::max();
  if (!step || step > maximum || static_cast<std::uint64_t>(current) > maximum - step)
    return false;
  next = static_cast<std::uint32_t>(static_cast<std::uint64_t>(current) + step);
  return true;
}

template <typename UInt>
inline bool next_nonce_batch(UInt first, std::uint64_t count, std::uint64_t lane_stride,
                             std::uint64_t batch_stride, UInt protected_mask,
                             UInt& last, UInt& next) {
  if (!count || !nonce_with_offset(
        first, count - 1, lane_stride, protected_mask, last)) {
    return false;
  }
  UInt next_last = 0;
  return nonce_with_offset(first, 1, batch_stride, protected_mask, next) &&
    nonce_with_offset(next, count - 1, lane_stride, protected_mask, next_last);
}

template <typename UInt>
inline UInt load_nonce(const std::uint8_t* data, bool big_endian) {
  static_assert(std::is_integral_v<UInt> && std::is_unsigned_v<UInt> &&
                !std::is_same_v<UInt, bool>,
                "UInt must be an unsigned integer type");
  UInt value = 0;
  if (big_endian) {
    for (std::size_t i = 0; i < sizeof(UInt); ++i)
      value = static_cast<UInt>((value << 8) | static_cast<UInt>(data[i]));
    return value;
  }
  for (std::size_t i = 0; i < sizeof(UInt); ++i)
    value = static_cast<UInt>(value | (static_cast<UInt>(data[i]) << (i * 8)));
  return value;
}

template <typename UInt>
inline void store_nonce(std::uint8_t* data, UInt value, bool big_endian) {
  static_assert(std::is_integral_v<UInt> && std::is_unsigned_v<UInt> &&
                !std::is_same_v<UInt, bool>,
                "UInt must be an unsigned integer type");
  if (big_endian) {
    for (std::size_t i = 0; i < sizeof(UInt); ++i) {
      const std::size_t shift = (sizeof(UInt) - 1 - i) * 8;
      data[i] = static_cast<std::uint8_t>(value >> shift);
    }
    return;
  }
  for (std::size_t i = 0; i < sizeof(UInt); ++i)
    data[i] = static_cast<std::uint8_t>(value >> (i * 8));
}

} // namespace mom::job_boundary
