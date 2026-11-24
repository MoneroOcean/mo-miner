// Copyright GNU GPLv3 (c) 2023-2026 MoneroOcean <support@moneroocean.stream>

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string_view>
#include <type_traits>

namespace mom::job_boundary {

// Parse a complete unsigned number in `base` (2..36), rejecting empty text, stray characters and
// overflow; `out` is written only on success. Hex input may carry a 0x prefix when allowed.
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

// Split "left<delimiter>right" where both sides are non-empty and the delimiter occurs once.
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

// Protocol byte order of the embedded nonce: Iron Fish FishHash headers, XelisHashV3, NexaPow and
// BeamHash III store it big-endian; every other embedded-nonce layout is little-endian.
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

// Accept only PearlHash matrix profiles whose dimensions keep tile, transcript and proof indices
// inside the 32-bit signed ranges the kernels use, and whose M*N*K product fits uint64_t.
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

// A counter step overflowed when it wrapped or changed a pool-owned (masked) prefix bit.
inline bool nonce_overflowed(std::uint64_t previous, std::uint64_t next,
                             std::uint64_t protected_mask) {
  return previous > next || (previous & protected_mask) != (next & protected_mask);
}

// result = base + index * stride without wrapping UInt or changing the masked prefix bits.
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

// Visit each target hit a single-blob GPU entry point reported; sycl/lib.h places hit i at
// slot i of its nonce and hash arrays. A count outside [0, capacity] breaks that ABI, so it
// visits nothing rather than reading unwritten or out-of-bounds slots.
template <typename Visit>
inline bool for_each_gpu_hit(int hits, unsigned capacity, Visit&& visit) {
  if (hits < 0 || static_cast<unsigned>(hits) > capacity)
    return false;
  for (unsigned i = 0; i < static_cast<unsigned>(hits); ++i)
    visit(i);
  return true;
}

// For a batch of `count` nonces starting at `first` with `lane_stride` between lanes, set `last`
// to its final nonce and `next` to the following batch start (`first + batch_stride`). Returns
// false when this batch or the next one leaves the counter range; `last` is then still valid if
// only the next batch overflowed.
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

// Read a sizeof(UInt)-byte counter in the given byte order.
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

// Write a sizeof(UInt)-byte counter in the given byte order.
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

// Decode one ASCII hex digit.
inline bool hex_digit(const char character, unsigned& value) {
  if (character >= '0' && character <= '9')
    value = static_cast<unsigned>(character - '0');
  else if (character >= 'a' && character <= 'f')
    value = static_cast<unsigned>(character - 'a') + 10;
  else if (character >= 'A' && character <= 'F')
    value = static_cast<unsigned>(character - 'A') + 10;
  else
    return false;
  return true;
}

// Decode into hex.size() / 2 bytes. Reject a partial byte before writing to the output.
inline bool decode_hex(std::string_view hex, std::uint8_t* out) {
  if (hex.size() % 2 != 0)
    return false;
  for (std::size_t i = 0; i < hex.size(); i += 2) {
    unsigned high = 0;
    unsigned low = 0;
    if (!hex_digit(hex[i], high) || !hex_digit(hex[i + 1], low))
      return false;
    *out++ = static_cast<std::uint8_t>(high << 4 | low);
  }
  return true;
}

inline std::string_view strip_hex_prefix(std::string_view hex) {
  const bool prefixed = hex.size() >= 2 && hex[0] == '0' && (hex[1] == 'x' || hex[1] == 'X');
  return prefixed ? hex.substr(2) : hex;
}

// Equihash proof lengths fit the one- or three-byte form of Bitcoin CompactSize.
inline unsigned write_compact_size16(const std::uint16_t size, std::uint8_t* out) {
  if (size < 253) {
    out[0] = static_cast<std::uint8_t>(size);
    return 1;
  }
  out[0] = 253;
  out[1] = static_cast<std::uint8_t>(size);
  out[2] = static_cast<std::uint8_t>(size >> 8);
  return 3;
}

// Parse a nonzero 256-bit big-endian target of 2..64 hex digits (even length, no prefix),
// left-padding shorter values with zeros.
inline bool parse_big_target(std::string_view hex, std::uint8_t (&out)[32]) {
  if (hex.empty() || hex.size() > 64 || (hex.size() & 1))
    return false;
  std::fill(std::begin(out), std::end(out), std::uint8_t{0});
  if (!decode_hex(hex, out + 32 - hex.size() / 2))
    return false;
  return std::any_of(std::begin(out), std::end(out), [](std::uint8_t byte) { return byte != 0; });
}

// A pool share target as the workers compare it. Algorithms whose kernels test the full 256-bit
// value use `bytes` (word is then 1, marking an active target); the CryptoNight, RandomX, C29 and
// KawPoW paths compare one 64-bit word against the hash's most-significant word.
struct ShareTarget {
  bool full = false;
  std::uint64_t word = 0;
  std::uint8_t bytes[32]{};
};

inline bool uses_full_target(std::string_view algo) {
  return algo == "etchash" || algo == "octopus" || algo == "autolykos2" || algo == "pearlhash" ||
    algo == "fishhash" || algo == "c30" || algo == "equihash192_7" || algo == "zelhash" ||
    algo == "zhash" || algo == "beamhash3" || algo == "karlsenhashv2" || algo == "hoohash" ||
    algo == "walahash" || algo == "verthash" || algo == "xelishashv3" || algo == "nexapow";
}

// Parse the job's target hex (without 0x prefix). Full targets are big-endian on the wire; the
// Kaspa-family and Verthash kernels compare little-endian hash words, so their bytes are reversed
// (byte 31 becomes the most significant). Word targets: KawPoW and values longer than 16 digits
// are the most-significant word of a big-endian 256-bit target, 9..16 digits a little-endian
// uint64, and up to 8 digits a little-endian uint32 compact target expanded to the equivalent
// 64-bit threshold. A zero target is invalid.
inline bool parse_share_target(std::string_view algo, std::string_view hex, ShareTarget& target) {
  target = ShareTarget{};
  if (uses_full_target(algo)) {
    if (!parse_big_target(hex, target.bytes))
      return false;
    if (algo == "karlsenhashv2" || algo == "hoohash" || algo == "walahash" || algo == "verthash")
      std::reverse(std::begin(target.bytes), std::end(target.bytes));
    target.full = true;
    target.word = 1;
    return true;
  }
  std::uint8_t bytes[32]{};
  if (hex.empty() || hex.size() > 64 || !decode_hex(hex, bytes))
    return false;
  const bool high_word = algo == "kawpow" || hex.size() > 16;
  const std::uint64_t value = load_nonce<std::uint64_t>(bytes, high_word);
  if (!value)
    return false;
  constexpr std::uint64_t compact_limit = std::numeric_limits<std::uint32_t>::max();
  target.word = high_word || hex.size() > 8
    ? value : std::numeric_limits<std::uint64_t>::max() / (compact_limit / value);
  return true;
}

} // namespace mom::job_boundary
