#include "job-boundary.h"

#include <array>
#include <cassert>
#include <cstdint>
#include <limits>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using mom::job_boundary::fishhash_nonce_offset;
using mom::job_boundary::load_nonce;
using mom::job_boundary::nonce_overflowed;
using mom::job_boundary::nonce_big_endian;
using mom::job_boundary::nonce_with_offset;
using mom::job_boundary::next_nonce_batch;
using mom::job_boundary::padded_u32_range_fits;
using mom::job_boundary::parse_unsigned;
using mom::job_boundary::pearl_seed_next;
using mom::job_boundary::pearl_seed_start;
using mom::job_boundary::split_once;
using mom::job_boundary::store_nonce;
using mom::job_boundary::valid_beamhash3_input;
using mom::job_boundary::valid_cn_gpu_input;
using mom::job_boundary::valid_fishhash_layout;
using mom::job_boundary::valid_karlsenhashv2_layout;
using mom::job_boundary::valid_nexapow_layout;
using mom::job_boundary::valid_pearlhash_shape;
using mom::job_boundary::c29_completed_graphs;
using mom::job_boundary::for_each_gpu_hit;
using mom::job_boundary::decode_hex;
using mom::job_boundary::parse_big_target;
using mom::job_boundary::parse_share_target;
using mom::job_boundary::ShareTarget;
using mom::job_boundary::strip_hex_prefix;

void test_parse_unsigned() {
  std::uint32_t value32 = 7;
  assert(parse_unsigned("0", value32));
  assert(value32 == 0);
  assert(parse_unsigned("4294967295", value32));
  assert(value32 == std::numeric_limits<std::uint32_t>::max());
  assert(!parse_unsigned("", value32));
  assert(!parse_unsigned("+1", value32));
  assert(!parse_unsigned("-1", value32));
  assert(!parse_unsigned(" 1", value32));
  assert(!parse_unsigned("1 ", value32));
  assert(!parse_unsigned("12x", value32));
  assert(!parse_unsigned("4294967296", value32));
  assert(!parse_unsigned("0x10", value32));
  assert(parse_unsigned("0x10", value32, 16, true));
  assert(value32 == 16);
  assert(parse_unsigned("0XFF", value32, 16, true));
  assert(value32 == 255);
  assert(!parse_unsigned("0x", value32, 16, true));

  std::uint64_t value64 = 9;
  assert(parse_unsigned("0", value64));
  assert(value64 == 0);
  assert(parse_unsigned("18446744073709551615", value64));
  assert(value64 == std::numeric_limits<std::uint64_t>::max());
  assert(!parse_unsigned("", value64));
  assert(!parse_unsigned("+1", value64));
  assert(!parse_unsigned("-1", value64));
  assert(!parse_unsigned(" ", value64));
  assert(!parse_unsigned("\t1", value64));
  assert(!parse_unsigned("1\n", value64));
  assert(!parse_unsigned("1z", value64));
  assert(!parse_unsigned("18446744073709551616", value64));
}

void test_split_once() {
  std::string_view left;
  std::string_view right;
  const std::string_view text = "alpha:beta";
  assert(split_once(text, ':', left, right));
  assert(left == "alpha");
  assert(right == "beta");
  assert(left.data() == text.data());
  assert(right.data() == text.data() + 6);
  assert(!split_once(":beta", ':', left, right));
  assert(!split_once("alpha:", ':', left, right));
  assert(!split_once("alpha::beta", ':', left, right));
  assert(!split_once("alpha:beta:gamma", ':', left, right));
  assert(!split_once("alpha", ':', left, right));
  assert(!split_once("", ':', left, right));
}

void test_layouts() {
  assert(fishhash_nonce_offset(40) == 32);
  assert(valid_fishhash_layout(40, 32));
  assert(!valid_fishhash_layout(40, 0));
  assert(fishhash_nonce_offset(39) == std::numeric_limits<std::size_t>::max());
  assert(!valid_fishhash_layout(39, 32));
  assert(!valid_fishhash_layout(41, 32));
  assert(fishhash_nonce_offset(180) == 172);
  assert(valid_fishhash_layout(180, 172));
  assert(!valid_fishhash_layout(180, 0));
  assert(!valid_fishhash_layout(180, 32));
  assert(!valid_fishhash_layout(140, 172));
  assert(!valid_fishhash_layout(179, 172));
  assert(!valid_fishhash_layout(181, 172));
  assert(valid_karlsenhashv2_layout(80, 72));
  assert(!valid_karlsenhashv2_layout(79, 72));
  assert(!valid_karlsenhashv2_layout(81, 72));
  assert(valid_nexapow_layout(40, 32));
  assert(valid_nexapow_layout(44, 36));
  assert(valid_nexapow_layout(48, 40));
  assert(!valid_nexapow_layout(40, 40));
  assert(!valid_nexapow_layout(44, 32));
  assert(!valid_nexapow_layout(44, 40));
  assert(!valid_nexapow_layout(48, 32));
  assert(!valid_nexapow_layout(39, 32));
  assert(!valid_nexapow_layout(41, 32));
  assert(!valid_nexapow_layout(43, 36));
  assert(!valid_nexapow_layout(45, 36));
  assert(!valid_nexapow_layout(47, 40));
  assert(!valid_nexapow_layout(49, 40));
  assert(!nonce_big_endian("fishhash", 40));
  assert(nonce_big_endian("fishhash", 180));
  assert(!nonce_big_endian("fishhash", 140));
  assert(nonce_big_endian("beamhash3", 0));
  assert(nonce_big_endian("xelishashv3", 0));
  assert(nonce_big_endian("nexapow", 0));
  assert(!nonce_big_endian("control", 180));
}

void test_nonce_bytes() {
  constexpr std::uint64_t value = 0x0102030405060708ULL;
  std::array<std::uint8_t, 20> storage{};
  storage.fill(0xa5);
  store_nonce(storage.data() + 1, value, true);
  assert(storage[0] == 0xa5);
  assert(storage[1] == 0x01);
  assert(storage[2] == 0x02);
  assert(storage[8] == 0x08);
  assert(storage[9] == 0xa5);
  assert(load_nonce<std::uint64_t>(storage.data() + 1, true) == value);

  storage.fill(0xa5);
  store_nonce(storage.data() + 1, value, false);
  assert(storage[0] == 0xa5);
  assert(storage[1] == 0x08);
  assert(storage[2] == 0x07);
  assert(storage[8] == 0x01);
  assert(storage[9] == 0xa5);
  assert(load_nonce<std::uint64_t>(storage.data() + 1, false) == value);

  constexpr std::uint32_t value32 = 0x11223344U;
  store_nonce(storage.data() + 3, value32, true);
  assert(load_nonce<std::uint32_t>(storage.data() + 3, true) == value32);
  store_nonce(storage.data() + 3, value32, false);
  assert(load_nonce<std::uint32_t>(storage.data() + 3, false) == value32);

  constexpr std::uint64_t beam_first = 0xa1b2010000000000ULL;
  store_nonce(storage.data() + 1, beam_first,
              nonce_big_endian("beamhash3", 44));
  const std::array<std::uint8_t, 8> expected_first = {
    0xa1, 0xb2, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
  };
  for (std::size_t i = 0; i < expected_first.size(); ++i) {
    assert(storage[i + 1] == expected_first[i]);
  }

  store_nonce(storage.data() + 1, beam_first + 1,
              nonce_big_endian("beamhash3", 44));
  const std::array<std::uint8_t, 8> expected_next = {
    0xa1, 0xb2, 0x01, 0x00, 0x00, 0x00, 0x00, 0x01,
  };
  for (std::size_t i = 0; i < expected_next.size(); ++i) {
    assert(storage[i + 1] == expected_next[i]);
  }

  std::array<std::uint8_t, 180> fish_header{};
  const std::size_t fish_offset = fishhash_nonce_offset(fish_header.size());
  store_nonce(fish_header.data() + fish_offset, value,
              nonce_big_endian("fishhash", fish_header.size()));
  assert(fish_offset == 172);
  assert(load_nonce<std::uint64_t>(fish_header.data() + fish_offset, true) == value);

  for (const std::size_t nexa_size : {std::size_t{40}, std::size_t{44}, std::size_t{48}}) {
    std::array<std::uint8_t, 48> nexa_header{};
    const std::size_t nexa_offset = nexa_size - sizeof(value);
    assert(valid_nexapow_layout(nexa_size, nexa_offset));
    store_nonce(nexa_header.data() + nexa_offset, value,
                nonce_big_endian("nexapow", nexa_size));
    assert(load_nonce<std::uint64_t>(nexa_header.data() + nexa_offset, true) == value);
  }
}

void test_nonce_overflow() {
  assert(!nonce_overflowed(1, 2, 0));
  assert(nonce_overflowed(std::numeric_limits<std::uint32_t>::max(), 0, 0));
  assert(!nonce_overflowed(0xab000001, 0xab000002, 0xff000000));
  assert(nonce_overflowed(0xabffffff, 0xac000000, 0xff000000));
  assert(nonce_overflowed(0xabffffff, 0xab000000, 0xff000000));

  std::uint32_t value32 = 0;
  assert(nonce_with_offset<std::uint32_t>(0xfffffffe, 1, 1, 0, value32));
  assert(value32 == std::numeric_limits<std::uint32_t>::max());
  assert(!nonce_with_offset<std::uint32_t>(0xfffffffe, 2, 1, 0, value32));
  assert(nonce_with_offset<std::uint32_t>(0xabfffffd, 2, 1, 0xff000000, value32));
  assert(!nonce_with_offset<std::uint32_t>(0xabfffffd, 3, 1, 0xff000000, value32));

  std::uint64_t value64 = 0;
  assert(nonce_with_offset<std::uint64_t>(
    std::numeric_limits<std::uint64_t>::max(), 0, 9, 0, value64));
  assert(!nonce_with_offset<std::uint64_t>(
    std::numeric_limits<std::uint64_t>::max(), 1, 1, 0, value64));
  assert(!nonce_with_offset<std::uint64_t>(
    0, std::numeric_limits<std::uint64_t>::max(), 2, 0, value64));

  std::uint32_t last = 0;
  std::uint32_t next = 0;
  assert(next_nonce_batch<std::uint32_t>(0xfffffff8, 4, 1, 4, 0, last, next));
  assert(last == 0xfffffffb);
  assert(next == 0xfffffffc);
  assert(!next_nonce_batch<std::uint32_t>(0xfffffffc, 4, 1, 4, 0, last, next));
  assert(last == std::numeric_limits<std::uint32_t>::max());
}

void test_pearl_seed_partition() {
  std::uint32_t start = 0;
  assert(pearl_seed_start(0, 0, 256, 256, start));
  assert(start == 0);
  constexpr std::uint64_t step = 65536;
  std::uint32_t seed = start;
  for (unsigned i = 0; i != 4; ++i) {
    assert(seed % 256 == 0);
    assert(pearl_seed_next(seed, step, seed));
  }
  assert(seed == 4 * 65536);

  std::set<std::uint32_t> seen;
  for (std::uint32_t thread_id = 0; thread_id != 256; ++thread_id) {
    std::uint32_t thread_start = 0;
    assert(pearl_seed_start(0, thread_id, 256, 256, thread_start));
    for (unsigned iteration = 0; iteration != 8; ++iteration) {
      assert(seen.insert(thread_start).second);
      if (iteration + 1 != 8) {
        assert(pearl_seed_next(thread_start, step, thread_start));
      }
    }
  }
  assert(seen.size() == 256 * 8);

  assert(!pearl_seed_start(0, 0, 1, 0, start));
  assert(!pearl_seed_start(std::numeric_limits<std::uint32_t>::max(), 1, 2, 1,
                           start));
  assert(pearl_seed_start(std::numeric_limits<std::uint32_t>::max(), 0, 1, 1,
                          start));
  assert(!pearl_seed_next(start, step, start));
  assert(pearl_seed_start(0, 0, 1, std::numeric_limits<std::uint32_t>::max(),
                          start));
  assert(pearl_seed_next(start, std::numeric_limits<std::uint32_t>::max(), start));
  assert(start == std::numeric_limits<std::uint32_t>::max());
  assert(!pearl_seed_next(start, std::numeric_limits<std::uint32_t>::max(), start));
}

void test_padded_ranges() {
  constexpr std::uint32_t maximum = std::numeric_limits<std::uint32_t>::max();
  assert(!padded_u32_range_fits(1, 0));
  assert(padded_u32_range_fits(maximum, 1));
  assert(padded_u32_range_fits(maximum - 127, 128));
  assert(!padded_u32_range_fits(maximum - 126, 128));
  assert(padded_u32_range_fits(maximum - 255, 256));
  assert(!padded_u32_range_fits(maximum - 254, 256));
}

void test_cn_input_lengths() {
  assert(!valid_cn_gpu_input(0));
  assert(valid_cn_gpu_input(1));
  assert(valid_cn_gpu_input(135));
  assert(!valid_cn_gpu_input(136));
  assert(!valid_cn_gpu_input(200));
  assert(!valid_beamhash3_input(43));
  assert(valid_beamhash3_input(44));
  assert(!valid_beamhash3_input(45));
}

void test_pearlhash_shapes() {
  constexpr std::uint64_t max_dimension = std::uint64_t{1} << 24;
  constexpr std::uint64_t max_m =
      (static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()) / 2048 / 32) * 32;
  constexpr std::uint64_t max_n =
      (static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()) / 128 / 32) * 32;
  assert(valid_pearlhash_shape(128, 128, 2048, 128));
  assert(valid_pearlhash_shape(160, 160, 2048, 128));
  assert(valid_pearlhash_shape(131072, 524288, 8192, 128));
  assert(valid_pearlhash_shape(128, 128, 16384, 1024));
  assert(valid_pearlhash_shape(128, 128, 65536, 1024));
  assert(valid_pearlhash_shape(131072, 524288, 8320, 128));
  assert(valid_pearlhash_shape(131072, 131072, 4096, 256));
  assert(valid_pearlhash_shape(32768, 32768, 2048, 128));
  assert(valid_pearlhash_shape(16384, 16384, 2048, 128));
  assert(valid_pearlhash_shape(max_m, 128, 2048, 128));
  assert(valid_pearlhash_shape(128, max_n, 2048, 128));
  assert(!valid_pearlhash_shape(96, 128, 2048, 128));
  assert(!valid_pearlhash_shape(128, 96, 2048, 128));
  assert(!valid_pearlhash_shape(161, 128, 2048, 128));
  assert(!valid_pearlhash_shape(128, 161, 2048, 128));
  assert(!valid_pearlhash_shape(max_dimension + 32, 128, 2048, 128));
  assert(!valid_pearlhash_shape(128, max_dimension + 32, 2048, 128));
  assert(!valid_pearlhash_shape(max_dimension, max_dimension, 2048, 128));
  assert(!valid_pearlhash_shape(max_m + 32, 128, 2048, 128));
  assert(!valid_pearlhash_shape(128, max_n + 32, 2048, 128));
  assert(!valid_pearlhash_shape(131072, 4194304, 2048, 128));
  assert(!valid_pearlhash_shape(128, 128, 960, 128));
  assert(!valid_pearlhash_shape(128, 128, 1024, 128));
  assert(!valid_pearlhash_shape(128, 128, 1025, 128));
  assert(!valid_pearlhash_shape(128, 128, 65600, 1024));
  assert(valid_pearlhash_shape(128, 128, 4160, 256));
  assert(!valid_pearlhash_shape(128, 128, 32768, 2048));
  assert(!valid_pearlhash_shape(128, 128, 3072, 192));
  assert(!valid_pearlhash_shape(128, 128, 16, 16));
  assert(!valid_pearlhash_shape(128, 128, 48, 16));
  assert(!valid_pearlhash_shape(128, 128, 32, 16));
  assert(!valid_pearlhash_shape(128, 128, 1024, 32));
  assert(!valid_pearlhash_shape(128, 128, std::uint64_t{1} << 63, std::uint64_t{1} << 63));
}

void test_c29_completed_graphs() {
  assert(c29_completed_graphs(0) == 1);  // new graph
  assert(c29_completed_graphs(1) == 0);  // recovery
  assert(c29_completed_graphs(-1) == 0); // drained

  std::uint64_t total = 0;
  for (const int result : {0, 1, 1, 0, -1})
    total += c29_completed_graphs(result);
  assert(total == 2);
}

// A fake single-blob dispatch that finds two target hits in one batch must get both reported,
// each once, in its slot order; impossible counts must not expose any slot.
void test_gpu_hits() {
  constexpr unsigned capacity = 15;
  std::uint64_t nonces[capacity]{};
  std::uint8_t hashes[capacity * 32]{};
  const auto dispatch = [&](std::uint64_t start) {
    nonces[0] = start + 3;
    hashes[0] = 3;
    nonces[1] = start + 7;
    hashes[32] = 7;
    return 2;
  };
  std::vector<std::pair<std::uint64_t, std::uint8_t>> reported;
  const auto report = [&](unsigned slot) {
    reported.emplace_back(nonces[slot], hashes[slot * 32]);
  };
  assert(for_each_gpu_hit(dispatch(100), capacity, report));
  assert((reported == std::vector<std::pair<std::uint64_t, std::uint8_t>>{{103, 3}, {107, 7}}));

  reported.clear();
  assert(for_each_gpu_hit(0, capacity, report));
  assert(!for_each_gpu_hit(-1, capacity, report));
  assert(!for_each_gpu_hit(capacity + 1, capacity, report));
  assert(reported.empty());
}

std::uint64_t word_target(std::string_view algo, std::string_view hex) {
  ShareTarget target;
  assert(parse_share_target(algo, hex, target));
  assert(!target.full);
  return target.word;
}

void test_share_targets() {
  // Up to 8 digits: little-endian uint32 compact target expanded to a 64-bit threshold.
  assert(word_target("rx/0", "b88d0600") == 0x68db8bac710cbULL);
  assert(word_target("cn/r", "ffffff00") == 0xffffffffffffffULL);
  // 9..16 digits: little-endian uint64, right-padded.
  assert(word_target("rx/0", "0011223344") == 0x4433221100ULL);
  assert(word_target("rx/0", "0011223344556677") == 0x7766554433221100ULL);
  // Longer targets and KawPoW: the most-significant big-endian word.
  assert(word_target("c29", "00000000ffff00000000000000000000") == 0xffff0000ULL);
  assert(word_target("kawpow", "ff") == 0xff00000000000000ULL);

  ShareTarget target;
  assert(parse_share_target("etchash", "01", target) && target.full && target.word == 1);
  assert(target.bytes[31] == 1 && target.bytes[0] == 0);
  // Kaspa-family and Verthash kernels compare little-endian words: byte 0 becomes the MSB.
  assert(parse_share_target("walahash", "0100", target) && target.bytes[1] == 1);
  assert(parse_share_target("verthash", "01" + std::string(62, '0'), target));
  assert(target.bytes[31] == 1 && target.bytes[0] == 0);

  for (const std::string_view bad : {"", "0", "abc", "zz", "00", "00000000"})
    assert(!parse_share_target("rx/0", bad, target));
  assert(!parse_share_target("etchash", "0000", target));
  assert(!parse_share_target("etchash", std::string(66, '1'), target));
  assert(!parse_share_target("kawpow", "0000000000000000ff", target));

  std::uint8_t bytes[32];
  assert(parse_big_target(std::string(64, 'f'), bytes) && bytes[0] == 0xff && bytes[31] == 0xff);
  assert(!parse_big_target("0g", bytes));
  std::uint8_t decoded[2]{};
  assert(decode_hex("aB0f", decoded) && decoded[0] == 0xab && decoded[1] == 0x0f);
  assert(strip_hex_prefix("0Xff") == "ff" && strip_hex_prefix("ff") == "ff");
}

} // namespace

int main() {
  test_parse_unsigned();
  test_split_once();
  test_layouts();
  test_nonce_bytes();
  test_nonce_overflow();
  test_pearl_seed_partition();
  test_padded_ranges();
  test_cn_input_lengths();
  test_pearlhash_shapes();
  test_c29_completed_graphs();
  test_gpu_hits();
  test_share_targets();
  return 0;
}
