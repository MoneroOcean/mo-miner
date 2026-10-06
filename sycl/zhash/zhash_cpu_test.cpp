// Host-only regression for the ZHash adapter.  It deliberately duplicates only the authoritative
// BTG fixture bytes; the validator implementation is the shared adapter under test.

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "equihash_core.hpp"

namespace {

template <std::size_t N, std::size_t Characters>
void from_hex(const char (&text)[Characters], std::uint8_t (&out)[N]) {
  static_assert(Characters == 2 * N + 1, "fixture hex length mismatch");
  for (std::size_t i = 0; i < N; ++i) {
    const auto digit = [](const char c) -> unsigned {
      return c >= '0' && c <= '9'   ? static_cast<unsigned>(c - '0')
             : c >= 'a' && c <= 'f' ? static_cast<unsigned>(c - 'a' + 10)
             : c >= 'A' && c <= 'F' ? static_cast<unsigned>(c - 'A' + 10)
                                    : 0xffu;
    };
    const unsigned high = digit(text[2 * i]), low = digit(text[2 * i + 1]);
    if (high > 15 || low > 15)
      std::abort();
    out[i] = static_cast<std::uint8_t>((high << 4) | low);
  }
}

} // namespace

namespace {

struct SmallBuckets {
  static constexpr unsigned bucket_count = 2;
  static constexpr unsigned local_bins = 4;
  static constexpr unsigned bucket_slot_capacity = 4;
  static constexpr unsigned local_bits = 2;
  static constexpr unsigned bucket_for(const std::uint32_t field) {
    return field >> local_bits;
  }
  static constexpr unsigned local_bin(const std::uint32_t field) {
    return field & 3u;
  }
};

struct TinyScatter {
  static constexpr unsigned bucket_count = 2;
  static constexpr unsigned local_bins = 4;
  static constexpr unsigned bucket_slot_capacity = 2;
  static constexpr unsigned local_bits = 2;
  static constexpr unsigned bucket_for(const std::uint32_t field) {
    return field >> local_bits;
  }
  static constexpr unsigned local_bin(const std::uint32_t field) {
    return field & 3u;
  }
};

struct SyntheticTree {
  static constexpr unsigned bucket_count = 1;
  static constexpr unsigned local_bins = 256;
  static constexpr unsigned bucket_slot_capacity = 64;
  static constexpr unsigned local_bits = 8;
  static constexpr unsigned collision_work_group = 256;
  static constexpr unsigned bucket_for(const std::uint32_t field) {
    return field >> local_bits;
  }
  static constexpr unsigned local_bin(const std::uint32_t field) {
    return field & (local_bins - 1u);
  }
};

template <typename Spec>
mom_equihash::PackedLevel0Record<Spec> level0(const std::uint32_t (&fields)[Spec::rounds],
                                              const std::uint32_t leaf) {
  mom_equihash::PackedLevel0Record<Spec> record{};
  mom_equihash::store_level0<Spec>(record, fields, leaf);
  return record;
}

int test_bucket_level0_packing() {
  using Spec = mom_equihash::ZHash144_5;
  using Record = mom_equihash::PackedBucketLevel0Record<Spec, 12>;
  constexpr std::uint32_t bucket = 0x345u;
  const std::uint32_t fields[Spec::rounds] = {
      0x345abcu, 0x102030u, 0x405060u, 0x708090u, 0xa0b0c0u, 0xd0e0f0u};
  constexpr std::uint32_t leaf = 0x1abcdeu;
  Record record{};
  mom_equihash::store_bucket_level0<Spec, 12>(record, fields, leaf);

  if (mom_equihash::load_bucket_level0_field<Spec, 12>(record, bucket, 0) != fields[0])
    return 1;
  for (unsigned field = 1; field < Spec::rounds - 1; ++field)
    if (mom_equihash::load_bucket_level0_field<Spec, 12>(record, bucket, field) != fields[field])
      return 1 + static_cast<int>(field);
  const std::uint32_t expected_final =
      Record::compact_selection_head ? fields[Spec::rounds - 1] & 0xfffffu
                                     : fields[Spec::rounds - 1];
  if (mom_equihash::load_bucket_level0_field<Spec, 12>(record, bucket, Spec::rounds - 1) !=
      expected_final)
    return 6;
  if (mom_equihash::load_bucket_level0_leaf<Spec, 12>(record) != leaf)
    return 7;
  if constexpr (Record::compact_selection_head) {
    if ((record.words[0] & 0xfffu) != (fields[0] & 0xfffu)) {
      return 8;
    }
  }

  using Record11 = mom_equihash::PackedBucketLevel0Record<Spec, 11>;
  const std::uint32_t bucket11 = fields[0] >> 11;
  Record11 record11{};
  mom_equihash::store_bucket_level0<Spec, 11>(record11, fields, leaf);
  if (mom_equihash::load_bucket_level0_field<Spec, 11>(record11, bucket11, 0) != fields[0])
    return 9;
  for (unsigned field = 1; field < Spec::rounds - 1; ++field)
    if (mom_equihash::load_bucket_level0_field<Spec, 11>(record11, bucket11, field) !=
        fields[field])
      return 9 + static_cast<int>(field);
  const std::uint32_t expected_final11 =
      Record11::compact_selection_head ? fields[Spec::rounds - 1] & 0x1fffffu
                                       : fields[Spec::rounds - 1];
  if (mom_equihash::load_bucket_level0_field<Spec, 11>(record11, bucket11, Spec::rounds - 1) !=
      expected_final11)
    return 14;
  if (mom_equihash::load_bucket_level0_leaf<Spec, 11>(record11) != leaf)
    return 15;
  if constexpr (Record11::compact_selection_head) {
    if ((record11.words[0] & 0x7ffu) != (fields[0] & 0x7ffu)) {
      return 16;
    }
  }
  return 0;
}

template <unsigned LocalBits, unsigned ParentBucketBits, unsigned ParentSlots>
int test_round3_record(const std::uint32_t bucket) {
  using Spec = mom_equihash::ZHash144_5;
  using Record = mom_equihash::PackedBucketCollisionRecord<
      Spec, 3, LocalBits, ParentBucketBits, ParentSlots>;
  const std::uint32_t fields[3] = {
      (bucket << LocalBits) | 0x345u, 0x123456u, 0xfedcbau};
  constexpr std::uint32_t left = ParentSlots - 2;
  constexpr std::uint32_t right = ParentSlots - 1;
  Record record{};
  mom_equihash::store_bucket_collision(record, fields, bucket, left, right);
  if (mom_equihash::load_bucket_collision_field(record, bucket, 0) != fields[0] ||
      mom_equihash::load_bucket_collision_field(record, bucket, 1) != fields[1])
    return 1;
  const std::uint32_t trailing_mask = static_cast<std::uint32_t>(
      mom_equihash::packed_mask(Spec::collision_bits - Record::trailing_bits_omitted));
  if (mom_equihash::load_bucket_collision_field(record, bucket, 2) !=
      (fields[2] & trailing_mask))
    return 2;
  if (mom_equihash::load_bucket_collision_parent(record, false) != bucket * ParentSlots + left ||
      mom_equihash::load_bucket_collision_parent(record, true) != bucket * ParentSlots + right)
    return 3;
#if defined(MOM_SYCL_HAS_HIP)
  static_assert(sizeof(Record) == 12);
#else
  static_assert(sizeof(Record) == 16);
#endif
  return 0;
}

int test_round3_packing() {
  if (const int result = test_round3_record<12, 12, 8704>(0x345u))
    return result;
  if (const int result = test_round3_record<11, 13, 5632>(0x1234u))
    return 10 + result;
  return 0;
}

int test_bucket_stage() {
  using Spec = mom_equihash::ZHash144_5;
  using Record = mom_equihash::PackedLevel0Record<Spec>;
  using Collision = mom_equihash::PackedCollisionRecord<Spec>;
  static_assert(sizeof(Record) == 24 && sizeof(Collision) == 24);

  const std::uint32_t a[Spec::rounds] = {1, 0x010203, 0x112233, 0x445566, 0x778899, 0xaabbcc};
  const std::uint32_t b[Spec::rounds] = {1, 0x102030, 0x203040, 0x506070, 0x8090a0, 0xb0c0d0};
  const std::uint32_t c[Spec::rounds] = {2, 1, 2, 3, 4, 5};
  const std::uint32_t d[Spec::rounds] = {5, 6, 7, 8, 9, 10};
  const std::uint32_t e[Spec::rounds] = {5, 0xfedcba, 0x13579b, 0x2468ac, 0xabcdef, 0x0badf0};
  Record input[SmallBuckets::bucket_count * SmallBuckets::bucket_slot_capacity]{};
  input[0] = level0<Spec>(a, 100);
  input[1] = level0<Spec>(b, 200);
  input[2] = level0<Spec>(c, 300);
  input[4] = level0<Spec>(d, 50);
  input[5] = level0<Spec>(e, 40);
  std::uint32_t counts[SmallBuckets::bucket_count] = {3, 2};
  Collision output[2]{};
  const std::uint32_t total =
      mom_equihash::bucket_round_reference<Spec, SmallBuckets>(input, counts, output, 2);
  if (total != 2)
    return 1;
  if (mom_equihash::load_collision_left<Spec>(output[0]) != 0 ||
      mom_equihash::load_collision_right<Spec>(output[0]) != 1)
    return 2;
  if (mom_equihash::load_collision_left<Spec>(output[1]) != 5 ||
      mom_equihash::load_collision_right<Spec>(output[1]) != 4)
    return 3;
  for (unsigned i = 0; i < Spec::rounds - 1; ++i) {
    if (mom_equihash::load_collision_field<Spec>(output[0], i) != (a[i + 1] ^ b[i + 1]))
      return 4;
    if (mom_equihash::load_collision_field<Spec>(output[1], i) != (d[i + 1] ^ e[i + 1]))
      return 5;
  }

  Collision one[1]{};
  if (mom_equihash::bucket_round_reference<Spec, SmallBuckets>(input, counts, one, 1) != 2)
    return 6;
  return 0;
}

int test_scatter_overflow_and_two_rounds() {
  using Spec = mom_equihash::ZHash144_5;
  using Level0 = mom_equihash::PackedLevel0Record<Spec>;
  using First = mom_equihash::PackedCollisionRecord<Spec>;
  using Second = mom_equihash::PackedCollisionRecord<Spec, Spec::rounds - 2>;

  const std::uint32_t scatter_fields[4][Spec::rounds] = {{1, 10, 20, 30, 40, 50},
                                                         {2, 11, 21, 31, 41, 51},
                                                         {3, 12, 22, 32, 42, 52},
                                                         {4, 13, 23, 33, 43, 53}};
  Level0 scatter_input[4],
      scatter_output[TinyScatter::bucket_count * TinyScatter::bucket_slot_capacity]{};
  for (unsigned i = 0; i < 4; ++i)
    scatter_input[i] = level0<Spec>(scatter_fields[i], 100 + i);
  std::uint32_t scatter_counts[TinyScatter::bucket_count] = {}, scatter_overflow = 0;
  const std::uint32_t stored = mom_equihash::scatter_level0_reference<Spec, TinyScatter>(
      scatter_input, 4, scatter_output, scatter_counts, &scatter_overflow);
  if (stored != 3 || scatter_overflow != 1 || scatter_counts[0] != 3 || scatter_counts[1] != 1)
    return 1;
  if (mom_equihash::load_level0_leaf<Spec>(scatter_output[0]) != 100 ||
      mom_equihash::load_level0_leaf<Spec>(scatter_output[1]) != 101 ||
      mom_equihash::load_level0_leaf<Spec>(scatter_output[2]) != 103)
    return 2;

  const std::uint32_t round_fields[4][Spec::rounds] = {{1, 4, 10, 20, 30, 40},
                                                       {1, 6, 15, 25, 35, 45},
                                                       {2, 7, 8, 30, 40, 50},
                                                       {2, 5, 13, 35, 45, 55}};
  Level0 level0_input[TinyScatter::bucket_count * SmallBuckets::bucket_slot_capacity]{};
  for (unsigned i = 0; i < 4; ++i)
    level0_input[i] = level0<Spec>(round_fields[i], 100 + i * 100);
  const std::uint32_t level0_counts[SmallBuckets::bucket_count] = {4, 0};
  First first_round[8]{};
  if (mom_equihash::bucket_round_reference<Spec, SmallBuckets>(level0_input, level0_counts,
                                                               first_round, 8) != 2)
    return 3;
  if (mom_equihash::load_collision_left<Spec>(first_round[0]) != 0 ||
      mom_equihash::load_collision_right<Spec>(first_round[0]) != 1 ||
      mom_equihash::load_collision_left<Spec>(first_round[1]) != 2 ||
      mom_equihash::load_collision_right<Spec>(first_round[1]) != 3)
    return 4;

  std::uint32_t first_min[2] = {100, 300};
  First first_arena[SmallBuckets::bucket_count * SmallBuckets::bucket_slot_capacity]{};
  std::uint32_t first_arena_min[SmallBuckets::bucket_count * SmallBuckets::bucket_slot_capacity]{};
  std::uint32_t first_arena_counts[SmallBuckets::bucket_count] = {}, first_overflow = 0;
  if (mom_equihash::scatter_collision_reference<Spec, Spec::rounds - 1, SmallBuckets>(
          first_round, first_min, 2, first_arena, first_arena_min, first_arena_counts,
          &first_overflow) != 2 ||
      first_overflow != 0 || first_arena_counts[0] != 2)
    return 5;
  Second second_round[2]{};
  std::uint32_t second_min[2]{};
  if (mom_equihash::collision_round_reference<Spec, Spec::rounds - 1, SmallBuckets>(
          first_arena, first_arena_min, first_arena_counts, second_round, second_min, 2) != 1)
    return 6;
  if (second_min[0] != 100 ||
      mom_equihash::load_collision_left<Spec, Spec::rounds - 2>(second_round[0]) != 0 ||
      mom_equihash::load_collision_right<Spec, Spec::rounds - 2>(second_round[0]) != 1)
    return 7;
  for (unsigned i = 0; i < Spec::rounds - 2; ++i) {
    const std::uint32_t expected =
        mom_equihash::load_collision_field<Spec, Spec::rounds - 1>(first_arena[0], i + 1) ^
        mom_equihash::load_collision_field<Spec, Spec::rounds - 1>(first_arena[1], i + 1);
    if (mom_equihash::load_collision_field<Spec, Spec::rounds - 2>(second_round[0], i) != expected)
      return 8;
  }
  First overlap[2] = {first_arena[0], first_arena[0]};
  const std::uint32_t overlap_min[2] = {100, 200},
                      overlap_counts[SmallBuckets::bucket_count] = {2, 0};
  if (mom_equihash::collision_round_reference<Spec, Spec::rounds - 1, SmallBuckets>(
          overlap, overlap_min, overlap_counts, second_round, second_min, 2) != 0)
    return 9;
  return 0;
}

int test_complete_synthetic_tree() {
  using Spec = mom_equihash::ZHash144_5;
  using Level0 = mom_equihash::PackedLevel0Record<Spec>;
  using Round1 = mom_equihash::CollisionRoundRecord<Spec, 1>;
  using Round2 = mom_equihash::CollisionRoundRecord<Spec, 2>;
  using Round3 = mom_equihash::CollisionRoundRecord<Spec, 3>;
  using Round4 = mom_equihash::CollisionRoundRecord<Spec, 4>;
  using Root = mom_equihash::ZHashRootRecord<Spec>;
  constexpr unsigned arena_size = SyntheticTree::bucket_slot_capacity;
  static_assert(sizeof(Root) == 12 && Spec::solution_length == 100);

  // For field j, one designated leaf in each 2^j subtree carries the unique key of its parent
  // pair.  The sibling subtree carries the same key, so each of the five joins has exactly one
  // match per intended binary-tree node and no accidental cross-pair match.
  Level0 level0[arena_size]{};
  for (std::uint32_t leaf = 0; leaf < Spec::proof_indices; ++leaf) {
    std::uint32_t fields[Spec::rounds]{};
    for (unsigned field = 0; field < Spec::rounds - 1; ++field) {
      const unsigned width = 1u << field;
      const bool designated = (leaf & (width - 1u)) == 0;
      fields[field] = designated ? (32u * (field + 1u) + (leaf >> (field + 1u)) + 1u) : 0u;
    }
    fields[Spec::rounds - 1] = 0;
    mom_equihash::store_level0<Spec>(level0[leaf], fields, leaf);
  }
  const std::uint32_t level0_counts[SyntheticTree::bucket_count] = {Spec::proof_indices};

  mom_equihash::ProvenanceView<Spec> view{};
  view.level0 = level0;
  view.level0_capacity = arena_size;
  Round1 first_flat[arena_size]{};
  std::uint32_t first_flat_min[arena_size]{};
  if (mom_equihash::bucket_round_reference_checked<Spec, SyntheticTree>(
          level0, level0_counts, first_flat, first_flat_min, arena_size, view) != 16)
    return 1;

  Round1 round1[arena_size]{};
  std::uint32_t round1_min[arena_size]{}, round1_counts[SyntheticTree::bucket_count] = {},
                                          overflow = 0;
  if (mom_equihash::scatter_collision_reference<Spec, Spec::rounds - 1, SyntheticTree>(
          first_flat, first_flat_min, 16, round1, round1_min, round1_counts, &overflow) != 16 ||
      overflow != 0)
    return 2;
  view.round1 = round1;
  view.round1_capacity = arena_size;

  Round2 second_flat[arena_size]{};
  std::uint32_t second_flat_min[arena_size]{};
  if (mom_equihash::collision_round_reference_checked<Spec, Spec::rounds - 1, SyntheticTree>(
          round1, round1_min, round1_counts, second_flat, second_flat_min, arena_size, view) != 8)
    return 3;
  Round2 round2[arena_size]{};
  std::uint32_t round2_min[arena_size]{}, round2_counts[SyntheticTree::bucket_count] = {};
  overflow = 0;
  if (mom_equihash::scatter_collision_reference<Spec, Spec::rounds - 2, SyntheticTree>(
          second_flat, second_flat_min, 8, round2, round2_min, round2_counts, &overflow) != 8 ||
      overflow != 0)
    return 4;
  view.round2 = round2;
  view.round2_capacity = arena_size;

  Round3 third_flat[arena_size]{};
  std::uint32_t third_flat_min[arena_size]{};
  if (mom_equihash::collision_round_reference_checked<Spec, Spec::rounds - 2, SyntheticTree>(
          round2, round2_min, round2_counts, third_flat, third_flat_min, arena_size, view) != 4)
    return 5;
  Round3 round3[arena_size]{};
  std::uint32_t round3_min[arena_size]{}, round3_counts[SyntheticTree::bucket_count] = {};
  overflow = 0;
  if (mom_equihash::scatter_collision_reference<Spec, Spec::rounds - 3, SyntheticTree>(
          third_flat, third_flat_min, 4, round3, round3_min, round3_counts, &overflow) != 4 ||
      overflow != 0)
    return 6;
  view.round3 = round3;
  view.round3_capacity = arena_size;

  Round4 fourth_flat[arena_size]{};
  std::uint32_t fourth_flat_min[arena_size]{};
  if (mom_equihash::collision_round_reference_checked<Spec, Spec::rounds - 3, SyntheticTree>(
          round3, round3_min, round3_counts, fourth_flat, fourth_flat_min, arena_size, view) != 2)
    return 7;
  Round4 round4[arena_size]{};
  std::uint32_t round4_min[arena_size]{}, round4_counts[SyntheticTree::bucket_count] = {};
  overflow = 0;
  if (mom_equihash::scatter_collision_reference<Spec, Spec::rounds - 4, SyntheticTree>(
          fourth_flat, fourth_flat_min, 2, round4, round4_min, round4_counts, &overflow) != 2 ||
      overflow != 0)
    return 8;
  view.round4 = round4;
  view.round4_capacity = arena_size;

  Root roots[arena_size]{};
  std::uint32_t root_min[arena_size]{};
  if (mom_equihash::collision_round_reference_checked<Spec, 2, SyntheticTree>(
          round4, round4_min, round4_counts, roots, root_min, arena_size, view) != 1)
    return 9;
  if (!mom_equihash::is_zero_root<Spec>(roots[0]))
    return 10;

  std::uint32_t leaves[Spec::proof_indices]{};
  std::uint8_t solution[Spec::solution_length]{}, expected_solution[Spec::solution_length]{};
  if (!mom_equihash::recover_solution<Spec>(view, roots[0], leaves, solution))
    return 11;
  for (unsigned i = 0; i < Spec::proof_indices; ++i)
    if (leaves[i] != i)
      return 12;
  mom_equihash::encode_indices<Spec>(leaves, expected_solution);
  if (std::memcmp(solution, expected_solution, sizeof(solution)) != 0)
    return 13;

  Root nonzero = roots[0];
  const std::uint32_t one[1] = {1};
  mom_equihash::store_collision<Spec, 1>(nonzero, one,
                                         mom_equihash::load_collision_left<Spec, 1>(roots[0]),
                                         mom_equihash::load_collision_right<Spec, 1>(roots[0]));
  if (mom_equihash::is_zero_root<Spec>(nonzero) ||
      mom_equihash::recover_solution<Spec>(view, nonzero, leaves, solution))
    return 14;

  Round4 overlap_round4[arena_size] = {};
  overlap_round4[0] = round4[0];
  overlap_round4[1] = round4[0];
  mom_equihash::ProvenanceView<Spec> overlap_view = view;
  overlap_view.round4 = overlap_round4;
  if (mom_equihash::recover_solution<Spec>(overlap_view, roots[0], leaves, solution))
    return 15;
  return 0;
}

template <typename Spec>
int test_blake2b_pair() {
  std::uint8_t header[Spec::header_length];
  for (unsigned i = 0; i < Spec::header_length; ++i)
    header[i] = static_cast<std::uint8_t>(11u + 37u * i);
  std::uint64_t midstate[8];
  mom_equihash::hash_header_midstate<Spec>(header, midstate);
  const std::uint64_t first = mom_equihash::load64_le(header + 128);
  const std::uint32_t second = mom_equihash::load32_le(header + 136);
  const std::uint32_t indices[] = {0u, 1u, Spec::hash_count / 2u, Spec::hash_count - 1u};
  for (unsigned i = 0; i < sizeof(indices) / sizeof(indices[0]); ++i) {
    std::uint64_t scalar[8], pair[8];
    mom_equihash::hash_words_from_tail<Spec>(midstate, first, second, indices[i], scalar);
    mom_equihash::hash_words_from_tail_pair(midstate, first, second, indices[i], pair);
    if (std::memcmp(scalar, pair, sizeof(scalar)) != 0)
      return static_cast<int>(i + 1u);
  }
  return 0;
}

} // namespace

int main() {
  using Spec = mom_equihash::ZHash144_5;
  static_assert(Spec::row_count == (1u << 25) && Spec::hash_count == 11184811);
  static_assert(Spec::level0_words == 6 && Spec::collision_record_words == 6);
  static_assert(mom_equihash::PackedCollisionRecord<Spec, Spec::rounds - 2>::word_count == 5);
  if (const int result = test_bucket_stage())
    return 5 + result;
  if (const int result = test_bucket_level0_packing())
    return 80 + result;
  if (const int result = test_round3_packing())
    return 100 + result;
  if (const int result = test_scatter_overflow_and_two_rounds())
    return 20 + result;
  if (const int result = test_complete_synthetic_tree())
    return 40 + result;
  if (const int result = test_blake2b_pair<mom_equihash::ZHash144_5>())
    return 60 + result;
  std::uint8_t header[Spec::header_length], solution[Spec::solution_length];
  from_hex("0400000008e9694cc2120ec1b5733cc12687b609058eec4f7046a521ad1d1e3049b400003e7420ed6f40659"
           "de0305ef9b7ec037f4380ed9848bc1c015691c90aa16ff39300000000000000000000000000000000000000"
           "00000000000000000000000000c9310d5874e0001f000000000000000000000000000000010b00000000000"
           "0000000000000666666",
           header);
  from_hex("01629b3779fd498defb2b0a551f7e111a8a003711acfe129622eb80bc98df66b9d8178b9670bacdc972b250"
           "fcb6715f437eb0addf858f9419c03f93a1be742e6377d4dcc4b9196afd811592ee4589cecfa321e7a9d5675"
           "338e7834923fe12b49f743a8d4",
           solution);
  if (!mom_equihash::verify_solution<Spec>(header, solution))
    return 1;
  std::uint32_t indices[Spec::proof_indices];
  mom_equihash::decode_indices<Spec>(solution, indices);
  const std::uint32_t first = indices[0];
  indices[0] = indices[1];
  indices[1] = first;
  mom_equihash::encode_indices<Spec>(indices, solution);
  if (mom_equihash::verify_solution<Spec>(header, solution))
    return 2;
  mom_equihash::encode_indices<Spec>(indices, solution);
  const std::uint32_t second = indices[0];
  indices[0] = indices[1];
  indices[1] = second;
  mom_equihash::encode_indices<Spec>(indices, solution);
  if (!mom_equihash::verify_solution<Spec>(header, solution))
    return 3;
  solution[0] ^= 0x80;
  if (mom_equihash::verify_solution<Spec>(header, solution))
    return 4;
  return 0;
}
